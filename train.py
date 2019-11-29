import logging
import math
import os

import click
import numpy as np
import torch.utils.data
import torchvision.transforms as T
from all_the_tools.config import Config
from all_the_tools.metrics import Mean
from all_the_tools.transforms import ApplyTo, Extract
from tensorboardX import SummaryWriter
from ticpfptp.torch import fix_seed
from tqdm import tqdm

import data_builders.market1501
from losses import triplet_loss
from metrics import rank_k, mean_average_precision
from models.resnet import ResNet
from samplers import RandomIdentityBatchSampler
# TODO: visualize ranks
# TODO: remove logging
from transforms import CheckSize

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


@click.command()
@click.option('--experiment-path', type=click.Path(), default='./tf_log')
@click.option('--dataset-path', type=click.Path(), required=True)
@click.option('--config-path', type=click.Path(), required=True)
@click.option('--restore-path', type=click.Path())
@click.option('--workers', type=click.INT, default=os.cpu_count())
def main(experiment_path, dataset_path, config_path, restore_path, workers):
    logging.basicConfig(level=logging.INFO)
    config = Config.from_json(config_path)
    fix_seed(config.seed)

    train_transform = T.Compose([
        ApplyTo('image', T.Compose([
            CheckSize((128, 64)),

            # T.RandomCrop((96, 48)),
            # T.Resize((128, 64)),

            T.RandomHorizontalFlip(),

            T.ColorJitter(0.1, 0.1, 0.1),

            T.ToTensor(),
        ])),
        Extract(['image', 'id'])
    ])
    eval_transform = T.Compose([
        ApplyTo('image', T.Compose([
            CheckSize((128, 64)),
            T.ToTensor()
        ])),
        Extract(['image', 'id'])
    ])

    dataset_builder = data_builders.market1501.DatasetBuilder(dataset_path)
    datasets = {
        'train': dataset_builder.build_train(transform=train_transform),
        'query': dataset_builder.build_query(transform=eval_transform),
        'gallery': dataset_builder.build_gallery(transform=eval_transform),
    }
    data_loaders = {
        'train': torch.utils.data.DataLoader(
            datasets['train'],
            batch_sampler=RandomIdentityBatchSampler(datasets['train'].ids, config.train.batch_size),
            num_workers=workers),
        'query': torch.utils.data.DataLoader(
            datasets['query'],
            batch_size=config.eval.batch_size,
            num_workers=workers),
        'gallery': torch.utils.data.DataLoader(
            datasets['gallery'],
            batch_size=config.eval.batch_size,
            num_workers=workers),
    }

    model = ResNet(34)
    model.to(DEVICE)

    optimizer = build_optimizer(model.parameters(), config)
    scheduler = build_scheduler(optimizer, len(data_loaders['train']), config)

    # ==================================================================================================================
    # main loop

    train_writer = SummaryWriter(os.path.join(experiment_path, 'train'))
    eval_writer = SummaryWriter(os.path.join(experiment_path, 'eval'))
    # best_score = 0

    for epoch in range(config.epochs):
        if epoch % 10 == 0:
            logging.info(experiment_path)

        # ==============================================================================================================
        # training

        metrics = {
            'loss': Mean(),
        }

        model.train()
        for images, ids in tqdm(data_loaders['train'], desc='epoch {} train'.format(epoch), smoothing=0.01):
            images, ids = images.to(DEVICE), ids.to(DEVICE)

            features = model(images)

            loss = compute_loss(features, ids)
            metrics['loss'].update(loss.data.cpu().numpy())

            lr = np.squeeze(scheduler.get_lr())

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            metrics = {k: metrics[k].compute_and_reset() for k in metrics}
            logging.info('[EPOCH {}][TRAIN] {}'.format(
                epoch, ', '.join('{}: {:.4f}'.format(k, metrics[k]) for k in metrics)))
            for k in metrics:
                train_writer.add_scalar(k, metrics[k], global_step=epoch)
            train_writer.add_scalar('learning_rate', lr, global_step=epoch)

        # ==============================================================================================================
        # evaluation

        metrics = {
            'loss': Mean(),
            'mAP': Mean(),
            'rank/1': Mean(),
            'rank/5': Mean(),
            'rank/10': Mean(),
        }

        visualization_samples = []

        model.eval()
        with torch.no_grad():
            gallery_images, gallery_ids, gallery_features = collect_features(data_loaders['gallery'], model)

            for images, ids in tqdm(data_loaders['query'], desc='epoch {} eval'.format(epoch), smoothing=0.01):
                images, ids = images.to(DEVICE), ids.to(DEVICE)

                features = model(images)

                loss = compute_loss(features, ids)
                metrics['loss'].update(loss.data.cpu().numpy())

                distances = compute_distance(features, gallery_features)
                metric, sort_indices = compute_metric(distances, ids, gallery_ids)
                for k in metric:
                    metrics[k].update(metric[k].data.cpu().numpy())

                visualization_samples.append((images[0], sort_indices[0]))
                visualization_samples = visualization_samples[:32]

            metrics = {k: metrics[k].compute_and_reset() for k in metrics}
            logging.info('[EPOCH {}][EVAL] {}'.format(
                epoch, ', '.join('{}: {:.4f}'.format(k, metrics[k]) for k in metrics)))
            for k in metrics:
                eval_writer.add_scalar(k, metrics[k], global_step=epoch)

            images, sort_indices = [torch.stack(x, 0) for x in zip(*visualization_samples[:32])]
            eval_writer.add_image(
                'images',
                visualize_ranks(images, gallery_images, sort_indices, k=10),
                global_step=epoch)

            del images, features, gallery_images, gallery_features
           
        # saver.save(os.path.join(experiment_path, 'model.pth'))
        # if metrics['wer'] < best_score:
        #     best_score = metrics['wer']
        #     save_model(model_to_save, mkdir(os.path.join(experiment_path, 'best')))


def build_optimizer(parameters, config):
    if config.train.opt.type == 'adam':
        optimizer = torch.optim.Adam(parameters, config.train.opt.lr, weight_decay=1e-4)
    elif config.train.opt.type == 'sgd':
        optimizer = torch.optim.SGD(parameters, config.train.opt.lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise AssertionError('invalid config.opt.type {}'.format(config.train.opt.type))

    return optimizer


def build_scheduler(optimizer, epoch_size, config):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_size * config.epochs)

    return scheduler


def visualize_ranks(query_images, gallery_images, sort_indices, k):
    gallery_images = gallery_images[sort_indices[:, :k]]

    gallery_images = gallery_images.permute(0, 2, 3, 1, 4)
    b, c, h, n, w = gallery_images.size()
    gallery_images = gallery_images.contiguous().view(b, c, h, n * w)

    images = torch.cat([query_images, gallery_images], 3)
    images = images.permute(1, 0, 2, 3)
    c, n, h, w = images.size()
    images = images.contiguous().view(c, n * h, w)

    return images


def compute_loss(input, target):
    loss = triplet_loss(input, target)
    # loss = lsep_loss(input, target)

    return loss


def compute_metric(distances, query_ids, gallery_ids):
    sort_indices = distances.argsort(1)
    sorted_gallery_ids = gallery_ids[sort_indices]
    eq = query_ids.unsqueeze(1) == sorted_gallery_ids

    metric = {
        'mAP': mean_average_precision(eq),
        'rank/1': rank_k(eq, 1),
        'rank/5': rank_k(eq, 5),
        'rank/10': rank_k(eq, 10),
    }

    return metric, sort_indices


def compute_distance(a, b):
    return torch.norm(a.unsqueeze(1) - b.unsqueeze(0), 2, 2)


def collect_features(data_loader, model):
    all_images = []
    all_ids = []
    all_features = []
    for images, ids in tqdm(data_loader, desc='collecting features'):
        images, ids = images.to(DEVICE), ids.to(DEVICE)
        features = model(images)

        all_images.append(images)
        all_ids.append(ids)
        all_features.append(features)

    all_images = torch.cat(all_images, 0)
    all_ids = torch.cat(all_ids, 0)
    all_features = torch.cat(all_features, 0)

    return all_images, all_ids, all_features


def compute_nrow(images):
    b, _, h, w = images.size()
    nrow = math.ceil(math.sqrt(h * b / w))

    return nrow


if __name__ == '__main__':
    main()
