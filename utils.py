import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


def visualize_ranks(query_images, gallery_images, sort_indices, eq, k):
    def denormalize(images):
        return (images - min_pixel) / (max_pixel - min_pixel)

    def add_border(images, color, border_size=3):
        bn = images.size()[:-3]
        if len(bn) == 2:
            images = images.view(bn[0] * bn[1], *images.size()[-3:])

        # edge = torch.zeros(images.size(0), 3, 1, 1, dtype=images.dtype, device=images.device)
        border = torch.tensor(color, dtype=images.dtype, device=images.device) \
            .view(1, 3, 1, 1).repeat(images.size(0), 1, 1, 1)

        images = torch.cat([
            # edge.repeat(1, 1, 1, images.size(3)),
            border.repeat(1, 1, border_size + 1, images.size(3)),
            images,
            border.repeat(1, 1, border_size + 1, images.size(3)),
            # edge.repeat(1, 1, 1, images.size(3)),
        ], 2)
        images = torch.cat([
            # edge.repeat(1, 1, images.size(2), 1),
            border.repeat(1, 1, images.size(2), border_size + 1),
            images,
            border.repeat(1, 1, images.size(2), border_size + 1),
            # edge.repeat(1, 1, images.size(2), 1),
        ], 3)

        images[:, :, [0, -1], :] = 0.
        images[:, :, :, [0, -1]] = 0.

        if len(bn) == 2:
            images = images.view(*bn, *images.size()[-3:])

        return images

    eq = eq[:, :k]
    gallery_images = gallery_images[sort_indices[:, :k]]

    min_pixel = min(query_images.min(), gallery_images.min())
    max_pixel = max(query_images.max(), gallery_images.max())
    query_images = denormalize(query_images)
    gallery_images = denormalize(gallery_images)

    query_images = add_border(query_images, (0, 0, 0))
    gallery_images = torch.where(
        eq.view(*eq.size(), 1, 1, 1),
        add_border(gallery_images, (0, 1, 0)),
        add_border(gallery_images, (1, 0, 0)))

    gallery_images = gallery_images.permute(0, 2, 3, 1, 4)
    b, c, h, n, w = gallery_images.size()
    gallery_images = gallery_images.contiguous().view(b, c, h, n * w)

    images = torch.cat([query_images, gallery_images], 3)
    images = images.permute(1, 0, 2, 3)
    c, n, h, w = images.size()
    images = images.contiguous().view(c, n * h, w)

    return images


def cmc_curve_plot(cmc):
    fig = plt.figure()
    plt.plot(np.arange(1, cmc.shape[0] + 1), cmc)
    plt.xlim(0, cmc.shape[0] + 1)
    plt.ylim(0, 1)

    return fig


def distance_plot(distances, eq):
    fig = plt.figure()

    kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40, ec='k')
    plt.hist(distances[eq].data.cpu().numpy(), label='pos', **kwargs)
    plt.hist(distances[~eq].data.cpu().numpy(), label='neg', **kwargs)
    plt.legend()

    return fig


def encode_category(dfs, column):
    le = LabelEncoder()
    le.fit(pd.concat([df[column] for df in dfs]))
    for df in dfs:
        df[column] = le.transform(df[column])
