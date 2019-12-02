import os

import pandas as pd
import torch.utils.data
from PIL import Image

from utils import encode_category


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        row = self.data.loc[index]

        image = Image.open(row['path'])

        input = {
            'image': image,
            'id': row['id'],
        }

        if self.transform is not None:
            input = self.transform(input)

        return input

    def __len__(self):
        return len(self.data)

    @property
    def ids(self):
        return self.data['id']


class DatasetBuilder(object):
    def __init__(self, path):
        self.train, self.query, self.gallery = load_data(path)

    def build_train(self, **kwargs):
        return Dataset(self.train, **kwargs)

    def build_query(self, **kwargs):
        return Dataset(self.query, **kwargs)

    def build_gallery(self, **kwargs):
        return Dataset(self.gallery, **kwargs)


def load_data(path):
    train = load_subset(os.path.join(path, 'peopleDevTrain.txt'))
    test = load_subset(os.path.join(path, 'peopleDevTest.txt'))

    train['path'] = path
    test['path'] = path

    train, = flatten_and_split(train, 1)
    query, gallery = flatten_and_split(test, 2)

    encode_category([train], 'id')
    encode_category([query, gallery], 'id')

    return train, query, gallery


def load_subset(path):
    with open(path) as f:
        next(f)
        data = pd.read_csv(f, sep='\s+', names=['id', 'num_images'])

    return data


def flatten_and_split(data, num_splits):
    splits = [[] for _ in range(num_splits)]

    for _, row in data.iterrows():
        if row['num_images'] < num_splits:
            continue
       
        group = pd.DataFrame({
            'path': [
                os.path.join(row['path'], row['id'], '{}_{:04}.jpg'.format(row['id'], i + 1))
                for i in range(row['num_images'])],
        })
        group['id'] = row['id']

        for i, split in enumerate(splits):
            split.append(group.iloc[i::num_splits])

    splits = [pd.concat(split) for split in splits]
    for split in splits:
        split.index = range(len(split))

    return splits
