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
    train = load_subset(path, 'train')
    query = load_subset(path, 'probe')
    gallery = load_subset(path, 'gallery')

    encode_category([train], 'id')
    encode_category([query, gallery], 'id')
    encode_category([train, query, gallery], 'cam')

    return train, query, gallery


def load_subset(root, subset):
    data = pd.read_csv(
        os.path.join(root, 'vric_{}.txt'.format(subset)),
        sep='\s+',
        names=['path', 'id', 'cam'])
    data['path'] = data['path'].apply(lambda x: os.path.join(root, '{}_images'.format(subset), x))
    data.index = range(len(data))

    return data
