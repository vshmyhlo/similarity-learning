import glob
import os

import pandas as pd
import torch.utils.data
from PIL import Image
from sklearn.preprocessing import LabelEncoder


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
            'cam': row['cam'],
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
    train = load_folder(os.path.join(path, 'bounding_box_train'))
    query = load_folder(os.path.join(path, 'query'))
    gallery = load_folder(os.path.join(path, 'bounding_box_test'))

    encode_category([train], 'id')
    encode_category([query, gallery], 'id')
    encode_category([train, query, gallery], 'cam')

    return train, query, gallery


def load_folder(path):
    data = pd.DataFrame({
        'path': sorted(glob.glob(os.path.join(path, '*.jpg'))),
    })

    meta = data['path'].apply(lambda x: os.path.split(x)[1].split('_')[:2])

    data['id'], data['cam'] = zip(*meta)
    data['id'] = data['id'].apply(lambda x: int(x))
    data['cam'] = data['cam'].apply(lambda x: int(x[1]))

    data = data[data['id'] != -1]
    data.index = range(len(data))

    return data


def encode_category(dfs, column):
    le = LabelEncoder()
    le.fit(pd.concat([df[column] for df in dfs]))
    for df in dfs:
        df[column] = le.transform(df[column])
