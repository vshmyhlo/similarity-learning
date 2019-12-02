import math
import random

import numpy as np
import torch.utils


class RandomIdentityBatchSampler(torch.utils.data.Sampler):
    def __init__(self, data, batch_size, drop_last, fill_with, num_identities=4):
        super().__init__(data)

        assert batch_size % num_identities == 0

        self.data = data
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.fill_with = fill_with
        self.num_instances = num_identities

    def __len__(self):
        if self.fill_with == 'instance':
            num_batches = self.data.nunique() / (self.batch_size / self.num_instances)
        elif self.fill_with == 'identity':
            num_batches = len(self.data) / self.batch_size
        else:
            raise AssertionError('invalid fill_with {}'.format(self.fill_with))

        if self.drop_last:
            return math.floor(num_batches)
        else:
            return math.ceil(num_batches)

    def __iter__(self):
        if self.fill_with == 'instance':
            batches = self.filled_with_instance()
        elif self.fill_with == 'identity':
            batches = self.filled_with_identity()
        else:
            raise AssertionError('invalid fill_with {}'.format(self.fill_with))

        if self.drop_last:
            batches = [batch for batch in batches if len(batch) == self.batch_size]

        assert len(batches) == len(self)
        random.shuffle(batches)

        return iter(batches)

    def filled_with_instance(self):
        id_to_indices = [[] for _ in range(self.data.max() + 1)]
        for index, id in self.data.iteritems():
            id_to_indices[id].append(index)

        for indices in id_to_indices:
            random.shuffle(indices)
        random.shuffle(id_to_indices)

        batches = []
        batch = []
        for indices in id_to_indices:
            replace = len(indices) < self.num_instances
            batch.extend(np.random.choice(indices, self.num_instances, replace=replace))

            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []

        batches.append(batch)
        del batch

        return batches

    def filled_with_identity(self):
        id_to_indices = [[] for _ in range(self.data.max() + 1)]
        for index, id in self.data.iteritems():
            id_to_indices[id].append(index)

        for indices in id_to_indices:
            random.shuffle(indices)
        random.shuffle(id_to_indices)

        batches = []
        batch = []
        while len(id_to_indices) > 0:
            for indices in id_to_indices:
                num_identities = min(self.num_identities, len(indices))
                for _ in range(num_identities):
                    batch.append(indices.pop())

                    if len(batch) == self.batch_size:
                        batches.append(batch)
                        batch = []

            id_to_indices = [indices for indices in id_to_indices if len(indices) > 0]
        batches.append(batch)
        del batch

        return batches
