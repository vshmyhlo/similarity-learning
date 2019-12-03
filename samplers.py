import math
import random

import numpy as np
import torch.utils


class RandomIdentityBatchSampler(torch.utils.data.Sampler):
    def __init__(self, data, batch_size, drop_last, num_instances=4, pad_instances=False):
        super().__init__(data)

        assert batch_size % num_instances == 0

        self.data = data
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_instances = num_instances
        self.pad_instances = pad_instances

    def __len__(self):
        batches = self.build_batches()

        return len(batches)

    def __iter__(self):
        batches = self.build_batches()

        return iter(batches)

    def build_batches(self):
        def pad(indices):
            target_len = math.ceil(len(indices) / self.num_instances) * self.num_instances
            pad_len = target_len - len(indices)
            padding = np.random.choice(
                indices,
                pad_len,
                replace=pad_len > len(indices)).tolist()
            return indices + padding

        id_to_indices = [[] for _ in range(self.data.max() + 1)]
        for index, id in self.data.iteritems():
            id_to_indices[id].append(index)

        if self.pad_instances:
            id_to_indices = [pad(indices) for indices in id_to_indices]

        batches = []
        batch = []
        while len(id_to_indices) > 0:
            random.shuffle(id_to_indices)
            for indices in id_to_indices:
                random.shuffle(indices)

            for indices in id_to_indices:
                assert len(indices) % self.num_instances == 0
                for _ in range(self.num_instances):
                    batch.append(indices.pop())

                    if len(batch) == self.batch_size:
                        batches.append(batch)
                        batch = []

            id_to_indices = [indices for indices in id_to_indices if len(indices) > 0]
        batches.append(batch)
        del batch

        if self.drop_last:
            batches = [batch for batch in batches if len(batch) == self.batch_size]

        random.shuffle(batches)

        return batches
