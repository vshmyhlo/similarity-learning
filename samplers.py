import math
import random

import torch.utils


class RandomIdentityBatchSampler(torch.utils.data.Sampler):
    def __init__(self, data, batch_size, drop_last, num_instances=4, balance_identities=False):
        super().__init__(data)

        assert batch_size % num_instances == 0

        self.data = data
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_instances = num_instances
        self.balance_identities = balance_identities

    def __len__(self):
        batches = self.build_batches()

        return len(batches)

    def __iter__(self):
        batches = self.build_batches()

        return iter(batches)

    def build_batches(self):
        id_to_indices = [[] for _ in range(self.data.max() + 1)]
        for index, id in self.data.iteritems():
            id_to_indices[id].append(index)

        if self.balance_identities:
            id_to_indices = self.balance(id_to_indices)

        batches = []
        batch = []
        while len(id_to_indices) > 0:
            random.shuffle(id_to_indices)
            for indices in id_to_indices:
                random.shuffle(indices)

            for indices in id_to_indices:
                num_instances = min(self.num_instances, len(indices))
                for _ in range(num_instances):
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

    def balance(self, id_to_indices):
        instances_per_identity = round(len(self.data) / len(id_to_indices))
        instances_per_identity = math.ceil(instances_per_identity / self.num_instances) * self.num_instances
        id_to_indices = [
            (indices * math.ceil(instances_per_identity / len(indices)))[:instances_per_identity]
            for indices in id_to_indices]

        return id_to_indices
