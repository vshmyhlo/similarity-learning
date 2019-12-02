from __future__ import absolute_import
from __future__ import division

import torch


def triplet_loss(input, target, margin=1.):
    distances = pairwise_distances(input, input)

    mask_pos = build_pos_mask(target)
    hardest_pos_dist = (distances * mask_pos.float()).max(dim=1)[0]

    mask_neg = build_neg_mask(target)
    max_neg_dist = distances.max(dim=1, keepdim=True)[0]
    hardest_neg_dist = (distances + max_neg_dist * (~mask_neg).float()).min(dim=1)[0]

    # loss = torch.log1p(torch.exp(hardest_pos_dist - hardest_neg_dist))
    loss = torch.clamp(hardest_pos_dist - hardest_neg_dist + margin, min=0)

    return loss


def lsep_loss(input, target):
    distances = pairwise_distances(input, input)

    mask_pos = build_pos_mask(target)
    mask_neg = build_neg_mask(target)

    dist_pos = distances[mask_pos]
    dist_neg = distances[mask_neg]

    dist_pos = dist_pos.unsqueeze(1)
    dist_neg = dist_neg.unsqueeze(0)

    loss = torch.log1p(torch.exp(dist_pos - dist_neg))

    return loss


def pairwise_distances(a, b):
    delta = a.unsqueeze(1) - b.unsqueeze(0)
    return torch.norm(delta, 2, 2)


# def pairwise_distances(a, b):
#     a = F.normalize(a, 2, 1)
#     b = F.normalize(b, 2, 1)
#
#     dot = (a.unsqueeze(1) * b.unsqueeze(0)).sum(2)
#     # dist = torch.acos(dot * (1 - 1e-7)) / math.pi
#     dist = dot
#
#     return dist


def build_pos_mask(target):
    indices_equal = torch.eye(target.size(0), dtype=torch.bool, device=target.device)
    index_not_eq = ~indices_equal

    target_eq = torch.eq(target.unsqueeze(1), target.unsqueeze(0))
    mask = index_not_eq & target_eq

    return mask


def build_neg_mask(target):
    indices_equal = torch.eye(target.size(0), dtype=torch.bool, device=target.device)
    index_not_eq = ~indices_equal

    target_not_eq = torch.ne(target.unsqueeze(1), target.unsqueeze(0))
    mask = index_not_eq & target_not_eq

    return mask
