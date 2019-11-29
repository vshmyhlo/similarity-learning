import torch


def rank_k(eq, k):
    return torch.any(eq[:, :k], 1)


def cmc(eq, k):
    return eq[:, :k].float().cumsum(1) > 0.


def map(eq):
    eq = eq.float()

    cs = eq.cumsum(1)
    cs = cs * eq
    cs = cs / torch.arange(1, eq.size(1) + 1, dtype=torch.float, device=eq.device).unsqueeze(0)

    map = cs.sum(1) / eq.sum(1)

    return map
