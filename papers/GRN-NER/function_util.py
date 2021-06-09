import torch


def log_sum_exp(x, dim=None, keepdim=False):
    """
    checked
    Calculate the log of the sum of the exponential of x, along dimension "dim"
    :param x: tensor
    :param dim: int, dimension index
    :param keepdim: bool, keep the size or not
    :return: log of the sum of the exponential of x, along dimension "dim"
    """
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    x = torch.where(
        (xm == float('inf')) | (xm == float('-inf')),
        xm,
        xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True)))
    return x if keepdim else x.squeeze(dim)
