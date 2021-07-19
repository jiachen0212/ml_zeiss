# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import _reduction as _Reduction


class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class my_mse_loss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(my_mse_loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # 重点频段加权
        # a, b = input.detach().numpy(), target.detach().numpy()
        # a, b = input.data, target.data
        diff = input - target
        mse = torch.square(diff)
        weights = [2, 1.5, 1, 1, 1.5, 2, 1.5, 1, 1, 1, 1, 1.5, 2, 1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5, 2, 1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
, 1.5, 2, 1.5, 1, 1, 1, 1.5, 2]
        weights = torch.Tensor(np.array(weights))
        mse *= weights
        mse = torch.mean(mse)
        return mse



class my_mse_loss1(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(my_mse_loss1, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = input - target
        mse = torch.square(diff)
        weights = [2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
        1, 2, 2, 2]
        weights = torch.Tensor(np.array(weights))
        mse *= weights
        mse = torch.mean(mse)
        return mse



def topn_sample(diff, n):
    diff_sort = np.sort(diff)
    diff = diff.tolist()
    res = []
    for a in diff_sort[:n]:
        res.append(diff.index(a))
    return res


class my_mse_loss2(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(my_mse_loss2, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor, n: int) -> Tensor:
        diff = input - target
        diff_square = np.square(diff.detach().numpy())
        diff_square = np.mean(diff_square, axis=1)
        res = topn_sample(diff_square, n)
        mse = torch.square(diff)
        weights = [1]*4
        weights = torch.Tensor(np.array(weights))
        mse *= weights
        mse = torch.mean(mse)
        return mse, res



class my_mse_loss3(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(my_mse_loss3, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = input - target
        mse = torch.square(diff)
        weights = [2,1,2,1]
        weights = torch.Tensor(np.array(weights))
        mse *= weights
        mse = torch.mean(mse)
        return mse

# w = [1]*81
# ind = [0,5,12,52,74,78, 79, 80]
# for i in ind:
#     w[i] = 2
# # ind = [1,4,6,11,13,51,53,73,75,79]
# # for i in ind:
# #     w[i] = 1.5
# print(w)