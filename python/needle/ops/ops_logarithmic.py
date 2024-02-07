from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        if self.axes is None:
            self.axes = tuple(range(len(Z.shape)))
        max_Z = array_api.max(Z, axis=self.axes)
        max_Z_keepdim = array_api.max(Z, axis=self.axes, keepdims=True)
        return array_api.log(array_api.sum(array_api.exp(Z - max_Z_keepdim), axis=self.axes)) + max_Z

    def gradient(self, out_grad, node):
        Z = node.inputs[0]
        exp_t = exp(Z - Z.realize_cached_data().max(axis=self.axes, keepdims=True))
        sum_t = summation(exp_t, axes=self.axes)

        g1 = out_grad / sum_t

        if not self.axes:
            g2 = broadcast_to(g1, exp_t.shape)
        else:
            exp_t_shape = list(exp_t.shape)
            for i in self.axes:
                exp_t_shape[i] = 1
            exp_t_shape = tuple(exp_t_shape)
            g2 = broadcast_to(g1.reshape(exp_t_shape), exp_t.shape)

        return g2 * exp_t



def logsumexp(a, axes=None):
    if axes is None:
        axes = tuple(range(len(a.shape)))
    max_a = maximum(a, axes)
    max_a_keepdim = broadcast_to(reshape(max_a, [1 if i in axes else a.shape[i] for i in range(len(a.shape))]), a.shape)
    return log(summation(exp(a - max_a_keepdim), axes)) + max_a

