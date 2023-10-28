import numpy as n
from ThinkAutoGrad2.Tensor import Tensor, check_grad_outs
from ThinkAutoGrad2 import Utils


def check_losses_inps_shape_equal(func):
    def wrap(self, prds, labs, *args, **kwargs):
        prds_shape = tuple(prds.shape)
        labs_shape = tuple(labs.shape)
        if prds_shape != labs_shape:
            raise Exception('prds与labs必须相同: prds.shape={} labs.shape={}'.format(prds_shape, labs_shape))
        ret = func(self, prds, labs, *args, **kwargs)
        return ret
    return wrap


# 2级
class MSE:
    @check_losses_inps_shape_equal
    def __init__(self, prds, labs):
        self.prds = prds
        self.labs = labs

    def forward(self):
        prds = self.prds
        labs = self.labs
        c = Tensor(n.array([1 / prds.shape[0], ]))
        loss = c * (labs - prds) * (labs - prds)
        return loss


# 1级
class CrossEntropyLoss:
    @check_losses_inps_shape_equal
    def __init__(self, prds, labs, axis):
        self.prds = prds
        self.labs = labs
        self.axis = axis

    def forward(self):
        prds = self.prds
        labs = self.labs
        axis = self.axis
        batch = prds.shape[0]
        prds_max = n.max(prds.arr, axis=axis, keepdims=True)
        prds_max = n.repeat(prds_max, prds.shape[axis], axis=axis)
        prds_max = Tensor(prds_max)
        prds = (prds - prds_max)
        eps = Utils.exp(prds)
        sum_p = Utils.sum(eps, axis=1)
        sum_p = Utils.repeat(sum_p, prds.shape[axis], axis=1)
        log_softmax = prds - Utils.log(sum_p)
        nll = Tensor(n.array([0.0])) - labs * log_softmax
        c = Tensor(n.array([1 / batch]))
        ret = c * nll
        return ret


# 1级
# 该CrossEntropyLoss参照网络代码实现,问题在于输入和梯度形状不一致
class CrossEntropyLoss2:
    @check_losses_inps_shape_equal
    def __init__(self, prds, labs, axis):
        self.prds = prds
        self.labs = labs
        self.axis = axis

    def __call__(self):
        prds = self.prds
        labs = self.labs
        axis = self.axis
        batch = prds.shape[0]
        eps = n.exp(prds.arr - n.max(prds.arr, axis=axis, keepdims=True))
        p = eps / n.sum(eps, axis=axis, keepdims=True)
        nll = - n.log(n.sum(p * labs.arr, axis=axis))
        nll = nll / batch
        z = Tensor(nll, self, (prds,))
        return z

    @check_grad_outs
    def backward(self, grad):
        batch = self.prds.shape[0]
        gz = self.prds.arr          # 可以不需要grad
        gz -= self.labs.arr
        gz = gz / batch
        return (gz,)










