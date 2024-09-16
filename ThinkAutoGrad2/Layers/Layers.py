from ..Tensor import Tensor, check_grad_outs
import numpy as n


# 1级测试
class Flatten:
    def __init__(self, x):
        self.x = x
        self.x_shape = x.arr.shape

    def forward(self):
        import functools
        z = self.x.reshape((self.x_shape[0], functools.reduce(lambda x, y: x * y, self.x_shape[1:])))
        return z


# 1级测试
class UpSample2d:
    def __init__(self, x, stride):
        self.x = x
        self.x_shape = x.shape
        self.stride_hw = stride

    def forward(self):
        inputs = self.x.arr
        n_samples, channels, height, width = self.x_shape
        stride_h, stride_w = self.stride_hw
        z = n.zeros((n_samples, channels, height * stride_h, width * stride_w))
        for hi in range(height):
            for wi in range(width):
                z[:, :, hi*stride_h:(hi+1)*stride_h, wi*stride_w:(wi+1)*stride_w] = \
                    n.ones((n_samples, channels, stride_h, stride_w)) * inputs[:, :, hi:hi+1, wi:wi+1]
        z = Tensor(z, self, (self.x, ))
        return z

    @check_grad_outs
    def backward(self, grad):
        n_samples, channels, grad_height, grad_width = grad.shape
        stride_h, stride_w = self.stride_hw
        if grad_height % stride_h != 0 or grad_width % stride_w != 0:
            raise Exception('error')
        out_height = int(grad_height / stride_h)
        out_width = int(grad_width / stride_w)
        gz = n.zeros((n_samples, channels, out_height, out_width))
        for hi in range(out_height):
            for wi in range(out_width):
                temp = n.sum(grad[:, :, hi*stride_h:(hi+1)*stride_h, wi*stride_w:(wi+1)*stride_w], axis=3, keepdims=True)
                temp = n.sum(temp, axis=2, keepdims=True)
                gz[:, :, hi:hi+1, wi:wi+1] = temp
        return (gz, )










