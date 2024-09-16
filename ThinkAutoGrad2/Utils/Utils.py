from ThinkAutoGrad2.Tensor import Tensor, check_grad_outs
import numpy as n


# 1级测试
class Concat:
    def __init__(self, xls, axis):
        self.xls = xls
        self.axis = axis

    def forward(self):
        z = Tensor(n.concatenate([i.arr for i in self.xls], self.axis), self, self.xls)
        return z

    @check_grad_outs
    def backward(self, grad):
        shape_ls = [i.arr.shape[self.axis] for i in self.xls]
        shape_len = len(self.xls[0].arr.shape)

        shape_new = 0
        gz = []
        head = [slice(None, None, None) for i in range(self.axis)]
        tail = [slice(None, None, None) for i in range(shape_len - self.axis - 1)]
        for s in shape_ls:
            shape_last = shape_new
            shape_new = shape_new + s
            this = [slice(shape_last, shape_new, )]
            slice_ = tuple(head + this + tail)
            gz.append(grad[slice_])
        return tuple(gz)


# 1级
class Exp:
    def __init__(self, x):
        self.x = x
        self.shape = x.shape

    def forward(self):
        z = Tensor(n.exp(self.x.arr), self, (self.x,))
        return z

    @check_grad_outs
    def backward(self, grad):
        gz = grad * n.exp(self.x.arr)
        return (gz, )


# 1级
class Log:
    def __init__(self, x):
        self.x = x
        self.shape = x.shape

    def forward(self):
        z = Tensor(n.log(self.x.arr), self, (self.x,))
        return z

    @check_grad_outs
    def backward(self, grad):
        gz = grad / self.x.arr
        return (gz, )
    

# 0级
class Sqrt:
    def __init__(self, x):
        self.x = x
        self.shape = x.shape

    def forward(self):
        z = Tensor(n.sqrt(self.x.arr), self, (self.x,))
        return z
    
    @check_grad_outs
    def backward(self, grad):
        gz = grad * 0.5 * self.x.arr**0.5
        return (gz, )


# 1级
class Repeat:
    def __init__(self, x, reps, axis):
        self.x = x
        self.x_shape = x.shape
        self.reps = reps
        self.axis = axis

    def forward(self):
        z = Tensor(n.repeat(self.x.arr, self.reps, self.axis), self, (self.x,))
        return z

    @check_grad_outs
    def backward(self, grad):
        gz = n.zeros(self.x_shape, dtype='float32')
        sli1 = [slice(None)] * self.axis + \
              [None] + \
              [slice(None)] * (len(self.x_shape) - self.axis - 1)
        sli2 = [slice(None)] * self.axis + \
              [None] + \
              [slice(None)] * (len(self.x_shape) - self.axis - 1)
        s = self.x_shape[self.axis]
        for i in range(s):
            sli1_ = slice(i, i+1, None)
            sli1[self.axis] = sli1_
            sli2_ = slice(i, None, s)
            sli2[self.axis] = sli2_
            gz[tuple(sli1)] = n.sum(grad[tuple(sli2)], axis=self.axis, keepdims=True)

        return (gz,)


# 1级
class Sum:
    def __init__(self, x, axis):
        self.x = x
        self.x_shape = x.shape
        self.axis = axis
        
    def forward(self):
        z = Tensor(n.sum(self.x.arr, self.axis, keepdims=True), self, (self.x,))
        return z

    @check_grad_outs
    def backward(self, grad):
        gz = n.repeat(grad, self.x_shape[self.axis], axis=self.axis)
        return (gz,)


# 0级
class Tile:
    def __init__(self, x, reps):
        self.x = x
        self.reps = reps
        self.x_shape = x.shape

    def forward(self):
        z = Tensor(n.tile(self.x.arr, self.reps), self, (self.x,))
        return z

    @check_grad_outs
    def backward(self, grad):
        import functools
        flg, res_ls = self.walk(self.reps, 0, [], grad, [])
        gz = functools.reduce(lambda x, y: x + y, res_ls)
        return (gz,)

    def walk(self, s, i, sli_ls, ori, res_ls):
        if i >= len(s):
            return True, tuple(sli_ls)
        for j in range(s[i]):
            sli_ls.append(slice(j * self.x_shape[i], (j + 1) * self.x_shape[i], None))
            flg, sli_tup = self.walk(s, i + 1, sli_ls, ori, res_ls)
            if flg:
                res_ls.append(ori[sli_tup])
            sli_ls.pop(-1)
        return False, res_ls











