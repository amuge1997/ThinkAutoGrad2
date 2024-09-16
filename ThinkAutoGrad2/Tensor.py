import numpy as n


# 张量的定义与实现, 张量的算子的定义与实现, 张量的成员函数的定义与实现操作

# 测试级别
# 0. 无测试
# 1. 简单算例测试
# 2. 一般应用算例测试
# 3. 复杂应用算例测试

# 广播机制:
# 默认高维度数组的后部维度与低维度数组相同


class Tensor:
    def __init__(self, arr, opt=None, ts_tp=tuple(), require_grad=False):
        self.shape = arr.shape
        self.arr = arr.astype('float32')
        self.grad = n.zeros(self.shape, dtype='float32')
        self.opt = opt          # 算子
        self.ts_ls = ts_tp      # 子元组
        self.is_grad = require_grad  # 是否记录梯度

    def __add__(self, other):
        return Add(self, other).forward()
    
    def __sub__(self, other):
        return Sub(self, other).forward()
    
    def __mul__(self, other):
        return Mul(self, other).forward()
    
    def __truediv__(self, other):
        return Div(self, other).forward()
    
    def __matmul__(self, other):
        return Matmul(self, other).forward()
    
    def __str__(self):
        return self.arr.__str__()
    
    def __getitem__(self, sc):
        return Slice(self, sc).forward()

    def copy(self):
        return copy(self.arr)
    
    def reshape(self, new_s):
        return Reshape(self, new_s).forward()
    
    def transpose(self, new_s):
        return Transpose(self, new_s).forward()

    def grad_zeros(self):
        self.grad = n.float32(n.zeros(self.shape))

    def numpy(self):
        return self.arr

    def is_require_grad(self, select):
        if select:
            self.is_grad = True
        else:
            self.is_grad = False

    def backward(self, grad):
        if self.is_grad:
            self.grad += grad

        if self.opt is not None:
            grads = self.opt.backward(grad)
            for i, v in enumerate(self.ts_ls):
                v.backward(grads[i])


def check_grad_outs(func):
    def wrap(self, grad):
        ret = func(self, grad)
        if not isinstance(ret, tuple):
            raise Exception('返回的梯度必须是元组形式')
        return ret
    return wrap


class Operator:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        self.x_shape = x.arr.shape
        if y is not None:
            self.y_shape = y.arr.shape

    def forward(self):
        pass

    def __call__(self):
        pass

    @check_grad_outs
    def backward(self, grad):
        pass

    def inputs_check(self):
        pass


# 3级测试
class Add(Operator):
    def __init__(self, x, y):
        super().__init__(x, y)

    def forward(self):
        z = Tensor(arr=self.x.arr + self.y.arr, opt=self, ts_tp=(self.x, self.y))
        return z

    @check_grad_outs
    def backward(self, grad):
        if len(self.x_shape) > len(self.y_shape):
            long_shape = self.x_shape
            long_shape_len = len(self.x_shape)
            short_shape_len = len(self.y_shape)
            long_arr = self.x.arr
            short_arr = self.y.arr

            dims1 = list(long_shape[:long_shape_len - short_shape_len])
            dims1 = dims1 + [1] * short_shape_len
            short_arr = n.tile(short_arr, dims1)

            gx = grad * n.ones_like(long_arr)
            gy = grad * n.ones_like(short_arr)
            for i in range(long_shape_len - short_shape_len):
                gy = n.sum(gy, axis=0)
        else:
            long_shape = self.y_shape
            long_shape_len = len(self.y_shape)
            short_shape_len = len(self.x_shape)
            long_arr = self.y.arr
            short_arr = self.x.arr

            dims1 = list(long_shape[:long_shape_len - short_shape_len])
            dims1 = dims1 + [1] * short_shape_len
            short_arr = n.tile(short_arr, dims1)

            gx = grad * n.ones_like(short_arr)
            for i in range(long_shape_len - short_shape_len):
                gx = n.sum(gx, axis=0)
            gy = grad * n.ones_like(long_arr)
        ret = (gx, gy)
        return ret


# 3级测试
class Sub(Operator):
    def __init__(self, x, y):
        super().__init__(x, y)

    def forward(self):
        z = Tensor(arr=self.x.arr - self.y.arr, opt=self, ts_tp=(self.x, self.y))
        return z

    @check_grad_outs
    def backward(self, grad):
        if len(self.x_shape) > len(self.y_shape):
            long_shape = self.x_shape
            long_shape_len = len(self.x_shape)
            short_shape_len = len(self.y_shape)
            long_arr = self.x.arr
            short_arr = self.y.arr

            dims1 = list(long_shape[:long_shape_len - short_shape_len])
            dims1 = dims1 + [1] * short_shape_len
            short_arr = n.tile(short_arr, dims1)

            gx = grad * n.ones_like(long_arr)
            gy = - grad * n.ones_like(short_arr)
            for i in range(long_shape_len - short_shape_len):
                gy = n.sum(gy, axis=0)
        else:
            long_shape = self.y_shape
            long_shape_len = len(self.y_shape)
            short_shape_len = len(self.x_shape)
            long_arr = self.y.arr
            short_arr = self.x.arr

            dims1 = list(long_shape[:long_shape_len - short_shape_len])
            dims1 = dims1 + [1] * short_shape_len
            short_arr = n.tile(short_arr, dims1)

            gx = grad * n.ones_like(short_arr)
            for i in range(long_shape_len - short_shape_len):
                gx = n.sum(gx, axis=0)
            gy = - grad * n.ones_like(long_arr)
        ret = (gx, gy)
        return ret


# 3级测试
class Mul(Operator):
    def __init__(self, x, y):
        super().__init__(x, y)

    def forward(self):
        z = Tensor(arr=self.x.arr * self.y.arr, opt=self, ts_tp=(self.x, self.y))
        return z

    @check_grad_outs
    def backward(self, grad):
        if len(self.x_shape) > len(self.y_shape):
            long_shape = self.x_shape
            long_shape_len = len(self.x_shape)
            short_shape_len = len(self.y_shape)
            long_arr = self.x.arr
            short_arr = self.y.arr

            dims1 = list(long_shape[:long_shape_len - short_shape_len])
            dims1 = dims1 + [1] * short_shape_len
            short_arr = n.tile(short_arr, dims1)

            gx = grad * short_arr
            gy = grad * long_arr
            for i in range(long_shape_len - short_shape_len):
                gy = n.sum(gy, axis=0)
        else:
            long_shape = self.y_shape
            long_shape_len = len(self.y_shape)
            short_shape_len = len(self.x_shape)
            long_arr = self.y.arr
            short_arr = self.x.arr

            dims1 = list(long_shape[:long_shape_len - short_shape_len])
            dims1 = dims1 + [1] * short_shape_len
            short_arr = n.tile(short_arr, dims1)

            gx = grad * long_arr
            for i in range(long_shape_len - short_shape_len):
                gx = n.sum(gx, axis=0)
            gy = grad * short_arr
        ret = (gx, gy)
        return ret


# 1级测试
class Div(Operator):
    def __init__(self, x, y):
        super().__init__(x, y)

    def forward(self):
        z = Tensor(arr=self.x.arr / self.y.arr, opt=self, ts_tp=(self.x, self.y))
        return z

    @check_grad_outs
    def backward(self, grad):

        if len(self.x_shape) > len(self.y_shape):
            long_shape = self.x_shape
            long_shape_len = len(self.x_shape)
            short_shape_len = len(self.y_shape)
            long_arr_x = self.x.arr
            short_arr_y = self.y.arr

            dims1 = list(long_shape[:long_shape_len - short_shape_len])
            dims1 = dims1 + [1] * short_shape_len
            short_arr_y = n.tile(short_arr_y, dims1)

            gx = grad * short_arr_y
            gy = grad * -1 * long_arr_x / short_arr_y ** 2
            for i in range(long_shape_len - short_shape_len):
                gy = n.sum(gy, axis=0)
        else:
            long_shape = self.y_shape
            long_shape_len = len(self.y_shape)
            short_shape_len = len(self.x_shape)
            short_arr_x = self.x.arr
            long_arr_y = self.y.arr

            dims1 = list(long_shape[:long_shape_len - short_shape_len])
            dims1 = dims1 + [1] * short_shape_len
            short_arr_x = n.tile(short_arr_x, dims1)

            gx = grad * long_arr_y
            for i in range(long_shape_len - short_shape_len):
                gx = n.sum(gx, axis=0)
            gy = grad * -1 * short_arr_x / long_arr_y ** 2
        ret = (gx, gy)
        return ret


# 2级测试
class Matmul(Operator):
    def __init__(self, x, y):
        super().__init__(x, y)

    def forward(self):
        z = Tensor(arr=self.x.arr @ self.y.arr, opt=self, ts_tp=(self.x, self.y))
        return z

    @check_grad_outs
    def backward(self, grad):
        dims = [i for i in range(len(self.y.arr.shape))]
        temp = dims[-2]
        dims[-2] = dims[-1]
        dims[-1] = temp
        tx = n.transpose(self.y.arr, dims)
        gx = grad @ tx

        dims = [i for i in range(len(self.x.arr.shape))]
        temp = dims[-2]
        dims[-2] = dims[-1]
        dims[-1] = temp
        ty = n.transpose(self.x.arr, dims)
        gy = ty @ grad

        # 广播问题
        len_x = len(self.x.shape)
        len_y = len(self.y.shape)
        if len_x > len_y:
            for i in range(len_x - len_y):
                gy = n.sum(gy, axis=0)
        else:
            for i in range(len_y - len_x):
                gx = n.sum(gx, axis=0)
        ret = (gx, gy)
        return ret


# 2级测试
def copy(arr):
    return Tensor(arr)


# 1级测试
class Reshape:
    def __init__(self, x, s):
        self.x = x
        self.x_shape = x.shape
        self.new_s = s
        self.ori_s = self.x.arr.shape

    def forward(self):
        z = Tensor(n.reshape(self.x.arr, self.new_s), self, (self.x,))
        return z

    @check_grad_outs
    def backward(self, grad):
        gz = n.reshape(grad, self.ori_s)
        return (gz,)


# 2级测试
class Slice:
    def __init__(self, x, sc):
        self.x = x
        self.x_shape = x.shape
        self.sc = sc

    def forward(self):
        z = Tensor(self.x.arr[self.sc], self, ts_tp=(self.x,))
        return z

    @check_grad_outs
    def backward(self, grad):
        gz = n.float32(n.zeros(self.x_shape))
        gz[self.sc] = grad
        return (gz,)


# 3级测试
class Transpose:
    def __init__(self, x, s):
        self.x = x
        self.x_shape = x.shape
        self.s = s

    def forward(self):
        z = Tensor(n.transpose(self.x.arr, self.s), self, (self.x,))
        return z

    @check_grad_outs
    def backward(self, grad):
        now_s = self.s
        old_s = [None] * len(self.s)
        for i, v in enumerate(now_s):
            old_s[v] = i
        gz = n.transpose(grad, old_s)
        return (gz,)


if __name__ == "__main__":

    a = Tensor(n.ones((2, 3, 2)), require_grad=True)
    b = Tensor(n.ones((3, 2)), require_grad=True)

    c = a / b
    g = n.ones_like(c.arr, dtype=n.float32)
    print(g.shape)
    c.backward(g)
    print(b.grad)

    print()

    a = Tensor(n.ones((2, 3, 2)), require_grad=True)
    b = Tensor(n.ones((3, 2)), require_grad=True)

    c = a.transpose([0, 2, 1])

    g = n.float32(n.array([
        [
            [1, 0, 0],
            [1, 0, 0]
        ],
        [
            [1, 0, 0],
            [1, 0, 0]
        ],
    ]))
    print(g.shape)
    c.backward(g)
    print(a.grad)
    print()

    a = Tensor(n.ones((1, 2)), require_grad=True)
    print(a)
    c = a.tile([3, 2])
    print(c)

    g = n.ones(c.shape)
    print(g)
    c.backward(g)
    print(a.grad)


