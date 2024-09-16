from ..Tensor import Tensor
import numpy as n


class GD:
    def __init__(self, lr):
        self.lr = lr
    
    def run(self, lis):
        def walk(li):
            for i in li:
                if isinstance(i, Tensor):
                    self.run_(i)
                elif isinstance(i, list):
                    walk(i)
        walk(lis)
    
    def run_(self, w):
        lr = self.lr
        arr = w.arr
        grad = w.grad
        arr -= lr * grad



class Adam:
    def __init__(self, lr, p1=0.9, p2=0.999):
        self.dc = dict()

        self.p1 = p1
        self.p2 = p2
        self.e = 1e-8
        self.lr = lr

    def run(self, lis):
        # if isinstance(w, list):
        #     for i in w:
        #         if isinstance(i, list):
        #             for j in
        #             self.run_(i)
        #         else:
        #             self.run_(i)
        # elif isinstance(w, Tensor):
        #     self.run_(w)
        def walk(li):
            for i in li:
                if isinstance(i, Tensor):
                    self.run_(i)
                elif isinstance(i, list):
                    walk(i)

        walk(lis)


    def run_(self, w):
        # type(w) = Tensor
        w_id = id(w)
        dc = self.dc
        p1 = self.p1
        p2 = self.p2
        e = self.e
        lr = self.lr
        arr = w.arr
        grad = w.grad
        if w_id not in dc:
            dc[w_id] = {
                'epoch': 0,
                's': n.zeros(w.shape),
                'r': n.zeros(w.shape)
            }
        w_dc = dc[w_id]

        w_dc['epoch'] += 1
        w_dc['s'] = p1 * w_dc['s'] + (1 - p1) * grad
        w_dc['r'] = p2 * w_dc['r'] + (1 - p2) * grad ** 2

        s = w_dc['s'] / (1 - p1 ** w_dc['epoch'])
        r = w_dc['r'] / (1 - p2 ** w_dc['epoch'])

        arr += - lr * s / (n.sqrt(r) + e)











