from ThinkAutoGrad2 import Init, Layers, Losses, Optimizer, Utils, Tensor, Activate, backward
from nn import Model, Linear, Conv2D

# ------------------------------------   TEST  ------------------------------------------ #

class Test3(Model):
    def __init__(self, in_size, out_size):
        super(Test3, self).__init__()
        w = Init.xavier((in_size, out_size), 4, 1, is_grad=True)
        self.data = [w]

    def forward(self, x):
        # x.shape = ..., in_size
        # y.shape = ..., out_size
        y = x @ self.data
        return y


class Test2(Model):
    def __init__(self, in_size, out_size):
        super(Test2, self).__init__()
        w = Init.xavier((in_size, out_size), 4, 1, is_grad=True)
        self.data = [w, w]
        self.fc3 = Test3(3, 3)

    def forward(self, x):
        # x.shape = ..., in_size
        # y.shape = ..., out_size
        y = x @ self.data
        return y


class Test1(Model):
    def __init__(self, in_size, out_size):
        super(Test1, self).__init__()
        w = Init.xavier((in_size, out_size), 4, 1, is_grad=True)
        self.data = [w]
        self.fc1 = Test2(3, 3)
        self.fc2 = Test2(3, 3)

    def forward(self, x):
        # x.shape = ..., in_size
        # y.shape = ..., out_size
        y = x @ self.data
        return y


class Test(Model):
    def __init__(self):
        super(Test, self).__init__()
        self.fc = Test1(3, 3)


def test1():
    import numpy as n
    # m = Linear(5, 1)
    # y = m.forward(Tensor(n.ones((2, 5)), require_grad=True))

    qm = Test()
    w = qm.get_weights(is_numpy=False, is_return_tree=False)
    print(w)
    # print(w)

    # w[1][1][0] = Tensor(n.ones((3, 3)), require_grad=True)
    # print(w)
    w[1][0] = Tensor(n.ones((3, 3)), require_grad=True)
    # print(w)

    # print(qm.get_weights(is_numpy=True, is_return_tree=True))
    # qm.set_weights(w, is_tree=False)
    # print(qm.get_weights(is_numpy=True, is_return_tree=True))

    # print(qm.get_model_tree(is_return_tree=True)[1][2][0].data.is_grad)
    # qm.is_require_grad(False)
    # print(qm.get_model_tree(is_return_tree=True)[1][2][0].data.is_grad)

    # for m in qm.get_model_tree(is_return_tree=False):
    #     print(m.data)
    for m in qm.get_weights(is_numpy=True, is_return_tree=False):
        print(m)
    # for m in qm.get_model_tree(is_return_tree=False):
    #     i = m.data
    #     if len(i) == 0:
    #         print(None)
    #     for j in i:
    #         print(j.is_grad)
    qm.save_weights('../model/qm.pt')
    qm.load_weights('../model/qm.pt')
    print()
    # for m in qm.get_model_tree(is_return_tree=False):
    #     print(m.data)
    for m in qm.get_weights(is_numpy=True, is_return_tree=False):
        print(m)
    # qm.is_require_grad(False)
    # for m in qm.get_model_tree(is_return_tree=False):
    #     i = m.data
    #     if len(i) == 0:
    #         print(None)
    #     for j in i:
    #         print(j.is_grad)


def test2():
    import numpy as n
    conv = Conv2D(3, 4, 2, (2, 2), False)
    y = conv(Tensor(n.ones((1, 3, 10, 10)), require_grad=True))
    print(y.shape)


if __name__ == '__main__':
    # test1()

    test2()




















