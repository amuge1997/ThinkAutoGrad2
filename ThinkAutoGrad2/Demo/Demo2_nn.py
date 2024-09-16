
from ThinkAutoGrad2 import Init, Layers, Losses, Optimizer, Utils, Tensor, Activate, backward
import numpy as n


inps = Tensor(n.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
]))
labs = Tensor(n.array([
    [1],
    [1],
    [0],
    [0]
]))

w1 = Tensor(n.random.randn(2, 3), require_grad=True)
b1 = Tensor(n.zeros((3,)), require_grad=True)

w2 = Tensor(n.random.randn(3, 1), require_grad=True)
b2 = Tensor(n.zeros((1,)), require_grad=True)

c = Tensor(n.array(1 / 4))

y = Activate.relu(inps @ w1 + b1)
outs = Activate.sigmoid(y @ w2 + b2)
loss = c * (labs - outs) * (labs - outs)
g = n.ones(loss.shape, dtype=n.float32)

for i in range(5000):
    y = Activate.relu(inps @ w1 + b1)
    outs = Activate.sigmoid(y @ w2 + b2)
    loss = c * (labs - outs) * (labs - outs)
    loss.backward(g)

    w1.arr -= 1e-1 * w1.grad
    b1.arr -= 1e-1 * b1.grad

    w2.arr -= 1e-1 * w2.grad
    b2.arr -= 1e-1 * b2.grad

    w1.grad_zeros()
    b1.grad_zeros()
    w2.grad_zeros()
    b2.grad_zeros()

    print('{} loss - {}'.format(i+1, n.sum(loss.arr)))

y = Activate.relu(inps @ w1 + b1)
outs = Activate.sigmoid(y @ w2 + b2)

print('outs')
print(outs)















