

def relu(x):
    from .Activate import Relu
    return Relu(x).forward()


def sigmoid(x):
    from .Activate import Sigmoid
    return Sigmoid(x).forward()


def tanh(x):
    from .Activate import Tanh
    return Tanh(x).forward()











