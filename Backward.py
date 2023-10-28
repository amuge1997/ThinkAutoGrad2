
import numpy as n


def backward(tensor):
    g = n.ones(tensor.shape)
    tensor.backward(g)







