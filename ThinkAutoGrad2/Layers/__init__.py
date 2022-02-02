

def conv2d(in_features, kernels, bias, stride=(1, 1), is_padding=False):
    from .Conv2d import Conv2d
    return Conv2d(in_features, kernels, bias, stride=stride, is_padding=is_padding).forward()


def flatten(x):
    from .Layers import Flatten
    return Flatten(x).forward()


def gru(x, h, w, wz, wr):
    from .RNN import GRU
    return GRU(x, h, w, wz, wr).forward()


def lstm(x, h, c, wf, bf, wi, bi, wc, bc, wo, bo):
    from .RNN import LSTM
    return LSTM(x, h, c, wf, bf, wi, bi, wc, bc, wo, bo).forward()


def rnn(x, h, u, w, b):
    from .RNN import RNN
    return RNN(x, h, u, w, b).forward()


def up_sample2d(x, stride):
    from .Layers import UpSample2d
    return UpSample2d(x, stride).forward()














