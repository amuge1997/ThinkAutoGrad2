import numpy as n
n.random.seed(0)

from ThinkAutoGrad2 import Init, Layers, Losses, Optimizer, Utils, Tensor, Activate, backward
from sklearn.metrics import accuracy_score


def test1():
    in_c = 1
    out_c = 2
    image = Tensor(n.ones((1, in_c, 5, 5)))
    kernel = Tensor(n.ones((out_c, in_c, 3, 3)))
    bias = Tensor(n.ones(out_c))
    image2 = Layers.conv2d(image, kernel, bias, (1, 1), is_padding=False)

    grad = n.ones_like(image2.arr)
    image2.backward(grad)

    print(image2.shape)


def one_hot_encoding(labels, num_class=None):
    if num_class is None:
        num_class = n.max(labels) + 1
    one_hot_labels = n.zeros((len(labels), num_class))
    one_hot_labels[n.arange(len(labels)), labels] = 1
    return one_hot_labels.astype('int')


def load_data():
    import cv2 as cv

    dc = n.load('./ThinkAutoGrad2/Demo/mnist.npz')
    data_x, data_y = dc['x_train'], dc['y_train']

    x_ls = []
    for i in data_x:
        x = cv.resize(i, (16, 24))
        x = x[n.newaxis, ...]
        x_ls.append(x)
    data_x = n.concatenate(x_ls)

    data_x = data_x
    data_y = data_y

    cls = n.max(data_y) + 1
    data_y = one_hot_encoding(data_y, cls)

    return data_x, data_y


def test2():
    data_x, data_y = load_data()

    data_x = data_x[:, n.newaxis, ...]
    data_x = data_x / 255

    n_samples = 1000
    data_x = data_x[:n_samples]
    data_y = data_y[:n_samples]

    ts_data_x = Tensor(data_x)
    ts_data_y = Tensor(data_y)

    out_c = data_y.shape[-1]

    lr = 1e-3
    batch_size = 24
    epochs = 200
    epochs_show = 10

    ts_kernels1 = Tensor(n.random.randn(4, 1, 3, 2) / n.sqrt(4+1), require_grad=True)
    ts_bias1 = Tensor(n.zeros((4,)), require_grad=True)
    ts_kernels2 = Tensor(n.random.randn(8, 4, 2, 2) / n.sqrt(8+4), require_grad=True)
    ts_bias2 = Tensor(n.zeros((8,)), require_grad=True)

    ts_kernels4 = Tensor(n.random.randn(16, 8, 2, 2) / n.sqrt(16+8), require_grad=True)
    ts_bias4 = Tensor(n.zeros((16,)), require_grad=True)

    ts_kernels5 = Tensor(n.random.randn(32, 16, 3, 3) / n.sqrt(32+16), require_grad=True)
    ts_bias5 = Tensor(n.zeros((32,)), require_grad=True)

    ts_weights3 = Tensor(n.random.randn(32*4*4, out_c) / n.sqrt(32*4*4 + out_c), require_grad=True)
    ts_bias3 = Tensor(n.zeros((out_c, )), require_grad=True)
    c = Tensor(n.array([1 / batch_size, ]))

    weights_list = [
        ts_kernels1, ts_bias1, ts_kernels2, ts_bias2,
        ts_kernels4, ts_bias4, ts_kernels5, ts_bias5,
        ts_weights3, ts_bias3
    ]

    batch_i = n.random.randint(0, n_samples, batch_size)
    ts_batch_x = ts_data_x[batch_i]
    ts_batch_y = ts_data_y[batch_i]
    y0 = Layers.up_sample2d(ts_batch_x, stride=(2, 2))
    y1 = Activate.relu(Layers.conv2d(y0, ts_kernels1, ts_bias1, stride=(3, 2)))  # 16
    y2 = Activate.relu(Layers.conv2d(y1, ts_kernels2, ts_bias2, stride=(2, 2)))  # 8
    y6 = Activate.relu(Layers.conv2d(y2, ts_kernels4, ts_bias4, stride=(2, 2)))  # 4
    y7 = Activate.relu(Layers.conv2d(y6, ts_kernels5, ts_bias5, stride=(1, 1), is_padding=True))  # 4
    y3 = Layers.flatten(y7)
    y4 = y3 @ ts_weights3 + ts_bias3
    y5 = Activate.sigmoid(y4)
    loss = c * (ts_batch_y - y5) * (ts_batch_y - y5)

    g = n.ones(loss.shape)

    adam = Optimizer.Adam(lr)

    for i in range(epochs):

        batch_i = n.random.randint(0, n_samples, batch_size)
        ts_batch_x = ts_data_x[batch_i]
        ts_batch_y = ts_data_y[batch_i]

        y0 = Layers.up_sample2d(ts_batch_x, stride=(2, 2))
        y1 = Activate.relu(Layers.conv2d(y0, ts_kernels1, ts_bias1, stride=(3, 2)))  # 16
        y2 = Activate.relu(Layers.conv2d(y1, ts_kernels2, ts_bias2, stride=(2, 2)))  # 8
        y6 = Activate.relu(Layers.conv2d(y2, ts_kernels4, ts_bias4, stride=(2, 2)))  # 4
        y7 = Activate.relu(Layers.conv2d(y6, ts_kernels5, ts_bias5, stride=(1, 1), is_padding=True))  # 4
        y3 = Layers.flatten(y7)
        y4 = y3 @ ts_weights3 + ts_bias3
        y5 = Activate.sigmoid(y4)
        loss = c * (ts_batch_y - y5) * (ts_batch_y - y5)

        loss.backward(g)

        adam.run(weights_list)

        for w in weights_list:
            w.grad_zeros()

        if (i+1) % epochs_show == 0:
            print('{} loss - {}'.format(i + 1, n.sum(loss.arr)))

    batch_i = n.array(range(32))
    ts_batch_x = ts_data_x[batch_i]
    ts_batch_y = ts_data_y[batch_i]
    y0 = Layers.up_sample2d(ts_batch_x, stride=(2, 2))
    y1 = Activate.relu(Layers.conv2d(y0, ts_kernels1, ts_bias1, stride=(3, 2)))  # 16
    y2 = Activate.relu(Layers.conv2d(y1, ts_kernels2, ts_bias2, stride=(2, 2)))  # 8
    y6 = Activate.relu(Layers.conv2d(y2, ts_kernels4, ts_bias4, stride=(2, 2)))  # 4
    y7 = Activate.relu(Layers.conv2d(y6, ts_kernels5, ts_bias5, stride=(1, 1), is_padding=True))  # 4
    y3 = Layers.flatten(y7)
    y4 = y3 @ ts_weights3 + ts_bias3
    y5 = Activate.sigmoid(y4)

    print()
    ls1 = [n.argmax(i) for i in ts_batch_y.arr]
    ls2 = [n.argmax(i) for i in y5.arr]
    print(ls1)
    print(ls2)
    print()
    acc = accuracy_score(ls1, ls2)
    print('acc - {}'.format(n.round(acc, 3)))

    # 全连接16个单元,训练卷积层   acc = 0.781
    # 全连接16个单元,不训练卷积层  acc = 0.156

    # 函数adam 1.3331531286239624 0.20902830362319946
    # 类adam   1.3331531286239624 0.2090282142162323


if __name__ == '__main__':
    test2()




