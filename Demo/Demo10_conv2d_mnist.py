import sys
from ThinkAutoGrad2 import Init, Layers, Losses, Optimizer, Utils, Tensor, Activate, backward
from ThinkAutoGrad2 import nn
import numpy as n

n.random.seed(0)


class Net(nn.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2D(1, 4, 2, (2, 2))
        self.conv2 = nn.Conv2D(4, 8, 2, (2, 2))
        self.conv3 = nn.Conv2D(8, 16, 2, (2, 2))
        self.conv4 = nn.Conv2D(16, 32, 3, (1, 1), is_padding=True)
        self.fc = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        y1 = Activate.relu(self.conv1(x))   # 16,16
        y2 = Activate.relu(self.conv2(y1))  # 8,8
        y6 = Activate.relu(self.conv3(y2))  # 4,4
        y7 = Activate.relu(self.conv4(y6))  # 4,4
        y3 = Layers.flatten(y7)
        y4 = self.fc(y3)
        y5 = Activate.sigmoid(y4)
        return y5


def one_hot_encoding(labels, num_class=None):
    if num_class is None:
        num_class = n.max(labels) + 1
    one_hot_labels = n.zeros((len(labels), num_class))
    one_hot_labels[n.arange(len(labels)), labels] = 1
    return one_hot_labels.astype('int')


def resize(img, size):
    from PIL import Image
    img_ = Image.fromarray(img)
    img_ = img_.resize(size)
    img = n.array(img_)
    return img


def load_data():
    dc = n.load('./ThinkAutoGrad2/Demo/mnist.npz')
    data_x, data_y = dc['x_train'], dc['y_train']

    x_ls = []
    for i in data_x:
        x = resize(i, (32, 32))
        x = x[n.newaxis, ...]
        x_ls.append(x)
    data_x = n.concatenate(x_ls)

    data_x = data_x
    data_y = data_y

    cls = n.max(data_y) + 1
    data_y = one_hot_encoding(data_y, cls)

    return data_x, data_y


def train():
    data_x, data_y = load_data()

    data_x = data_x[:, n.newaxis, ...]
    data_x = data_x / 255

    n_samples = 100
    data_x = data_x[:n_samples]
    data_y = data_y[:n_samples]

    data_x_ts = Tensor(data_x)
    data_y_ts = Tensor(data_y)

    lr = 1e-3
    batch_size = 8
    epochs = 500
    epochs_show = 5
    weights_path = 'net.pt'

    net = Net()

    adam = Optimizer.Adam(lr)

    for i in range(epochs):

        batch_i = n.random.randint(0, n_samples, batch_size)
        batch_x_ts = data_x_ts[batch_i]
        batch_y_ts = data_y_ts[batch_i]

        predict_y = net(batch_x_ts)
        loss = Losses.mse(predict_y, batch_y_ts)

        backward(loss)
        adam.run(net.get_weights(is_numpy=False, is_return_tree=False))
        net.grad_zeros()

        if (i + 1) % epochs_show == 0:
            print('{} loss - {}'.format(i + 1, n.sum(loss.arr)))

    batch_i = n.array(range(16))
    batch_x_ts = data_x_ts[batch_i]
    batch_y_ts = data_y_ts[batch_i]
    predict_y = net(batch_x_ts)

    print()
    loss1 = [n.argmax(i) for i in batch_y_ts.arr]
    loss2 = [n.argmax(i) for i in predict_y.arr]
    print(loss1)
    print(loss2)
    print()
    net.save_weights(weights_path)


def test():
    data_x, data_y = load_data()

    data_x = data_x[:, n.newaxis, ...]
    data_x = data_x / 255

    n_samples = 100
    data_x = data_x[:n_samples]
    data_y = data_y[:n_samples]

    data_x_ts = Tensor(data_x)
    data_y_ts = Tensor(data_y)

    weights_path = 'net.pt'

    net = Net()
    net.load_weights(weights_path)

    batch_size = 16

    batch_i = n.random.randint(0, n_samples, batch_size)
    batch_x_ts = data_x_ts[batch_i]
    batch_y_ts = data_y_ts[batch_i]
    predict_y = net(batch_x_ts)

    print()
    loss1 = [n.argmax(i) for i in batch_y_ts.arr]
    loss2 = [n.argmax(i) for i in predict_y.arr]
    print(loss1)
    print(loss2)
    print()


if __name__ == '__main__':
    train()
    test()













































