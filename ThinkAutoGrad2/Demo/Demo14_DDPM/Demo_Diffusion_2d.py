# 自动梯度框架
from ThinkAutoGrad2 import nn, Losses, Optimizer, Tensor, Activate, backward, Utils

# 基础库
import numpy as n
import random
from PIL import Image

from DDPM import DDPM


# 加载mnist数据集
def load_data():
    data = n.load("./workspace/data.npz")
    x = data['x']
    y = data['y']
    xy = n.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
    return xy


class Net(nn.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_channels=2, out_channels=16)
        self.fc2 = nn.Linear(in_channels=16, out_channels=32)
        self.fc3 = nn.Linear(in_channels=32*2, out_channels=32)
        self.fc4 = nn.Linear(in_channels=32, out_channels=16)
        self.fc5 = nn.Linear(in_channels=16, out_channels=2)

        self.fct = nn.Linear(in_channels=1, out_channels=32)
    
    def forward(self, x, t):
        # x.shape = batch_size, 784
        # t.shape = batch_size,
        # print(x.shape, t.shape)
        t = t.reshape((t.shape[0], 1))
        t = Activate.relu(self.fct(t))
        m = Activate.relu(self.fc1(x))
        m = Activate.relu(self.fc2(m))
        m = Utils.concat([m, t], axis=1)
        m = Activate.relu(self.fc3(m))
        m = Activate.relu(self.fc4(m))
        y = self.fc5(m)
        return y


def train(batch_size, epochs, lr, save_per_epochs, continue_train):

    # 加载数据
    data_x = load_data()
    nums = data_x.shape[0]

    # 网络结构
    net = Net()

    # 加载参数
    if continue_train:
        net.load_weights(net_path)

    opt = Optimizer.Adam(lr)
    
    # 训练
    for ep in range(epochs):
        # 随机挑选真样本
        samples_index = random.sample(range(nums), batch_size)
        x = Tensor(data_x[samples_index, ...])

        t = n.random.randint(0, ddpm.n_steps, (batch_size, ))
        eps = n.random.randn(*x.shape)
        x_t = ddpm.sample_forward(x, t, eps)
        eps_theta = net(x_t, Tensor(t))
        loss = Losses.mse(eps_theta, Tensor(eps))

        net.grad_zeros()
        backward(loss)
        opt.run(net.get_weights(is_numpy=False, is_return_tree=False))

        # 保存模型参数
        if ep % save_per_epochs == 0:
            net.save_weights(net_path)
            print("{:>5}/{:>5}   loss: {:.5f}".format(ep, epochs, loss.arr.mean()))
    

def sample(net, simple_var=True):
    n_sample = 100
    shape = (n_sample, 2)  # n, 2
    result = ddpm.sample_backward(shape, net, simple_var=simple_var)

    from generate_diffusion_2d_data import sin_scal, x_max, x_min
    plot_data_sin(result, sin_scal, x_min, x_max)


def plot_data_sin(data, a, x_min, x_max):
    import matplotlib.pyplot as plt
    import numpy as np
    
    xy_train = load_data()

    # 提取 x 和 y 数据
    x_data = data[:, 0]
    y_data = data[:, 1]
    
    # 生成绘制曲线时的 x 值，范围在 [x_min, x_max]
    x_curve = np.linspace(x_min, x_max, 500)
    y_curve = np.sin(a * x_curve)
    
    # 绘制数据点和曲线
    plt.scatter(x_data, y_data, label="Generate Data", color="blue", s=10)
    # plt.scatter(xy_train[:,0], xy_train[:,1], label="Data", color="green", s=10)
    plt.plot(x_curve, y_curve, label=f"y={a}*sin", color="red")

    plt.grid(True)
    plt.gca().tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)

    
    # 添加图例和标签
    # plt.xlabel("x")
    # plt.ylabel("y")
    plt.legend()
    plt.show()


def test():
    net = Net()
    net.load_weights(net_path)
    sample(net)


if __name__ == '__main__':
    ddpm = DDPM(1000)
    net_path = './workspace/net_2d.pt'
    train(
        batch_size=128, 
        epochs=1000, 
        lr=1e-3, 
        save_per_epochs=500, 
        continue_train=True
    )
    test()























