
# 自动梯度框架
from ThinkAutoGrad2 import nn, Losses, Optimizer, Tensor, Activate, backward

# 基础库
import numpy as n
import random
from PIL import Image


# 加载mnist数字数据集数据
def load_data():
    dc = n.load('./ThinkAutoGrad2/Demo/mnist.npz')
    data_x = dc['x_train']
    x_ls = []
    for i in data_x:
        x = Image.fromarray(i)
        x = x.resize((28, 28))
        x = n.array(x)
        x = x[n.newaxis, ...]
        x_ls.append(x)
    data_x = n.concatenate(x_ls)
    return data_x


# 展示前数组9个图像
def show_image9(x):
    import matplotlib.pyplot as p
    nums = 9

    for i in range(nums):
        xi = x[i]   # s,s,1 or 3
        p.subplot(3, 3, i+1)
        p.axis('off')
        p.imshow(xi)
    p.show()


# 生成网络
class GN(nn.Model):
    def __init__(self, seed_size):
        super().__init__()
        self.fc1 = nn.Linear(in_channels=seed_size, out_channels=128)
        self.fc2 = nn.Linear(in_channels=128, out_channels=784)
    
    def forward(self, x):
        y0 = x
        y1 = Activate.relu(self.fc1(y0))
        y2 = self.fc2(y1)
        return y2


# 判别网络
class DN(nn.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_channels=784, out_channels=32)
        self.fc2 = nn.Linear(in_channels=32, out_channels=1)
    
    def forward(self, x):
        nums = x.shape[0]
        x = x.reshape((nums, 784))
        y0 = x
        y1 = Activate.relu(self.fc1(y0))
        y2 = self.fc2(y1)
        return y2


# 训练
def train(batch, epochs, save_per_epochs, continue_train=True):
    seed_size = 16      # 采样维度
    g_path = 'g.pt'     # 生成器保存路径
    d_path = 'd.pt'     # 判别器保存路径

    # 加载数据
    data_x = load_data()
    data_x = data_x / 255
    nums = data_x.shape[0]

    # 网络结构
    G = GN(seed_size)
    D = DN()

    # 加载参数
    if continue_train:
        G.load_weights(g_path)
        D.load_weights(d_path)

    # 优化器
    opt = Optimizer.GD(1e-2)
    
    # 训练
    for ep in range(epochs):
        # 正太分布随机采样
        seed = Tensor(n.random.randn(batch*seed_size).reshape(batch, seed_size))
        # 生成器生成假图像
        fake = G(seed)

        # 随机挑选真样本
        samples_index = random.sample(range(nums), batch)
        real = Tensor(data_x[samples_index, ...])

        # 判别器判别
        real_dis = D(real)
        fake_dis = D(fake)
        
        # 判别器loss
        d_loss = Losses.mse(real_dis, Tensor(n.ones_like(real_dis.arr))) + Losses.mse(fake_dis, Tensor(n.zeros_like(fake_dis.arr)))
        # 生成器loss
        g_loss = Losses.mse(fake_dis, Tensor(n.ones_like(fake_dis.arr)))
        
        # 判别器优化
        D.grad_zeros()
        G.grad_zeros()
        backward(d_loss)
        dw = D.get_weights(is_numpy=False, is_return_tree=False)
        opt.run(dw)
        
        # 生成器优化
        D.grad_zeros()
        G.grad_zeros()
        backward(g_loss)
        dg = G.get_weights(is_numpy=False, is_return_tree=False)
        opt.run(dg)

        # 保存模型参数
        if ep % save_per_epochs == 0:
            G.save_weights(g_path)
            D.save_weights(d_path)
            print("{:>5}/{:>5}   Dloss: {:.5f}   Gloss: {:.5f}".format(ep, epochs, d_loss.arr.mean(), g_loss.arr.mean()))
    
    # 保存模型参数
    G.save_weights(g_path)
    D.save_weights(d_path)
    

# 展示图像
def show():
    seed_size = 16
    batch = 9
    g_path = 'g.pt'

    # 展示真实图像
    data_x = load_data()
    data_x = data_x / 255
    real = data_x[:batch].reshape(batch, 28, 28)
    show_image9(real)

    # 展示生成图像
    G = GN(seed_size)
    G.load_weights(g_path)
    seed = Tensor(n.random.randn(batch*seed_size).reshape(batch, seed_size))
    fake = G(seed).arr.reshape(batch, 28, 28)
    fake = n.where(fake > 1., 1., fake)
    fake = n.where(fake < 0., 0., fake)
    show_image9(fake)
    

if __name__ == '__main__':
    train(
        batch=16,
        epochs=5000,
        save_per_epochs=500
    )
    show()
































































