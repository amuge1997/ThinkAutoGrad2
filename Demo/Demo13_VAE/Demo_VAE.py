# 自动梯度框架
from ThinkAutoGrad2 import nn, Losses, Optimizer, Tensor, Activate, backward, Utils

# 基础库
import numpy as n
import random
from PIL import Image



# 加载mnist数据集
def load_data():
    dc = n.load('./ThinkAutoGrad2/Demo/mnist.npz')
    data_x, data_y = dc['x_train'], dc['y_train']
    x_ls = []
    for i in data_x:
        x = Image.fromarray(i)
        x = x.resize((28, 28))
        x = n.array(x)
        x = x[n.newaxis, ...]
        x_ls.append(x)
    data_x = n.concatenate(x_ls)

    # 取数字0严格不能
    data_x = data_x[data_y == 0]
    return data_x

# 展示数组前9个图像
def show_image9(x):
    import matplotlib.pyplot as p
    nums = 9

    for i in range(nums):
        xi = x[i]   # s,s,1 or 3
        p.subplot(3, 3, i+1)
        p.axis('off')
        p.imshow(xi, cmap='gray')
    p.show()


# 编码器
class Encoder(nn.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_channels=784, out_channels=128)
        self.fc2 = nn.Linear(in_channels=128, out_channels=64)
        self.fc3 = nn.Linear(in_channels=64, out_channels=32)

    def forward(self, x):
        y0 = x
        y1 = Activate.relu(self.fc1(y0))
        y2 = Activate.relu(self.fc2(y1))
        y3 = Activate.relu(self.fc3(y2))

        mu, sig = y3[:, :16], y3[:, 16:]
        
        sig = Utils.exp(sig)    # 映射到正数域内

        eps = Tensor(n.random.randn(*sig.shape))

        z = mu + eps * Utils.sqrt(sig)

        return z, mu, sig


# 解码器
class Decoder(nn.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_channels=16, out_channels=64)
        self.fc2 = nn.Linear(in_channels=64, out_channels=128)
        self.fc3 = nn.Linear(in_channels=128, out_channels=784)

    def forward(self, x):
        y0 = x
        y1 = Activate.relu(self.fc1(y0))
        y2 = Activate.relu(self.fc2(y1))
        y3 = Activate.sigmoid(self.fc3(y2))
        return y3


def train(batch, epochs, lr, save_per_epochs, continue_train):
    encoder_path = 'encoder.pt'     # 生成器保存路径
    decoder_path = 'decoder.pt'     # 判别器保存路径

    # 加载数据
    data_x = load_data()
    data_x = data_x / 255
    nums = data_x.shape[0]

    # 网络结构
    encoder = Encoder()
    decoder = Decoder()

    # 加载参数
    if continue_train:
        encoder.load_weights(encoder_path)
        decoder.load_weights(decoder_path)


    opt = Optimizer.GD(lr)
    
    # 训练
    for ep in range(epochs):
        # 随机挑选真样本
        samples_index = random.sample(range(nums), batch)
        image = Tensor(data_x[samples_index, ...])
        nums = image.shape[0]
        image = image.reshape((nums, 784))

        z, mu, sig = encoder(image)
        rec_image = decoder(z)
        
        t1 = Tensor(n.ones(sig.shape))
        t01 = Tensor(n.ones(sig.shape)*1e-8)
        tc = Tensor(n.ones(sig.shape)*0.5 / batch)
        
        loss1 = Losses.mse(rec_image, image)
        loss1 = Utils.sum(loss1, axis=1)
        loss2 = tc * (sig + mu * mu - Utils.log(sig + t01) - t1)
        loss2 = Utils.sum(loss2, axis=1)
        loss = loss1 + loss2

        # 判别器优化
        encoder.grad_zeros()
        decoder.grad_zeros()
        backward(loss)
        opt.run(encoder.get_weights(is_numpy=False, is_return_tree=False))
        opt.run(decoder.get_weights(is_numpy=False, is_return_tree=False))

        # 保存模型参数
        if ep % save_per_epochs == 0:
            encoder.save_weights(encoder_path)
            decoder.save_weights(decoder_path)
            print("{:>5}/{:>5}   loss1: {:.5f}   loss2: {:.5f}".format(ep, epochs, loss1.arr.mean(), loss2.arr.mean()))
    

# 展示图像
def show():
    z_dims = 16
    batch = 9
    decoder_path = 'decoder.pt'

    # 展示生成图像
    decoder = Decoder()
    decoder.load_weights(decoder_path)
    z = Tensor(n.random.randn(batch*z_dims).reshape(batch, z_dims))
    fake = decoder(z).arr.reshape(batch, 28, 28)
    fake = n.where(fake > 1., 1., fake)
    fake = n.where(fake < 0., 0., fake)
    show_image9(fake)

    # 展示真实图像
    data_x = load_data()
    data_x = data_x / 255
    real = data_x[:batch].reshape(batch, 28, 28)
    show_image9(real)


if __name__ == '__main__':
    train(
        batch=16,
        epochs=500,
        lr=1e-2,
        save_per_epochs=100,
        continue_train=True
    )
    show()














