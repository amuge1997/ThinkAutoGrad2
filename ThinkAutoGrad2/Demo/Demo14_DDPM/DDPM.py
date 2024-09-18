# 自动梯度框架
from ThinkAutoGrad2 import nn, Losses, Optimizer, Tensor, Activate, backward, Utils

# 基础库
import numpy as n


class DDPM():
    def __init__(self, n_steps: int, min_beta: float = 0.0001, max_beta: float = 0.02):
        betas = n.linspace(min_beta, max_beta, n_steps)
        alphas = 1 - betas
        alpha_bars = n.zeros_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars
    
    # 前向过程
    def sample_forward(self, x, t, eps):
        # x.shape = batch_size, 784
        # t.shape = batch_size,
        alpha_bar = self.alpha_bars[t].reshape(-1, 1)
        alpha_bar = n.tile(alpha_bar, (1, x.shape[1]))

        assert alpha_bar.shape == eps.shape, "alpha_bar和eps形状不相同"

        eps = n.array(eps)
        a = n.sqrt(1 - alpha_bar)
        b = n.sqrt(alpha_bar)

        eps = Tensor(eps)
        a = Tensor(a)
        b = Tensor(b)

        res = b * x + a * eps 
        return res
    
    # 反向过程
    def sample_backward(self, img_shape, net, simple_var=True):
        # img_shape : n_samples, n_dims
        x = n.random.randn(*img_shape)
        x = Tensor(x)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var)
        return x.numpy()

    # 反向过程单步
    def sample_backward_step(self, x_t, t, net, simple_var=True):
        n_samples = x_t.shape[0]
        t_tensor = Tensor(n.array([t]*n_samples))
        
        eps = net(x_t, t_tensor)

        if t == 0:
            noise = n.zeros_like(x_t)
        else:
            if simple_var:
                var = self.betas[t]
                noise = n.random.randn(*x_t.shape)
                noise = noise * n.sqrt(var)
            else:
                var = (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]
                noise = n.random.randn(*x_t.shape)
                noise = noise * n.sqrt(var)

        a = (1 - self.alphas[t])
        b = n.sqrt(1 - self.alpha_bars[t]) 
        d = 1 / n.sqrt(self.alphas[t])
        c = a / b

        c = Tensor(c)
        d = Tensor(d)
        noise = Tensor(noise)
        
        x_t = d * (x_t - c * eps) + noise
        return x_t














