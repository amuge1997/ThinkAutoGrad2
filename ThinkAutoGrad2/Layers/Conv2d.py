
import numpy as n
from ..Tensor import Tensor, check_grad_outs


# 测试
def matmul_backward(grad, features_col, kernels_col):
    n_samples, channels, height, width = grad.shape
    grad = n.reshape(grad, [n_samples, channels, height * width])
    grad = n.transpose(grad, [0, 2, 1])
    grad_feature_col = grad @ kernels_col
    grad_kernel_col = features_col.transpose([0, 2, 1]) @ grad
    grad_kernel_col = n.sum(grad_kernel_col, axis=0)
    grad_kernel_col = n.transpose(grad_kernel_col, [1, 0])
    grad_bias = n.sum(n.sum(grad, axis=0), axis=0)
    return grad_feature_col, grad_kernel_col, grad_bias


# 测试
def matmul_forward(features_col, kernels_col, bias, out_hw):
    out_height, out_width = out_hw
    n_samples = features_col.shape[0]
    ret = features_col @ n.transpose(kernels_col, [1, 0]) + bias
    ret = n.transpose(ret, [0, 2, 1])
    out_channels = ret.shape[1]
    ret = n.reshape(ret, [n_samples, out_channels, out_height, out_width])
    return ret


# 测试
def grad_col_to_kel(grad_kernels_col, in_channels, kernel_size):
    kernel_height, kernel_width = kernel_size
    output_channels = grad_kernels_col.shape[0]
    grad_kernels = n.reshape(grad_kernels_col, [output_channels, in_channels, kernel_height, kernel_width])
    return grad_kernels


# 测试
def kel_to_col(kernels):
    kernels = n.float32(kernels)
    out_channels, in_channels, kernel_height, kernel_width = kernels.shape
    ret = n.reshape(kernels, [out_channels, in_channels * kernel_height * kernel_width])
    return ret


# 测试
def grad_col_to_img(features_col, kernel_size, in_channels, in_shape, out_shape, stride):
    kernel_height, kernel_width = kernel_size
    n_samples = features_col.shape[0]
    ih = features_col.shape[1]
    in_height, in_width = in_shape
    out_height, out_width = out_shape
    stride_h, stride_w = stride
    ret = n.zeros((n_samples, in_channels, in_height, in_width))
    for ihi in range(ih):
        patch = features_col[:, ihi, :]
        patch = n.reshape(patch, [n_samples, in_channels, kernel_height, kernel_width])
        hi = int(ihi / out_width)
        wi = ihi % out_width
        anchor_h = hi * stride_h
        anchor_w = wi * stride_w
        ret[:, :, anchor_h:anchor_h+kernel_height, anchor_w:anchor_w+kernel_width] = patch
    return ret


# 将图像转为向量
def img_to_col(features, kernel_size, stride):
    kernel_height, kernel_width = kernel_size
    stride_height, stride_width = stride
    n_samples, channels, in_height, in_width = features.shape
    height_able = in_height - kernel_height + 1
    width_able = in_width - kernel_width + 1
    if (height_able - 1) % stride_height != 0 or (width_able - 1) % stride_width != 0:
        raise Exception('error')
    out_height = int((height_able - 1) / stride_height) + 1
    out_width = int((width_able - 1) / stride_width) + 1
    ret = n.zeros((n_samples, out_height * out_width, kernel_height * kernel_width * channels))
    for hi in range(out_height):
        for wi in range(out_width):
            h_anchor = hi * stride_height
            w_anchor = wi * stride_width
            patch = features[:, :, h_anchor:h_anchor+kernel_height, w_anchor:w_anchor+kernel_width]
            patch = n.reshape(patch, [n_samples, 1, -1])
            ihi = hi * out_width + wi
            ret[:, ihi:ihi+1, :] = patch
    out_height = out_height
    out_width = out_width
    out_hw = (out_height, out_width)
    return ret, out_hw


# 去除补零
def re_padding2d(image, pad):
    pad_h, pad_w = pad
    h0 = pad_h[0]
    h1 = -pad_h[1]
    w0 = pad_w[0]
    w1 = -pad_w[1]
    # 以下操作主要是为了防止-0的情况
    if h1 != 0:
        hs = slice(h0, h1)
    else:
        hs = slice(h0, None)
    if w1 != 0:
        ws = slice(w0, w1)
    else:
        ws = slice(w0, None)
    return image[:, :, hs, ws]


# 补零
def padding2d(image, kernel_size, stride_hw):
    kernel_height, kernel_width = kernel_size
    stride_h, stride_w = stride_hw
    n_samples, channels, output_height, output_width = image.shape
    input_height = output_height * stride_h - stride_h + kernel_height
    input_width = output_width * stride_w - stride_w + kernel_width
    pad_sum = input_height - output_height
    pad_half1 = int(pad_sum / 2)
    pad_half2 = pad_sum - pad_half1
    pad_h = (pad_half1, pad_half2)
    pad_sum = input_width - output_width
    pad_half1 = int(pad_sum / 2)
    pad_half2 = pad_sum - pad_half1
    pad_w = (pad_half1, pad_half2)
    ret = n.pad(image, pad_width=[(0, 0), (0, 0), pad_h, pad_w])
    return ret, pad_h, pad_w


# 卷积
class Conv2d:
    def __init__(self, in_features, kernels, bias, stride, is_padding):

        # in_features.shape = n_samples, in_channels, in_height, in_width
        # kernel.shape = out_channels, in_channels, kernel_size, kernel_size
        # bias.shape = out_channels

        # out_features.shape = n_samples, out_channels, out_height, out_width

        # features * kernel + bias

        self.ts_in_features = in_features
        self.ts_kernels = kernels
        self.ts_bias = bias

        self.in_features = in_features.arr
        self.kernels = kernels.arr
        self.bias = bias.arr

        kernel_shape = kernels.shape
        k_o, k_i, k_h, k_w = kernel_shape

        input_shape = in_features.shape
        n_samples, in_channels, in_height, in_width = input_shape

        # 已使用
        self.in_channels = in_channels      # 输入通道
        self.kernel_size = (k_h, k_w)
        self.stride_hw = stride             # 步长
        self.is_padding = is_padding
        self.pad_in_features = None         # 补零后的输入
        self.pad_in_hw = None               # 输入补零后的高宽
        self.pad_out_hw = None              # 根据补零后的输入得到的输出的高宽
        self.pad = None                     # 补零参数,记录了高宽两个维度的补零

    def forward(self):

        in_features = self.in_features
        kernels = self.kernels
        bias = self.bias

        kernel_size = self.kernel_size
        stride_hw = self.stride_hw

        # 展开为核向量
        kernels_col = kel_to_col(kernels)

        # 输入补零
        if self.is_padding:
            pad_in_features, pad_h, pad_w = padding2d(in_features, kernel_size, stride_hw)
            self.pad = (pad_h, pad_w)
        else:
            pad_in_features = in_features
            self.pad = ((0, 0), (0, 0))

        self.pad_in_features = pad_in_features
        self.pad_in_hw = (pad_in_features.shape[2], pad_in_features.shape[3])
        pad_in_features_col, pad_out_hw = img_to_col(pad_in_features, kernel_size, stride_hw)
        self.pad_out_hw = pad_out_hw
        out_features = matmul_forward(pad_in_features_col, kernels_col, bias, self.pad_out_hw)

        z = Tensor(out_features, self, (self.ts_in_features, self.ts_kernels, self.ts_bias))

        return z

    @check_grad_outs
    def backward(self, grad):
        pad_in_features = self.pad_in_features
        kernel_size = self.kernel_size
        in_channels = self.in_channels
        pad_in_hw = self.pad_in_hw
        pad_out_hw = self.pad_out_hw
        stride_hw = self.stride_hw

        kernels = self.kernels

        pad_features_col, _ = img_to_col(features=pad_in_features, kernel_size=kernel_size, stride=stride_hw)
        kernel_col = kel_to_col(kernels)
        gard_features_col, grad_kernels_col, grad_bias = matmul_backward(grad, pad_features_col, kernel_col)
        gard_features = grad_col_to_img(  # 梯度向量转梯度图
            gard_features_col,
            kernel_size,
            in_channels,
            pad_in_hw,
            pad_out_hw,
            stride_hw
        )

        grad_kernels = grad_col_to_kel(grad_kernels_col, in_channels, kernel_size)
        gard_features = re_padding2d(gard_features, self.pad)
        gz = (gard_features, grad_kernels, grad_bias)
        return gz















