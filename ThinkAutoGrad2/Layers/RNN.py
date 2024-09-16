from .. import Activate
from .. import Utils
from ..Tensor import Tensor, check_grad_outs
import numpy as n


# 1级
class RNN:
    def __init__(self, x, h, u, w, b):
        # x.shape = batch, time_step, in_dims
        # h.shape = batch, 1, hi_dims
        # u.shape = in_dims, hi_dims
        # w.shape = hi_dims, hi_dims
        # b.shape = hi_dims
        x, h, u, w, b = self.check_rnn_inps(x, h, u, w, b)
        self.x = x
        self.h = h
        self.u = u
        self.w = w
        self.b = b

    @staticmethod
    def check_rnn_inps(x, h, u, w, b):
        condition = [len(x.shape) == 3, len(h.shape) == 3,
                     h.shape[1] == 1, len(u.shape) == 2,
                     len(w.shape) == 2, len(b.shape) == 1
                     ]
        if False in condition:
            raise Exception('维度错误, {}'.format(condition))
        return x, h, u, w, b

    def forward(self):
        batch, time_step = self.x.shape[0], self.x.shape[1]
        tsp_h = self.h
        for tsp in range(time_step):
            p1 = self.x[:, tsp:tsp+1, :] @ self.u
            p2 = tsp_h @ self.w
            p3 = p1 + p2 + self.b
            tsp_h = Activate.tanh(p3)
        z = tsp_h
        return z


# 1级测试
class LSTMCell:
    def __init__(self, x, hc, wf, bf, wi, bi, wc, bc, wo, bo):
        x, hc, wf, bf, wi, bi, wc, bc, wo, bo = self.check_lstm_inps(x, hc, wf, bf, wi, bi, wc, bc, wo, bo)
        # x.shape = batch, 1, in_dims
        # hc.shape = batch, 1, hi_dims + ci_dims
        # hi_dims = ci_dims
        self.x = x
        self.hc = hc
        # 遗忘门
        # wf.shape = hi_dims+in_dims, ci_dims
        # bf.shape = ci_dims
        self.wf = wf
        self.bf = bf
        # 输入门
        # wi.shape = hi_dims+in_dims, ci_dims
        # bi.shape = ci_dims
        self.wi = wi
        self.bi = bi
        # wc.shape = hi_dims+in_dims, ci_dims
        # bc.shape = ci_dims
        self.wc = wc
        self.bc = bc
        # 输出门
        # wo.shape = hi_dims+in_dims, ci_dims
        # bo.shape = ci_dims
        self.wo = wo
        self.bo = bo

        # 梯度隔离

        # 输出隔离
        self.ht_ct = None
        # 输入隔离
        self.x_copy = self.x.copy()
        self.hc_copy = self.hc.copy()
        self.wf_copy = self.wf.copy()
        self.bf_copy = self.bf.copy()
        self.wi_copy = self.wi.copy()
        self.bi_copy = self.bi.copy()
        self.wc_copy = self.wc.copy()
        self.bc_copy = self.bc.copy()
        self.wo_copy = self.wo.copy()
        self.bo_copy = self.bo.copy()

    @staticmethod
    def check_lstm_inps(x, hc, wf, bf, wi, bi, wc, bc, wo, bo):
        condition = [
            len(x.shape) == 3, len(hc.shape) == 3,
            len(wf.shape) == 2, len(bf.shape) == 1,
            len(wi.shape) == 2, len(bi.shape) == 1,
            len(wc.shape) == 2, len(bc.shape) == 1,
            len(wo.shape) == 2, len(bo.shape) == 1,
        ]
        if False in condition:
            raise Exception('维度错误, {}'.format(condition))
        return x, hc, wf, bf, wi, bi, wc, bc, wo, bo

    def forward(self):
        x = self.x_copy
        hc = self.hc_copy
        wf = self.wf_copy
        bf = self.bf_copy
        wi = self.wi_copy
        bi = self.bi_copy
        wc = self.wc_copy
        bc = self.bc_copy
        wo = self.wo_copy
        bo = self.bo_copy

        half_size = int(hc.shape[2] / 2)
        h, c = hc[:, :, :half_size], hc[:, :, half_size:]
        hx = Utils.concat([h, x], 2)
        # 遗忘门
        ft = Activate.sigmoid(hx @ wf + bf)    # 遗忘比例
        ct = c * ft
        # 输入门
        it = Activate.sigmoid(hx @ wi + bi)    # 记忆比例
        ct_ = Activate.tanh(hx @ wc + bc)      # 信息
        ct = ct + it * ct_
        # 输出门
        ot = Activate.sigmoid(hx @ wo + bo)
        ht = ot * Activate.tanh(ct)
        self.ht_ct = Utils.concat([ht, ct], 2)
        z = Tensor(self.ht_ct.arr, self, (self.x,  self.hc,     # hc(h和c合并)是复杂度为O(n)的关键
                                          self.wf, self.bf,
                                          self.wi, self.bi,
                                          self.wc, self.bc,
                                          self.wo, self.bo))
        return z

    @check_grad_outs
    def backward(self, grad):
        self.ht_ct.backward(grad)
        gz = (
            self.x_copy.grad,  self.hc_copy.grad,
            self.wf_copy.grad, self.bf_copy.grad,
            self.wi_copy.grad, self.bi_copy.grad,
            self.wc_copy.grad, self.bc_copy.grad,
            self.wo_copy.grad, self.bo_copy.grad
        )
        return gz


# 1级测试
class LSTM:
    def __init__(self, x, h, c, wf, bf, wi, bi, wc, bc, wo, bo):
        # x.shape = batch, time_step, in_dims
        # h.shape = batch, 1, hi_dims
        # c.shape = batch, 1, ci_dims
        # hi_dims = ci_dims
        self.x = x
        self.h = h
        self.c = c
        self.hc = Utils.concat([h, c], axis=2)          # hc(h和c合并)是复杂度为O(n)的关键
        # 遗忘门
        # wf.shape = hi_dims+in_dims, ci_dims
        # bf.shape = ci_dims
        self.wf = wf
        self.bf = bf
        # 输入门
        # wi.shape = hi_dims+in_dims, ci_dims
        # bi.shape = ci_dims
        self.wi = wi
        self.bi = bi
        # wc.shape = hi_dims+in_dims, ci_dims
        # bc.shape = ci_dims
        self.wc = wc
        self.bc = bc
        # 输出门
        # wo.shape = hi_dims+in_dims, ci_dims
        # bo.shape = ci_dims
        self.wo = wo
        self.bo = bo

    def forward(self):
        batch, time_step, in_dims = self.x.shape

        tsp_hc = self.hc
        for tsp in range(time_step):
            tsp_x = self.x[:, tsp:tsp + 1, :]
            tsp_hc = LSTMCell(tsp_x, tsp_hc, self.wf, self.bf, self.wi, self.bi, self.wc, self.bc, self.wo, self.bo).forward()

        half_size = int(self.hc.shape[2] / 2)
        last_h = tsp_hc[:, :, :half_size]
        last_c = tsp_hc[:, :, half_size:]
        return last_h, last_c


class GRUCell:
    def __init__(self, x, h, w, wz, wr):
        # x.shape = batch, 1, x_dims
        # h.shape = batch, 1, h_dims
        # w.shape = h_dims+x_dims, h_dims
        # wz.shape = h_dims+x_dims, h_dims
        # wr.shape = h_dims+x_dims, h_dims
        self.x = x
        self.h = h
        self.w = w
        self.wz = wz
        self.wr = wr

        self.x_copy = x.copy()
        self.h_copy = h.copy()
        self.w_copy = w.copy()
        self.wz_copy = wz.copy()
        self.wr_copy = wr.copy()

        self.ht = None

    def forward(self):
        x = self.x_copy
        h = self.h_copy
        w = self.w_copy
        wz = self.wz_copy
        wr = self.wr_copy

        one = Tensor(n.array([1.]))

        hx = Utils.concat([h, x], 2)
        zt = Activate.sigmoid(hx @ wz)
        rt = Activate.sigmoid(hx @ wr)
        rth = Utils.concat([rt * h, x], 2)
        h_ = Activate.tanh(rth @ w)
        ht = (one - zt) * h + zt * h_
        self.ht = ht
        z = Tensor(ht.arr, self, (self.x, self.h, self.w, self.wz, self.wr))
        return z

    def backward(self, grad):
        self.ht.backward(grad)
        gz = (self.x_copy.grad, self.h_copy.grad, self.w_copy.grad, self.wz_copy.grad, self.wr_copy.grad)
        return gz


class GRU:
    def __init__(self, x, h, w, wz, wr):
        # x.shape = batch, time_steps, x_dims
        # h.shape = batch, 1, h_dims
        # w.shape = h_dims+x_dims, h_dims
        # wz.shape = h_dims+x_dims, h_dims
        # wr.shape = h_dims+x_dims, h_dims
        self.x = x
        self.h = h
        self.w = w
        self.wz = wz
        self.wr = wr

    def forward(self):
        batch, time_step, in_dims = self.x.shape

        tsp_h = self.h
        for tsp in range(time_step):
            tsp_x = self.x[:, tsp:tsp + 1, :]
            tsp_h = GRUCell(tsp_x, tsp_h, self.w, self.wz, self.wr).forward()
        return tsp_h


# 效率极低
# rnn递归函数调用复杂度为O(n),常数级
# lstm递归函数调用复杂度至少为O(2^n),指数级
class LSTMUnable:
    def __init__(self, x, h, c, wf, bf, wi, bi, wc, bc, wo, bo):
        # x.shape = batch, time_step, in_dims
        # h.shape = batch, hi_dims
        # c.shape = batch, ci_dims
        # hi_dims = ci_dims
        self.x = x
        self.h = h
        self.c = c
        # 遗忘门
        # wf.shape = hi_dims+in_dims, ci_dims
        # bf.shape = ci_dims
        self.wf = wf
        self.bf = bf
        # 输入门
        # wi.shape = hi_dims+in_dims, ci_dims
        # bi.shape = ci_dims
        self.wi = wi
        self.bi = bi
        # wc.shape = hi_dims+in_dims, ci_dims
        # bc.shape = ci_dims
        self.wc = wc
        self.bc = bc
        # 输出门
        # wo.shape = hi_dims+in_dims, ci_dims
        # bo.shape = ci_dims
        self.wo = wo
        self.bo = bo

    def __call__(self):
        batch, time_step = self.x.shape[0], self.x.shape[1]
        last_c = []
        last_h = []

        for bah in range(batch):
            tsp_h = self.h[bah:bah+1, :]
            tsp_c = self.c[bah:bah+1, :]

            for tsp in range(time_step):
                tsp_x = self.x[bah:bah + 1, tsp, :]     # 维度降低,去除time_step维度
                hx_cat = Utils.concat([tsp_h, tsp_x], 1)()
                # 遗忘门
                ft = Activate.sigmoid(hx_cat @ self.wf + self.bf)
                ct = ft * tsp_c
                # 输入门
                it = Activate.sigmoid(hx_cat @ self.wi + self.bi)
                c_ = Activate.tanh(hx_cat @ self.wc + self.bc)
                ct = ct + it * c_
                # 输出门
                ot = Activate.sigmoid(hx_cat @ self.wo + self.bo)
                ht = ot * Activate.tanh(ct)()

                tsp_h = ht
                tsp_c = ct
            last_h.append(tsp_h)
            last_c.append(tsp_c)
        ret_h = Utils.concat(last_h, 0)()
        ret_c = Utils.concat(last_c, 0)()
        return ret_h, ret_c













