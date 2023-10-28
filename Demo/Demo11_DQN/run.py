from envi import Envi
import numpy as n
from ThinkAutoGrad2 import Tensor, Losses, Optimizer, backward, Utils
import random


class MM:
    def __init__(self):
        from model import QModel
        self.Q = QModel()
        self.QD = QModel()
        self.path = 'model/Q.pt'

        self.samples = self.make_samples()

    def qd_un_require_grad(self):
        self.QD.is_require_grad(False)

    def qd_forward(self, s):
        # s.shape = n, 2
        q = self.QD(s)
        max_a = q.numpy().argmax(axis=1)

        indexes = list(range(s.shape[0]))

        max_q = q[indexes, max_a].reshape((q.shape[0], 1))
        return q, max_a, max_q

    def q_forward(self, s):
        # q 模型
        # s.shape = n, 2
        # a = [int]

        q = self.Q(s)     # n, 4
        max_a = q.numpy().argmax()
        return q, max_a

    def make_samples(self):
        en = Envi()

        epochs = 1000
        steps = 10

        samples = []

        for ep in range(epochs):
            s = en.init_state()
            for step in range(steps):
                if en.is_random_select():
                    a = en.random_select()
                else:
                    s_ = Tensor(n.array(s)[n.newaxis, ...])
                    _, a, _ = self.qd_forward(s_)
                    a = a[0]
                s_next = en.state_step(s, a)
                v = en.value_step(s_next)
                sample = {
                    's': s,
                    'a': a,
                    'v': v,
                    's_next': s_next
                }
                samples.append(sample)
                if en.is_terminal(s_next):
                    break
                s = s_next
        return samples

    def trans_paras(self):
        weights_ = self.Q.get_weights(is_numpy=True, is_return_tree=False)

        weights = []
        for data in weights_:
            di = []
            for di_ in data:
                if not issubclass(di_.__class__, Tensor):
                    di.append(Tensor(di_, require_grad=True))
            weights.append(di)

        self.QD.set_weights(weights, is_tree=False)

    def loader(self):
        batch_size = 32
        samples = random.sample(self.samples, batch_size)
        ss = [sam['s'] for sam in samples]
        ss = n.array(ss)

        vs = [sam['v'] for sam in samples]
        vs = n.array(vs)[..., n.newaxis]

        acts = [sam['a'] for sam in samples]

        sns = [sam['s_next'] for sam in samples]
        sns = n.array(sns)
        return ss, vs, acts, sns

    def save(self):
        self.QD.save_weights(self.path)

    def load(self):
        self.Q.load_weights(self.path)
        self.QD.load_weights(self.path)

    def train(self, is_continue):
        epochs = 200
        lr = 1e-2
        c = 5

        if is_continue:
            self.load()

        r = 0.3

        adam = Optimizer.Adam(lr)

        for ep in range(epochs):
            print()
            ss, vs, acts, sns = self.loader()
            sns = Tensor(sns)
            _, _, q_qd_max = self.qd_forward(sns)
            q_q, _ = self.q_forward(Tensor(ss))

            indexes = list(range(len(acts)))
            q_q = q_q[indexes, acts]
            q_q = q_q.reshape((q_q.shape[0], 1))

            self.QD.is_require_grad(False)
            self.Q.is_require_grad(True)
            self.Q.grad_zeros()

            loss = Losses.mse(q_q, (Tensor(n.array(vs)) + Tensor(n.array(r)) * q_qd_max))
            backward(loss)

            print('EPOCHS: {:<3} LOSS: {:<5.9f}'.format(ep, loss.numpy().mean()))

            adam.run(self.Q.get_weights(is_numpy=False, is_return_tree=False))
            if ep % c == 0:
                self.trans_paras()

        self.save()

    def predict(self, s):
        # s = (x, y)
        s = Tensor(n.array(s)[n.newaxis, ...])
        self.load()
        q, max_a = self.q_forward(s)
        return max_a


def predict():
    en = Envi()

    mm = MM()

    x = 0
    y = 0
    s = (x, y)
    for i in range(10):
        a = mm.predict(s)
        s = en.state_step(s, a)
        print(s)
        if en.is_terminal(s):
            break


def train(is_continue):
    mm = MM()
    mm.train(is_continue)


if __name__ == '__main__':
    train(
        is_continue=True
    )
    predict()

















