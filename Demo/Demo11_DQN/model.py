from ThinkAutoGrad2 import Init, Layers, Losses, Optimizer, Utils, Tensor, Activate, backward
from ThinkAutoGrad2.nn import Linear, Model


class QModel(Model):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 16)
        self.fc2 = Linear(16, 32)
        self.fc3 = Linear(32, 16)
        self.fc4 = Linear(16, 4)

    def forward(self, x):
        # x.shape = n, 2
        # y.shape = n, 4
        x = Activate.tanh(self.fc1(x))
        x = Activate.tanh(self.fc2(x))
        x = Activate.tanh(self.fc3(x))
        x = self.fc4(x)
        return x


if __name__ == '__main__':
    qm = QModel()























