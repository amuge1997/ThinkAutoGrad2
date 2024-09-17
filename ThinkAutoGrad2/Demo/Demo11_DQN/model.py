from ThinkAutoGrad2 import Init, Layers, Losses, Optimizer, Utils, Tensor, Activate, backward
from ThinkAutoGrad2.nn import Linear, Model


class QModelBig(Model):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 32)
        self.fc2 = Linear(32, 32)
        self.fc3 = Linear(32, 32)
        self.fc4 = Linear(32, 4)

    def forward(self, x):
        # x.shape = n, 2
        # y.shape = n, 4
        x = Activate.relu(self.fc1(x))
        x = Activate.relu(self.fc2(x))
        x = Activate.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class QModel(Model):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 32)
        self.fc4 = Linear(32, 4)

    def forward(self, x):
        # x.shape = n, 2
        # y.shape = n, 4
        x = Activate.relu(self.fc1(x))
        x = self.fc4(x)
        return x


if __name__ == '__main__':
    qm = QModel()























