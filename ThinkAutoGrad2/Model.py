from ThinkAutoGrad2.Tensor import Tensor
import pickle


class Model:
    def __init__(self):
        self.weights_list = []

    def __call__(self, inps):
        return self.forward(inps)

    def forward(self, inps):
        pass

    def append_weights(self, *weights):
        for w in weights:
            if type(w) != Tensor:
                raise Exception('非 Tensor 类型')
            self.weights_list.append(w)

    def get_weights(self):
        return self.weights_list

    def grad_zeros(self):
        for w in self.weights_list:
            w.grad_zeros()

    def save_weights(self, path):
        ws = self.weights_list
        with open(path, 'wb') as fp:
            pickle.dump(ws, fp)

    def load_weights(self, path):
        with open(path, 'rb') as fp:
            ws = pickle.load(fp)
        for w in ws:
            if type(w) != Tensor:
                raise Exception('非 Tensor 类型')
        self.weights_list = ws












