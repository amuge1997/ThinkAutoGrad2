

class Model:
    def __init__(self):
        self.weights_list = []

    def __call__(self, inps):
        return self.forward(inps)

    def forward(self, inps):
        pass

    def get_weights(self):
        return self.weights_list

    def grad_zeros(self):
        for w in self.weights_list:
            w.grad_zeros()






