from .. import Init, Layers, Losses, Optimizer, Utils, Tensor, Activate, backward


class Model:
    def __init__(self):
        self.data = []
        self.data_tree = []

    def get_model_tree(self, is_return_tree):
        model_tree = [self]
        attrs = dir(self)
        for att in attrs:
            att = getattr(self, att)
            if issubclass(att.__class__.__base__, Model):
                if not is_return_tree:
                    model_tree += att.get_model_tree(is_return_tree)
                else:
                    model_tree.append(att.get_model_tree(is_return_tree))
        return model_tree

    def grad_zeros(self):
        model_tree = self.get_model_tree(is_return_tree=False)
        for m in model_tree:
            if len(m.data) > 0:
                for d in m.data:
                    d.grad_zeros()

    def is_require_grad(self, select, all_data=True):
        for di in self.data:
            di.is_require_grad(select)
        if all_data:
            model_tree = self.get_model_tree(is_return_tree=False)
            for m in model_tree:
                data = m.data
                for di in data:
                    di.is_require_grad(select)

    def get_weights(self, is_numpy, is_return_tree=False):
        if is_numpy:
            data = [di.numpy() for di in self.data]
        else:
            data = self.data
        self.data_tree = [data]

        attrs = dir(self)
        for att in attrs:
            att = getattr(self, att)
            if issubclass(att.__class__.__base__, Model):
                if not is_return_tree:
                    self.data_tree += att.get_weights(is_numpy, is_return_tree)
                else:
                    self.data_tree.append(att.get_weights(is_numpy, is_return_tree))
        return self.data_tree

    def set_weights(self, data, is_tree=False):
        if not is_tree:
            model_tree = self.get_model_tree(is_return_tree=False)
            for m, w in zip(model_tree, data):
                m.data = w
        else:
            self.data = data[0]
            index = 1
            attrs = dir(self)
            for att in attrs:
                att = getattr(self, att)
                if issubclass(att.__class__.__base__, Model):
                    att.set_weights(data[index], is_tree)
                    index += 1

    def load_weights(self, path, use_pickle=True):
        import pickle
        if use_pickle:
            with open(path, 'rb') as fp:
                data_tree_ = pickle.load(fp)
        else:
            import joblib
            data_tree_ = joblib.load(path)
        data_tree = []
        for data in data_tree_:
            di = []
            for di_ in data:
                if not issubclass(di_.__class__, Tensor):
                    di.append(Tensor(di_, require_grad=True))
            data_tree.append(di)
        self.set_weights(data_tree)

    def save_weights(self, path, use_pickle=True):
        import pickle
        data_tree = self.get_weights(is_numpy=True, is_return_tree=False)
        if use_pickle:
            with open(path, 'wb') as fp:
                pickle.dump(data_tree, fp)
        else:
            import joblib
            joblib.dump(data_tree, path)

    def forward(self, *args, **kwargs):
        import numpy as n
        return Tensor(n.zeros(1, dtype='float32'))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Conv2D(Model):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_padding=False):
        super(Conv2D, self).__init__()
        w = Init.xavier((out_channels, in_channels, kernel_size, kernel_size), in_channels, out_channels, is_grad=True)
        b = Init.zeros((out_channels,), is_grad=True)
        self.data = [w, b]
        self.stride = stride
        self.is_padding = is_padding

    def forward(self, x):
        # x.shape = n, c, h, w
        [w, b] = self.data
        return Layers.conv2d(x, w, b, stride=self.stride, is_padding=self.is_padding)


class Linear(Model):
    def __init__(self, in_channels, out_channels):
        super(Linear, self).__init__()
        w = Init.xavier((in_channels, out_channels), in_channels, out_channels, is_grad=True)
        b = Init.zeros((out_channels,), is_grad=True)
        self.data = [w, b]

    def forward(self, x):
        # x.shape = ..., in_size
        # y.shape = ..., out_size
        [w, b] = self.data
        y = x @ w + b
        return y





















