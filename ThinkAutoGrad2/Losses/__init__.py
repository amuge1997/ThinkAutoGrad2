

def mse(prds, labs):
    from .Losses import MSE
    return MSE(prds, labs).forward()


def cross_entropy_loss(prds, labs, axis):
    from .Losses import CrossEntropyLoss
    return CrossEntropyLoss(prds, labs, axis).forward()



