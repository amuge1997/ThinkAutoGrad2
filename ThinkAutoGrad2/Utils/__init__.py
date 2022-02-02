

def concat(xls, axis):
    from .Utils import Concat
    return Concat(xls, axis).forward()


def exp(x):
    from .Utils import Exp
    return Exp(x).forward()


def log(x):
    from .Utils import Log
    return Log(x).forward()


def repeat(x, reps, axis):
    from .Utils import Repeat
    return Repeat(x, reps, axis).forward()


def sum(x, axis):
    from .Utils import Sum
    return Sum(x, axis).forward()


def tile(x, reps):
    from .Utils import Tile
    return Tile(x, reps).forward()






