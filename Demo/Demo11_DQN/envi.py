import random


class Envi:
    def __init__(self):
        pass

    @staticmethod
    def random(prob):
        return True if random.random() < prob else False

    @staticmethod
    def is_random_select():
        return Envi.random(0.8)

    @staticmethod
    def random_select():
        acts = list(range(0, 4))
        return random.choice(acts)

    @staticmethod
    def init_state():
        x = random.randint(0, 5-1)
        y = random.randint(0, 5-1)
        return x, y

    @staticmethod
    def state_step(s, a):
        x, y = s
        if a == 0:
            x -= 1
        elif a == 1:
            x += 1
        elif a == 2:
            y -= 1
        elif a == 3:
            y += 1
        else:
            raise Exception
        return x, y

    @staticmethod
    def is_terminal(s):
        x, y = s
        if not 0 <= x < 5 or not 0 <= y < 5 or (x == 4 and y == 4):
            return True
        return False

    @staticmethod
    def translate_act(a):
        if a == 0:
            return '左'
        elif a == 1:
            return '右'
        elif a == 2:
            return '上'
        elif a == 3:
            return '下'
        else:
            raise Exception

    @staticmethod
    def value_step(s):
        x, y = s
        if not 0 <= x < 5 or not 0 <= y < 5:
            return -1
        elif x == 4 and y == 4:
            return 1
        else:
            return 0









