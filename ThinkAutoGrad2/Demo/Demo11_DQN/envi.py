import random


class Envi:
    
    target_x = 4
    target_y = 4

    x_min = 0
    x_max = 5
    y_min = 0
    y_max = 5

    obs_xy = [(0, 1), (2, 0), (1, 4), (2, 2), (1,2), (3, 2), (3,3)]
    # obs_xy = [(0, 2), (3, 0), (4, 2), (2, 4), (2, 2)]
    # obs_xy = [(0, 2), (3, 0), (4, 2), (2, 4)]
    # obs_xy = []

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
        if not Envi.x_min <= x < Envi.x_max or not Envi.y_min <= y < Envi.y_max or (x == Envi.target_x and y == Envi.target_y) or Envi.is_obs(x, y):
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
    def is_obs(x, y):
        for ox, oy in Envi.obs_xy:
            if x == ox and y == oy:
                return True
        return False

    @staticmethod
    def value_step(s):
        x, y = s
        if not Envi.x_min <= x < Envi.x_max or not Envi.y_min <= y < Envi.y_max:        # 越界惩罚
            return -1
        elif Envi.is_obs(x, y):                                                         # 障碍惩罚
            return -1
        elif x == Envi.target_x and y == Envi.target_y:                                 # 目标奖励
            return 1
        else:
            return 0








