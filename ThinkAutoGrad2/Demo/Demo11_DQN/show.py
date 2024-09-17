import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from run import train, predict, Envi

def draw_grid(ax, grid_size):
    # 创建一个 5x5 的网格
    # 绘制网格
    for x in range(grid_size):
        for y in range(grid_size):
            color = 'white'  # 默认颜色
            if (x, y) == (0, 0):
                color = 'green'
            elif (x, y) == (Envi.target_x, Envi.target_y):
                color = 'blue'
            elif Envi.is_obs(x, y):
                color = 'red'
            rect = plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='black')
            ax.add_patch(rect)


def draw_smooth_curve(control_points, ax):
    # 提取控制点的 x 和 y 坐标
    control_points = np.array(control_points)
    x = control_points[:, 0]
    y = control_points[:, 1]
    # 使用 splprep 进行参数化样条插值
    tck, u = splprep([x, y], s=0)
    u_fine = np.linspace(0, 1, 500)
    x_smooth, y_smooth = splev(u_fine, tck)
    ax.plot(x_smooth, y_smooth, color='black')


def get_center(points):
    return [(x+0.5, y+0.5) for x,y in points]


def show(ret):
    print(ret)
    grid_size = 5

    fig, ax = plt.subplots()
    draw_grid(ax, grid_size)
    control_points = get_center(ret)
    draw_smooth_curve(control_points, ax)
    
    # 设置坐标轴的范围
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    # ax.set_xticks(np.arange(0, grid_size + 1, 1))
    # ax.set_yticks(np.arange(0, grid_size + 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    train(
        epochs = 1000,
        is_continue=True
    )
    ret = predict()
    show(ret)


