import numpy as np
import matplotlib.pyplot as plt


sin_scal = 4
x_min = -1
x_max = 1

def generate_noisy_data(noise_level=1.0, num_points=100):
    x = np.linspace(x_min, x_max, num_points)
    y = np.sin(sin_scal*x)
    noise = np.random.normal(0, noise_level * 0.01, num_points)
    y_noisy = y + noise
    noise = np.random.normal(0, noise_level * 0.01, num_points)
    x_noise = x + noise
    return x, x_noise, y_noisy

def run_generate():
    x_data, x_noise, y_noisy = generate_noisy_data(noise_level=1.0, num_points=1000)

    np.savez('./workspace/data.npz', x=x_noise, y=y_noisy)

    # 绘制带噪声的数据
    plt.scatter(x_noise, y_noisy, label="Noisy data", color="blue", s=10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def read_data():
    data = np.load("./workspace/data.npz")
    x = data['x']
    y = data['y']
    xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
    print(xy.shape)
    plt.scatter(x, y, label="Noisy data", color="blue", s=10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run_generate()
    read_data()





