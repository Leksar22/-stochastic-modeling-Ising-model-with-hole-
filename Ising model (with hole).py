import numpy as np
from matplotlib import pyplot as plt, animation, colors
from numba import jit
from tqdm import tqdm
import matplotlib as mpl


def animate(x):
    step, chart = next(my_gen)
    plt.title(f'Шаг Монте-Карло: {step}')
    if -1 not in chart:
        sctr = plt.pcolormesh(chart, cmap=colors.ListedColormap(['black', 'red']), edgecolors='w', linewidth=0)
        return sctr,
    if 1 not in chart:
        sctr = plt.pcolormesh(chart, cmap=colors.ListedColormap(['blue', 'black']), edgecolors='w', linewidth=0)
        return sctr,
    sctr = plt.pcolormesh(chart, cmap=colors.ListedColormap(['blue', 'black', 'red']), edgecolors='w', linewidth=0)
    return sctr


def draw(mas, H, T):
    N = mas.shape[0]
    CustomCmap = colors.ListedColormap(['blue', 'black', 'red'])
    if -1 not in mas:
        CustomCmap = colors.ListedColormap(['black', 'red'])
    if 1 not in mas:
        CustomCmap = colors.ListedColormap(['blue', 'black'])

    colors.Normalize(vmin=-1, vmax=1)
    fig, ax = plt.subplots()
    plt.style.use("ggplot")
    plt.xlim(0, N)
    plt.ylim(0, N)
    plt.gca().invert_yaxis()
    ax.set_aspect('equal')
    plt.title(f'H = {H}, T = {T}')
    plt.pcolormesh(mas, cmap=CustomCmap, edgecolors='w', linewidth=0)
    plt.savefig(f'H={H}, T={T}.png')
##    plt.show()


def initialize(N, radius):
    np.random.seed(666)
    config = np.random.choice([-1, 1], (N, N))
    config[N//2 - radius:N//2 + radius, N//2 - radius:N//2 + radius] = np.random.choice([1, 0, -1, 0, 0], (2*radius, 2*radius))
    return config


@jit(nopython=True)
def simulate(H, T, S):
    for step in range(N_mc):  # Монте-Карло
        for x in range(N):
            for y in range(N):
                dE = 2 * H * S[x, y] + 2 * J * S[x, y] * (S[(x + 1) % N, y] + S[x, (y + 1) % N] + S[(x - 1) % N, y] + S[x, (y - 1) % N])
                if dE < 0:
                    S[x, y] *= -1
                elif dE >= 0 and np.random.rand() < 1 / tau * np.exp(-dE / T):
                    S[x, y] *= -1
        yield step, S
##    return S

if __name__ == '__main__':
    N = 64
    N_mc = 60000
    radius = 5  # размер дырки
    tau = 0.5

    J = 0.5
    H = [0, 0.5]
    T = [0.1, 0.2, 0.3, 0.8]
    S = initialize(N, radius)

    for i in tqdm(H):
        for j in tqdm(T):
            S = initialize(N, radius)
            
    ##        res = simulate(i, j, S)
    ##        draw(res, H=i, T=j)

            
            my_gen = simulate(i, j, S)
            fig, ax = plt.subplots()
            plt.style.use("ggplot")
            plt.xlim(0, N)
            plt.ylim(0, N)
            plt.gca().invert_yaxis()
            ax.set_aspect('equal')
            anim = animation.FuncAnimation(fig=fig, func=animate, interval=10, blit=False, repeat=False)
            plt.show()
    ##        anim.save(f'111H={i}, T={j}.gif', writer='ffmpeg', fps=15)
            


        



        
