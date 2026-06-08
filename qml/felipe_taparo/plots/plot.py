import sys
from pathlib import Path
parent_dir_1 = Path(__file__).parents[1]
parent_dir_2 = Path(__file__).parents[2]
sys.path.append(str(parent_dir_1))
sys.path.append(str(parent_dir_2))

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


def draw(arrs: np.ndarray | list,
         x_init: float=0,
         x_scale: float=1,
         xlabel: str='x',
         ylabel: str='y',
         title: str='') -> Figure:
    fig, ax = plt.subplots()

    for arr in arrs:
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        n = arr.shape[0]
        ci = 1.96 * std / np.sqrt(n)
        x = [(x_scale*i)+x_init for i in range(len(mean))]
        plt.xticks(x)
        ax.plot(x, mean, marker='.')
        ax.fill_between(x, mean-ci, mean+ci, alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def main():
    if len(sys.argv) != 2:
        print("Usage :python3 plot.py <directory>\nExample: python3 plot.py qcnn_ae_clean")
        exit()

    path = Path(sys.argv[1])
    acc_path = list(path.glob('*.txt'))
    acc = []
    for acc_file in acc_path:
        with open(acc_file, 'r') as f:
            acc.append([float(x) for x in f.read().splitlines()])
    fig = draw(np.array([acc]), x_init=1, xlabel='Épocas de treinamento', ylabel='Acurácia nos dados de teste')
    save_path = f'./{path.name}.png'
    fig.savefig(save_path)
    print(save_path)


if __name__ == '__main__':
    main()
