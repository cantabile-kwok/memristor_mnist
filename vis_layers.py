import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np
import json

if __name__ == '__main__':
    data = pd.read_csv('layers_acc.csv')
    plt.figure(figsize=(7, 5))
    sns.violinplot(
        x='layers',
        y='acc',
        scale='width',
        data=data
    )
    with open('n2acc.json', 'r') as f:
        n2acc = json.load(f)
    xs = list(n2acc.keys())
    xs = list(map(eval, xs))
    xs = list(map(lambda x: x - 1, xs))
    ys = list(n2acc.values())
    plt.plot(xs, ys,
             linestyle='-.',
             marker='x',
             color='green',
             markersize=10,
             label='Clean Acc'
             )
    plt.grid('on', linestyle=':')
    plt.yticks(np.arange(-5, 105, 5))
    plt.legend()
    plt.title('MLP layers vs. Acc')
    plt.savefig('figs/layers_vs_acc.png', dpi=100)
    plt.show()
