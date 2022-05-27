import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import json

with open(f"bsz_lr_model_clean.json") as f:
    clean_acc = json.load(f)

plt.figure(figsize=(8, 6))
data = pd.read_csv('bsz_lr_model.csv')
data = data[data.lr == 0.001]
sns.violinplot(x='bsz', y='acc', hue='model', scale='width',
               data=data, split=True, palette="muted",
               inner="quartile"
               )
plt.grid('on', linestyle=':')

plt.xlabel('Batch size')
plt.title('Batch size vs. Acc (lr=0.001, std=1.2)')
# plt.scatter(2, 50, s=10)

for i, bsz in enumerate([8, 16, 32, 64, 128, 256, 512]):
    exp_dir = f"[b]{bsz}-[lr]0.001-[m]MLP"
    acc = clean_acc[exp_dir]
    xs = np.arange(i-0.4, i, 0.01)
    if i == 0:
        plt.plot(xs, [acc] * len(xs), color='red', label='clean Acc', linestyle='-.')
    else:
        plt.plot(xs, [acc] * len(xs), color='red', linestyle='-.')
    exp_dir = f"[b]{bsz}-[lr]0.001-[m]CNN"

    acc = clean_acc[exp_dir]
    xs = np.arange(i, i+0.4, 0.01)
    plt.plot(xs, [acc] * len(xs), color='red', linestyle='-.')

plt.yticks(np.arange(-3, 105, 5))
plt.legend(loc=(0.01, .75))
# plt.show()

plt.savefig('figs/bsz_acc.png', dpi=100)
