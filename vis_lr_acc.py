import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import json

with open(f"bsz_lr_model_clean.json") as f:
    clean_acc = json.load(f)

plt.figure(figsize=(8, 6))
data = pd.read_csv('bsz_lr_model.csv')
data = data[data.bsz == 16]
sns.violinplot(x='lr', y='acc', hue='model', scale='width',
               data=data, split=True, palette="muted",
               inner="quartile")

for i, lr in enumerate([0.0001, 0.001, 0.01, 0.1]):
    exp_dir = f"[b]16-[lr]{lr}-[m]MLP"
    acc = clean_acc[exp_dir]
    xs = np.arange(i - 0.45, i, 0.01)
    if i == 0:
        plt.plot(xs, [acc] * len(xs), color='red', label='clean Acc', linestyle='-.')
    else:
        plt.plot(xs, [acc] * len(xs), color='red', linestyle='-.')
    exp_dir = f"[b]16-[lr]{lr}-[m]CNN"

    acc = clean_acc[exp_dir]
    xs = np.arange(i, i + 0.45, 0.01)
    plt.plot(xs, [acc] * len(xs), color='red', linestyle='-.')

plt.grid('on', linestyle=':')

plt.yticks(np.arange(-3, 103, 5))
plt.legend(loc=(0.8, 0.8))
plt.xlabel('Learning rate')
plt.title('Learning rate vs. Acc (batch size=16, std=1.2)')
# plt.show()
plt.savefig('figs/lr_acc.png', dpi=100)
