import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from train_lenet import LeNet5

data = pd.read_csv('lenet_layers.csv')
plt.figure(figsize=(8, 5))
sns.violinplot(
    x='module',
    y='acc',
    scale='width',
    data=data, palette="muted"
)
lenet = LeNet5()
modules = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
numels = []
for m in modules:
    mod = getattr(lenet, m)
    numels.append(sum([p.numel() for p in mod.parameters()]))
plt.yticks(np.arange(0, 125, 5))
plt.grid('on', linestyle=':')

plt.twinx()
plt.plot(np.arange(len(modules)), numels,
         linestyle='-.', color='green', marker='+', markersize=10,
         label='Parameters')
plt.ylabel('#Parameters')
plt.legend()
plt.title('Module vs. Acc & #parameters (std=1.2)')
plt.savefig('figs/module_acc.png')
plt.show()
