import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('std_vs_acc.csv')
# data['std'] = list(map(eval, data['std']))
plt.figure(figsize=(7, 7))
ax = sns.violinplot(x='std', y='acc', data=data, scale='width')
# tips = sns.load_dataset("tips")
# ax = sns.violinplot(x="day", y="total_bill", hue="sex",
#
#                     data=tips, palette="Set2", split=True, linestyle=':',
#
#                     scale="count", inner="quartile")
plt.grid('on', linestyle=':')
plt.yticks(np.arange(-10, 125, 5))
plt.xlabel('Std', fontsize=12
)
plt.ylabel('Acc', fontsize=12)
plt.title('Std vs. Acc')
# plt.show()
plt.savefig('figs/std_acc.png', dpi=100)
