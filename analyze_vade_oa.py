# coding: utf-8
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

########## wrong regularizer
df1 = pd.read_csv('/Users/jan/sciebo/output/segmentation/vade-tv/IndianPines_w/test_scores_100_wrong_reg.csv', index_col=False, header=None, names=['reg', 'oa'], sep=',')

# plt.boxplot(df['oa'])

########## without regularizer
rootdir = '/Users/jan/sciebo/output/segmentation/vade-tv/IndianPines_wo'

oa = np.zeros((100, 2))
i = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file == 'exp_stats.csv':
            print(f'Process data from experiment {subdir}.')
            df2 = pd.read_csv(f'{subdir}/{file}', index_col=False, header=0, sep=',')
            oa[i, 1] = df2.iloc[-1]['accuracy']
            print(f'Extracted value {i + 1}.')
            i += 1

oa[:, 0] = df1['oa']

print(f'Minimum w/ reg: {oa[:, 0].min():.4f}.')
print(f'Mean w/ reg: {oa[:, 0].mean():.4f}.')
print(f'Maximum w/ reg: {oa[:, 0].max():.4f}.')
print(f'Standard deviation w/ reg: {oa[:, 0].std():.4f}.')
print(f'Minimum w/o reg: {oa[:, 1].min():.4f}.')
print(f'Mean w/o reg: {oa[:, 1].mean():.4f}.')
print(f'Maximum w/o reg: {oa[:, 1].max():.4f}.')
print(f'Standard deviation w/o reg: {oa[:, 1].std():.4f}.')

fig, ax = plt.subplots()
ax.boxplot(oa)
ax.set_xticks([1, 2], ['w/ reg', 'w/o reg'])
plt.show()

# fig2, ax2 = plt.subplots()
# ax2.scatter(np.ones(100), oa[:, 0], marker='x')
# ax2.scatter(2 * np.ones(100), oa[:, 1], marker='x')
# ax2.set_xticks([1, 2], ['w/ reg', 'w/o reg'])
# plt.show()
