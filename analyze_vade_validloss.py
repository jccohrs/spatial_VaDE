# coding: utf-8
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

validloss = np.zeros(100)

########## without regularizer
rootdir = os.path.expandvars('${OUTPUT_DIR}') + '/segmentation/vade-tv/IndianPines_wo'

i = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file == 'exp_stats.csv':
            print(f'Process data from experiment {subdir}.')
            df = pd.read_csv(f'{subdir}/{file}', index_col=False, header=0, sep=',')
            validloss[i] = df.iloc[-1]['valid loss']
            print(f'Extracted value {i + 1}.')
            i += 1

print(f'Minimum w/o reg: {validloss.min():.4f}.')
print(f'Mean w/o reg: {validloss.mean():.4f}.')
print(f'Maximum w/o reg: {validloss.max():.4f}.')
print(f'Standard deviation w/o reg: {validloss.std():.4f}.')

fig, ax = plt.subplots()
ax.boxplot(validloss)
ax.set_xticks([1], ['w/o reg'])
plt.show()

# fig2, ax2 = plt.subplots()
# ax2.scatter(np.ones(100), oa[:, 0], marker='x')
# ax2.scatter(2 * np.ones(100), oa[:, 1], marker='x')
# ax2.set_xticks([1, 2], ['w/ reg', 'w/o reg'])
# plt.show()
