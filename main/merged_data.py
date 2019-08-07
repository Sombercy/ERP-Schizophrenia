import pandas
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

df = pandas.read_csv('mergedTrialData.csv', index_col = 'subject')
#print(df.head())
trial_1 = df.loc['1']
#print(trial_1)
#print(trial_1[:][trial_1['condition'] == 1])
#time_1 = np.cumsum(trial_1.loc[:, 'ITI'])
#print(time_1)
bpt_1 = trial_1[:][trial_1['condition'] == 1]
x = np.cumsum(bpt_1.loc[:, 'ITI'])
#print(x)
cols = list(bpt_1.loc[:,'Fz_N100':'CP4_B1'].columns.values.tolist())
fig, ax = plt.subplots(figsize=(12, 12), facecolor = 'w')
#v = [np.mean(bpt_1[i][cols]) for i in bpt_1[:]['trial']]
for y in cols[:1]:

    ax.plot(x, bpt_1[:][y], marker = 'o')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()
