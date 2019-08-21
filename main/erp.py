import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import computations as comp
import functional as func
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


erps = pd.read_csv('ERPdata.csv', index_col = 'subject')
#data = pd.read_csv('mergedTrialData.csv', index_col = 'subject')
demo = pd.read_csv('demographic.csv', index_col = 'subject')

"""t, HC, SZ = map(np.array, comp.pwr_time(erps, demo, 1, -100, 300, export = True))
y = HC
M = len(y.tolist())
fe = 1
feh = 2*np.pi/M
D = func.delta(M,fe,feh)
hh = 15
yscsa, kappa, Nh, psinnor = func.scsa(y, D, hh);
fig, ax = plt.subplots(figsize=(10, 7), facecolor = 'w')
ax.plot(t[:], yscsa[:] + 30, label = 'Right')
ax.plot(t[:], y[:])
ax.plot(t[:], y-yscsa - 30)
plt.show()
kappa = np.array(kappa)
print(kappa.diagonal())"""

t, X, Y = comp.dataset(erps, demo, 1, 0, 300)
#subs = list(set(data.index.values.tolist()))
#subs = [i for i in subs if (list(demo.loc[i])[0] == 0)]
lamda = []
Nh = []
for i in range(len(Y)):
    #print(X[i])
    kappa, n = comp.eigen(np.array(X[i]))
    lamda.append(kappa.diagonal())
    Nh.append(n)
""" Ind = np.array([1 if X[i][j]<0 else 0 for j in range(len(X[i]))])
    y = X[i] - Ind*min(X[i])"""
         
Nh = min(Nh)
print(Nh)

features = np.array([x[:Nh] for x in lamda])
cols = ['eigenvalue_%s' % x for x in range(Nh)]
data = pd.DataFrame(features, columns = cols)
data['Type'] = np.array(Y)
#data = data.sample(frac=1).reset_index(drop=True)
X_train, X_test, y_train, y_test = train_test_split(
        data[cols], data['Type'], test_size = 0.2)
neigh = KNeighborsClassifier(n_neighbors=5, weights = 'distance')
neigh.fit(X_train, y_train) 
print(neigh.score(X_test, y_test))
#comp.rl_plt(erps, demo, 1, 1)
