import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import computations as comp
import functional as func
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import feature_selection as fs


data = pd.read_csv('ERPdata.csv', index_col = 'subject')
#data = pd.read_csv('mergedTrialData.csv', index_col = 'subject')
demo = pd.read_csv('demographic.csv', index_col = 'subject')
time = pd.read_csv('time.csv')
"""t, HC, SZ = map(np.array, comp.pwr_time(erps, demo, 1, 0, 300, export = True))
y = SZ
time = pd.read_csv('time.csv')
data = data[data['condition'] == 1].drop('condition',axis = 1)  
time = time[:][time['time_ms'] >= 0]
time = time[:][time['time_ms'] <= 300]
data = data[data['time_ms'] <= 300]
data = data[data['time_ms'] >= 0].drop('time_ms', axis = 1)"""

""""temp = data.loc[5]

temp.index = range(temp.shape[0])
temp = temp**2
temp = temp.sum(axis = 1, skipna = True)
temp = np.sqrt(temp)
temp = [temp[x]*100/max(temp) for x in range(len(temp))]
fig, ax = plt.subplots(figsize=(10, 7), facecolor = 'w')
ax.plot(time['time_ms'], temp)
y = np.array(temp)
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
ax.plot(t, psinnor)
kappa = np.array(kappa)
kappa = kappa.diagonal())
print(Nh)
#print(kappa.diagonal())"""
tmin = 0
tmax = 300
time = time[time['time_ms'] >= tmin]
time = time[time['time_ms'] <= tmax]
time = time['time_ms']
time.index = range(time.shape[0])
timepoints = len(time)
X, Y = comp.dataset(data, demo, 1, tmin, tmax, timepoints)
#subs = list(set(data.index.values.tolist()))
#subs = [i for i in subs if (list(demo.loc[i])[0] == 0)]
lamda = []
Nh = []
psi = [] 
size = len(Y)

for i in range(size):
    #print(X[i])
    kappa, n, psinnor = comp.eigen(np.array(X[i]))
    lamda.append(kappa.diagonal())
    Nh.append(n)
    psi.append(psinnor)
# Ind = np.array([1 if X[i][j]<0 else 0 for j in range(len(X[i]))])
#    y = X[i] - Ind*min(X[i])

"""set1 = np.array([fs.t_comp(time, lamda[i], psi[i]) for i in range(size)])
set2 = np.array([fs.ngreates(31, lamda[i], psi[i]) for i in range(size)])
eigenvals = [lamda[i][set1[i]].tolist() for i in range(size)]
time_points = [time.loc[set2[i]].values.tolist() for i in range(size)]
cols1 = ['eigenvalue_%s' % x for x in range(set1.shape[1])]
cols2 = ['feature_%s' % x for x in range(set2.shape[1])]
dataset1 = pd.DataFrame(np.array(eigenvals).tolist(),columns = np.array(cols1))
dataset2 = pd.DataFrame(np.array(time_points).tolist(), columns = np.array(cols2))   
dataset1['Type'] = np.array(Y)   
dataset2['Type'] = np.array(Y)   """
Nh = min(Nh)
#print(Nh)
features = np.array([x[:Nh] for x in lamda])
cols = ['eigenvalue_%s' % x for x in range(Nh)]
data = pd.DataFrame(features, columns = cols)
data['Type'] = np.array(Y)
#data = data.sample(frac=1).reset_index(drop=True)"""
X_train, X_test, y_train, y_test = train_test_split(
        data[cols], data['Type'], test_size = 0.25, random_state = 42)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
print(neigh.score(X_test, y_test))
#comp.rl_plt(erps, demo, 1, 1)"""
