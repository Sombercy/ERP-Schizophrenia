
import pandas
import numpy as np
import matplotlib.pyplot as plt
import computations as comp
import functional as func
import sklearn
erps = pandas.read_csv('ERPdata.csv', index_col = 'subject')

demo = pandas.read_csv('demographic.csv', index_col = 'subject')

t, HC, SZ = map(np.array, comp.pwr_time(erps, demo, 1, 0, 300, export = True))
y = SZ
M = len(y.tolist())
fe = 1
feh = 2*np.pi/M
D = func.delta(M,fe,feh)
hh = 15
yscsa, kappa, Nh, psinnor = func.scsa(y, D, hh);
"""fig, ax = plt.subplots(figsize=(10, 7), facecolor = 'w')
ax.plot(t[:], yscsa[:] + 30, label = 'Right')
ax.plot(t[:], y[:])
ax.plot(t[:], y-yscsa - 30)
plt.show()"""
kappa = np.array(kappa)
print(kappa.shape)
print(kappa[:5])
psinnor = np.array(psinnor)
print(psinnor.shape)
print(psinnor[:5])
"""t, X, Y = comp.dataset(erps, demo, 1, -100, 300)
subs = list(set(erps.index.values.tolist()))
subs = [i for i in subs if (list(demo.loc[i])[0] == 0)]
print(len(Y))
psinnor = []
Nh = []
for i in range(len(Y)):
    #print(X[i])
    n, psi = comp.eigen(np.array(X[i]))
    print(np.array(psi).shape)
    psinnor.append(psi)
    Nh.append(n)
Nh = min(Nh)
features = psi[:Nh, :]
print(np.array(features).shape)"""
#comp.rl_plt(erps, demo, 1, 1)
