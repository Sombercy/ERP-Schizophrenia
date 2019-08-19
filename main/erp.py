import pandas
import numpy as np
import matplotlib.pyplot as plt
import computations as comp
import functional as func

erps = pandas.read_csv('ERPdata.csv', index_col = 'subject')

demo = pandas.read_csv('demographic.csv', index_col = 'subject')

t, HC, SZ = map(np.array, comp.pwr_time(erps, demo, 1, -100, 300, export = True))
y = SZ
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

#comp.rl_plt(erps, demo, 1, 1)

