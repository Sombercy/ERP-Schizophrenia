import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def A_plot(data, num, cond):
    """This function returns am amplitude plot of certain subject for
       the whole observation period"""
    data = data.loc[num]
    data = data[:][data['condition'] == cond].drop('condition', axis = 1)
    x = data.loc[:, 'time_ms']
    data = data.drop('time_ms', axis = 1)
    data.index = range(data.shape[0])
    fig, ax = plt.subplots(figsize=(12, 12), facecolor = 'w')
    ax.plot(x[:], data[:][:])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()
    return 0

def rl_plt(data, demo, cond, group):
    subs = subjects(data, demo, cond, group)
    #average quantity of time-points of the given period
    n = mpoints(data, demo, cond, subs, -100, 300) #average quantity of time-points of the given period
    #building a boxplot with mean values of the given period
    right = data.loc[:, ['FC4', 'C4', 'CP4', 'condition', 'time_ms']]
    left = data.loc[:, ['FC3', 'C3', 'CP3', 'condition', 'time_ms']]
    time0, pwr0 = mplot(right, demo, cond, n, subs, -100)
    time1, pwr1 = mplot(left, demo, cond, n, subs, -100)
    fig, ax = plt.subplots(figsize=(13, 10), facecolor = 'w')
    ax.plot(time0[:], pwr0[:], color = 'k', label = 'Right')
    ax.plot(time1[:], pwr1[:], color = 'b', label = 'Left')
    ax.legend( loc='best', borderaxespad=1)
    plt.show()
    return 0

def power_plt(data, num, cond):
    """This function returns power plots from different channels areas
       for a certain subject in a certain condition"""
    data = data.loc[num]
    data = data[:][data['condition'] == cond].drop('condition', axis = 1)
    x = data.loc[:, 'time_ms']
    data = data.drop('time_ms', axis = 1)
    data.index = range(data.shape[0])
    info = demo.loc[num]

    if (list(info)[0] == 0):
        for area in ['l', 'r', None]:
            if (area == 'l'):
                y = data.loc[:, ['FC3', 'C3', 'CP3']]
                y = y**2
            elif (area == 'r'):
                y = data.loc[:, ['FC4', 'C4', 'CP4']]
                y = y**2
            else:
                y = data**2
            y = y.sum(axis = 1, skipna = True)
            f = np.sqrt(y)
            y = (y/y.max())*100
            f = (f/f.max())*100
            fig, ax = plt.subplots(figsize=(13, 10), facecolor = 'w')
            ax.plot(x[:], y[:], color = 'k', label = 'power')
            ax.plot(x[:], f[:], color = 'r', label = 'firing')
            ax.set(xlabel='Time, ms', ylabel='Power, %', title='Summa of EP power')
            ax.annotate(f'{info}', (900, 90))
            ax.legend( loc='best', borderaxespad=1)
            if (area != None):
                plt.savefig(f'SoP-{num}-{cond}-{area}.png', orientation='landscape')
            else:
                plt.savefig(f'SoP-{num}-{cond}.png', orientation='landscape')
            plt.close(fig)
    return 0

def subjects(data, demo, cond, group):
    """This function returns a list with numbers of subjects related to
       a certain group"""
    subs = list(set(data.index.values.tolist()))
    subs = [i for i in subs if (list(demo.loc[i])[0] == group)]
    return subs

def mpoints(data, demo, cond, subs, tmin, tmax):
    """This functrion returns quantity of time points appropriet
       for all subjects of the group for a given period (tmin; rmax)"""
    data = data.loc[subs]
    data = data[:][data['condition'] == cond].drop('condition', axis = 1)
    data = data[:][data['time_ms'] >= tmin]
    data = data[:][data['time_ms'] <= tmax]
    n = [data.loc[i].shape[0] for i in subs]
    n = sum(n)
    return n//len(subs)

def mplot(data, demo, cond, points, subs, tmin):
    """This function returns lists of averaged around the subjects of the group
       power points and corresponding time points (also averaged)"""
    time = [0] * points
    pwr = [0] * points
    k = len(subs)
    data = data.loc[subs]
    data = data[:][data['condition'] == cond].drop('condition', axis = 1)
    data = data[:][data['time_ms'] >= tmin]
    for q in range(len(subs)):
        i = subs[q]
        y = data.loc[i]
        y.index = range(y.shape[0])
        y = y.loc[:points-1, :]
        x = y.loc[:, 'time_ms']
        y = y.drop('time_ms', axis = 1)
        x.index = range(x.shape[0])
        time = [time[j] + x[j] for j in range (x.shape[0])]
        #y = y**2
        y = y.sum(axis = 1, skipna = True)
        pwr = [pwr[j] + y[j] for j in range(y.shape[0])]
    time = [time[i]/k for i in range(len(time))]
    pwr = [pwr[i]/k for i in range(len(pwr))]
    #max_pwr = max(pwr)
    #pwr = [pwr[i]*100/max_pwr for i in range(len(pwr))]
    return time, pwr

def mbox(data, demo, cond, points, subs, tmin):
    """This function returns a list of averaged power values for a given period
       for each subject of the group"""
    k = len(subs)
    pmax = 0
    pwr = []
    data = data.loc[subs]
    data = data[:][data['condition'] == cond].drop('condition', axis = 1)
    data = data[:][data['time_ms'] >= tmin]
    for q in range(len(subs)):
        i = subs[q]
        y = data.loc[i]
        y.index = range(y.shape[0])
        y = y.loc[:points-1, :]
        y = y.drop('time_ms', axis = 1)
        #y = y**2
        y = y.sum(axis = 1, skipna = True)
        #if (max(y) > pmax): pmax = max(y)
        pwr.append(np.mean(y))
#    pwr = [pwr[i]*100/pmax for i in range(len(pwr))]

    return pwr
def pwr_time(data, demo, cond, tmin, tmax, boxplot = False):
    #HC sujects
    subs0 = subjects(data, demo, cond, 0)
    #SZ subjects
    subs1 = subjects(data, demo, cond, 1)
    #average quantity of time-points of the given period
    n = (mpoints(data, demo, cond, subs0, tmin, tmax) + mpoints(data, demo, cond, subs1, tmin, tmax))//2 #average quantity of time-points of the given period
    #building a boxplot with mean values of the given period
    if boxplot:
        df = pd.DataFrame(columns = ['Amplitude', 'Group', 'Condition'])
        for cond in range(1,4,1):
            ex0 = mbox(data, demo, cond, n, subs0, tmin)
            ex1 = mbox(data, demo, cond, n, subs1, tmin)
            temp = pd.DataFrame({'Amplitude': ex0+ex1})
            temp['Group'] =['HC' if i < len(subs0) else 'SZ' for i in range(len(ex0+ex1))]
            temp['Condition'] = ['Button Tone' if (cond == 1)  else "Play Tone" if (cond == 2)  else "Button Alone" for i in subs0+subs1]
            df = df.append(temp)
        sns.set(style="whitegrid")
        sns.boxplot(x = 'Condition', y = 'Amplitude', hue='Group', data = df)
        #plt.savefig('mean-boxplot-P200.png', orientation='landscape')
        plt.show()
    #building simple plot with averaged power for each group
    else:
        time0, pwr0 = mplot(data, demo, cond, n, subs0, tmin)
        time1, pwr1 = mplot(data, demo, cond, n, subs1, tmin)
        #pwr0 = np.sqrt(pwr0)
        #m0 = max(pwr0)
        #pwr0 = [pwr0[i]*100/m0 for i in range(len(pwr0))]
        #pwr1 = np.sqrt(pwr1)
        #m1 = max(pwr1)
        #pwr1 = [pwr1[i]*100/m1 for i in range(len(pwr1))]
        fig, ax = plt.subplots(figsize=(13, 10), facecolor = 'w')
        ax.plot(time0[:], pwr0[:], color = 'k', label = 'HC')
        ax.plot(time1[:], pwr1[:], color = 'b', label = 'SZ')
        ax.legend( loc='bsubs0 = subjects(data, demo, cond, 0)est', borderaxespad=1)
        plt.show()
    return 0

def ttest-in(data, demo, cond, group, tmin, tmax):
    subs = subjects(data, demo, cond, group)
