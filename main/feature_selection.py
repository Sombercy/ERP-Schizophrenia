#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:04:17 2019

@author: savenko
"""
import numpy as np

def t_comp(time, lamda, psi):
    #returns indexes of eigenvalues related to major time component eigenfunctions
    PN100 = []
    N100 = []
    P200 = []
    P300 = []
    n = len(lamda)
    for i in range(n):
        #getting indexes of 
        #print(psi[list(
         #       set(time[time >= 40].index).intersection(set(time[time <= 60].index))), i])
        
        PN100.append(time[time >= 40].index[0] + np.argmax(psi[list(
                set(time[time >= 40].index).intersection(set(time[time <= 60].index))), i], axis = 0))
        N100.append(time[time >= 80].index[0] + np.argmax(psi[list(
                set(time[time >= 80].index).intersection(set(time[time <= 120].index))), i], axis = 0))
        P200.append(time[time >= 150].index[0] + np.argmax(psi[list(
                set(time[time >= 150].index).intersection(set(time[time <= 200].index))), i], axis = 0))
        P300.append(time[time >= 250].index[0] + np.argmax(psi[list(
                set(time[time >= 250].index).intersection(set(time[time <= 300].index))), i], axis = 0))
    #print(N100)
    PN100 = np.argmax([psi[PN100[i], i] for i in range(n)])
    N100 = np.argmax([psi[N100[i], i] for i in range(n)])
    P200 = np.argmax([psi[P200[i], i] for i in range(n)])
    P300 = np.argmax([psi[P300[i], i] for i in range(n)])
    return [PN100, N100, P200, P300]

def ngreates(n, lamda, psi):
    #returns indexes of eigenfunction time components with greatest eigenvalues
    index = np.argsort(lamda)[::-1]
    index = index[:n]
    lamda = [lamda[i] for i in index]
    psi = psi[:, index]
    features = [np.argmax(psi[:, i], axis = 0) for i in range(psi.shape[1]) ]
    return (np.array(features)%100).tolist()
    