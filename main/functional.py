#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 12:17:34 2019

@author: savenko
"""
import numpy as np
from scipy.sparse import diags
from scipy.special import gamma

def delta(n, fex, feh):
    ex = np.kron([x for x in range(n-1, 0, -1)], np.ones((n,1)))
    if (n%2) == 0:
        dx = -np.pi**2/(3*feh**2)-(1/6)*np.ones((n,1))
        test_bx = -(-1)**ex*(0.5)/(np.sin(ex*feh*0.5)**2)
        test_tx =  -(-1)*(-ex)*(0.5)/(np.sin((-ex)*feh*0.5)**2)
    else:
        dx = -np.pi**2/(3*feh**2)-(1/12)*np.ones((n,1))
        test_bx = -0.5*((-1)**ex)*np.tan(ex*feh*0.5)**-1/(np.sin(ex*feh*0.5))
        test_tx = -0.5*((-1)**(-ex))*np.tan((-ex)*feh*0.5)**-1/(np.sin((-ex)*feh*0.5))
    
    rng = [x for x in range(-n+1, 1, 1)] + [y for y in range(n-1, 0, -1)]    
    Ex = diags(np.array(np.concatenate((test_bx, dx, test_tx), axis = 1)).T, np.array(rng), n, n).toarray()
    Dx=(feh/fex)**2*Ex
    return Dx


def scsa(y, D, h):
    Y = np.diag(y)
    gm = 0.5
    Lcl = (1/(2*(np.pi)**2))*(gamma(gm+1)/gamma(gm+(3/2)))
    SC = -h*h*D-Y
    psi,lamda = np.linalg.eig(SC)
    temp = np.diag(lamda)
    temp = temp[temp<0]
    kappa = np.diag((-temp)**gm); 
    Nh = kappa.shape[0]
    psin = psi(:,ind(:,1));
    
    