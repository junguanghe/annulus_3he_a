#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:28:35 2020

@author: junguanghe
"""

import numpy as np
from scipy import optimize
from scipy import integrate
import matplotlib.pyplot as plt

############################# The GL equation
def GL(u,h,dtheta):
    eta1 = u[0,:]+u[1,:]*1j
    eta2 = u[2,:]+u[3,:]*1j
    eta1 = np.concatenate((np.array([eta1[0]]),eta1,np.array([eta1[-1]])))
    eta2 = np.concatenate((np.array([0]),eta2,np.array([0])))
    
    ees = eta1*np.conj(eta1) + eta2*np.conj(eta2)
    ee  = eta1**2 + eta2**2
    
    G1 = -eta1 + 0.5*ees*eta1 + 0.25*ee*np.conj(eta1) + dtheta**2*eta1 - 2*(-dtheta**2*eta1+1j*dtheta*np.gradient(eta2,h))
    G2 = -eta2 + 0.5*ees*eta2 + 0.25*ee*np.conj(eta2) + dtheta**2*eta2 - 2*(1j*dtheta*np.gradient(eta1,h))
    G1 = G1[1:-1] - np.diff(eta1, n=2)/h**2
    G2 = G2[1:-1] - 3*np.diff(eta2, n=2)/h**2
    
    G = np.array([G1.real, G1.imag, G2.real, G2.imag])
    
    return G
############################ The GL equation

############################ solve GL
def lateral_flow(h,dtheta,eta0):
    u0 = np.array([eta0[0,1:-1].real, eta0[0,1:-1].imag, eta0[1,1:-1].real, eta0[1,1:-1].imag])
    sol = optimize.root(GL, u0, args=(h,dtheta), method='krylov',options={'disp':False, 'fatol':6e-12})
    u = sol.x
    
    eta1 = u[0,:]+u[1,:]*1j
    eta2 = u[2,:]+u[3,:]*1j
    eta1 = np.concatenate((np.array([eta1[0]]),eta1,np.array([eta1[-1]])))
    eta2 = np.concatenate((np.array([0]),eta2,np.array([0])))
    
    ees = eta1*np.conj(eta1) + eta2*np.conj(eta2)
    ee  = eta1**2 + eta2**2
    
    de1 = np.gradient(eta1,h)
    de2 = np.gradient(eta2,h)
    
    f = -ees + 0.25*ees**2 + 0.125*ee*np.conj(ee) + ees*dtheta**2\
    +de1*np.conj(de1) + 3*de2*np.conj(de2) + 2*eta1*np.conj(eta1)*dtheta**2\
    +1j*dtheta*(eta1*np.conj(de2)+eta2*np.conj(de1)-np.conj(eta1)*de2-np.conj(eta2)*de1)
    
    F = integrate.trapz(f, dx=h)
#    print("Free Energy=", F)
    
    j1 = 3*np.conj(eta1)*eta1*(1j*dtheta) + np.conj(eta2)*eta2*(1j*dtheta)\
    +np.conj(eta2)*de1 + np.conj(eta1)*de2
    
    j2 = 3*np.conj(eta2)*de2 + np.conj(eta1)*de1 + np.conj(eta1)*eta2*(1j*dtheta)\
    +np.conj(eta2)*eta1*(1j*dtheta)
    
    return eta1, eta2, f, F, j1, j2
############################ solve GL
    
############################ phase transition
def F_diff_polar_chiral(D,dtheta):
    ny = np.max([301, np.ceil(D/0.1)+1]) # ny no less than 301, h no larger than 0.1
    y, h = np.linspace(-D/2, D/2, ny, retstep=True)
    
    ############################ initial guess
    eta0_chiral = np.array([np.ones_like(y), np.tanh(D/2-np.abs(y))*1j], dtype=complex)        # chiral phase p+ip
    ############################ initial guess
    
    F_polar = -2/3*D*(3*(dtheta)**2-1)**2
    _,eta2,_,F,_,_ = lateral_flow(h,dtheta,eta0_chiral)
    if np.amax(np.abs(eta2))<1e-6:
        F = 0
    return F.real-F_polar
############################ phase transition 

n = 10
dtheta_span = np.linspace(0,0.3,n)
D_trans = np.zeros_like(dtheta_span)

for ii in range(n):
    dtheta = dtheta_span[ii]
    D_trans[ii] = optimize.newton(F_diff_polar_chiral,7,args=(dtheta,),tol=1e-3)
    print(ii, D_trans[ii], dtheta)
    
plt.plot(D_trans,dtheta_span)
plt.xlim(0,30)
plt.ylim(0,1/np.sqrt(3))
plt.show()

np.savez('polar_vs_chiral.npz',D_trans=D_trans,dtheta_span=dtheta_span)