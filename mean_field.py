# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import os
import math
from math import factorial  # for multinomial distribution
from mpl_toolkits import mplot3d  # for 3D plotd

#%%

def null_eigenvector_eq(k1,k2,v1,v2):
    xB = 1/(1 + (k1*v2*2/(k2*v1)))
    xA = xC = (1-xB)/2
    return xA, xB, xC


def null_eigenvector_neq(k1,k2,v1,v2, N):
    xB = k2*v1/(v2 - v1)

    B = - (v2*(k1 + N - k1**2) - v1*(N + k2))/(v2 - v1)
    C = (v2*k1*k1*(1 + v1/(v2 - v1)))/(v2 - v1)

    xA_1 = (-B + math.sqrt(B**2 - 4*C))/(2*N)
    xA_2 = (-B - math.sqrt(B**2 - 4*C))/(2*N)

    if (xA_1 < 0) or (xA_1 > 1):
        print("A problem occured")

    if (xA_2 < 0) or (xA_2 > 1):
        print("A problem occured")
    
    xB = xB/N

    xC_1 = 1 - xA_1 - xB
    xC_2 = 1 - xA_2 - xB
    
    return xA_1, xB, xC_1, xA_2, xC_2

#%%
def p_star_old(x, y, n_C):
    pi_AA = 1 - K_2*x/(K_1*K_2 + K_1*y + K_2*x)
    pi_AB = v_2*K_1/(K_1*K_2 + K_1*y + K_2*n_C)
    pi_AC = 0  # k*(N - n_A - n_B)/N
    pi_BA = K_2*v_1/(K_1*K_2 + K_1*y + K_2*x)
    pi_BB = 1 - y*K_1/(K_1*K_2 + K_1*y + K_2*n_C) - y*K_1/(K_1*K_2 + K_1*y + K_2*x)
    pi_BC = K_2*v_1/(K_1*K_2 + K_2*n_C + K_1*y)
    pi_CA = 0  # k*n_A/N
    pi_CB = v_2*K_1/(K_1*K_2 + K_1*y + K_2*x)
    pi_CC = 0#1 - K_2*n_C/(K_1*K_2 + K_2*n_C + K_1*y)
    Z = pi_AC*pi_AB + pi_AB*pi_BC + pi_AC*pi_CB + \
        pi_BC*pi_CA + pi_BA*pi_AC + pi_CA*pi_CB + \
        pi_CA*pi_AB + pi_BC*pi_BA + pi_CB*pi_BA
    p_A_n = (pi_AB*pi_BC + pi_AC*pi_AB + pi_AC*pi_CB)/Z
    p_B_n = (pi_BC*pi_BA + pi_BC*pi_CA + pi_BA*pi_AC)/Z
    p_C_n = 1 - p_A_n - p_B_n
    
    return p_A_n, p_B_n, p_C_n

#%%

def N_threshold(k1,k2,v1,v2):
    N_high = (v1*k2 + v2*k1 + v2*(k1**2))/(v2-v1)
    N_low = (v1*k2 - 3*v2*k1 + v2*(k1**2))/(v2-v1)
    return [N_low, N_high]

#%%
K_1 = K_4 = 1.
K_2 = K_3 = 2.
v_1 = v_4 = 1.
v_2_values = [1.05]#, 1.15] #1.15
N_min = 85
N_max = 105
N_step = 10
K_1_values = [1.0]#, 1.5]
peaks = ['C']
points = ['eq', 'neq']

#%%
for v_2 in v_2_values:
    v_3 = v_2
    for K_1 in K_1_values:
        th = N_threshold(K_1,K_2,v_1,v_2)
        
        for N in np.arange(N_min, N_max + 1, N_step):   
            if ((N <= th[0]) or (N >= th[1])):

                xA1_star, xB_star, xC1_star, xA2_star, xC2_star = null_eigenvector_neq(K_1,K_2,v_1,v_2,N)
                p_1_n_star = p_star_old(xA1_star*N, xB_star*N, xC1_star*N)
                print("Bistability with N: {}   p*({:.2f}, {:.2f}, {:.2f})   p({:.2f}, {:.2f}, {:.2f})".format(N, xA1_star, xB_star, xC1_star, p_1_n_star[0],p_1_n_star[1],p_1_n_star[2]))

            else:
                xA_star, xB_star, xC_star = null_eigenvector_eq(K_1,K_2,v_1,v_2)
                p_n_star = p_star_old(xA_star*N, xB_star*N, xC_star*N)
                print("Monostability with N: {}   p*({:.2f}, {:.2f}, {:.2f})   p({:.2f}, {:.2f}, {:.2f})".format(N, xA_star, xB_star, xC_star, p_n_star[0],p_n_star[1],p_n_star[2]))
               

# %%
