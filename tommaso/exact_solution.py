#%%

import numpy as np
from numpy import linalg as LA
import math
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import itertools
import scipy

#%%
#----------------- Global parameters ---------------#
#---------------------------------------------------#

N = 15

K_1 = K_4 = 0.1
K_2 = K_3 = 1.
v_1 = v_4 = 1.
v_3 = v_2 = 1.8

k1 = K_1
k2 = K_2 
k3 = K_3 
k4 = K_4 
v1 = v_1 
v2 = v_2 
v3 = v_3 
v4 = v_4 

#%%
#----------------- Transition rates ----------------#
#---------------------------------------------------#

def pi_AA(xA,xB,xC, k1, k2, v1, v2):
    AA = 0.
    return AA

def pi_AB(xA,xB,xC, k1, k2, v1, v2):
    AB = k1*v2/(k2*k1 + k1*xB + k2*xC)
    return AB

def pi_AC(xA,xB,xC, k1, k2, v1, v2):
    AC = 0. # k*(N - n_A - n_B)/N
    return AC

def pi_BA(xA,xB,xC, k1, k2, v1, v2):
    BA = k2*v1/(k1*k2 + k1*xB + k2*xA)
    return BA

def pi_BB(xA,xB,xC, k1, k2, v1, v2):
    BB = 0.
    return BB

def pi_BC(xA,xB,xC, k1, k2, v1, v2):
    BC = k2*v1/(k2*k1 + k1*xB + k2*xC)
    return BC

def pi_CA(xA,xB,xC, k1, k2, v1, v2):
    CA = 0.  # k*n_A/N
    return CA

def pi_CB(xA,xB,xC, k1, k2, v1, v2):
    CB = k1*v2/(k1*k2 + k1*xB + k2*xA)
    return CB

def pi_CC(xA,xB,xC, k1, k2, v1, v2):
    CC = 0.
    return CC

#%%
#----------------- Rates decision ------------------#
#---------------------------------------------------#

def return_rate(i,j,k,l,m,n):
    assert((j+l+n) == N)

    a1 = i-j
    a2 = k-l
    a3 = m-n

    assert((a1+a2+a3) == 0)

    if a1 == 0:
        if ((a2 == 1) and (a3 == -1)):
            g = pi_BC(j/N, l/N, n/N, k1, k2, v1, v2)
        elif ((a2 == -1) and (a3 == 1)):
            g = pi_CB(j/N, l/N, n/N, k1, k2, v1, v2)
        elif ((a2 == 0) and (a3 == 0)):
            AB = pi_AB(j/N, l/N, n/N, k1, k2, v1, v2)
            BA = pi_BA(j/N, l/N, n/N, k1, k2, v1, v2)
            AC = pi_AC(j/N, l/N, n/N, k1, k2, v1, v2)
            CA = pi_CA(j/N, l/N, n/N, k1, k2, v1, v2)
            BC = pi_BC(j/N, l/N, n/N, k1, k2, v1, v2)
            CB = pi_CB(j/N, l/N, n/N, k1, k2, v1, v2)
            g = - (AB+BA+AC+CA+BC+CB)
        else:
            raise ValueError('Encountered impossible combination of alphas (1).')
    elif a1 == 1:
        if ((a2 == -1) and (a3 == 0)):
            g = pi_AB(j/N, l/N, n/N, k1, k2, v1, v2)
        elif ((a2 == 0) and (a3 == -1)):
            g = pi_AC(j/N, l/N, n/N, k1, k2, v1, v2)
        else:
            raise ValueError('Encountered impossible combination of alphas (2).')
    elif a1 == -1:
        if ((a2 == 1) and (a3 == 0)):
            g = pi_BA(j/N, l/N, n/N, k1, k2, v1, v2)
        elif ((a2 == 0) and (a3 == 1)):
            g = pi_CA(j/N, l/N, n/N, k1, k2, v1, v2)
        else:
            raise ValueError('Encountered impossible combination of alphas (3).')

    return g    


#%%
#----------------- Define G matrix -----------------#
#---------------------------------------------------#

G = np.zeros((((N+1)**3),((N+1)**3)))

dim_1 = list(np.arange((N+1)**3))

cnt_1_x = 0
cnt_2_x = 0
cnt_3_x = 0

occupation_filled = False

for i in dim_1:
    cnt_1_y = 0
    cnt_2_y = 0
    cnt_3_y = 0

    occupation = np.zeros(((N+1)**3))

    for j in dim_1:
        
        alpha_1 = cnt_1_x - cnt_1_y
        alpha_2 = cnt_2_x - cnt_2_y
        alpha_3 = cnt_3_x - cnt_3_y
        
        if ((cnt_1_y + cnt_2_y + cnt_3_y) != N):
            cnt_3_y += 1
            if (cnt_3_y == (N+1)):
                cnt_3_y = 0
                cnt_2_y += 1
                if (cnt_2_y == (N+1)):
                    cnt_2_y = 0
                    cnt_1_y += 1
            continue 

        occupation[j] = 1.

        if ((alpha_3 < -1) or (alpha_3 > 1)):
            cnt_3_y += 1
            if (cnt_3_y == (N+1)):
                cnt_3_y = 0
                cnt_2_y += 1
                if (cnt_2_y == (N+1)):
                    cnt_2_y = 0
                    cnt_1_y += 1
            continue
        
        if ((alpha_2 < -1) or (alpha_2 > 1)):
            cnt_3_y += 1
            if (cnt_3_y == (N+1)):
                cnt_3_y = 0
                cnt_2_y += 1
                if (cnt_2_y == (N+1)):
                    cnt_2_y = 0
                    cnt_1_y += 1
            continue
        
        if ((alpha_1 < -1) or (alpha_1 > 1)):
            cnt_3_y += 1
            if (cnt_3_y == (N+1)):
                cnt_3_y = 0
                cnt_2_y += 1
                if (cnt_2_y == (N+1)):
                    cnt_2_y = 0
                    cnt_1_y += 1
            continue

        if ((alpha_1 + alpha_2 + alpha_3) != 0):
            cnt_3_y += 1
            if (cnt_3_y == (N+1)):
                cnt_3_y = 0
                cnt_2_y += 1
                if (cnt_2_y == (N+1)):
                    cnt_2_y = 0
                    cnt_1_y += 1
            continue

        G[i,j] = return_rate(cnt_1_x, cnt_1_y, cnt_2_x, cnt_2_y, cnt_3_x, cnt_3_y)

        cnt_3_y += 1
        if (cnt_3_y == (N+1)):
            cnt_3_y = 0
            cnt_2_y += 1
            if (cnt_2_y == (N+1)):
                cnt_2_y = 0
                cnt_1_y += 1

    #NEW METHOD
    if not occupation_filled:
        if np.all(G[i] == 0):
            G[i] = occupation
            occupation_filled = True
            idx = i.copy()
            occ = occupation.copy()
    
    cnt_3_x += 1
    if (cnt_3_x == (N+1)):
        cnt_3_x = 0
        cnt_2_x += 1
        if (cnt_2_x == (N+1)):
            cnt_2_x = 0
            cnt_1_x += 1
            
#%%
#--------------- Constraint addition ---------------#
#---------------------------------------------------#

b = np.zeros(((N+1)**3))

# OLD METHOD
#idx = np.where(~G.any(axis=1))[0][0]
#G[idx] = 1.

b[idx] = 1


#%%
#-------------------- Solution ---------------------#
#---------------------------------------------------#

P = scipy.linalg.lstsq(G,b)
solution = P[0]
solution = solution.reshape((N+1,N+1,N+1))

dim = list(np.arange(N+1))
for i, j, k in itertools.product(dim, dim, dim):
    if ((i+j+k) != N):
        solution[i,j,k] = 0

Z = solution.max(axis = -2)

#%%
#------------------ Plot solution ------------------#
#---------------------------------------------------#

fig = plt.figure(figsize=(8,6))
plt.imshow(Z.T/Z.sum(),origin='lower',interpolation='nearest')
plt.colorbar()
plt.title(u"Solution with $K_1 = {}$ , $K_2 = {}$, $v_1 = {}$, $v_2 = {}$ and $N = {}$".format(K_1, K_2, v_1, v_2, N))
plt.xlabel("$n_A$",size=14)
plt.ylabel("$n_C$",size=14)
plt.tight_layout()
plt.show()


#%%
#-------------- Gaussian elimination ---------------#
#---------------------------------------------------#

#P = scipy.linalg.solve(G,np.zeros(((N+1)**3)))  #Not working! singular matrix
P = scipy.linalg.lstsq(G,np.zeros(((N+1)**3)))
print(P[0])

#%%
#----------------- LU decomposition ----------------#
#---------------------------------------------------#

_,_,U = scipy.linalg.lu(G)

#P = scipy.linalg.solve(U,np.zeros(((N+1)**3)))  #Not working! singular matrix
P = scipy.linalg.lstsq(U,np.zeros(((N+1)**3)))
print(P[0])

#%%
#------------------ OLD BLOCK METHOD ---------------#
#---------------------------------------------------#

"""
def second_level(i, j, alpha_1, k, l, alpha_2):
    G_sl = np.zeros((N+1,N+1), dtype=object)
    for m in dim:
        for n in dim:
            if ((j + l + n) != N):
                continue
            alpha_3 = m-n
            if ((alpha_3 < -1) or (alpha_3 > 1)):
                continue
            elif ((alpha_1 + alpha_2 + alpha_3) != 0):
                continue
            else:        
                G_sl[m,n] = return_rate(i, j, k, l, m, n)
    return G_sl

def first_level(i, j, alpha_1):
    
    G_fl = np.zeros((N+1,N+1), dtype=object)

    for k in dim:
        for l in dim:
            if ((j + l) > N):
                continue
            alpha_2 = k-l
            if ((alpha_2 < -1) or (alpha_2 > 1)):
                continue
            elif ((alpha_1 != 0) and (alpha_2 == alpha_1)):
                continue
            else:
                G_fl[k,l] = second_level(i, j, alpha_1, k, l, alpha_2) 
    return G_fl


dim = list(np.arange(N+1))

G = np.zeros((N+1,N+1), dtype=object)

for i in dim:
    for j in dim:
        alpha_1 = i-j
        if ((alpha_1 < -1) or (alpha_1 > 1)):
            continue
        else:
            G[i,j] = first_level(i, j, alpha_1)


zero = np.zeros((N+1), dtype=object)
cnv = np.zeros((N+1), dtype=object)
for i in dim:
    zero[i] = cnv
    for j in dim:
        zero[i][j] = cnv

"""


#%%
#-------------- TENSOR-LIKE METHOD -----------------#
#---------------------------------------------------#

"""
dim = list(np.arange(N+1))

G = np.zeros((N+1,N+1,N+1,N+1,N+1,N+1))

for i in dim:
    for j in dim:
        alpha_1 = i-j
        if ((alpha_1 < -1) or (alpha_1 > 1)):
            continue
        else:
            for k in dim:
                for l in dim:
                    if ((j + l) > N):
                        continue
                    alpha_2 = k-l
                    if ((alpha_2 < -1) or (alpha_2 > 1)):
                        continue
                    elif ((alpha_1 != 0) and (alpha_2 == alpha_1)):
                        continue
                    else:
                        for m in dim:
                            for n in dim:
                                if ((j + l + n) != N):
                                    continue
                                alpha_3 = m-n
                                if ((alpha_3 < -1) or (alpha_3 > 1)):
                                    continue
                                elif ((alpha_1 + alpha_2 + alpha_3) != 0):
                                    continue
                                else:        
                                    G[i,j,k,l,m,n] = return_rate(i, j, k, l, m, n)


G1 = np.zeros((N+1,N+1,N+1,N+1,N+1,N+1))
for i, j in itertools.product(dim, dim):
    alpha_1 = i-j
    if ((alpha_1 < -1) or (alpha_1 > 1)):
        continue
    else:
        for k, l in itertools.product(dim, dim):
            if ((j + l) > N):
                continue
            alpha_2 = k-l
            if ((alpha_2 < -1) or (alpha_2 > 1)):
                continue
            elif ((alpha_1 != 0) and (alpha_2 == alpha_1)):
                continue
            else:
                for m, n in itertools.product(dim, dim):
                    if ((j + l + n) != N):
                        continue
                    alpha_3 = m-n
                    if ((alpha_3 < -1) or (alpha_3 > 1)):
                        continue
                    elif ((alpha_1 + alpha_2 + alpha_3) != 0):
                        continue
                    else:        
                        G1[i,j,k,l,m,n] = return_rate(i, j, k, l, m, n)
"""
