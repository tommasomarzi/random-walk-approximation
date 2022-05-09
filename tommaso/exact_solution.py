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
import sympy

#%%
#----------------- Global parameters ---------------#
#---------------------------------------------------#


N = 3

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
    AB = xB*k1*v2/(k2*k1 + k1*xB + k2*xC)
    return AB

def pi_AC(xA,xB,xC, k1, k2, v1, v2):
    AC = 0. # k*(N - n_A - n_B)/N
    return AC

def pi_BA(xA,xB,xC, k1, k2, v1, v2):
    BA = xA*k2*v1/(k1*k2 + k1*xB + k2*xA)
    return BA

def pi_BB(xA,xB,xC, k1, k2, v1, v2):
    BB = 0.
    return BB

def pi_BC(xA,xB,xC, k1, k2, v1, v2):
    BC = xC*k2*v1/(k2*k1 + k1*xB + k2*xC)
    return BC

def pi_CA(xA,xB,xC, k1, k2, v1, v2):
    CA = 0.  # k*n_A/N
    return CA

def pi_CB(xA,xB,xC, k1, k2, v1, v2):
    CB = xB*k1*v2/(k1*k2 + k1*xB + k2*xA)
    return CB

def pi_CC(xA,xB,xC, k1, k2, v1, v2):
    CC = 0.
    return CC


#%%
#----------------- Rates decision ------------------#
#---------------------------------------------------#

def return_rate(i,j,k,l):

    a1 = i-j
    a3 = k-l
    a2 = -(a1+a3)

    assert((a1+a2+a3) == 0)

    BA = pi_BA(j/N, 1-(j/N + l/N), l/N, k1, k2, v1, v2)
    CA = pi_CA(j/N, 1-(j/N + l/N), l/N, k1, k2, v1, v2)
    AB = pi_AB(j/N, 1-(j/N + l/N), l/N, k1, k2, v1, v2)
    CB = pi_CB(j/N, 1-(j/N + l/N), l/N, k1, k2, v1, v2)
    AC = pi_AC(j/N, 1-(j/N + l/N), l/N, k1, k2, v1, v2)
    BC = pi_BC(j/N, 1-(j/N + l/N), l/N, k1, k2, v1, v2)
    g = 0.
    if a1 == 0:
        if ((a2 == 1) and (a3 == -1)):
            g = BC
        elif ((a2 == -1) and (a3 == 1)):
            g = CB
        elif ((a2 == 0) and (a3 == 0)):
            if j != 0:
                g -= BA 
                g -= CA
            if l != 0:
                g -= AC
                g -= BC
            if (N-j-l) != 0:
                g -= AB
                g -= CB
        else:
            g = 0.
    elif a1 == 1:
        if ((a2 == -1) and (a3 == 0)):
            g = AB
        elif ((a2 == 0) and (a3 == -1)):
            g = AC
        else:
            g = 0.
    elif a1 == -1:
        if ((a2 == 1) and (a3 == 0)):
            g = BA
        elif ((a2 == 0) and (a3 == 1)):
            g = CA
        else:
            g = 0.

    return g    


#%%
#----------------- Define G matrix -----------------#
#---------------------------------------------------#

G = np.zeros((((N+1)**2),((N+1)**2)))

dim_1 = list(np.arange((N+1)**2))

cnt_1_x = 0
cnt_2_x = 0

do_occupation = True
occupation_filled = False
disallowed = True

for i in dim_1:
    cnt_1_y = 0
    cnt_2_y = 0

    occupation = np.zeros(((N+1)**2))

    for j in dim_1:
        
        alpha_1 = cnt_1_x - cnt_1_y
        alpha_2 = cnt_2_x - cnt_2_y
        alpha_3 = - (alpha_1 + alpha_2)
        
        if ((cnt_1_y + cnt_2_y) > N):
            cnt_2_y += 1
            if (cnt_2_y == (N+1)):
                cnt_2_y = 0
                cnt_1_y += 1
            continue 

        occupation[j] = 1.

        if ((alpha_2 < -1) or (alpha_2 > 1)):
            cnt_2_y += 1
            if (cnt_2_y == (N+1)):
                cnt_2_y = 0
                cnt_1_y += 1
            continue
        
        if ((alpha_1 < -1) or (alpha_1 > 1)):
            cnt_2_y += 1
            if (cnt_2_y == (N+1)):
                cnt_2_y = 0
                cnt_1_y += 1
            continue

        G[i,j] = return_rate(cnt_1_x, cnt_1_y, cnt_2_x, cnt_2_y)

        cnt_2_y += 1
        if (cnt_2_y == (N+1)):
            cnt_2_y = 0
            cnt_1_y += 1
    
    if do_occupation:
        if not occupation_filled:
            if np.all(G[i] == 0):
                occupation_filled = True
                idx = i.copy()
                occ = occupation.copy()
    
    cnt_2_x += 1
    if (cnt_2_x == (N+1)):
        cnt_2_x = 0
        cnt_1_x += 1

#%%
#------------ Mass-balance constraint --------------#
#---------------------------------------------------#

idx_list = np.where(occ == 0)[0]

G_del = np.delete(G, idx_list, axis = 0)
G_del = np.delete(G_del, idx_list, axis = 1)

assert(np.allclose(np.sum(G_del, axis = 0), np.zeros(G_del.shape[0])))

#%%
#--------------- Gaussian elimination --------------#
#---------------------------------------------------#
b = np.zeros(G_del.shape[0])
n = len(b)
x = np.zeros(n, float)

offset = 1e-15

for k in range(n-1):
    if np.abs(G_del[k,k]) < 1.0e-12:
        for i in range(k+1, n):
            if np.abs(G_del[i,k]) > np.abs(G_del[k,k]):
                G_del[[k,i]] = G_del[[i,k]]
                b[[k,i]] = b[[i,k]]
                break

    for i in range(k+1,n):
        if G_del[i,k] == 0:continue

        #factor = G_del[k,k]/G_del[i,k]
        factor_1 = G_del[i,k].copy() 
        factor_2 = G_del[k,k].copy()

        for j in range(k,n):
            G_del[i,j] = G_del[k,j]*factor_1 - G_del[i,j]*factor_2
            #G_del[i,j] = G_del[k,j] - G_del[i,j]*factor
        
        G_del[i, np.abs(G_del[i,:]) < offset] = 0
        
        b[i] = b[k]*factor_1 - b[i]*factor_2

#%%
#------------ Normalization constraint -------------#
#---------------------------------------------------#

empty_idx = np.where(~G_del.any(axis=1))[0][0]
G_del[empty_idx] = 1.0
b[empty_idx] = 1.0



#%%
#----------------- Find solution -------------------#
#---------------------------------------------------#

P = scipy.linalg.solve(G_del,b)

solution = occ.copy()
solution[occ == 1] = P
solution = solution.reshape((N+1,N+1))


#%%
#------------------ Plot solution ------------------#
#---------------------------------------------------#

fig = plt.figure(figsize=(8,6))
plt.imshow(solution.T,origin='lower',interpolation='nearest')
plt.colorbar()
plt.title(u"Solution with $K_1 = {}$ , $K_2 = {}$, $v_1 = {}$, $v_2 = {}$ and $N = {}$".format(K_1, K_2, v_1, v_2, N))
plt.xlabel("$n_A$",size=14)
plt.ylabel("$n_C$",size=14)
plt.tight_layout()
plt.show()

#%%
#-------------------- Solution ---------------------#
#---------------------------------------------------#

P = scipy.linalg.lstsq(G_del,b)
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
