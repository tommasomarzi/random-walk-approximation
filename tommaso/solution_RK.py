#%%

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import RK45

#%%
#----------------- Global parameters ---------------#
#---------------------------------------------------#

N = 35

K_1 = K_4 = 0.1
K_2 = K_3 = 1.
v_1 = v_4 = 1.
v_3 = v_2 = 3.0

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

def build_G():
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
    return G, occ


#%%
#----------------- Define function -----------------#
#---------------------------------------------------#

def P_dot(t,P):
    return np.matmul(G, P)


# %%
#----------------- Initialization ------------------#
#---------------------------------------------------#

G, occupation_vector = build_G()
P_0 = occupation_vector/occupation_vector.sum()


# %%
#------------------ RUNGE-KUTTA --------------------#
#---------------------------------------------------#

res = RK45(P_dot, t0 = 0, y0 = P_0, t_bound = 1000)

t_values = []
P_values = []
for i in range(100):
    res.step()
    t_values.append(res.t)
    P_values.append(res.y)

    if res.status == 'finished':
        break


#%%
#------------------ Get solution -------------------#
#---------------------------------------------------#

solution = P_values[-1]/(np.sum(P_values[-1]))
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
