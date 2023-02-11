import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt
from scipy.integrate import RK45
from tqdm import tqdm
from tqdm.contrib.itertools import product
import os
from time import sleep
import itertools


#----------------- Global parameters ---------------#
#---------------------------------------------------#

K_1 = K_4 = 0.1
K_2 = K_3 = 1.
v_1 = v_4 = 1.
v_2_range = np.linspace(1.5,3.52,26) 
N_min = 5
N_max = 295 
N_step = 10
N_range = np.arange(N_min, N_max+1, N_step)
v_2_range = [1.8, 2.48,3.04]

# RK parameters
t_max = 3000
i_osc = 30                  # period over which we test the error
error_shift = 150           # delay
error_offset = 2.2e-9       # difference

fiedler_list = []


#----------------- Transition rates ----------------#
#---------------------------------------------------#
"""   Functions for the rates of the dual PdPc.    """   

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


#-------------- Generalized Laplacian --------------#
#---------------------------------------------------#

def return_rate(i,j,k,l,N):
    """ 
    Given indexes of the matrix, compute associated rate for the generalized Laplacian.
    Check reference [26] (https://www.sciencedirect.com/science/article/pii/S0377042715005075?via%3Dihub)
    for a detailed description of how the matrix is build.
    Arguments:
        i: first row index (# of exchanged particles first species)
        j: first column index (# of particles first species)
        k: second row index (# of exchanged particles first species)
        l: second column index (# of particles second species)
        N: number of species
    Returns:
        Corresponding value of the generalized Laplacian.
    """   
    a1 = i-j
    a3 = k-l
    a2 = -(a1+a3)

    BA = pi_BA(j/N, 1-(j/N + l/N), l/N, K_1, K_2, v_1, v_2)
    CA = pi_CA(j/N, 1-(j/N + l/N), l/N, K_1, K_2, v_1, v_2)
    AB = pi_AB(j/N, 1-(j/N + l/N), l/N, K_1, K_2, v_1, v_2)
    CB = pi_CB(j/N, 1-(j/N + l/N), l/N, K_1, K_2, v_1, v_2)
    AC = pi_AC(j/N, 1-(j/N + l/N), l/N, K_1, K_2, v_1, v_2)
    BC = pi_BC(j/N, 1-(j/N + l/N), l/N, K_1, K_2, v_1, v_2)
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


def allowed_configurations(N):
    """
    Build allowed configurations.
    Returns:
        occupation: vector with 1 if the configuration is allowed, 0 otherwise
        occupation_states: list of allowed couples
    """
    dim = list(np.arange((N+1)))
    occupation = np.zeros(((N+1)**2))
    occupation_states = []
    for i,j in itertools.product(dim, dim):
        if (i+j <= N):
            occupation[i*(N+1)+j] = 1.
            occupation_states.append((i,j))

    return occupation, occupation_states


def custom_sum(i, N):
    """
    Custom sum for indexes iteration in filtered generalized Laplacian.
    """
    if i == 0:
        res = 0
    else:
        res = np.sum([(N + 1 - idx) for idx in np.arange(i)])
    return res


def build_G_filtered(occupation_states, N):
    """
    Build generalized Laplacian filtered with the allowed configurations.
    Check reference [26] (https://www.sciencedirect.com/science/article/pii/S0377042715005075?via%3Dihub)
    for a detailed description of how the matrix is build.
    The filtering strongly reduces the dimensionality of the matrix.
    """
    filtered_dim = len(occupation_states)

    G = np.zeros((filtered_dim,filtered_dim))

    for i,j in tqdm(list(itertools.product(occupation_states, occupation_states))):
        alpha_1 = i[0] - j[0]
        alpha_2 = i[1] - j[1]

        if ((alpha_2 < -1) or (alpha_2 > 1)):
            continue
        if ((alpha_1 < -1) or (alpha_1 > 1)):
            continue

        G[custom_sum(i[0], N)+i[1],custom_sum(j[0], N)+j[1]] = return_rate(i[0], j[0], i[1], j[1], N)

    return G


#--------------------- Utils -----------------------#
#---------------------------------------------------#

def get_fiedler(matrix):
    """
    Get fiedler eigenvalue for given matrix.
    """
    eigvals, _ = LA.eig(matrix)
    fielder = np.sort(np.abs(eigvals))[1]
    return fielder


def get_eigvals(matrix):
    """
    Get spectrum for given matrix.
    """
    eigvals, _ = LA.eig(matrix)
    return eigvals


def P_dot(t,P):
    """
    Define positive autonomous linear system to be solved with RK.
    Reference to equation A3
    """
    return np.matmul(G, P)

def error_L1(p_curr, p_prev, N):
    """
    L1 norm
    """
    return np.sum(np.abs(p_curr-p_prev))

def error_L2(p_curr, p_prev, N):
    """
    L2 norm
    """
    return np.sqrt(np.sum((p_curr-p_prev)**2))


#--------------- Spectral analysis -----------------#
#---------------------------------------------------#

eigvals_list = []

for v_2 in v_2_range:
    for N in N_range:
        v_3 = v_2
        vec, states = allowed_configurations(N)
        G = build_G_filtered(states,N)
        eigv = get_eigvals(G)
        eigvals_list.append(eigv)


#---------------- Run RK algorithm -----------------#
#---------------------------------------------------#

L1_values = []
L2_values = []

for v_2 in v_2_range:
    for N in N_range:
        v_3 = v_2

        vec, states = allowed_configurations(N)
        G = build_G_filtered(states,N)
        P_0 = np.ones(len(states))
        P_0 = P_0/np.sum(P_0)
        f = get_fiedler(G)
        fiedler_list.append(f)

        res = RK45(P_dot, t0 = 0, y0 = P_0, t_bound = t_max)

        t_values = []
        P_values = []
        error_values_L1 = []
        error_values_L2 = []
        P_values.append(P_0)

        for i in tqdm(range(t_max)):
            res.step()
            t_values.append(res.t)
            P_values.append(res.y)
            
            error_values_L1.append(error_L1(P_values[i],P_values[i-1],N))
            error_values_L2.append(error_L2(P_values[i],P_values[i-1],N))
            if i > error_shift + i_osc:
                if np.alltrue(np.abs(np.subtract(error_values_L1[(i-i_osc):i],
                error_values_L1[(i-i_osc - error_shift):(i - error_shift)])) < error_offset):
                    break
            
            if res.status == 'finished':
                break

        solution = np.zeros(vec.shape)
        solution[vec == 1] = P_values[-1]/(np.sum(P_values[-1]))
        solution = solution.reshape((N+1,N+1))
        fig = plt.figure(figsize=(8,6))
        plt.imshow(solution,origin='lower',interpolation='nearest')
        plt.colorbar()
        plt.title(u"Solution with $K_1 = {}$ , $K_2 = {}$, $v_1 = {}$, $v_2 = {}$ and $N = {}$".format(K_1, K_2, v_1, f"{v_2:.2f}", N))
        plt.xlabel("$n_A$",size=14)
        plt.ylabel("$n_C$",size=14)
        plt.tight_layout()
        plt.show()
        L1_values.append(error_values_L1)
        L2_values.append(error_values_L2)


#------------- Convergence analysis ----------------#
#---------------------------------------------------#

cutoff = [2,800]
fig, ax = plt.subplots()
for idx, data in enumerate(L1_values):
    ax.plot(data[cutoff[0]: cutoff[1]],label='v_2 = {}'.format(v_2_range[idx]))
ax.set_title("Convergence in time")
ax.set_ylabel("Error",size=14)
ax.set_xlabel("Time",size=14)
ax.set_yscale('log')
ax.set_ylim([1e-4,1e-1])
ax.legend()
fig.tight_layout()
fig.show()
