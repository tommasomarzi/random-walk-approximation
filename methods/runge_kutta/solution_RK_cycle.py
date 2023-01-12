# %%
import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt
from scipy.integrate import RK45
from tqdm import tqdm
from tqdm.contrib.itertools import product
import os
#os.chdir("/home/PERSONALE/stefano.polizzi/spolizzi/laplacian graph theory")
#os.chdir("/home/PERSONALE/stefano.polizzi")
#path = os.path.abspath(os.getcwd()) # %pwd
from time import sleep
import itertools

# %%

K_1 = K_4 = 0.1
K_2 = K_3 = 1.
v_1 = v_4 = 1.
v_2_range = np.linspace(1.5,3.52,26) # for server bio07 to reduce time
#np.linspace(1.5,3.52,51)
#np.linspace(1.62,3.52,51)  #after server interruption
N_min = 5
N_max = 295 #205 for server
N_step = 10
N_range = np.arange(N_min, N_max+1, N_step)
N_range = [100]
v_2_range = [1.8, 2.48,3.04]

t_max = 3000
i_osc = 30 #period over which we test the error
error_shift = 150           #delay
error_offset = 2.2e-9        #difference

fiedler_list = []


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


def return_rate(i,j,k,l,N):

    a1 = i-j
    a3 = k-l
    a2 = -(a1+a3)

    assert((a1+a2+a3) == 0)

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


def custom_sum(i, N):
    if i == 0:
        res = 0
    else:
        res = np.sum([(N + 1 - idx) for idx in np.arange(i)])
    return res


def build_occ(N):
    dim = list(np.arange((N+1)))
    occupation = np.zeros(((N+1)**2))
    occ_states = []
    for i,j in itertools.product(dim, dim):
        if (i+j <= N):
            occupation[i*(N+1)+j] = 1.
            occ_states.append((i,j))

    return occupation, occ_states


def build_G_filtered(occupation_states, N):
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


def build_G(N):
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

            G[i,j] = return_rate(cnt_1_x, cnt_1_y, cnt_2_x, cnt_2_y, N)

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


def get_fiedler(matrix):
    eigvals, _ = LA.eig(matrix)
    fielder = np.sort(eigvals)[-2]
    return fielder


def get_eigvals(matrix):
    eigvals, _ = LA.eig(matrix)
    return eigvals


def P_dot(t,P):
    return np.matmul(G, P)

def error_L1(p_curr, p_prev, N):
    return np.sum(np.abs(p_curr-p_prev))

def error_L2(p_curr, p_prev, N):
    return np.sqrt(np.sum((p_curr-p_prev)**2))

#%%

eigvals_list = []

for v_2 in v_2_range:
    for N in N_range:
        v_3 = v_2

        vec, states = build_occ(N)
        G = build_G_filtered(states,N)
        eigv = get_eigvals(G)
        eigvals_list.append(eigv)

#%%
L1_values = []
L2_values = []

for v_2 in v_2_range:
    for N in N_range:
        v_3 = v_2

        vec, states = build_occ(N)
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
        #folder = os.path.join(path,"RKI_results")
        #namefile = os.path.join(folder,"monostable", \
        #            "{}_monostable_RKI_v2_{}.txt".format(N, f"{v_2:.2f}")) if v_2 < 2.5 \
        #            else os.path.join(folder, "bistable","{}_bistable_RKI_v2_{}.txt".format(N, f"{v_2:.2f}"))
        #namefile_G = os.path.join(folder,"monostable", \
        #            "{}_monostable_RKI_v2_{}_Fiedler.txt".format(N, f"{v_2:.2f}")) if v_2 < 2.5 \
        #            else os.path.join(folder, "bistable","{}_bistable_RKI_v2_{}_Fiedler.txt".format(N, f"{v_2:.2f}"))
        #np.savetxt(namefile, solution)
        #np.savetxt(namefile_G,np.array([f]))
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


# %%

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
fig.savefig("convergence.png", dpi=300, facecolor='white', transparent=False)
#fig.show()

# %%