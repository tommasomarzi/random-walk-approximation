#%%
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
from tqdm import tqdm

#%%

v_2_range = np.linspace(1.5,3.52,26)
#v_2_range = np.linspace(2.55,3.52,13)
v_2_threshold = 2.50
#v_2_range = [1.5,2.0]

K_1 = 0.1
K_2 = 1.
v_1 = 1.

N_min = 5
N_max = 205
N_step = 10
N_range = np.arange(N_min, N_max+1, N_step)
#N_range = [20, 30]

#%%

def build_L(x,y,z,v2):
    pi_AA = 0 #1 - K_2*x/(K_1*K_2 + K_1*y + K_2*x)
    pi_AB = v2*K_1/(K_1*K_2 + K_1*y + K_2*z)
    pi_AC = 0  # k*(N - n_A - n_B)/N
    pi_BA = K_2*v_1/(K_1*K_2 + K_1*y + K_2*x)
    pi_BB = 1 - y*K_1/(K_1*K_2 + K_1*y + K_2*z) - y*K_1/(K_1*K_2 + K_1*y + K_2*x)
    pi_BC = K_2*v_1/(K_1*K_2 + K_2*z + K_1*y)
    pi_CA = 0  # k*n_A/N
    pi_CB = v2*K_1/(K_1*K_2 + K_1*y + K_2*x)
    pi_CC = 0#1 - K_2*n_C/(K_1*K_2 + K_2*n_C + K_1*y)
    L = np.array([[  pi_BA + pi_CA, - pi_AB,         - pi_AC],
                  [- pi_BA,           pi_AB + pi_CB, - pi_BC],     
                  [- pi_CA,         - pi_CB,           pi_AC + pi_BC]])
    return L


def get_fiedler(matrix):
    eigvals, _ = LA.eig(matrix)
    fielder = (np.sort(np.abs(eigvals)))[1]
    return fielder


# %%

list_N = []

for v_2 in tqdm(v_2_range):
    list_v2 = []
    for N in N_range:
        list_fiedler = []
        for i in range(N+1):
            for j in range(N+1-i):
                L = build_L(i/N, j/N, (N-i-j)/N, v_2)
                f = get_fiedler(L)
                list_fiedler.append(f)
        list_v2.append(np.min(list_fiedler))
    list_N.append(list_v2)


#%%

to_plot = list(map(list, zip(*list_N)))
fig,ax = plt.subplots(figsize=(15,8))
for idx, n in enumerate(list_N):
    #ax.plot(v_2_range, n, label = N_range[idx])
    ax.plot(N_range, n, label = v_2_range[idx])
#ax.set_yscale('log')
ax.set_xlabel(r'$v_2$')
ax.set_ylabel(r'|$\lambda_f$|')
ax.legend()
#ax.set_ylim([0.665,0.67])
fig.tight_layout()


# %%
