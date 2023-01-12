#%%
import numpy as np 
import matplotlib.pyplot as plt

#%%

v_2_range = np.linspace(1.5,3.52,26)
#v_2_range = np.linspace(2.55,3.52,13)
v_2_threshold = 2.50

N_min = 5
N_max = 205
N_step = 10
N_range = np.arange(N_min, N_max+1, N_step)

exception = [185,195,205]

#N_range = [5]
#v_2_range = [1.50, 2.55]

# %%
stability = None
list_N = []

for N in N_range:
    list_v2 = []
    for v_2 in v_2_range:
        if v_2 > v_2_threshold:
            stability = "bistable"
        else:
            stability = "monostable"
        #filename = stability + "/" + str(N) + "_" + stability + "_RKI_v2_" + str(v_2) + "_Fiedler.txt"
        if v_2 == 1.50 and exception.count(N):
            list_v2.append(1e-10)
        else:
            filename = "../{}/{}_{}_RKI_v2_{:.2f}_Fiedler.txt".format(stability,N,stability,v_2)
            v = np.loadtxt(filename, dtype=complex)
            list_v2.append(np.abs(v.real))
    list_N.append(list_v2)

# %%
skip_N = 0

fig,ax = plt.subplots(figsize=(15,8))
for idx, n in enumerate(list_N[skip_N:]):
    ax.plot(v_2_range, n, label = N_range[idx+skip_N])
#ax.set_yscale('log')
ax.set_xlabel(r'$v_2$')
ax.set_ylabel(r'|$\lambda_f$|')
ax.legend()
fig.tight_layout()
#fig.savefig('fiedler.png', dpi=300, facecolor='white', transparent=False)

# %%
