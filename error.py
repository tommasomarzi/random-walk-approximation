#%%

import numpy as np
import matplotlib.pyplot as plt


#%%
def KL(y_true, y_pred, eps = 0.00001):
    """
    Compute KL divergence (a small epsilon is added to avoid discontinuities).
    """
    y_true_shift = y_true+eps
    y_true_shift = y_true_shift/y_true_shift.sum()

    y_pred_shift = y_pred+eps
    y_pred_shift = y_pred_shift/y_pred_shift.sum()

    error = np.sum(y_true_shift*np.log(y_true_shift/y_pred_shift))

    return error

def L1(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred))


def L2(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred)**2))

# %%

N_min = 5
N_max = 45
N_step = 10
K_2_values = [0.001, 0.05]
methods = ["LNA", "MUL"]#,"MUL_int"]

select_error = "L2"

errors_mono = {}
errors_bi = {}

for met in methods:
    errors_mono[met] = []
    errors_bi[met] = []

#%%
for K_2 in K_2_values:

    if K_2 == 0.001:
        stability = "monostable"
    elif K_2 == 0.05:
        stability = "bistable"

    for N in np.arange(N_min, N_max + 1, N_step):
        
        filename = stability + "/" + str(N) + "_" + stability + "_"
        GIL_data = np.loadtxt("../results_new/GIL_results/"+ filename + "GIL.txt")
        
        for met in methods:
            filemet = "../results_new/" + met + "_results/" + filename 
            
            if met == "MUL_int":
                filemet += "MUL.txt"
            else:
                filemet = filemet + met + ".txt"
            
            met_data = np.loadtxt(filemet)

            if select_error == "KL":
                err = KL(GIL_data, met_data)
            elif select_error == "L1":
                err = L1(GIL_data, met_data)
            elif select_error == "L2":
                err = L2(GIL_data, met_data)
           
            if stability == "monostable":
                errors_mono[met].append(err)
            if stability == "bistable":
                errors_bi[met].append(err)

# %%

th = 0			

fig,ax = plt.subplots()

for met in methods:
    ax.plot(np.arange(N_min, N_max + 1, N_step)[th:], errors_mono[met][th:], label = met)
ax.legend()
ax.set_title("{} for monostable".format(select_error))
ax.grid()
ax.set_xlabel("N")
ax.set_xticks(np.arange(th*N_step + 5, N_max+1,N_step))
ax.set_ylabel("Error")
fig.show()
#fig.savefig('{}_monostable.png'.format(select_error), dpi=200, facecolor='white', transparent=False)

fig,ax = plt.subplots()
for met in methods:
    ax.plot(np.arange(N_min, N_max + 1, N_step)[th:], errors_bi[met][th:], label = met)
ax.legend()
ax.grid()
ax.set_xlabel("N")
ax.set_xticks(np.arange(th*N_step + 5, N_max+1,N_step))
ax.set_ylabel("Error")
ax.set_title("{} for bistable".format(select_error))
#fig.savefig('{}_bistable.png'.format(select_error), dpi=200, facecolor='white', transparent=False)

# %%