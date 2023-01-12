# file to explore behaviour next to the critical point for laplacian Bazzani project
# use kernel of server bio14 named kernel_py or env_D

# %%
import matplotlib
%matplotlib qt5
import matplotlib.pyplot as plt
plt.plot()
plt.close()
import numpy as np
from matplotlib import rc
import os
from scipy.spatial.distance import squareform, pdist # for Wasserstein distance
from scipy.optimize import linprog  # for Wasserstein distance
from math import factorial  # for multinomial distribution
from mpl_toolkits import mplot3d  # for 3D plotd
rc('text', usetex=True)
os.chdir("C:\\Users\\stefano.polizzi\\OneDrive - \
Alma Mater Studiorum Università di Bologna\\back_up\\Post-doc Bologna\progetto laplacian graph theory")
#os.chdir('D:\spoli\Documents\seagate\Post-doc Bologna\progetto laplacian graph theory')
path = %pwd

%matplotlib
def null_eigenvector_k(k_1, k_2, v_1, v_2 , N):  # rates corrected
    xA = xC = 0.5*(1-v_1*k_2/(2*v_2*k_1 + v_1*k_2))
    xB = v_1*k_2/(2*v_2*k_1 + v_1*k_2)
    xB_temp = - k_1*k_2**2*v_1/(k_1*k_2*v_1 - v_2*k_1*k_2)
    b = xB_temp*v_2*N*k_1*N*k_2 - v_1*N**2*k_2**2*k_1 - v_1*N*k_1*N*k_2*xB_temp - v_1*N**2*k_2**2*(1 - xB_temp)
    a = v_1*N**2*k_2**2
    c = xB_temp*v_2*N**2*k_1**2*k_2 + xB_temp**2*v_2*N**2*k_1**2
    xA_temp = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    xC_temp = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    if (b**2 - 4*a*c > 0) & (0<=xA_temp<=1) & (0<=xB_temp<=1) & (0<=xC_temp<=1):
        xB = xB_temp
        xA = xA_temp
        xC = xC_temp
        return [(xA, xB, xC),(xC, xB, xA)]
    else:
        return xA, xB, xC

# %%
#Parameters
N = 155
k_1 = 0.1
k_2 = 1
v_1 = v_4 = 1
N_min = 55
N_max = 505
N_step = 10
# %%

N_values = np.arange(N_min, N_max, N_step)
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', ['y', 'c', 'b', 'r'])

# %%

##############################################################################################
################### Plot of the relative error of the standard deviation versus N ############
##############################################################################################
v_2 = 1.80
error_crit_MUL = np.zeros(len(N_values))
error_crit_GIL = np.zeros(len(N_values))
relative_error = np.zeros(len(N_values))
plt.figure()
for v_2 in [1.50, 1.80, 2.20, 2.40]:
    v_2_s = "{:0.2f}".format(v_2)
    for i, N in enumerate(N_values):
        x = np.arange(0,N+1)
        for name in ["MUL_files_norm_sum", "GIL_files_ergo"]:
            if name == "MUL_files_norm_sum":
                namefile = os.path.join(path,name, "bistable", str(N) + "_bistable_MUL_v2_" + str(v_2_s)) if v_2 > 2.5 \
                    else  os.path.join(path,name,"monostable", str(N) + "_monostable_MUL_v2_" + str(v_2_s))
                data_MUL = np.loadtxt(namefile + '.txt')
                error_crit_MUL[i] = np.sqrt(np.sum(np.sum(data_MUL*x**2, axis = 0)) - np.sum(np.sum(data_MUL*x, axis = 0))**2)  #since the distribution is symmetric the variance is proportional to the identy
            if name == "GIL_files_ergo":
                namefile = os.path.join(path,name,"monostable", \
                        "{}_monostable_GIL_v2_{}".format(N, f"{v_2:.2f}")) if v_2 < 2.5 \
                        else os.path.join(path,name, "bistable","{}_bistable_GIL_v2_{}".format(N, f"{v_2:.2f}"))
                data_GIL = np.loadtxt(namefile + '_AC_time_vel_k.txt')
                error_crit_GIL[i] = np.sqrt(np.sum(np.sum(data_GIL*x**2, axis = 0)) - np.sum(np.sum(data_GIL*x, axis = 0))**2)
            relative_error[i] = (error_crit_MUL[i] - error_crit_GIL[i])/error_crit_GIL[i]
    plt.plot(N_values, relative_error, '--', label = "$v_2 =$ " + str(v_2_s))
    plt.xlabel("$N$", size = 18)
    plt.ylabel("Error on $\\sigma$ in $\%$", size = 18)
    plt.tight_layout()
plt.legend(fontsize = 14, loc = 0)
plt.savefig('Error_std_vs_N_monostable.png', dpi=300, facecolor='white', transparent=False)
# %%
