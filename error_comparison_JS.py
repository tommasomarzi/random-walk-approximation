# file computing errors for laplacian Bazzani project
# use kernel of server bio14 names kernel_py or env_D

# %%
import matplotlib
%matplotlib qt5
import matplotlib.pyplot as plt
plt.plot()
plt.close()
import numpy as np
from matplotlib import rc
import os
import cv2 #resize images
from scipy.interpolate import interp2d
from scipy.special import kl_div
from scipy.spatial.distance import squareform, pdist, jensenshannon # for Wasserstein distance
from scipy.optimize import linprog  # for Wasserstein distance
from math import factorial  # for multinomial distribution
from mpl_toolkits import mplot3d  # for 3D plotd
rc('text', usetex=True)
os.chdir("C:\\Users\\stefano.polizzi\\OneDrive - \
Alma Mater Studiorum Università di Bologna\\back_up\\Post-doc Bologna\progetto laplacian graph theory")
#os.chdir('D:\spoli\Documents\seagate\Post-doc Bologna\progetto laplacian graph theory')
path = %pwd

%matplotlib

def KL(y_true, y_pred, base = 2, eps = 0.0000001):
    """
    Compute KL divergence (a small epsilon is added to remove discontinuities).
    base = base of the log (default 2) to have it in bits
    """
    y_true = np.where(y_true < 0, 0, y_true)  # added to avoid negative distribution (values are actually 0)
    y_true_shift = y_true+eps
    y_true_shift = y_true_shift/y_true_shift.sum()
    y_pred_shift = y_pred+eps
    y_pred_shift = y_pred_shift/y_pred_shift.sum()
    error = np.log(base)*np.sum(y_true_shift*np.log(y_true_shift/y_pred_shift))
    return error
# %%




#########################################################
### For new parameters of model of Bazzani's paper corrected with velocities
#########################################################
# %%
k_1 = 0.1
k_2 = 1
v_2 = v_3 = 1.82  # put integer
v_1 = v_4 = 1
N_min = 5
N_max = 505
N_step = 10
N = 165
error_L1_MUL = np.zeros(len(np.arange(N_min, N_max+1, N_step)))
error_L2_MUL = np.zeros(len(np.arange(N_min, N_max+1, N_step)))
error_KL_MUL = np.zeros(len(np.arange(N_min, N_max+1, N_step)))
error_L1_LNA = np.zeros(len(np.arange(N_min, N_max+1, N_step)))
error_L2_LNA = np.zeros(len(np.arange(N_min, N_max+1, N_step)))
error_KL_LNA = np.zeros(len(np.arange(N_min, N_max+1, N_step)))
error_L1_MUL_star = np.zeros(len(np.arange(N_min, N_max+1, N_step)))
error_L2_MUL_star = np.zeros(len(np.arange(N_min, N_max+1, N_step)))
error_KL_MUL_star = np.zeros(len(np.arange(N_min, N_max+1, N_step)))
error_L1_GIL = np.zeros(len(np.arange(N_min, N_max+1, N_step)))
error_L2_GIL = np.zeros(len(np.arange(N_min, N_max+1, N_step)))
error_KL_GIL = np.zeros(len(np.arange(N_min, N_max+1, N_step)))

# %%

for (v_2,v_3) in ([1.82, 1.82], [2.47, 2.47], [3.04, 3.04]):
    for i, N in enumerate(np.arange(N_min, N_max+1, N_step)):
        for name in ["MUL_files_norm_sum", "GIL_files_ergo","LNA_results", "MUL_files_star", "RKI_results"]:
            if name == "MUL_files_norm_sum":
                v_2_s = "{:0.2f}".format(v_2)
                namefile =  os.path.join(path, name, "bistable", str(N) + "_bistable_MUL_v2_" + str(v_2_s)) if v_2 >= \
                            2.5 else os.path.join(path, name, "monostable", str(N) + "_monostable_MUL_v2_" + str(v_2_s))
                data_MUL = np.loadtxt(namefile + '.txt')
            if name == "GIL_files_ergo":
                namefile = os.path.join(path,name,"monostable", \
                            "{}_monostable_GIL_v2_{}".format(N, f"{v_2:.2f}")) if v_2 < 2.5 \
                            else os.path.join(path,name, "bistable","{}_bistable_GIL_v2_{}".format(N, f"{v_2:.2f}"))
                try:
                    data_GIL = np.loadtxt(namefile + '_AC_time_vel_k.txt')
                except:
                    data_GIL = np.zeros(np.shape(data_MUL))
            if name == "LNA_results":
                namefile = os.path.join(path, name, "bistable", str(N) + "_bistable_LNA_v2_" + str(v_2) + "") if v_2 > 2.5 \
                            else os.path.join(path, name, "monostable", str(N) + "_monostable_LNA_v2_" + str(v_2) + "")
                try:
                    data_LNA = np.loadtxt(namefile + '.txt')
                except:
                    data_LNA = np.zeros(np.shape(data_MUL))
            if name == "MUL_files_star":
                namefile =  os.path.join(path,name, "bistable", str(N) + "_bistable_MUL_v2_" + str(v_2)) if v_2 > 2.5 \
                     else  os.path.join(path, name, "monostable", str(N) + "_monostable_MUL_v2_" + str(v_2))
                try:
                    data_MUL_star = np.loadtxt(namefile + '_const_star.txt')
                except:
                    data_MUL_star = np.zeros(np.shape(data_MUL))
            if name == "RKI_results":
                namefile =  os.path.join(path,name,"monostable", \
                            "{}_monostable_RKI_v2_{}.txt".format(N, f"{v_2:.2f}")) if v_2 < 2.5 \
                            else os.path.join(path,name, "bistable","{}_bistable_RKI_v2_{}.txt".format(N, f"{v_2:.2f}"))
                try:
                    data_RKI = np.loadtxt(namefile)
                    data_RKI = np.where(data_RKI<0,0,data_RKI)
                except:
                    data_RKI = np.zeros(np.shape(data_MUL))

        if N < N_max:           #interpolation
            axis_cur = (np.arange(N+1))/N
            axis_max = (np.arange(N_max+1))/N_max  

            itp = interp2d(axis_cur, axis_cur, data_MUL)
            data_MUL = itp(axis_max, axis_max) 
            data_MUL = data_MUL/data_MUL.sum()

            itp = interp2d(axis_cur, axis_cur, data_RKI)
            data_RKI = itp(axis_max, axis_max)  
            data_RKI = data_RKI/data_RKI.sum()  

            itp = interp2d(axis_cur, axis_cur, data_LNA)
            data_LNA = itp(axis_max, axis_max)      
            data_LNA = data_LNA/data_LNA.sum()

            itp = interp2d(axis_cur, axis_cur, data_MUL_star)
            data_MUL_star = itp(axis_max, axis_max)      
            data_MUL_star = data_MUL_star/data_MUL_star.sum()

            itp = interp2d(axis_cur, axis_cur, data_GIL)
            data_GIL = itp(axis_max, axis_max)
            data_GIL = data_GIL/data_GIL.sum()
    
        error_L1_MUL[i] = np.mean(abs(data_MUL-data_RKI))
        error_L2_MUL[i] = np.sqrt(np.mean((data_MUL-data_RKI)**2))
        error_KL_MUL[i] = jensenshannon(data_RKI.flatten(), data_MUL.flatten(), base = 2)
        error_L1_LNA[i] = np.mean(abs(data_LNA-data_RKI))
        error_L2_LNA[i] = np.sqrt(np.mean((data_LNA-data_RKI)**2))
        error_KL_LNA[i] = jensenshannon(data_RKI.flatten(), data_LNA.flatten(),base = 2)
        error_L1_MUL_star[i] = np.mean(abs(data_MUL_star-data_RKI))
        error_L2_MUL_star[i] = np.sqrt(np.mean((data_MUL_star-data_RKI)**2))
        error_KL_MUL_star[i] = jensenshannon(data_RKI.flatten(), data_MUL_star.flatten(),base = 2)
        error_L1_GIL[i] = np.mean(abs(data_RKI-data_GIL))
        error_L2_GIL[i] = np.sqrt(np.mean((data_RKI-data_GIL)**2))
        error_KL_GIL[i] =  jensenshannon(data_RKI.flatten(), data_GIL.flatten(),base = 2)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(np.arange(N_min, N_max+1, N_step), error_L1_LNA, 'b', label = "L1 LNA")
    plt.plot(np.arange(N_min, N_max+1, N_step), error_L2_LNA, 'b--', label = "L2 LNA")
    #plt.plot(np.arange(N_min, N_max+1, N_step), error_KL_LNA, 'b-.', label = "JS LNA")

    plt.plot(np.arange(N_min, N_max+1, N_step), error_L1_MUL_star, 'y', label = "L1 MUL*")
    plt.plot(np.arange(N_min, N_max+1, N_step), error_L2_MUL_star, 'y--', label = "L2 MUL*")
    #plt.plot(np.arange(N_min, N_max+1, N_step), error_KL_MUL_star, 'y-.', label = "JS MUL*")

    plt.plot(np.arange(N_min, N_max+1, N_step), error_L1_MUL, 'c', label = "L1 RWA")
    plt.plot(np.arange(N_min, N_max+1, N_step), error_L2_MUL, 'c--', label = "L2 RWA")
    #plt.plot(np.arange(N_min, N_max+1, N_step), error_KL_MUL, 'c-.', label = "JS RWA")

    #plt.plot(np.arange(N_min, N_max+1, N_step), error_L1_GIL, 'r', label = "L1 GIL")
    #plt.plot(np.arange(N_min, N_max+1, N_step), error_L2_GIL, 'r--', label = "L2 GIL")
    #plt.plot(np.arange(N_min, N_max+1, N_step), error_KL_GIL, 'r-.', label = "JS GIL")

    plt.xlabel("$N$", size=19)
    plt.ylabel("Error", size=19)
    plt.xlim(-30, N_max+2)
    plt.tick_params("both", labelsize = 19)
    plt.legend(fontsize = 15, loc = 2)
    if v_2 < 2.5:
        plt.title(u"Monostable with $k_1 = {}$, $k_2 = {}$, $v_2 = v_3 = {}$".format(k_1, k_2, v_2),
        size = 19)
        print("Type 1 if you want to save the figure for monostable any number otherwise")
        if int(input()) == 1:
            plt.tight_layout()
            plt.savefig("Errors_monostable" + '_v2_' + str(v_2) + '_poster.png', dpi=300)
            plt.close()
    else:
        plt.title(u"Bistable with $k_1 = {}$, $k_2 = {}$, $v_2 = v_3 = {}$".format(k_1, k_2, v_2),
        size = 19)
        print("Type 1 if you want to save the figure for bistable any number otherwise")
        if int(input()) == 1:
            plt.tight_layout()
            plt.savefig("Errors_bistable" + '_v2_' + str(v_2) + '_poster.png', dpi=300)
            plt.close()
