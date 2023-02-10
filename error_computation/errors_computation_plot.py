# use kernel of server bio14 names kernel_py or env_D

# %%
import matplotlib
%matplotlib qt5
import matplotlib.pyplot as plt
plt.plot()
plt.close()
import string
import numpy as np
from matplotlib import rc
import seaborn as sns
import os
import cv2 #resize images
from scipy.interpolate import interp2d
from scipy.special import kl_div
from scipy.spatial.distance import squareform, pdist, jensenshannon # for Wasserstein distance
from scipy.optimize import linprog  # for Wasserstein distance
from math import factorial  # for multinomial distribution
from mpl_toolkits import mplot3d  # for 3D plotd
from itertools import cycle
rc('text', usetex=True)
path = %pwd

%matplotlib
# %%


def error_compute_plot(N_min=5,  N_max=295, N_step=10, k_1=0.1, k_2=1):
    '''
    Compute the errors between the different methods and plot them,
    saving the figure png file.
    Parameters:
    N_min, N_max, N_step: range of N to compute the errors
    k_1, k_2: parameters of the model    
    '''
    j = 0
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
    fig, axs = plt.subplots(1,3, sharey = True, figsize=(15, 5))
    for (v_2,v_3) in ([1.82, 1.82], [2.47, 2.47], [3.04, 3.04]):
        for i, N in enumerate(np.arange(N_min, N_max+1, N_step)):
            for name in ["MUL_files_norm_sum", "GIL_files_ergo","LNA_results", "MUL_files_star", "RKI_results"]: #put here and in the follwing the names of the corresponding folder for each method
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

            error_L1_MUL[i] = np.sum(abs(data_MUL-data_RKI))
            error_L2_MUL[i] = np.sqrt(np.sum((data_MUL-data_RKI)**2))
            error_KL_MUL[i] = jensenshannon(data_RKI.flatten(), data_MUL.flatten(), base = 2)
            error_L1_LNA[i] = np.sum(abs(data_LNA-data_RKI))
            error_L2_LNA[i] = np.sqrt(np.sum((data_LNA-data_RKI)**2))
            error_KL_LNA[i] = jensenshannon(data_RKI.flatten(), data_LNA.flatten(),base = 2)
            error_L1_MUL_star[i] = np.sum(abs(data_MUL_star-data_RKI))
            error_L2_MUL_star[i] = np.sqrt(np.sum((data_MUL_star-data_RKI)**2))
            error_KL_MUL_star[i] = jensenshannon(data_RKI.flatten(), data_MUL_star.flatten(),base = 2)
            error_L1_GIL[i] = np.sum(abs(data_RKI-data_GIL))
            error_L2_GIL[i] = np.sqrt(np.sum((data_RKI-data_GIL)**2))
            error_KL_GIL[i] =  jensenshannon(data_RKI.flatten(), data_GIL.flatten(),base = 2)
        with sns.plotting_context('paper', font_scale=2):

            axs[j].plot(np.arange(N_min, N_max+1, N_step), error_L1_LNA, 'b--',lw = 2, label = "$\ell^1$ SSE")
            axs[j].plot(np.arange(N_min, N_max+1, N_step), error_KL_LNA, 'b--',lw = 2, label = "JS SSE")

            axs[j].plot(np.arange(N_min, N_max+1, N_step), error_L1_MUL_star, 'y-.',lw = 2, label = "$\ell^1$ MUL*")
            axs[j].plot(np.arange(N_min, N_max+1, N_step), error_KL_MUL_star, 'y-.',lw = 2, label = "JS MUL*")

            axs[j].plot(np.arange(N_min, N_max+1, N_step), error_L1_MUL, 'c', lw = 2, label = "$\ell^1$ RWA")
            axs[j].plot(np.arange(N_min, N_max+1, N_step), error_KL_MUL, 'c', lw = 2, label = "JS RWA")

            plt.plot(np.arange(N_min, N_max+1, N_step), error_L1_GIL, 'r', label = "L1 GIL")
            plt.plot(np.arange(N_min, N_max+1, N_step), error_KL_GIL, 'r-.', label = "JS GIL")

            axs[j].set_xlabel("$N$", size=19)
            axs[0].set_ylabel("Error", size=19)
            axs[j].set_xlim(-10, N_max+5)
            axs[j].tick_params("both", labelsize = 19)
            axs[0].legend(loc = 2)

            if v_2 < 2.5:
                axs[j].set_title(u"$v_2 = v_3 = {}$".format(v_2))
            else:
                axs[j].set_title(u"$v_2 = v_3 = {}$".format( v_2))

            axs[j].text(-0.1, 1.05, string.ascii_lowercase[j] + ")", transform=axs[j].transAxes,
                    size=20)
            j +=1
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.119)
    plt.savefig("Errors_figure_paper.png", dpi=300)


