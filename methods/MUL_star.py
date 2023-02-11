import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import os
import fnmatch
from math import factorial          
from mpl_toolkits import mplot3d    
from pathlib import Path
from math import log               
from collections import Counter

rc('text', usetex=True)

def null_eigenvector_k(k_1, k_2, v_1, v_2 , N):  
    """
    Parameters:
        k_1, k_2: rates of the reactions
        v_1, v_2: velocities of the reactions
        N: number of particles
    Returns:
        analytical stationary solution of the system of ODEs (x^* in the paper)
    """
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
        

def mult_const(x, y, p_A, p_B, p_C, N):
    """
    Function returning the log of the multinomial distribution 
    """
    if x + y > N:
        return -np.inf
    n_c = N-x-y
    return log(factorial(N)/(factorial(x)*factorial(y)*factorial(n_c))) + \
        x*log(p_A) + y*log(p_B) + n_c*log(1-p_A-p_B)


def save_MUL_const(v_2, folder = "MUL_files_star", k_1 = 0.1, k_2 = 1, v_1 = 1, N_min = 5, N_max = 105, N_step = 10):
    """
    Function that saves the txt and png files of the multinomial (MUL_star) with the correct constant
    analytical parameters for the system of ODE given from the model dual-phosphorylation dephosphorylation
    with parameters k_1, k_2, v_1, v_2 (the control paramter). The default values of the parameters are the ones
    in the paper.
    Arguments:
        N_min: lower bound of the number of particles for which we want to compute the MUL_star
        N_max: upper bound of the number of particles for which we want to compute the MUL_star
        N_step: step of the number of particles for which we want to compute the MUL_star
        folder: root folder for the data, inside wich a monostable and bistable directory is created in path (project folder)
    """
    for N in np.arange(N_min, N_max+1, N_step):
        lnZ = -np.ones((N+1, N+1))*np.inf
        temp = null_eigenvector_k(k_1, k_2, v_1, v_2, N)
        if len(temp) < 3:
            p_A, p_B, p_C = temp[0]
            lnZ2 = -np.ones((N+1, N+1))*np.inf
            for i in np.arange(N+1):
                for j in np.arange(N+1-i):
                    lnZ[i, N - j - i] = mult_const(i, j, p_A, p_B, p_C, N)
            p_A, p_B, p_C = temp[1]
            for i in np.arange(N+1):
                for j in np.arange(N+1-i):
                    lnZ2[i, N - j - i] = mult_const(i, j, p_A, p_B, p_C, N)
            lnZ = np.where(np.isfinite(lnZ), lnZ, -np.inf)
            Z1 = np.exp(lnZ)
            lnZ2 = np.where(np.isfinite(lnZ2), lnZ2, -np.inf)
            Z2 = np.exp(lnZ2)
            Z = Z1 + Z2
            Z = Z/np.sum(Z)
        else:
            p_A, p_B, p_C = temp
            for i in np.arange(N+1):
                for j in np.arange(N+1-i):
                    lnZ[i, N - j - i] = mult_const(i, j, p_A, p_B, p_C, N)
            lnZ = np.where(np.isfinite(lnZ), lnZ, -np.inf)
            Z = np.exp(lnZ)
            Z = Z/np.sum(Z)
        namefile = os.path.join(folder,"bistable", str(N) + "_bistable_MUL_v2_" + str(v_2) + "_const") if v_2 > \
                    2.5 else os.path.join(folder, "monostable", str(N) + "_monostable_MUL_v2_" + str(v_2) + "_const")
        np.savetxt(namefile + "_star.txt", Z.transpose())
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(Z.transpose(), origin='lower', interpolation='nearest')
        plt.colorbar()
        if v_2 < 2.5:
            plt.title(u"Monostable with $k_1 = {}$, $k_2 = {}$, $v_2 = v_3 = {}$ and $N = {}$".format(k_1, k_2, v_2, N))
        else:
            plt.title(u"Bistable with $k_1 = {}$, $k_2 = {}$, $v_2 = v_3 = {}$ and $N = {}$".format(k_1, k_2, v_2, N))
        plt.xlabel("$n_A$", size=14)
        plt.ylabel("$n_C$", size=14)
        plt.tight_layout()
        plt.savefig(namefile + '_star.png', dpi=300, facecolor='white', transparent=False)
        plt.close()
        
# Example of use
save_MUL_const(v_2 = 3.04, folder = "MUL_files_star", k_1 = 0.1, k_2 = 1,
                v_1 = 1, N_min = 105, N_max = 495, N_step = 10)

