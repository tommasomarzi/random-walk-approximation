
# %%
# for log of integer(when numbers are too big for the multinomial)
from math import log
from collections import Counter
from pathlib import Path
from mpl_toolkits import mplot3d  # for 3D plotd
from math import factorial  # for multinomial distribution
import fnmatch
import os
from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
%matplotlib qt5
plt.plot()
plt.close()
rc('text', usetex=True)
os.chdir("C:\\Users\\stefano.polizzi\\OneDrive - \
Alma Mater Studiorum Università di Bologna\\back_up\\Post-doc Bologna\progetto laplacian graph theory")
# os.chdir('D:\spoli\Documents\seagate\Post-doc Bologna\progetto laplacian graph theory')
path = %pwd

%matplotlib


def p_A_n(x, y, z, N, v_1=1, v_4=1):
    '''
    Parameters:
                x, y and z are the number of the particles in each state
                N is the total number of particles
    Returns:
                probability for one particle of being in state A, i.e. p_A_n is an eigenvector with 0 eigenvalue of the Laplacian matrix
    '''
    pi_AA = 0  # 1 - k_2*x/(k_1*k_2 + k_1*y + k_2*x)
    pi_AB = v_2*k_1/(k_1*k_2 + k_1*y/N + k_2*z/N)
    pi_AC = 0  # k*(N - n_A - n_B)/N
    pi_BA = v_1*k_2/(k_1*k_2 + k_1*y/N + k_2*x/N)
    # 1 - y*k_1/(k_1*k_2 + k_1*y + k_2*n_C) - y*k_1/(k_1*k_2 + k_1*y + k_2*x)
    pi_BB = 0
    pi_BC = v_4*k_2/(k_1*k_2 + k_2*z/N + k_1*y/N)
    pi_CA = 0  # k*n_A/N
    pi_CB = v_3*k_1/(k_1*k_2 + k_1*y/N + k_2*x/N)
    pi_CC = 0  # 1 - k_2*n_C/(k_1*k_2 + k_2*n_C + k_1*y)
    Z = pi_AC*pi_AB + pi_AB*pi_BC + pi_AC*pi_CB + \
        pi_BC*pi_CA + pi_BA*pi_AC + pi_CA*pi_CB + \
        pi_CA*pi_AB + pi_BC*pi_BA + pi_CB*pi_BA
    return (pi_AB*pi_BC + pi_AC*pi_AB + pi_AC*pi_CB)/Z


def p_B_n(x, y, z, N, v_1=1, v_4=1):
    '''
    Parameters:
                x, y and z are the number of the particles in each state
                N is the total number of particles
    Returns:
                probability for one particle of being in state B, i.e. p_B_n is an eigenvector with 0 eigenvalue of the Laplacian matrix
    '''
    pi_AA = 0  # 1 - k_2*x/(k_1*k_2 + k_1*y + k_2*x)
    pi_AB = v_2*k_1/(k_1*k_2 + k_1*y/N + k_2*z/N)
    pi_AC = 0  # k*(N - n_A - n_B)/N
    pi_BA = v_1*k_2/(k_1*k_2 + k_1*y/N + k_2*x/N)
    # 1 - y*k_1/(k_1*k_2 + k_1*y + k_2*n_C) - y*k_1/(k_1*k_2 + k_1*y + k_2*x)
    pi_BB = 0
    pi_BC = v_4*k_2/(k_1*k_2 + k_2*z/N + k_1*y/N)
    pi_CA = 0  # k*n_A/N
    pi_CB = v_3*k_1/(k_1*k_2 + k_1*y/N + k_2*x/N)
    pi_CC = 0  # 1 - k_2*n_C/(k_1*k_2 + k_2*n_C + k_1*y)
    Z = pi_AC*pi_AB + pi_AB*pi_BC + pi_AC*pi_CB + \
        pi_BC*pi_CA + pi_BA*pi_AC + pi_CA*pi_CB + \
        pi_CA*pi_AB + pi_BC*pi_BA + pi_CB*pi_BA
    return (pi_BC*pi_BA + pi_BC*pi_CA + pi_BA*pi_AC)/Z


def p_C_n(x, y, z, N, v_1=1, v_4=1):
    '''
    Parameters:
                x, y and z are the number of the particles in each state
                N is the total number of particles
    Returns:
                probability for one particle of being in state B, i.e. p_B_n is an eigenvector with 0 eigenvalue of the Laplacian matrix
    '''
    pi_AA = 0  # 1 - k_2*x/(k_1*k_2 + k_1*y + k_2*x)
    pi_AB = v_2*k_1/(k_1*k_2*N + k_1*y + k_2*z)
    pi_AC = 0  # k*(N - n_A - n_B)/N
    pi_BA = v_1*k_2/(k_1*k_2*N + k_1*y + k_2*x)
    # 1 - y*k_1/(k_1*k_2 + k_1*y + k_2*n_C) - y*k_1/(k_1*k_2 + k_1*y + k_2*x)
    pi_BB = 0
    pi_BC = v_4*k_2/(k_1*k_2*N + k_2*z + k_1*y)
    pi_CA = 0  # k*n_A/N
    pi_CB = v_3*k_1/(k_1*k_2*N + k_1*y + k_2*x)
    pi_CC = 0  # 1 - k_2*n_C/(k_1*k_2 + k_2*n_C + k_1*y)
    Z = pi_AC*pi_AB + pi_AB*pi_BC + pi_AC*pi_CB + \
        pi_BC*pi_CA + pi_BA*pi_AC + pi_CA*pi_CB + \
        pi_CA*pi_AB + pi_BC*pi_BA + pi_CB*pi_BA
    return (pi_CA*pi_CB + pi_CA*pi_AB + pi_CB*pi_BA)/Z


def mult(x, z, N):
    if (x + z > N):
        return -np.inf
    y = N - x - z
    return log(factorial(N)//(factorial(int(x))*factorial(int(y))*factorial(int(z)))) + \
        x*log(p_A_n(x, y, z, N)) + y*log(p_B_n(x, y, z, N)) + \
        log(1-p_A_n(x, y, z, N)-p_B_n(x, y, z, N))*z


def RWA(N_min, N_max, N_step, v_2_min=1.5, v_2_max=2.49, v_2_length=100, k_1=0.1,
        k_2=1):
    '''
    Parameters:
                v_1, v_2, v_3 and v_4 are the rates of the transitions between the states
                N_min, N_max and N_step are the parameters for the number of particles in the system
    Returns:
                It saves the text files with the values of the probabilities for each possible state of the system
                along with the png files with the color images of the probabilities $\rho$

    '''
    for v_2 in (np.around(np.linspace(v_2_min,v_2_max,v_2_length),2)):  #put rounding for file names
        v_3 = v_2
        v_2_s = "{:0.2f}".format(v_2)
        for N in np.arange(N_min, N_max+1, N_step):
            lnZ = -np.ones((N+1, N+1))*np.inf
            for i in np.arange(N+1):
                for j in np.arange(N+1-i):
                    lnZ[i, j] = mult(i, j, N)
            lnZ = np.where(np.isfinite(lnZ), lnZ, -np.inf)
            Z = np.exp(lnZ)
            Z = Z/np.sum(Z)
            namefile = os.path.join("bistable", str(N) + "_bistable_MUL_v2_" + str(v_2_s)) if v_2 >= \
                        2.5 else os.path.join("monostable", str(N) + "_monostable_MUL_v2_" + str(v_2_s))
            np.savetxt(namefile + ".txt", Z.transpose())
            fig = plt.figure(figsize=(8, 6))
            plt.imshow(Z.transpose(), origin='lower', interpolation='nearest')
            plt.colorbar()
            if (k_1 == 0.001) | (k_2 == 50) | (v_2 <= 2.5):
                    plt.title(u"Monostable with $k_1 = {}$, $k_2 = {}$, $v_2 = v_3 = {}$ and $N = {}$".format(k_1, k_2, v_2, N))
            else:
                    plt.title(u"Bistable with $k_1 = {}$, $k_2 = {}$, $v_2 = v_3 = {}$ and $N = {}$".format(k_1, k_2, v_2, N))
            #plt.xlim(low_nA, high_nA)
            #plt.ylim(low_nB, high_nB)
            plt.xlabel("$n_A$", size=14)
            plt.ylabel("$n_C$", size=14)
            plt.tight_layout()
            plt.savefig(namefile + '.png', dpi=300, facecolor='white', transparent=False)
            plt.close()
# %%
