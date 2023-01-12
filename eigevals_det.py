#%%
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import math
import matplotlib as mpl 
from tqdm import tqdm

#%%
plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=["r", "g", "k","orange","b"])
plt.rcParams.update({'figure.figsize': [12,8]})
plt.rcParams.update({'axes.titlesize': 20})
plt.rcParams.update({'axes.labelsize': 20})
plt.rcParams.update({'xtick.labelsize': 14})
plt.rcParams.update({'ytick.labelsize': 14})
plt.rcParams.update({'xtick.major.size': 14})
plt.rcParams.update({'ytick.major.size': 14})
plt.rcParams.update({'lines.linewidth': 3})
plt.rcParams.update({'legend.fontsize': 14})

#%%

v_2_range = np.linspace(1.8,3.52,26)
#v_2_range = np.linspace(2.55,3.52,13)
v_2 = [1.8,2.5]

K_1 = K_4 = 0.1
K_2 = K_3 = 1.
v_1 = v_4 = 1.


#%%
#------------------ v_2 threshold ------------------#
#---------------------------------------------------#

def v2_threshold(k1,k2,v1,v2):
    lhs = v2 - v1*(1+k2)
    rhs = 2*k1*v2
    if lhs >= rhs or lhs <= -rhs:
        return True
    else:
        return False

    
#%%
#------------------ Exact solution -----------------#
#---------------------------------------------------#   

def null_eigenvector_eq(k1,k2,v1,v2):
    xB = 1/(1 + (k1*v2*2/(k2*v1)))
    xA = xC = (1-xB)/2

    return xA, xB, xC


def null_eigenvector_neq(k1,k2,v1,v2):
    xB = k2*v1/(v2 - v1)

    B = - (v2 - v1*(1 + k2))/(v2 - v1)
    #C = (v2*k1*k1*(1 + v1/(v2 - v1)))/(v2 - v1)
    C = (v2*k1*k1*v2)/((v2 - v1)**2)

    xA_1 = (-B + math.sqrt(B**2 - 4*C))/(2)
    xA_2 = (-B - math.sqrt(B**2 - 4*C))/(2)

    if (xA_1 < 0) or (xA_1 > 1):
        print("A problem occured, xA_1: {}".format(xA_1))

    if (xA_2 < 0) or (xA_2 > 1):
        print("A problem occured, xA_2: {}".format(xA_2))
    
    xC_1 = 1 - xA_1 - xB
    xC_2 = 1 - xA_2 - xB
    return xA_2, xB, xC_2#, xA_2, xC_2 
    #return xA_1, xB, xC_1#, xA_2, xC_2


#%%
#--------------------- Laplacian -------------------#
#---------------------------------------------------#   

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


def get_spectrum(matrix):
    eigvals, _ = LA.eig(matrix)
    return (np.sort((eigvals)))


# %%

list_eig = []

for v_2 in tqdm(v_2_range):
    v_3 = v_2
    check_bistability = v2_threshold(K_1,K_2,v_1,v_2)  

    if ((v_2 > v_1) and (v_2 >= (v_1*(K_2+1)))):
        if check_bistability:
            print("Bistable system")
            xA_star, xB_star, xC_star = null_eigenvector_neq(K_1,K_2,v_1,v_2)
        else:
            print("Monostable system")
            xA_star, xB_star, xC_star = null_eigenvector_eq(K_1,K_2,v_1,v_2)
    else:
        print("Monostable system")
        xA_star, xB_star, xC_star = null_eigenvector_eq(K_1,K_2,v_1,v_2)
    
    L = build_L(xA_star, xB_star, xC_star, v_2)
    spectrum = get_spectrum(L)
    list_eig.append(spectrum)


#%%

fig, ax = plt.subplots()
ax.plot(v_2_range, np.array(list_eig).T[1])
ax.grid(color = 'black',alpha = 0.5)
ax.set_xlabel(r"Control parameter $v_2$")
ax.set_ylabel(r"Fiedler eigenvalue")
ax.set_title('Fiedler eigenvalue')
#fig.savefig('fiedler_det.png', dpi=150, facecolor='white', transparent=False)

fig, ax = plt.subplots()
ax.plot(v_2_range, np.array(list_eig).T[2])
ax.grid(color = 'black',alpha = 0.5)
ax.set_xlabel(r"Control parameter $v_2$")
ax.set_ylabel(r"Largest eigenvalue")
ax.set_title('Largest eigenvalue')
#fig.savefig('largest_det.png', dpi=150, facecolor='white', transparent=False)


# %%
