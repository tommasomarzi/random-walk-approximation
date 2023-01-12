import numpy as np
from numpy import linalg as LA
from scipy import linalg 
from scipy import integrate
import math
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from scipy.optimize import fsolve
import os

#----------------- Global parameters ---------------#
#---------------------------------------------------#

K_1 = K_4 = 1.
K_2 = K_3 = 2.
v_1 = v_4 = 1.
v_2_values = [1.05, 1.15] #1.15
N_min = 10
N_max = 100
N_step = 1
K_1_values = [1.0, 1.5]
peaks = ['C']
points = ['eq', 'neq']

stability = {}
for v_2 in v_2_values:
    stability[v_2] = {}
    for K_1 in K_1_values:
        stability[v_2][K_1] = {'eq':[], 'neq':[]}


#-------------------- Threshold --------------------#
#---------------------------------------------------#

def N_threshold(k1,k2,v1,v2):
    N_high = (v1*k2 + v2*k1 + v2*(k1**2))/(v2-v1)
    N_low = (v1*k2 - 3*v2*k1 + v2*(k1**2))/(v2-v1)
    return [N_low, N_high]


#-------------------- Gradients --------------------#
#---------------------------------------------------#

def grad(x,y,z,k1,k2,v1,v2):
    a = 1.
    grad_1_x = -k2*v1*(k1*k2/a + k1*y)/((k1*k2/a + k1*y + k2*x)**2)
    grad_1_y =  k1*v2*(k1*k2/a + k2*z)/((k1*k2/a + k1*y + k2*z)**2) + (k2*v1*k1*x)/((k1*k2/a + k1*y + k2*x)**2)
    grad_1_z = -(k1*k2*v2*y)/((k1*k2/a + k1*y + k2*z)**2)
    
    grad_2_x = -(k1*k2*v2*y)/((k1*k2/a + k1*y + k2*x)**2)
    grad_2_y =  k1*v2*(k1*k2/a + k2*x)/((k1*k2/a + k1*y + k2*x)**2) + (k2*v1*k1*z)/((k1*k2/a + k1*y + k2*z)**2)
    grad_2_z = -k2*v1*(k1*k2/a + k1*y)/((k1*k2/a + k1*y + k2*z)**2)
    
    #grad_3_x = grad_3_y = grad_3_z = -1.

    grad_3_x = - (grad_1_x + grad_2_x)
    grad_3_y = - (grad_1_y + grad_2_y)
    grad_3_z = - (grad_1_z + grad_2_z)

    return np.array([[grad_1_x, grad_1_y, grad_1_z],
                     [grad_3_x, grad_3_y, grad_3_z],
                     [grad_2_x, grad_2_y, grad_2_z]])

#------------------ Exact solution -----------------#
#---------------------------------------------------#   

def null_eigenvector_eq(k1,k2,v1,v2):
    xB = 1/(1 + (k1*v2*2/(k2*v1)))
    xA = xC = (1-xB)/2
    return xA, xB, xC


def null_eigenvector_neq(k1,k2,v1,v2, N):
    xB = k2*v1/(v2 - v1)

    B = - (v2*(k1 + N - k1**2) - v1*(N + k2))/(v2 - v1)
    C = (v2*k1*k1*(1 + v1/(v2 - v1)))/(v2 - v1)

    xA_1 = (-B + math.sqrt(B**2 - 4*C))/(2*N)
    xA_2 = (-B - math.sqrt(B**2 - 4*C))/(2*N)

    if (xA_1 < 0) or (xA_1 > 1):
        print("A problem occured")

    if (xA_2 < 0) or (xA_2 > 1):
        print("A problem occured")
    
    xB = xB/N

    xC_1 = 1 - xA_1 - xB
    xC_2 = 1 - xA_2 - xB
    
    return xA_1, xB, xC_1, xA_2, xC_2


#---------------------- Cycles  --------------------#
#---------------------------------------------------#
for v_2 in v_2_values:
    v_3 = v_2
    for K_1 in K_1_values:
        th = N_threshold(K_1,K_2,v_1,v_2)
        
        print("Starting simulation with $v_2$ = {} and $K_1$ = {}".format(v_2,K_1))
        for N in np.arange(N_min, N_max + 1, N_step):

            xA_star, xB_star, xC_star = null_eigenvector_eq(K_1,K_2,v_1,v_2)
            jacobian_matrix = grad(xA_star*N, xB_star*N, xC_star*N,K_1,K_2,v_1,v_2)
            eig = LA.eigvals(jacobian_matrix)
            eig.real[np.isclose(eig.real, np.zeros(eig.shape))] = 0.

            if np.all(eig.real <= 0):
                stability[v_2][K_1]['eq'].append(v_2*K_1)
            else:
                stability[v_2][K_1]['eq'].append(-v_2*K_1)
            
            if ((N <= th[0]) or (N >= th[1])):
                
                xA1_star, xB_star, xC1_star, xA2_star, xC2_star = null_eigenvector_neq(K_1,K_2,v_1,v_2,N)
                
                jacobian_matrix_1 = grad(xA1_star*N, xB_star*N, xC1_star*N,K_1,K_2,v_1,v_2)
                eig_1 = LA.eigvals(jacobian_matrix_1)
                eig_1.real[np.isclose(eig_1.real, np.zeros(eig_1.shape))] = 0.
            
                jacobian_matrix_2 = grad(xA2_star*N, xB_star*N, xC2_star*N,K_1,K_2,v_1,v_2)
                eig_2 = LA.eigvals(jacobian_matrix_2)
                eig_2.real[np.isclose(eig_2.real, np.zeros(eig_2.shape))] = 0.

                if (np.all(eig_1.real <= 0) and np.all(eig_2.real <= 0)):
                    stability[v_2][K_1]['neq'].append(v_2*K_1)
                else:
                    stability[v_2][K_1]['neq'].append(-v_2*K_1)
            else:
                stability[v_2][K_1]['neq'].append(0)





fig = plt.figure(figsize=(8,6))

for v_2 in v_2_values:
    for K_1 in K_1_values:
        plt.scatter(np.arange(N_min, N_max + 1, N_step), stability[v_2][K_1]['eq'], marker = 'x', label = 'eq + v_2 = '+str(v_2)+ ' + K_1 = '+str(K_1))
        plt.scatter(np.arange(N_min, N_max + 1, N_step), stability[v_2][K_1]['neq'], marker = '.',label = 'neq + v_2 = '+str(v_2)+ ' + K_1 = '+str(K_1))

plt.title("Stability as a function of N")
plt.legend()

plt.ylim(-2,5.)
plt.xlabel("N",size=14)
plt.ylabel("Stability",size=14)
plt.xticks(np.arange(N_min, N_max + 1, 10))

plt.tight_layout()
plt.savefig('stability.png', dpi=300, facecolor='white', transparent=False)
plt.close()
