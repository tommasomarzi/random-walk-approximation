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
v_3 = v_2 = 1.15
N_min = 30
N_max = 105
N_step = 10
K_1_values = [1.0]
peaks = ['C']
points = ['eq', 'neq']
stability = {'eq':[], 'neq':[]}

#-------------------- Gradients --------------------#
#---------------------------------------------------#

def grad(x,y,z):
    a = 1.
    grad_1_x = -K_2*v_1*(K_1*K_2/a + K_1*y)/((K_1*K_2/a + K_1*y + K_2*x)**2)
    grad_1_y =  K_1*v_2*(K_1*K_2/a + K_2*z)/((K_1*K_2/a + K_1*y + K_2*z)**2) + (K_2*v_1*K_1*x)/((K_1*K_2/a + K_1*y + K_2*x)**2)
    grad_1_z = -(K_1*K_2*v_2*y)/((K_1*K_2/a + K_1*y + K_2*z)**2)
    
    grad_2_x = -(K_1*K_2*v_2*y)/((K_1*K_2/a + K_1*y + K_2*x)**2)
    grad_2_y =  K_1*v_2*(K_1*K_2/a + K_2*x)/((K_1*K_2/a + K_1*y + K_2*x)**2) + (K_2*v_1*K_1*z)/((K_1*K_2/a + K_1*y + K_2*z)**2)
    grad_2_z = -K_2*v_1*(K_1*K_2/a + K_1*y)/((K_1*K_2/a + K_1*y + K_2*z)**2)
    
    #grad_3_x = grad_3_y = grad_3_z = -1.

    grad_3_x = - (grad_1_x + grad_2_x)
    grad_3_y = - (grad_1_y + grad_2_y)
    grad_3_z = - (grad_1_z + grad_2_z)

    return np.array([[grad_1_x, grad_1_y, grad_1_z],
                     [grad_3_x, grad_3_y, grad_3_z],
                     [grad_2_x, grad_2_y, grad_2_z]])


#---------------------- Cycles  --------------------#
#---------------------------------------------------#
for p in points:
    for N in np.arange(N_min, N_max + 1, N_step):
        for peak in peaks:

            #--------------- Numerical solution ----------------#
            #---------------------------------------------------#   
            def null_eigenvector_ns(vector):
                x,y,z = vector
                return [K_3*v_1*x/(K_1*K_3/N + K_1*y + K_3*x) - K_4*v_2*y/(K_2*K_4/N + K_4*y + K_2*z),
                        K_2*v_4*z/(K_2*K_4/N + K_4*y + K_2*z) - K_1*v_3*y/(K_1*K_3/N + K_1*y + K_3*x),
                        x + y + z - 1]

            crit_ns = fsolve(null_eigenvector_ns, [1, 0, 0])


            #------------------ Exact solution -----------------#
            #---------------------------------------------------#   

            def null_eigenvector_eq():
                xB = 1/(1 + (K_1*v_2*2/(K_2*v_1)))
                xA = xC = (1-xB)/2
                return xA, xB, xC


            def null_eigenvector_neq():
                xB = K_2*v_1/(v_2 - v_1)

                B = - (v_2*(K_1 + N - K_1**2) - v_1*(N + K_2))/(v_2 - v_1)
                C = (v_2*K_1*K_1*(1 + v_1/(v_2 - v_1)))/(v_2 - v_1)
                xA = (-B - math.sqrt(B**2 - 4*C))/(2*N)

                if (xA < 0) or (xA > 1):
                    xA = (-B - math.sqrt(B**2 - 4*C))/(2*N)

                xB = xB/N

                xC = 1 - xA - xB
                
                if peak == 'A':
                    return xA, xB, xC
                elif peak == 'C':
                    return xC, xB, xA
                else:
                    raise ValueError('Chosen peak not supported')

            if p == 'eq':
                xA_star, xB_star, xC_star = null_eigenvector_eq()
            elif p == 'neq':
                xA_star, xB_star, xC_star = null_eigenvector_neq()
            else:
                raise ValueError('Chosen stationary point does not exist')

            jacobian_matrix = grad(xA_star*N, xB_star*N, xC_star*N)
            eig = LA.eigvals(jacobian_matrix)
            print(eig)
            eig.real[np.isclose(eig.real, np.zeros(eig.shape))] = 0.

            if np.all(eig.real <= 0):
                print("{} with N = {}:   stable".format(p,N))
                stability[p].append(1)
            else:
                print("{} with N = {}:   unstable".format(p,N))
                stability[p].append(-1)


fig = plt.figure(figsize=(8,6))

for p in points:
    plt.plot(np.arange(N_min, N_max + 1, N_step), stability[p], label = p)

plt.title("Stability (+1) or instability (-1) as a function of N")
plt.legend()

plt.ylim(-1.5,1.5)
plt.xlabel("N",size=14)
plt.ylabel("Stability",size=14)
plt.xticks(np.arange(N_min, N_max + 1, N_step))

plt.tight_layout()
plt.savefig('stability.png', dpi=300, facecolor='white', transparent=False)
plt.close()
