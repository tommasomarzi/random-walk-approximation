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

K_1 = K_4 = 0.9
K_2 = K_3 = 1.
v_1 = v_4 = 1.
#v_3 = v_2 = 0.4
N_min = 15
N_max = 105
N_step = 10
v_2_values = [1.8, 3.0]
peaks = ['A', 'C']
stability = None

#-------------------- Threshold --------------------#
#---------------------------------------------------#

def v2_threshold(k1,k2,v1,v2):
    v2_high = v1*(1+k2)/(1 - 2*k1)
    v2_low =  v1*(1+k2)/(1 + 2*k1)
    if v2_high > v2_low:
        return [v2_low, v2_high]
    elif v2_high < v2_low:
        return [v2_high, v2_low]


#----------------- Transition rates ----------------#
#---------------------------------------------------#   

def pi_AA(xA,xB,xC, k1, k2, v1, v2):
    AA = 0.
    return AA

def pi_AB(xA,xB,xC, k1, k2, v1, v2):
    AB = k1*v2/(k2*k1 + k1*xB + k2*xC)
    return AB

def pi_AC(xA,xB,xC, k1, k2, v1, v2):
    AC = 0. # k*(N - n_A - n_B)/N
    return AC

def pi_BA(xA,xB,xC, k1, k2, v1, v2):
    BA = k2*v1/(k1*k2 + k1*xB + k2*xA)
    return BA

def pi_BB(xA,xB,xC, k1, k2, v1, v2):
    BB = 0.
    return BB

def pi_BC(xA,xB,xC, k1, k2, v1, v2):
    BC = k2*v1/(k2*k1 + k1*xB + k2*xC)
    return BC

def pi_CA(xA,xB,xC, k1, k2, v1, v2):
    CA = 0.  # k*n_A/N
    return CA

def pi_CB(xA,xB,xC, k1, k2, v1, v2):
    CB = k1*v2/(k1*k2 + k1*xB + k2*xA)
    return CB

def pi_CC(xA,xB,xC, k1, k2, v1, v2):
    CC = 0.
    return CC


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
    
    return xA_1, xB, xC_1, xA_2, xC_2


#---------------------- Cycles  --------------------#
#---------------------------------------------------#
for v_2 in v_2_values:
    
    v_3 = v_2

    print("\nStarting simulation with v_1 = {}, v_2 = {}, K_1 = {}, K_2 = {}".format(v_1,v_2,K_1,K_2))     
    
    th = v2_threshold(K_1,K_2,v_1,v_2)

    print("The threshold is: (low = {}, high = {})".format(th[0],th[1]))
    
    for N in np.arange(N_min, N_max + 1, N_step):

        Z_list = []

        print("\nStarting simulation with N = {}".format(N))

        for peak in peaks:
            K_4 = K_1

            #--------------- Numerical solution ----------------#
            #---------------------------------------------------#   
            def null_eigenvector_ns(vector):
                x,y,z = vector
                return [K_3*v_1*x/(K_1*K_3 + K_1*y + K_3*x) - K_4*v_2*y/(K_2*K_4 + K_4*y + K_2*z),
                        K_2*v_4*z/(K_2*K_4 + K_4*y + K_2*z) - K_1*v_3*y/(K_1*K_3 + K_1*y + K_3*x),
                        x + y + z - 1]

            if ((v_2 <= th[0]) or (v_2 >= th[1])):
                if ((v_2 > v_1) and (v_2 >= (v_1*(K_2+1)))):
                    xA1_star, xB_star, xC1_star, xA2_star, xC2_star = null_eigenvector_neq(K_1,K_2,v_1,v_2)
                    stability = 'bistable'
                    if peak == 'A':
                        crit_ns = fsolve(null_eigenvector_ns, [1.,0.,0.])
                        xA_star = xA1_star
                        xC_star = xC1_star
                    elif peak == 'C':
                        crit_ns = fsolve(null_eigenvector_ns, [0.,0.,1.])
                        xA_star = xA2_star
                        xC_star = xC2_star
            else:
                stability = 'monostable'
                crit_ns = fsolve(null_eigenvector_ns, [0.,1.,0.])
                xA_star, xB_star, xC_star = null_eigenvector_eq(K_1,K_2,v_1,v_2)

            assert(np.isclose((xA_star + xB_star + xC_star),1.))

            print("\nAnalytical critical point is:\t ({:.3f}, {:.3f}, {:.3f})".format(xA_star,xB_star,xC_star))
            print("\nNumerical critical point is:\t ({:.3f}, {:.3f}, {:.3f})".format(crit_ns[0],crit_ns[1],crit_ns[2]))

            if np.allclose(crit_ns, [xA_star, xB_star, xC_star]):
                print("\nThe analytic solution is close to the numerical one.")


            #---------------- Derivatives of L -----------------#
            #---------------------------------------------------#

            den_A = (K_1*K_2 + K_1*xB_star + K_2*xA_star)
            den_C = (K_1*K_2 + K_1*xB_star + K_2*xC_star)

            pi_BA_der_A = -((K_2**2)*v_1)/(den_A**2)
            pi_BA_der_B = -(K_1*K_2*v_1)/(den_A**2)
            pi_BA_der_C = 0.

            pi_AB_der_A = 0.
            pi_AB_der_B = -((K_1**2)*v_2)/(den_C**2)
            pi_AB_der_C = -(K_1*K_2*v_2)/(den_C**2)

            pi_CB_der_A = -(K_1*K_2*v_2)/(den_A**2)
            pi_CB_der_B = -((K_1**2)*v_2)/(den_A**2)
            pi_CB_der_C = 0.

            pi_BC_der_A = 0.
            pi_BC_der_B = -(K_1*K_2*v_1)/(den_C**2)
            pi_BC_der_C = -((K_2**2)*v_1)/(den_C**2)

            pi_AC_der_A = pi_AC_der_B = pi_AC_der_C = 0.

            pi_CA_der_A = pi_CA_der_B = pi_CA_der_C = 0.


            #------------ Elements of L and L_bar --------------#
            #---------------------------------------------------#

            L_AA_bar =  (pi_BA_der_A + pi_CA_der_A)*xA_star - pi_AB_der_A*xB_star - pi_AC_der_A*xC_star
            L_BA_bar = -pi_BA_der_A*xA_star + (pi_AB_der_A + pi_CB_der_A)*xB_star - pi_BC_der_A*xC_star
            L_CA_bar = -pi_CA_der_A*xA_star - pi_CB_der_A*xB_star + (pi_AC_der_A + pi_BC_der_A)*xC_star 

            L_AB_bar =  (pi_BA_der_B + pi_CA_der_B)*xA_star - pi_AB_der_B*xB_star - pi_AC_der_B*xC_star
            L_BB_bar = -pi_BA_der_B*xA_star + (pi_AB_der_B + pi_CB_der_B)*xB_star - pi_BC_der_B*xC_star
            L_CB_bar = -pi_CA_der_B*xA_star - pi_CB_der_B*xB_star + (pi_AC_der_B + pi_BC_der_B)*xC_star 

            L_AC_bar =  (pi_BA_der_C + pi_CA_der_C)*xA_star - pi_AB_der_C*xB_star - pi_AC_der_C*xC_star
            L_BC_bar = -pi_BA_der_C*xA_star + (pi_AB_der_C + pi_CB_der_C)*xB_star - pi_BC_der_C*xC_star
            L_CC_bar = -pi_CA_der_C*xA_star - pi_CB_der_C*xB_star + (pi_AC_der_C + pi_BC_der_C)*xC_star 

            pi_BA_star = pi_BA(xA_star, xB_star, xC_star, K_1, K_2, v_1, v_2)
            pi_CA_star = pi_CA(xA_star, xB_star, xC_star, K_1, K_2, v_1, v_2)
            pi_AB_star = pi_AB(xA_star, xB_star, xC_star, K_1, K_2, v_1, v_2)
            pi_CB_star = pi_CB(xA_star, xB_star, xC_star, K_1, K_2, v_1, v_2)
            pi_AC_star = pi_AC(xA_star, xB_star, xC_star, K_1, K_2, v_1, v_2)
            pi_BC_star = pi_BC(xA_star, xB_star, xC_star, K_1, K_2, v_1, v_2)


            #------------------- Drift matrix ------------------#
            #---------------------------------------------------#

            L = np.array([[  pi_BA_star + pi_CA_star, - pi_AB_star,              - pi_AC_star],
                          [- pi_BA_star,                pi_AB_star + pi_CB_star, - pi_BC_star],     
                          [- pi_CA_star,              - pi_CB_star,                pi_AC_star+pi_BC_star]])

            L_bar = np.array([[L_AA_bar, L_AB_bar, L_AC_bar],
                              [L_BA_bar, L_BB_bar, L_BC_bar],     
                              [L_CA_bar, L_CB_bar, L_CC_bar]])

            A = L + L_bar

            #----------------- Diffusion matrix ----------------#
            #---------------------------------------------------#   

            D_AA =   pi_BA_star*xA_star + pi_AB_star*xB_star + pi_CA_star*xA_star + pi_AC_star*xC_star
            D_BA = - pi_BA_star*xA_star - pi_AB_star*xB_star
            D_CA = - pi_CA_star*xA_star - pi_AC_star*xC_star

            D_BB =   pi_BA_star*xA_star + pi_AB_star*xB_star + pi_CB_star*xB_star + pi_BC_star*xC_star
            D_AB = - pi_BA_star*xA_star - pi_AB_star*xB_star
            D_CB = - pi_CB_star*xB_star - pi_BC_star*xC_star

            D_CC =   pi_AC_star*xC_star + pi_CA_star*xA_star + pi_CB_star*xB_star + pi_BC_star*xC_star
            D_AC = - pi_AC_star*xC_star - pi_CA_star*xA_star
            D_BC = - pi_CB_star*xB_star - pi_BC_star*xC_star

            D = np.array([[ D_AA, D_AB, D_AC],
                          [ D_BA, D_BB, D_BC],
                          [ D_CA, D_CB, D_CC]])

            D = D/N


            #------------------- First moment ------------------#
            #---------------------------------------------------#   

            delta_xA_star = 0
            delta_xB_star = 0
            delta_xC_star = 0


            #------------------ Second moment ------------------#
            #---------------------------------------------------#   

            precision = 1e-14        #add offset to avoid negative (close to zero) eigenvalues

            sigma = linalg.solve_continuous_lyapunov(A + precision*np.eye(*A.shape), D + precision*np.eye(*D.shape))  

            if np.allclose(A.dot(sigma) + sigma.dot(A.T), D):
                print("\nThe covariance matrix solves the Lyapunov equation.")

            if np.all(LA.eigvals(sigma)>=0):
                print("\nThe covariance matrix is positive semidefinite.")

            #----------------- Gaussian solution ---------------#
            #---------------------------------------------------#   

            Z = -np.zeros((N+1, N+1))
            for i in np.arange(N+1):
                for j in np.arange(N+1-i):
                    assert(i+j <= N) 
                    #Z[i,j] =  multivariate_normal.pdf([i/N,j/N,1-i/N-j/N], mean=[xA_star,xB_star,xC_star], cov=sigma)
                    Z[i,j] =  multivariate_normal.pdf([i/N,1-i/N-j/N,j/N], mean=[xA_star,xB_star,xC_star], cov=sigma)
            
            Z_list.append(Z)

            if stability == 'monostable':
                break
        
        Z = np.abs(np.array(Z_list)).sum(axis=0)

        Z_norm = Z.T/Z.sum()


        #--------------- Save pic and file -----------------#
        #---------------------------------------------------#   
        namefile = str(N)
        if stability == "bistable":
            namefile = 'results/LNA_results/bistable/' + namefile
            namefile += '_bistable_LNA'
        else:
            namefile = 'results/LNA_results/monostable/' + namefile
            namefile += '_monostable_LNA'

        np.savetxt(namefile+'.txt', Z_norm)


        #------------------ Visualization ------------------#
        #---------------------------------------------------#
        if False:
            low_xA = int(N*xA_star)-50
            if low_xA < 0:
                low_xA = 0
            high_xA = int(N*xA_star)+51
            if high_xA > N:
                high_xA = N
            low_xB = int(N*xB_star)-50
            if low_xB < 0:
                low_xB = 0
            high_xB = int(N*xB_star)+51
            if high_xB > N:
                high_xB = N
        else:
            low_xA = 0
            high_xA = N
            low_xB = 0
            high_xB = N
        fig = plt.figure(figsize=(8,6))
        plt.imshow(Z_norm,origin='lower',interpolation='nearest')
        plt.colorbar()
        if stability == "bistable":
            plt.title(u"Bistable with $K_1 = {}$ , $K_2 = {}$, $v_1 = {}$, $v_2 = {}$ and $N = {}$".format(K_1, K_2, v_1, v_2, N))
        else:
            plt.title(u"Monostable with $K_1 = {}$ , $K_2 = {}$, $v_1 = {}$, $v_2 = {}$ and $N = {}$".format(K_1, K_2, v_1, v_2, N))
            
        plt.xlim(low_xA,high_xA)
        plt.ylim(low_xB,high_xB)
        plt.xlabel("$n_A$",size=14)
        plt.ylabel("$n_C$",size=14)
        plt.tight_layout()
        plt.savefig(namefile+'.png', dpi=300, facecolor='white', transparent=False)
        plt.close()

        print(20*"-")
        print("\n")
