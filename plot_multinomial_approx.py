# file stationary solution laplacian Bazzani project
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
import fnmatch
from math import factorial  # for multinomial distribution
from mpl_toolkits import mplot3d  # for 3D plotd
rc('text', usetex=True)
from pathlib import Path
from math import log #for log of integer(when numbers are too big for the multinomial)
from collections import Counter
os.chdir('D:\spoli\Documents\seagate\Post-doc Bologna\progetto laplacian graph theory')
path = %pwd

%matplotlib
# %%

# For rates as in the phosphorylation paper normalized by columns (but is not important
# for the stationary solution, because diagonal rates do not enter in the laplacian matrix)
# considering also reaction velocities (put them to 1 if you don't want them)

N = 2
n_A, n_B = 1, 0 # N/3., N/3.
K_1 = 0.1 #np.linspace(0.000001,0.05,200)
K_2 = 1 #np.linspace(0.00001,5,200)  #they are the small k of the paper
v_2 = v_3 = 1.8   #1.15 "bistable" 1.05 "monostable"
v_1 = v_4 = 1
n_C = N - n_A - n_B
pi_AA = 0 #old 1 - K_2*n_A/(K_1*K_2 + K_1*n_B + K_2*n_A)
pi_AB = v_2*K_1/(N*K_1*K_2 + K_1*n_B + K_2*n_C) #v_2*n_B*K_1/(K_1*K_2 + K_1*n_B + K_2*n_C)
pi_AC = 0  # k*(N - n_A - n_B)/N
pi_BA = v_1*K_2/(N*K_1*K_2 + K_1*n_B + K_2*n_A) #v_1*K_2*n_A/(K_1*K_2 + K_1*n_B + K_2*n_A)
pi_BB = 0 #old 1 -  n_B*K_1/(K_1*K_2 + K_1*n_B + K_2*n_C) - n_B*K_1/(K_1*K_2 + K_1*n_B + K_2*n_A)
pi_BC = v_4*K_2/(N*K_1*K_2 + K_2*n_C + K_1*n_B) #v_4*K_2*n_C/(K_1*K_2 + K_2*n_C + K_1*n_B)
pi_CA = 0  # k*n_A/N
pi_CB = v_3*K_1/(N*K_1*K_2 + K_1*n_B + K_2*n_A) # v_3*n_B*K_1/(K_1*K_2 + K_1*n_B + K_2*n_A)
pi_CC = 0 #old 1 - K_2*n_C/(K_1*K_2 + K_2*n_C + K_1*n_B)

# %%
Z = pi_AC*pi_AB + pi_AB*pi_BC + pi_AC*pi_CB + \
    pi_BC*pi_CA + pi_BA*pi_AC + pi_CA*pi_CB + \
    pi_CA*pi_AB + pi_BC*pi_BA + pi_CB*pi_BA
p_A = (pi_AB*pi_BC + pi_AC*pi_AB + pi_AC*pi_CB)/Z
p_B = (pi_BC*pi_BA + pi_BC*pi_CA + pi_BA*pi_AC)/Z
p_C = (pi_CA*pi_CB + pi_CA*pi_AB + pi_CB*pi_BA)/Z
p_A + p_B + p_C;
plt.figure()
plt.plot(K_2, p_B, label = "$p_B$")
plt.plot(K_2, p_A, label = "$p_A$")
plt.plot(K_2, p_C, label = "$p_C$")
plt.xlabel("$K_{M_2}$")
plt.legend()
#plt.savefig('stationary_phospho_vs_K2.png', format='png', dpi=300)

# %%

#####################    Plots of the rates to see how much they differ from constants    ################
### Rates are not constant at all

N = 45
Z = np.zeros((N+1, N+1))
X, Y = np.meshgrid(np.arange(N+1), np.arange(N+1), indexing = 'ij')
for j in np.arange(N+1):
    for i in np.arange(N-j + 1):
        Z[i,j] = v_2*K_1/(K_1*K_2 + K_1*Y[i,j] + K_2*(N-X[i,j]-Y[i,j])) # pi_AB
        #Z[i,j] = v_1*K_2/(K_1*K_2 + K_1*Y[i,j] + K_2*X[i,j])  #pi_BA
Z[43,2]
np.where(Z == np.max(Z))
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z, c=Z, cmap='Greens')
#ax.set_zlim(0, 5)
ax.set_xlabel('$n_A$')
ax.set_ylabel('$n_B$')
ax.set_zlabel('$\\rho$')
ax.set_title("rates $\pi_{AB}  v_2 = 0.8$")

# %%
# Verify mean field approx

N = 500
Z = -np.ones((N+1, N+1))
X, Y = np.meshgrid(np.arange(N+1), np.arange(N+1), indexing = 'ij')
for j in np.arange(N+1):
    for i in np.arange(N-j + 1):
        n_C = N - X[i,j] - Y[i,j]
        Z[i,j] = p_A_n(X[i,j], Y[i,j], n_C, N) # pi_AB
# Z = np.where(np.isfinite(Z), Z, -np.inf)
np.shape(Z)  #it's approximately the mean value for N = 105
np.where(Z == np.max(Z))
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z, c=Z, cmap='Greens')
#ax.set_zlim(0, 5)
ax.set_xlabel('$n_A$')
ax.set_ylabel('$n_B$')
ax.set_zlabel('$p_A$')
ax.set_title("probability $p_{A}$ $v_2 = 2.2$")


# %%

# %%
# global parameters
N = 45
K_1 = 1
K_2 = 2
v_2 = v_3 = 1.
v_1 = v_4 = 1
# %%
x = 0/N
y = 2/N
n_C
z = (1 - x - y)
p_A_n(x, y, z, N)
p_B_n(x, y, z, N)
p_C_n(x, y, z, N)


[v_1*K_2*x/(K_1*K_2/N + K_1*y + K_2*x) - v_2*K_1*y/(K_1*K_2/N + K_1*y + K_2*z),
        v_4*K_2*z/(K_1*K_2/N + K_1*y + K_2*z) - v_3*K_1*y/(K_1*K_2/N + K_1*y + K_2*x),
        x + y + z - 1]

def p_A_n(x, y, z, N): # I removed x, y , n_C from the numerator
    '''
    x, y and z are the number of the particles in each state
    '''
    pi_AA = 0 #1 - k_2*x/(k_1*k_2 + k_1*y + k_2*x)
    pi_AB = v_2*k_1/(k_1*k_2 + k_1*y/N + k_2*z/N)
    pi_AC = 0  # k*(N - n_A - n_B)/N
    pi_BA = v_1*k_2/(k_1*k_2 + k_1*y/N + k_2*x/N)
    pi_BB = 0 # 1 - y*k_1/(k_1*k_2 + k_1*y + k_2*n_C) - y*k_1/(k_1*k_2 + k_1*y + k_2*x)
    pi_BC = v_4*k_2/(k_1*k_2 + k_2*z/N + k_1*y/N)
    pi_CA = 0  # k*n_A/N
    pi_CB = v_3*k_1/(k_1*k_2 + k_1*y/N + k_2*x/N)
    pi_CC = 0 # 1 - k_2*n_C/(k_1*k_2 + k_2*n_C + k_1*y)
    Z = pi_AC*pi_AB + pi_AB*pi_BC + pi_AC*pi_CB + \
        pi_BC*pi_CA + pi_BA*pi_AC + pi_CA*pi_CB + \
        pi_CA*pi_AB + pi_BC*pi_BA + pi_CB*pi_BA
    return (pi_AB*pi_BC + pi_AC*pi_AB + pi_AC*pi_CB)/Z

def p_B_n(x, y, z, N): # I removed x, y , n_C from the numerator
    pi_AA = 0 # 1 - k_2*x/(k_1*k_2 + k_1*y + k_2*x)
    pi_AB = v_2*k_1/(k_1*k_2 + k_1*y/N + k_2*z/N)
    pi_AC = 0  # k*(N - n_A - n_B)/N
    pi_BA = v_1*k_2/(k_1*k_2 + k_1*y/N + k_2*x/N)
    pi_BB = 0 # 1 - y*k_1/(k_1*k_2 + k_1*y + k_2*n_C) - y*k_1/(k_1*k_2 + k_1*y + k_2*x)
    pi_BC = v_4*k_2/(k_1*k_2 + k_2*z/N + k_1*y/N)
    pi_CA = 0  # k*n_A/N
    pi_CB = v_3*k_1/(k_1*k_2 + k_1*y/N + k_2*x/N)
    pi_CC = 0 # 1 - k_2*n_C/(k_1*k_2 + k_2*n_C + k_1*y)
    Z = pi_AC*pi_AB + pi_AB*pi_BC + pi_AC*pi_CB + \
        pi_BC*pi_CA + pi_BA*pi_AC + pi_CA*pi_CB + \
        pi_CA*pi_AB + pi_BC*pi_BA + pi_CB*pi_BA
    return (pi_BC*pi_BA + pi_BC*pi_CA + pi_BA*pi_AC)/Z

def p_C_n(x, y, z, N): # I removed x, y , n_C from the numerator
    pi_AA = 0 # 1 - k_2*x/(k_1*k_2 + k_1*y + k_2*x)
    pi_AB = v_2*k_1/(k_1*k_2*N + k_1*y + k_2*z)
    pi_AC = 0  # k*(N - n_A - n_B)/N
    pi_BA = v_1*k_2/(k_1*k_2*N + k_1*y + k_2*x)
    pi_BB = 0 # 1 - y*k_1/(k_1*k_2 + k_1*y + k_2*n_C) - y*k_1/(k_1*k_2 + k_1*y + k_2*x)
    pi_BC = v_4*k_2/(k_1*k_2*N + k_2*z + k_1*y)
    pi_CA = 0  # k*n_A/N
    pi_CB = v_3*k_1/(k_1*k_2*N + k_1*y + k_2*x)
    pi_CC = 0 # 1 - k_2*n_C/(k_1*k_2 + k_2*n_C + k_1*y)
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

def mult_const(x, y, p_A, p_B, p_C, N):
    if x + y > N:
        return -np.inf
    n_c = N-x-y
    return log(factorial(N)/(factorial(x)*factorial(y)*factorial(n_c))) + \
            x*log(p_A) + y*log(p_B) + n_c*log(1-p_A-p_B)


#########------------   Distribution with sampled multinomial   ----------################
#########------------   this makes sense only with constant rates   ------################
# %%
#parameters
N = 30
N_min = 5
N_max = 105
N_step = 10
K_1 = 1
K_2 = 2
# parameters corresponding to K_1 = 3 and K_2 = 50, for N = 300 and n_A = 50, n_B = 120
p_A = 0.05192373281268383
p_B = 0.8397595215414214
p_C = 0.10831674564589476
# %%
p = Path(os.path.join(path, "MUL_sampled_files"))
p.mkdir(exist_ok=True)   # it creates a directory if doesn't exist
for K_2 in [0.5,50]:
    if K_2 == 0.5:
        p_b = Path(os.path.join(p, "bistable"))
        p_b.mkdir(exist_ok=True)
    else:
        p_m = Path(os.path.join(p, "monostable"))
        p_m.mkdir(exist_ok=True)   # it creates a directory if doesn't exist
    for N in np.arange(N_min, N_max+1, N_step):
        p_AB = np.zeros((N+1, N+1))
        ZZ = np.random.multinomial(N, [p_A, p_B, p_C], 2533200)
        distributionA = Counter()
        for i in np.arange(np.shape(ZZ)[0]):
                (A, B) = (ZZ[i, 0], ZZ[i, 1])
                distributionA[(A, B)] += 1
        for (A, B), val in distributionA.items():
            p_AB[A, B] = val
        p_AB = p_AB/np.sum(p_AB)
        if K_2 == 0.5:
            namefile = os.path.join(p_b, str(N) + "_bistable_MUL_s")
            np.savetxt(namefile + ".txt", p_AB.transpose())
        else: # it creates a directory if doesn't exist
            namefile = os.path.join(p_m, str(N) + "_monostable_MUL_s")
            np.savetxt(namefile + ".txt", p_AB.transpose())
        #np.savetxt("30_monostable_MUL_s.txt", p_AB.transpose())



########---------- LOOP for txt files with the distribution and the color plots  -------------##########
########################################################################################################

N_min = 105
N_max = 505
N_step = 10
k_1 = 0.1
k_2 = 1
v_2 = v_3 = 2.2
v_1 = v_4 = 1
pwd
cd MUL_files_norm_sum
N = 1000
for v_2 in (np.around(np.linspace(1.5,2.49,100),2)):  #put rounding for file names
    v_3 = v_2
    v_2_s = "{:0.2f}".format(v_2)
    for N in np.arange(N_min, N_max+1, N_step):
        lnZ = -np.ones((N+1, N+1))*np.inf
        for i in np.arange(N+1):
            for j in np.arange(N+1-i):
                lnZ[i, j] = mult(i, j, N)
        lnZ = np.where(np.isfinite(lnZ), lnZ, -np.inf)
        Z = np.exp(lnZ)
        #y = np.linspace(0,N+1,N+1)
        #x = np.linspace(0,N+1,N+1)
        #int_2d = np.trapz(np.trapz(Z, y, axis=0), x, axis=0)
        #Z = Z/int_2d
        Z = Z/np.sum(Z)
        namefile = "bistable/" + str(N) + "_bistable_MUL_v2_" + str(v_2_s) if v_2 >= \
                    2.6 else "monostable/" + str(N) + "_monostable_MUL_v2_" + str(v_2_s)
        np.savetxt(namefile + ".txt", Z.transpose())
        '''
        nA_star, nB_star = (np.where(Z == np.max(Z))[0][0], np.where(Z == np.max(Z))[1][0])
        low_nA = int(nA_star)-50
        if low_nA < 0:
            low_nA = 0
        high_nA = int(nA_star)+51
        if high_nA > N:
            high_nA = N
        low_nB = int(nB_star)-50
        if low_nB < 0:
            low_nB = 0
        high_nB = int(nB_star)+51
        if high_nB > N:
            high_nB = N
        '''
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(Z.transpose(), origin='lower', interpolation='nearest')
        plt.colorbar()
        if (k_1 == 0.001) | (k_2 == 50) | (v_2 <= 2.6):
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




########---------- LOOP for txt files with the distribution and the color plots  -------------##########
########---------- for constant rates
########################################################################################################
'''
pi_AB = 0.05135520684736091
pi_BA = 0.8305647840531561
pi_CB = 0.11960132890365449
pi_BC = 0.927246790299572
xA_star = 0.05192373281268383
xB_star = 0.8397595215414214
xC_star = 0.10831674564589476
'''
v_2 = 5
# %%
def null_eigenvector(K_1, K_2, v_1, v_2 , N):  # rates corrected
    xA = xC = 0.5*(1-v_1*K_2/(2*v_2*K_1 + v_1*K_2))
    xB = v_1*K_2/(2*v_2*K_1 + v_1*K_2)
    xB_temp = - K_1*K_2**2*v_1/(N*(K_1*K_2*v_1 - v_2*K_1*K_2))
    b = xB_temp*v_2*K_1*K_2 - v_1*K_2**2*K_1/N - v_1*K_1*K_2*xB_temp - v_1*K_2**2*(1 - xB_temp)
    a = v_1*K_2**2
    c = xB_temp*v_2*K_1**2*K_2/N + xB_temp**2*v_2*K_1**2
    xA_temp = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    xC_temp = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    if (b**2 - 4*a*c > 0) & (0<=xA_temp<=1) & (0<=xB_temp<=1) & (0<=xC_temp<=1):
        xB = xB_temp
        xA = xA_temp
        xC = xC_temp
        return [(xA, xB, xC),(xC, xB, xA)]
    else:
        return xA, xB, xC

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
os.chdir(path)
def save_MUL_const(v_2, folder = "MUL_files_norm_sum", k_1 = 0.1, k_2 = 1, v_1 = 1, N_min = 5, N_max = 105, N_step = 10):
    """
    Function that saves the txt and png files of the multinomial with the correct constant
    analytical parameters for the system of equations given from model of paper Bazzani (2012)
    with parameters k_1, k_2, v_1, v_2,
    folder: root folder for the data, inside wich a monostable and bistable directory is created
    in path (project folder)
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
                    2.6 else os.path.join(folder, "monostable", str(N) + "_monostable_MUL_v2_" + str(v_2) + "_const")
        np.savetxt(namefile + "_star.txt", Z.transpose())
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(Z.transpose(), origin='lower', interpolation='nearest')
        plt.colorbar()
        if v_2 < 2.6:
            plt.title(u"Monostable with $k_1 = {}$, $k_2 = {}$, $v_2 = v_3 = {}$ and $N = {}$".format(k_1, k_2, v_2, N))
        else:
            plt.title(u"Bistable with $k_1 = {}$, $k_2 = {}$, $v_2 = v_3 = {}$ and $N = {}$".format(k_1, k_2, v_2, N))
        plt.xlabel("$n_A$", size=14)
        plt.ylabel("$n_C$", size=14)
        plt.tight_layout()
        plt.savefig(namefile + '_star.png', dpi=300, facecolor='white', transparent=False)
        plt.close()

save_MUL_const(1.8, folder = "MUL_files_star", k_1 = 0.1, k_2 = 1,
                v_1 = 1, N_min = 5, N_max = 105, N_step = 10)

'''
N_min = 5
N_max = 105
N_step = 10
K_1 = 3
p_A = 0.05192373281268383
p_B = 0.8397595215414214
p_C = 0.10831674564589476
(p_A, p_B, p_C) = (0.16440654510304975, 0.6711869097939005, 0.16440654510304975)
'''

# %%
#cd MUL_files_norm_sum
N = 30
for K_2 in [0.5,50]:
    for N in np.arange(N_min, N_max+1, N_step):
        lnZ = -np.ones((N+1, N+1))*np.inf
        p_A, p_B, p_C = null_eigenvector(K_1, K_2)
        for i in np.arange(N+1):
            for j in np.arange(N+1-i):
                lnZ[i, j] = mult_const(i,j)
        lnZ = np.where(np.isfinite(lnZ), lnZ, -np.inf)
        Z = np.exp(lnZ)
        y = np.linspace(0,N+1,N+1)
        x = np.linspace(0,N+1,N+1)
        int_2d = np.trapz(np.trapz(Z, y, axis=0), x, axis=0)
        #Z = Z/int_2d
        Z = Z/np.sum(Z)
        namefile = "bistable/" + str(N) + "_bistable_MUL_const" if K_2 == \
                    0.5 else "monostable/" + str(N) + "_monostable_MUL_const"
        np.savetxt(namefile + "_star.txt", Z.transpose())
        nA_star, nB_star = (np.where(Z == np.max(Z))[0][0], np.where(Z == np.max(Z))[1][0])
        low_nA = int(nA_star)-50
        if low_nA < 0:
            low_nA = 0
        high_nA = int(nA_star)+51
        if high_nA > N:
            high_nA = N
        low_nB = int(nB_star)-50
        if low_nB < 0:
            low_nB = 0
        high_nB = int(nB_star)+51
        if high_nB > N:
            high_nB = N

        fig = plt.figure(figsize=(8, 6))
        plt.imshow(Z.transpose(), origin='lower', interpolation='nearest')
        plt.colorbar()
        if K_2 > 5:
            plt.title(u"Monostable with $K_2 = {}$ and $N = {}$".format(K_2,N))
        else:
            plt.title(u"Bistable with $K_2 = {}$ and $N = {}$".format(K_2,N))
        plt.xlim(low_nA, high_nA)
        plt.ylim(low_nB, high_nB)
        plt.xlabel("$n_A$", size=14)
        plt.ylabel("$n_B$", size=14)
        plt.tight_layout()
        plt.savefig(namefile + '.png', dpi=300, facecolor='white', transparent=False)
        plt.close()


os.chdir(path)
def plot_distr_2D(root_folder, K_1, K_2, v_2, type = ""):
    """
    Returns the 2D color plot of the requested model distribution.
        Parameters:
            root_folder (str): the root folder containing the data inside the project folder
            (data are already transposed to be used with imshow() function)
            type (str): it can be one of the following "time","const", "const_star", "time_vel",
            "AC_time_vel", "v2_--value of v2--". In general it is what is written
            in the file name after the 3 capital letters indicating the type of simulation
                "time" indicates the GIL distribution computed on the time spent on a state;
                "const" indicates the constant multinomial with as parameters the solution of
                the laplacian of the rates;
                "const_star" indicates the constant multinomial with constant parameters equal
                to the stationary analytic solution of the model;
                "time_const" indicates Gillespie distribution with constant rates and
                computed on the time spent on a state;
                "time_vel"  indicates Gillespie distribution with constant rates and
                computed on the time spent on a state, considering reaction velocities;
                "AC_time_vel" indicates Gillespie distribution with constant rates and
                computed on the time spent on a state, considering reaction velocities
                and plotted over (n_A, n_C) space;
                default is empty meaning the standard way depending on root_folder
                "v2_-- value --" indicates models with v2 = at -- value --
    """
    if (K_1 == 0.001) | (K_2 == 50) | (v_2 < 2.6):
        namefile = os.path.join(path,root_folder, "monostable")
    else:
        namefile = os.path.join(path,root_folder, "bistable")
    if (root_folder[-3:] == "try") & (K_2 != 0.01):
        raise ValueError('Parameters are not consistent')
    if type != "":
        type = "_" + type
    file_list = os.listdir(os.path.join(namefile))
    pattern = "*" + root_folder[:3] + type + ".txt"
    good_list = fnmatch.filter(file_list, pattern)
    for f in good_list:
        data_distr = np.loadtxt(os.path.join(namefile,f))
        N = [int(s) for s in f.split(sep = "_") if s.isdigit()][0]
        nA_star, nB_star = (np.where(data_distr == np.max(data_distr))[0][0],
                            np.where(data_distr == np.max(data_distr))[1][0])
        low_nA = int(nA_star)-50
        if low_nA < 0:
            low_nA = 0
        high_nA = int(nA_star)+51
        if high_nA > N:
            high_nA = N
        low_nB = int(nB_star)-50
        if low_nB < 0:
            low_nB = 0
        high_nB = int(nB_star)+51
        if high_nB > N:
            high_nB = N
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(data_distr, origin='lower', interpolation='nearest')
        plt.colorbar()
        if (K_1 == 0.001) | (K_2 == 50) | (v_2 < 1.1):
            plt.title(u"Monostable with $K_1 = {}$, $K_2 = {}$, $v_2 = v_3 = {}$ and $N = {}$".format(K_1, K_2, v_2, N))
        else:
            plt.title(u"Bistable with $K_1 = {}$, $K_2 = {}$, $v_2 = v_3 = {}$ and $N = {}$".format(K_1, K_2, v_2, N))
        #plt.xlim(low_nA, high_nA)
        #plt.ylim(low_nB, high_nB)
        plt.xlabel("$n_A$", size=14)
        plt.ylabel("$n_C$", size=14)
        plt.tight_layout()
        if (K_1 == 0.001) | (K_2 == 50) | (v_2 < 2.6):
            plt.savefig(os.path.join(namefile,str(N) + "_monostable_" + root_folder[:3] + type + ".png"),
                dpi=300, facecolor='white', transparent=False)
        else:
            plt.savefig(os.path.join(namefile,str(N) + "_bistable_" + root_folder[:3] + type + ".png"),
                    dpi=300, facecolor='white', transparent=False)
        plt.close()
plot_distr_2D(root_folder = "GIL_files_ergo", K_2 = 1, K_1 = 0.1, v_2 = 3, type  = "v2_3_AC_vel_k")
# sotto da cancellare
cd GIL_files_ergo_v2_2
cd long
file_list = os.listdir(os.getcwd())
    for f in file_list:
        namefile = f
        data_distr = np.loadtxt(f)
        N = [int(s) for s in f.split(sep = "_") if s.isdigit()][0]
        nA_star, nB_star = (np.where(data_distr == np.max(data_distr))[0][0],
                            np.where(data_distr == np.max(data_distr))[1][0])
        low_nA = int(nA_star)-50
        if low_nA < 0:
            low_nA = 0
        high_nA = int(nA_star)+51
        if high_nA > N:
            high_nA = N
        low_nB = int(nB_star)-50
        if low_nB < 0:
            low_nB = 0
        high_nB = int(nB_star)+51
        if high_nB > N:
            high_nB = N
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(data_distr, origin='lower', interpolation='nearest')
        plt.colorbar()
        if (K_1 == 0.001) | (K_2 == 50) | (v_2 == 1.05):
            plt.title(u"Monostable with $K_1 = {}$, $K_2 = {}$, $v_2 = v_3 = {}$ and $N = {}$".format(K_1, K_2, v_2, N))
        else:
            plt.title(u"Bistable with $K_1 = {}$, $K_2 = {}$, $v_2 = v_3 = {}$ and $N = {}$".format(K_1, K_2, v_2, N))
        #plt.xlim(low_nA, high_nA)
        #plt.ylim(low_nB, high_nB)
        plt.xlabel("$n_A$", size=14)
        plt.ylabel("$n_C$", size=14)
        plt.tight_layout()
        if (K_1 == 0.001) | (K_2 == 50) | (v_2 == 1.05):
            plt.savefig(os.path.join(namefile,str(N) + "_monostable_" + root_folder[:3] + ".png"),
                dpi=300, facecolor='white', transparent=False)
        else:
            plt.savefig(os.path.join(str(N) + "_bistable_" + root_folder[:3] + ".png"),
                    dpi=300, facecolor='white', transparent=False)
        plt.close()
