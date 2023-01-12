#%%
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

# %%

K_1 = 0.1
K_2 = 1.
v_1 = 1.
#v_2_range = np.linspace(1.5,3.52,51)
v_2_range = [1.82]#,2.5,3.04]
N_range = [105] #np.arange(15,206,10)

#%%
#----------------- Transition rates ----------------#
#---------------------------------------------------#

def rate_AA(xA,xB,xC,v2):
    AA = 0.
    return AA

def rate_AB(xA,xB,xC,v2):
    AB = K_1*v2/(K_2*K_1 + K_1*xB + K_2*xC)
    return AB

def rate_AC(xA,xB,xC,v2):
    AC = 0. # k*(N - n_A - n_B)/N
    return AC

def rate_BA(xA,xB,xC,v2):
    BA = K_2*v_1/(K_1*K_2 + K_1*xB + K_2*xA)
    return BA

def rate_BB(xA,xB,xC,v2):
    BB = 0.
    return BB

def rate_BC(xA,xB,xC,v2):
    BC = K_2*v_1/(K_2*K_1 + K_1*xB + K_2*xC)
    return BC

def rate_CA(xA,xB,xC,v2):
    CA = 0.  # k*n_A/N
    return CA

def rate_CB(xA,xB,xC,v2):
    CB = K_1*v2/(K_1*K_2 + K_1*xB + K_2*xA)
    return CB

def rate_CC(xA,xB,xC,v2):
    CC = 0.
    return CC


# %%
#---------------- Compute utilities ----------------#
#---------------------------------------------------#

def pi_AB_BC(x,y,z,v2):
    den = (K_1*K_2+K_1*y+K_2*z)**3
    der_A = 0.
    der_B = -(2*(K_1**2)*K_2*v_1*v2)/den
    der_C = -(2*K_1*(K_2**2)*v_1*v2)/den

    return der_A, der_B, der_C


def pi_BC_BA(x,y,z,v2):
    den_yx = (K_1*K_2+K_1*y+K_2*x)
    den_yz = (K_1*K_2+K_1*y+K_2*z)
    der_A = -(((K_2*v_1)**2)*K_2)/((den_yx**2)*den_yz)
    der_B = -(((K_2*v_1)**2)*(2*(K_1**2)*K_2 + 2*y*(K_1**2)+K_1*K_2*(x+z)))/((den_yx*den_yz)**2)
    der_C = -(((K_2*v_1)**2)*K_2)/(den_yx*(den_yz**2))

    return der_A, der_B, der_C


def pi_CB_BA(x,y,z,v2):
    den = (K_1*K_2+K_1*y+K_2*x)**3
    der_A = -(2*K_1*(K_2**2)*v_1*v2)/den
    der_B = -(2*(K_1**2)*K_2*v_1*v2)/den
    der_C = 0.

    return der_A, der_B, der_C


# %%
#--------------- Compute derivatives ---------------#
#---------------------------------------------------#

def compute_all(x,y,z,v2):
    pi_AB = rate_AB(x,y,z,v2)
    pi_AC = rate_AC(x,y,z,v2)
    pi_BA = rate_BA(x,y,z,v2)
    pi_BC = rate_BC(x,y,z,v2)
    pi_CA = rate_CA(x,y,z,v2)
    pi_CB = rate_CB(x,y,z,v2)

    AB_BC_der_a, AB_BC_der_b, AB_BC_der_c = pi_AB_BC(x,y,z,v2)
    BC_BA_der_a, BC_BA_der_b, BC_BA_der_c = pi_BC_BA(x,y,z,v2)
    CB_BA_der_a, CB_BA_der_b, CB_BA_der_c = pi_CB_BA(x,y,z,v2)

    Z = pi_AC*pi_AB + pi_AB*pi_BC + pi_AC*pi_CB + \
        pi_BC*pi_CA + pi_BA*pi_AC + pi_CA*pi_CB + \
        pi_CA*pi_AB + pi_BC*pi_BA + pi_CB*pi_BA

    Z_squared = Z**2

    Z_der_a = AB_BC_der_a + BC_BA_der_a + CB_BA_der_a
    Z_der_b = AB_BC_der_b + BC_BA_der_b + CB_BA_der_b
    Z_der_c = AB_BC_der_c + BC_BA_der_c + CB_BA_der_c

    p_A_der_a = (Z*AB_BC_der_a -  pi_AB*pi_BC*Z_der_a)/Z_squared
    p_A_der_b = (Z*AB_BC_der_b -  pi_AB*pi_BC*Z_der_b)/Z_squared
    p_A_der_c = (Z*AB_BC_der_c -  pi_AB*pi_BC*Z_der_c)/Z_squared
  
    p_B_der_a = (Z*BC_BA_der_a -  pi_BC*pi_BA*Z_der_a)/Z_squared
    p_B_der_b = (Z*BC_BA_der_b -  pi_BC*pi_BA*Z_der_b)/Z_squared
    p_B_der_c = (Z*BC_BA_der_c -  pi_BC*pi_BA*Z_der_c)/Z_squared
    
    p_C_der_a = (Z*CB_BA_der_a -  pi_CB*pi_BA*Z_der_a)/Z_squared
    p_C_der_b = (Z*CB_BA_der_b -  pi_CB*pi_BA*Z_der_b)/Z_squared
    p_C_der_c = (Z*CB_BA_der_c -  pi_CB*pi_BA*Z_der_c)/Z_squared

    return p_A_der_a,p_A_der_b,p_A_der_c,p_B_der_a,p_B_der_b,p_B_der_c,p_C_der_a,p_C_der_b,p_C_der_c


# %%
#-------------- Build Jacobian matrix --------------#
#---------------------------------------------------#

def build_J(x,y,z,v2):
    p_A_x,p_A_y,p_A_z,p_B_x,p_B_y,p_B_z,p_C_x,p_C_y,p_C_z = compute_all(x,y,z,v2)
    J = np.array([[p_A_x,p_A_y,p_A_z],
                  [p_B_x,p_B_y,p_B_z],
                  [p_C_x,p_C_y,p_C_z]])
    return J


# %%
#--------------- Get max eigenvalue ----------------#
#---------------------------------------------------#

def get_max_eig(matrix):
    eigvals, _ = LA.eig(matrix)
    if not (np.allclose(eigvals.imag , np.zeros(eigvals.shape))):
        raise ValueError("Complex eigenvalue")
    else:
        eigvals = eigvals.real
    max_eig = np.sort(eigvals)[-1] #(np.sort(np.abs(eigvals)))[1]

    #print(eigvals)
    return max_eig

# %%
#--------------------- Def plot --------------------#
#---------------------------------------------------#

def do_plot(eigvals,v2):
    fig, ax = plt.subplots()
    scale = ax.imshow(eigvals.T,origin='lower',interpolation='nearest')
    ax.set_xlabel("$n_A$",size=14)
    ax.set_ylabel("$n_C$",size=14)
    ax.set_title("Lamda max with v2 = {}".format(v2))
    fig.colorbar(scale, ax = ax)

# %%
#---------------------- Cycle ----------------------#
#---------------------------------------------------#

plot = True
search_cond = False

for N in N_range:
    eigvals_max = np.zeros((N+1, N+1))
    for v_2 in v_2_range:
        for i in np.arange(N+1):
            for j in np.arange(N+1-i):
                assert(i+j <= N)
                J = build_J(i/N,1-i/N-j/N,j/N,v_2)
                eigvals_max[i,j] = get_max_eig(J)
        if plot:
            do_plot(eigvals_max, v_2)

# %%
