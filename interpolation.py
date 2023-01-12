#%%

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp2d

#%%

stability = "bistable"
N = 125
N_max = 20
v_2 = 3.04

axis = (np.arange(N+1))/N
axis_max = (np.arange(N_max+1))/N_max

# %%

namefile = 'tommaso/LNA_results/{}/{}_{}_LNA_v2_{:.2f}'.format(stability,N,stability,v_2)
data = np.loadtxt(namefile + '.txt')

# %%

itp = interp2d(axis, axis, data)
data_new = itp(axis_max, axis_max)

# %%

fig = plt.figure(figsize=(8,6))
plt.imshow(data_new/data_new.sum(),origin='lower',interpolation='nearest')
plt.colorbar()
#if K_2 > K_1:
#    plt.title(u"Monostable with $K_2 = {}$ and $N = {}$".format(K_2,N))
#else:
#    plt.title(u"Bistable with $K_2 = {}$ and $N = {}$".format(K_2,N))
#plt.xlim(low_xA,high_xA)
#plt.ylim(low_xB,high_xB)
plt.xlabel("$n_A$",size=14)
plt.ylabel("$n_C$",size=14)
plt.tight_layout()
plt.show()

# %%

fig = plt.figure(figsize=(8,6))
plt.imshow(data/data.sum(),origin='lower',interpolation='nearest')
plt.colorbar()
#if K_2 > K_1:
#    plt.title(u"Monostable with $K_2 = {}$ and $N = {}$".format(K_2,N))
#else:
#    plt.title(u"Bistable with $K_2 = {}$ and $N = {}$".format(K_2,N))
#plt.xlim(low_xA,high_xA)
#plt.ylim(low_xB,high_xB)
plt.xlabel("$n_A$",size=14)
plt.ylabel("$n_C$",size=14)
plt.tight_layout()
plt.show()
