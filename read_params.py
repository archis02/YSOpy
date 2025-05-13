import numpy as np
import corner
import matplotlib.pyplot as plt
from pypeit.core import wave
import astropy.units as u
from astropy.io import ascii
from mcmc import total_spec
import sys
import warnings

warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

arr = np.load("04112024_v960_1000it.npy")
n_params = 5
n_walkers = 10
steps = 1000
x = np.zeros(shape=(steps*n_walkers,n_params))
for i in range(200):
    for j in range(n_walkers):
        x[i*n_walkers+j] = arr[i,j,:]

print(x)
# print(x)
# print("shape",arr.shape)
pars = []
for i in range(n_params):
    pars.append(np.median(x[:,i]))
print(pars)

#read the data
# path_to_valid = "../../../../validation_files/"
# data = ascii.read(path_to_valid+'HIRES_sci_42767_1/KOA_42767/HIRES/extracted/tbl/ccd0/flux/HI.20030211.26428_0_02_flux.tbl.gz')

#HBC 722
# path_to_valid = "../../../validation_files/"
# data = ascii.read(path_to_valid+'KOA_93088/HIRES/extracted/tbl/ccd1/flux/HI.20141209.56999_1_04_flux.tbl.gz')

# data = [data['wave'],data['Flux']/np.median(data['Flux']),data['Error']/np.median(data['Flux'])]
# #vac to air correction for given data
# wavelengths_air = wave.vactoair(data[0]*u.AA)
# data[0] = wavelengths_air
# # best_fit = total_spec(pars,data[0]*u.AA)
# print("done")

# fig = plt.figure(figsize=(16,10))
# ax = fig.add_subplot(111)
# ax.plot(data[0],data[1],label="observed")
# ax.plot(data[0],best_fit,label="model")
# plt.legend()
# plt.show()

# sys.exit(0)
# figure = corner.corner(
#     x,
#     labels=[
#         r'$M$',
#         r'log $\dot M$',
#         r'$B$', r'$\theta$',
#         r'log $n_e$',
#         r'$R_{\ast}$',
#         r'$T_o$',
#         r'$T_{slab}$',
#         r'$\tau$'
#     ],
#     quantiles=[0.16,0.5,0.84],
#     show_titles=True,
#     smooth=0.5
# )

figure = corner.corner(
    x,
    labels=[
        r'$M$',
        r'log $\dot M$',
        r'b',
        r'$\theta$',
        r'$T_o$',
    ],
    quantiles=[0.16,0.5,0.84],
    range=[[0.1,1.0],[-6,-4],[0.9,1.1],[0.1,90],[3600,4400]],
    show_titles=True,
    smooth=0.5
)

plt.show()
#print(arr)