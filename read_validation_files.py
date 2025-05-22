import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import ascii
import astropy.constants as const
from pypeit.core import wave
import os
import glob

def rad_vel_correction(wave_ax, vel):
    """
    Apply correction to wavelength for the doppler shift due to
    radial velocity of the star.
    """
    del_wav = (vel/const.c) * wave_ax
    return wave_ax - del_wav

#read the data, V960 Mon
path_to_valid = "../../../validation_files/"
# data = ascii.read(path_to_valid+'KOA_93088/HIRES/extracted/tbl/ccd1/flux/HI.20141209.56999_1_0_flux.tbl.gz')
# data = [data['wave'],data['Flux']/np.median(data['Flux']),data['Error']/np.median(data['Flux'])]    # median normalized
# wavelengths_air = wave.vactoair(data[0]*u.AA)   # vac to air correction for given data
# data[0] = rad_vel_correction(wavelengths_air, 40.3 * u.km / u.s)    # radial velocity correction to wavelength, from header file

# Define base path
path_to_valid = "../../../validation_files/"
flux_dir = os.path.join(path_to_valid, 'KOA_93088/HIRES/extracted/tbl/ccd1/flux/')
file_pattern = os.path.join(flux_dir, '*.tbl.gz')

# Get all files
flux_files = sorted(glob.glob(file_pattern))

for file in flux_files:
    data = ascii.read(file)
    data = [data['wave'],data['Flux']/np.median(data['Flux']),data['Error']/np.median(data['Flux'])]
    
    plt.figure(figsize=(20, 5))
    plt.plot(data[0], data[1], label='Normalized Flux', color='black')
    plt.fill_between(data[0], data[1] - data[2], data[1] + data[2], color='gray', alpha=0.4, label='Error')
    plt.xlabel("Wavelength")
    plt.ylabel("Normalized Flux")
    plt.title(f"file : {file}")
    plt.legend()
    plt.show()
