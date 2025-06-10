import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from pypeit.core import wave
import os
import glob
import sys

# Define base path, for archis
# path_to_valid = "/home/arch/yso/validation_files/"
path_to_valid = "/home/nius2022/observational_data/v960mon/"
# path_to_valid = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
flux_dir = os.path.join(path_to_valid, 'KOA_93088/HIRES/extracted/tbl/ccd1/flux/')
# For Marvin and Archis
file_pattern = os.path.join(flux_dir, '*.tbl.gz')
# for Gautam
# file_pattern = os.path.join(flux_dir, '*.tbl')

# Get all files
flux_files = sorted(glob.glob(file_pattern))

# First, read in all data
all_data = []
bounds = []

for file in flux_files:
    data = ascii.read(file)
    wave = np.array(data['wave'])
    flux = np.array(data['Flux'])
    err = np.array(data['Error'])

    all_data.append((wave, flux, err))
    bounds.append((wave[0], wave[-1]))
print(bounds)
# Now, stitch the data based on midpoints
wave_tot = np.array([])
flux_tot = np.array([])
err_tot = np.array([])

#store the stitch locations
stitch_locs = []

## simple stitching

for i in range(len(all_data)):
    wave, flux, err = all_data[i]

    if i == 0:
        # For first file: take from wave[0] to midpoint with next file
        next_lower = bounds[i+1][0]
        upper_limit = 0.5 * (bounds[i][1] + next_lower)
        mask = wave <= upper_limit
        stitch_locs.append(bounds[i][0])

    elif i == len(all_data) - 1:
        # For last file: take from midpoint with previous file to wave[-1]
        prev_upper = bounds[i-1][1]
        lower_limit = 0.5 * (bounds[i][0] + prev_upper)
        mask = wave > lower_limit
        stitch_locs.append(lower_limit)
        stitch_locs.append(bounds[i][1])

    else:
        # For intermediate files: take from midpoint(prev.upper, cur.lower) to midpoint(cur.upper, next.lower)
        prev_upper = bounds[i-1][1]
        next_lower = bounds[i+1][0]
        lower_limit = 0.5 * (bounds[i][0] + prev_upper)
        upper_limit = 0.5 * (bounds[i][1] + next_lower)
        mask = (wave > lower_limit) & (wave <= upper_limit)
        stitch_locs.append(lower_limit)

    wave_tot = np.append(wave_tot, wave[mask])
    flux_tot = np.append(flux_tot, flux[mask])
    err_tot = np.append(err_tot, err[mask])

# save the stitched array
data_tot = np.array([wave_tot,flux_tot,err_tot])
np.save(f"{path_to_valid}/stitched_HIRES_data_V960.npy",data_tot)
sys.exit(0)

print(f"stitch locations: {stitch_locs}")

plt.figure(figsize=(20, 5))

plt.scatter(wave_tot, flux_tot, label='Stitched Flux', s=20, alpha = 0.5, color='black')
plt.fill_between(wave_tot, flux_tot + err_tot, flux_tot-err_tot, color='gray', alpha=0.4, label='Error')

for i,file in enumerate(flux_files):
    data = ascii.read(file)
    wave = np.array(data['wave'])
    flux = np.array(data['Flux'])
    err = np.array(data['Error'])

    # plt.plot(wave,flux,label=f"{i}")

# plt.xlabel("Wavelength")
# plt.ylabel("Flux")

# plt.legend()
# plt.show()