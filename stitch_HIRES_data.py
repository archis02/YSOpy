import numpy as np
# import matplotlib.pyplot as plt
from astropy.io import ascii
from pypeit.core import wave
import os
import glob
import sys

# Define base path, for archis
# path_to_valid = "/home/arch/yso/validation_files/"
path_to_valid = "/home/nius2022/observational_data/"
# path_to_valid = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
# flux_dir = os.path.join(path_to_valid, 'v960mon/KOA_93088/HIRES/extracted/tbl/ccd1/flux/')  # V960 Mon
flux_dir = os.path.join(path_to_valid, "v899mon/KOA_90631/HIRES/extracted/tbl/ccd1/flux")  # V899 Mon # Joe Sir's Target  27/10/15
# For Marvin and Archis
file_pattern = os.path.join(flux_dir, '*.tbl.gz')
# for Gautam
# file_pattern = os.path.join(flux_dir, '*.tbl')

# Get all files
flux_files = sorted(glob.glob(file_pattern))
# print(flux_files)
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
# np.save(f"{path_to_valid}/stitched_HIRES_data_V960.npy",data_tot)
# sys.exit(0)

print(f"stitch locations: {stitch_locs}")

# plt.figure(figsize=(20, 5))

# plt.scatter(wave_tot, flux_tot, label='Stitched Flux', s=20, alpha = 0.5, color='black')
# plt.fill_between(wave_tot, flux_tot + err_tot, flux_tot-err_tot, color='gray', alpha=0.4, label='Error')

for i,file in enumerate(flux_files):
    data = ascii.read(file)
    wave = np.array(data['wave'])
    flux = np.array(data['Flux'])
    err = np.array(data['Error'])

    # plt.plot(wave,flux,label=f"{i}")

# plt.xlabel("Wavelength")
# plt.ylabel("Flux")
#
# plt.legend()
# plt.show()

def extract_in_window(window_wave, wave, flux):
    """
    This function trims the wavelength and flux arrays
    given a window in wavelength and flux arrays. Window
    should be specified in list format as [left_li, right_lim].
    returns:
    wave_trim, flux_trim
    """
    # Extract the limits from the window [left, right]
    left_lim = window_wave[0]
    right_lim = window_wave[-1]

    # Trimming
    wave_trim = np.where(wave > left_lim, wave, 0)
    flux_trim = np.where(wave > left_lim, flux, 0)
    flux_trim = np.where(wave < right_lim, flux_trim, 0)
    wave_trim = np.where(wave < right_lim, wave_trim, 0)

    # Shorten the arrays
    wave_trim = np.trim_zeros(wave_trim, "fb")
    flux_trim = np.trim_zeros(flux_trim, "fb")
    return wave_trim, flux_trim



def mask_in_window(window_wave, wave, flux, compress=False):
    """
    This function trims the wavelength and flux arrays
    given a window in wavelength and flux arrays. Window
    should be specified in list format as [left_li, right_lim].
    returns:
    wave_trim, flux_trim
    """
    # Extract the limits from the window [left, right]
    left_lim = window_wave[0]
    right_lim = window_wave[-1]

    mask = (wave >= left_lim) & (wave <= right_lim)
    wave_trim = np.ma.masked_where(mask, wave, copy=False)
    flux_trim = np.ma.masked_where(mask, flux, copy=False)
    if compress:
        wave_trim = np.ma.compressed(wave_trim)
        flux_trim = np.ma.compressed(flux_trim)
    return wave_trim, flux_trim


# Windows to mask
mask_window = [[4850, 4870],
               [4917, 4928],
               [4990, 4998],
               [5013, 5023],
               [5050, 5070],
               [5107, 5118],
               [5270, 5274],
               [5314, 5345],
               [5392, 5411],
               [5392, 5411],
               [5875, 5906]]  # Na line


window = [4780, 6130]  # for low mdot , Balmer jump
wave_arr = data_tot[0]
flux_arr = data_tot[1]
flx_error_arr = data_tot[2]
wave_trimmed, flux_trimmed = extract_in_window(window, wave_arr, flux_arr)
wave_trimmed1, error_trimmed = extract_in_window(window, wave_arr, flx_error_arr)

wave_trim = wave_trimmed
flux_trim = flux_trimmed
wave_err_trim = wave_trimmed
err_trim = error_trimmed
# plt.plot(wave_trim, flux_trim, label="Flux")
mask_window = np.array(mask_window)
for i in range(len(mask_window)):
    print("Masked window: ", mask_window[i])
    wave_trim, flux_trim = mask_in_window(mask_window[i], wave_trim, flux_trim, compress=True)
    wave_err_trim, err_trim = mask_in_window(mask_window[i], wave_err_trim, err_trim, compress=True)
# plt.plot(wave_trim, flux_trim, "k", label="Filtered")
# plt.show()

# GD
# save_loc = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
#Marvin
save_loc = "/home/nius2022/2025_mcmc_ysopy/Buffer/spectra_save"

data_reduced = np.zeros((3, len(wave_trim)))
data_reduced[0] = wave_trim
data_reduced[1] = flux_trim
data_reduced[2] = err_trim

np.save(f"{save_loc}/data_v899_mon.npy", data_reduced)

