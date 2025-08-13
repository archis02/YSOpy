import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from pypeit.core import wave
import os
import glob
import sys

# Define base path, for archis
# path_to_valid = "/home/arch/yso/validation_files/"
# path_to_valid = "/home/nius2022/observational_data/"
path_to_valid = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
flux_dir = os.path.join(path_to_valid, 'v960mon/KOA_93088/HIRES/extracted/tbl/ccd1/flux/')  # V960 Mon
# flux_dir = os.path.join(path_to_valid, "v899mon/KOA_90631/HIRES/extracted/tbl/ccd2/flux")  # V899 Mon # Joe Sir's Target  27/10/15
# flux_dir = os.path.join(path_to_valid, "hbc722/KOA_82942/HIRES/extracted/tbl/ccd2/flux")  # HBC 722 # 27/10/15--> Hillenbrand's obs 36frames

# For Marvin and Archis
# file_pattern = os.path.join(flux_dir, '*.tbl.gz')
# for Gautam # only v960 mon
file_pattern = os.path.join(flux_dir, '*.tbl')

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
#
# plt.scatter(wave_tot, flux_tot, label='Stitched Flux', s=20, alpha = 0.5, color='black')
# plt.fill_between(wave_tot, flux_tot + err_tot, flux_tot-err_tot, color='gray', alpha=0.4, label='Error')

for i,file in enumerate(flux_files):
    data = ascii.read(file)
    wave = np.array(data['wave'])
    flux = np.array(data['Flux'])
    err = np.array(data['Error'])

#     plt.plot(wave,flux,label=f"{i}")
#
# plt.xlabel("Wavelength")
# plt.ylabel("Flux")
#
# plt.legend()
# plt.show()
# exit(0)
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

def mask_outside_window(window_wave, wave, flux, compress=False):
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

    mask = (wave <= left_lim) | (wave >= right_lim)
    wave_trim = np.ma.masked_where(mask, wave, copy=False)
    flux_trim = np.ma.masked_where(mask, flux, copy=False)
    if compress:
        wave_trim = np.ma.compressed(wave_trim)
        flux_trim = np.ma.compressed(flux_trim)
    return wave_trim, flux_trim

# Windows to mask
# CCD1 HIRES V899
mask_window = [[4850, 4870],
               [4917, 4928],
               [4990, 4998],
               [5013, 5023],
               [5050, 5070],
               [5107, 5118],
               [5270, 5274],
               [5314, 5345],
               [5392, 5411],
               [5496, 5513],
               [5875, 5906]]  # Na line

# Valid window V960 Mon
valid_window = [[4779, 4850], [4854, 4917], [4921, 4986], [4990, 5056], [5060, 5129],
                [5133, 5204], [5208, 5281], [5285, 5361], [5365, 5442], [5446, 5527],
                [5531, 5614], [5618, 5704], [5708, 5797],
                # [5801, 5893], [5897, 5992], [5996, 6094]] # splitting first two windows into two and removing Na line
                [5801, 5844], [5846, 5888], [5899, 5935], [5937, 5991], [5996, 6094]]
# CCD2
# HBC722 CCD2
# valid_window = [[6233, 6308], [6315, 6422], [6429, 6539], [6570, 6660], [6671, 6787], [6799, 6866], [6934, 7050],
#                 [7073, 7166], [7216, 7340], [7367, 7493], [7524, 7593]]
# V899 CCD2
# valid_window = [[6367, 6423], [6440, 6540], [6600, 6660], [6683, 6767], [6803, 6867], [6943, 7033], [7073, 7143],
#                 [7216, 7292], [7392, 7494]]

# window =  # for CCD1 HIRES V960 Mon
# window = [4780, 6130]  # for CCD1 HIRES V899
# window = [6366, 7495]  # for CCD2 HIRES V899
# window = [6233, 7593]  # for CCD2 HIRES HBC722
wave_arr = data_tot[0]
flux_arr = data_tot[1]
flx_error_arr = data_tot[2]
# wave_trimmed, flux_trimmed = extract_in_window(window, wave_arr, flux_arr)
# wave_trimmed1, error_trimmed = extract_in_window(window, wave_arr, flx_error_arr)
#
# wave_trim = wave_trimmed
# flux_trim = flux_trimmed
# wave_err_trim = wave_trimmed
# err_trim = error_trimmed
# # plt.plot(wave_trim, flux_trim, "k", alpha=0.1, label="Flux")
# ##### This is for CCD1 Only
# mask_window = np.array(mask_window)
# for i in range(len(mask_window)):
#     print("Masked window: ", mask_window[i])
#     wave_trim, flux_trim = mask_in_window(mask_window[i], wave_trim, flux_trim, compress=True)
#     wave_err_trim, err_trim = mask_in_window(mask_window[i], wave_err_trim, err_trim, compress=True)
#     plt.plot(wave_trim, flux_trim)
# plt.show()
# # exit(0)
############

####### FOr CCD2 V899/HBC722
valid_window = np.array(valid_window)
# defining
wave_comb_window = np.array([0])
flux_comb_window = np.array([0])
wave_err_comb_window = np.array([0])
err_comb_window = np.array([0])

for i in range(len(valid_window)):
    print("Masked outside window: ", valid_window[i])
    wave_trim_1window, flux_trim1_1window = mask_outside_window(valid_window[i], wave_arr, flux_arr, compress=True)
    wave_err_trim_1window, err_trim_1window = mask_outside_window(valid_window[i], wave_arr, flx_error_arr, compress=True)
    wave_comb_window = np.concatenate((wave_comb_window, wave_trim_1window), axis=0)
    flux_comb_window = np.concatenate((flux_comb_window, flux_trim1_1window), axis=0)
    wave_err_comb_window = np.concatenate((wave_err_comb_window, wave_err_trim_1window), axis=0)
    err_comb_window = np.concatenate((err_comb_window, err_trim_1window), axis=0)
wave_comb_window = wave_comb_window[1:]
flux_comb_window = flux_comb_window[1:]
wave_err_comb_window = wave_err_comb_window[1:]
err_comb_window = err_comb_window[1:]
# plt.plot(wave_comb_window, flux_comb_window)
# plt.legend()
# plt.show()
# exit(0)

# GD
# save_loc = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
#Marvin
save_loc = "/home/nius2022/2025_mcmc_ysopy/Buffer/spectra_save"

# # Saving for CCD1 V899
# ################
# data_reduced = np.zeros((3, len(wave_trim)))
# data_reduced[0] = wave_trim
# data_reduced[1] = flux_trim
# data_reduced[2] = err_trim
# plt.plot(data_reduced[0], data_reduced[1], alpha=0.4, label="Filtered")
# plt.legend()
# plt.show()
#
# # np.save(f"{save_loc}/data_v960_mon_ccd1.npy", data_reduced)
# # np.save(f"{save_loc}/data_v899_mon_ccd1.npy", data_reduced)
# exit(0)
# ################

# Saving for CCD2 V899
################
data_reduced = np.zeros((3, len(wave_comb_window)))
data_reduced[0] = wave_comb_window
data_reduced[1] = flux_comb_window
data_reduced[2] = err_comb_window
# np.save(f"{save_loc}/data_v899_mon_ccd2.npy", data_reduced)
np.save(f"{save_loc}/data_v960mon_round2.npy", data_reduced)
################


