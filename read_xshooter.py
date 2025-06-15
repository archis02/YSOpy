import numpy as np
from astropy.io import fits
import astropy.units as u
# import matplotlib.pyplot as plt
import astropy.constants as const

# Gautam
# hdu = fits.open('/Users/tusharkantidas/github/tifr_2025/xshooter_spectra/archive_data/ADP.2023-04-05T09:17:48.818.fits')
# Marvin
hdu = fits.open("/home/nius2022/observational_data/ex_lupi/ADP.2023-04-05T09:17:48.818.fits")
print(hdu.info())
data = hdu[1].data
print(data.columns)

wave_arr = data['WAVE'] * u.nm
wave_arr = wave_arr.to(u.AA).value
flux_arr = data['FLUX_REDUCED'] #* u.erg * u.cm**(-2) * u.s**(-1) *u.AA**(-1)
flx_error_arr = data["ERR_REDUCED"] #* u.erg * u.cm**(-2) * u.s**(-1) *u.AA**(-1)
wave_arr = np.array(wave_arr).flatten()
flux_arr = np.array(flux_arr).flatten()
flx_error_arr = np.array(flx_error_arr).flatten()
# flux_photon = flux_arr * wave_arr / (const.h * const.c)
# err_photon = flx_error_arr * wave_arr / (const.h * const.c)
# plt.plot(wave_arr, flux_arr)
# plt.show()

####### Convolution
window_size = 51
flux_smooth = np.convolve(flux_arr, np.ones(window_size)/window_size, 'same')
error_smooth = np.convolve(flx_error_arr, np.ones(window_size)/window_size, 'same')


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


window = [3400, 4200]  # for low mdot , Balmer jump
wave_trimmed, flux_trimmed = extract_in_window(window, wave_arr, flux_arr)
wave_trimmed1, error_trimmed = extract_in_window(window, wave_arr, flx_error_arr)
wave_trimmed1, flux_smooth_trimmed = extract_in_window(window, wave_arr, flux_smooth)
wave_trimmed1, error_smooth_trimmed = extract_in_window(window, wave_arr, error_smooth)

# Windows to mask
mask_window = [[3702, 3706], [3710, 3713], [3719, 3723], [3731, 3737], [3744, 3746.5], [3747, 3752], [3757, 3760],
    [3767, 3774], [3794, 3801], [3826, 3842], [3880, 3898], [3928, 3938], [3960, 3979], [4088, 4112]]

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


# plt.plot(wave_trimmed, flux_trimmed, 'r', label="Unfiltered")
# plt.fill_between(wave_trimmed, flux_trimmed - error_trimmed, flux_trimmed + error_trimmed,color="blue", alpha=0.5)
wave_trim = wave_trimmed
flux_trim = flux_trimmed

wave_err_trim = wave_trimmed
err_trim = error_trimmed

for i in range(len(mask_window)):
    print(mask_window[i])
    wave_trim, flux_trim = mask_in_window(mask_window[i], wave_trim, flux_trim, compress=True)
    wave_err_trim, err_trim = mask_in_window(mask_window[i], wave_err_trim, err_trim, compress=True)
# plt.plot(wave_trim, flux_trim, "k", label="Filtered")
non_emission_smooth = np.convolve(flux_trim, np.ones(window_size)/window_size, 'smooth')
for i in range(9):
    non_emission_smooth = np.convolve(non_emission_smooth, np.ones(window_size)/window_size, 'same')
# plt.plot(wave_trim, non_emission_smooth, color='b', label="Smoothed")
# plt.fill_between(wave_trim, non_emission_smooth - err_trim, non_emission_smooth + err_trim, color='b', alpha=0.5)
# plt.legend()
# plt.show()
# plt.plot(wave_trimmed, error_trimmed, 'r', label="Error Unfiltered")
# plt.plot(wave_trimmed, error_smooth_trimmed, "k", label="Error Smoothed")
# plt.legend()
# plt.show()
# exit(0)

# plt.plot(wave_arr, flux_arr, label='photon flux')
# plt.fill_between(wave_arr, flux_arr - flx_error_arr, flux_arr+flx_error_arr, alpha=0.2)
# plt.legend()
# plt.show()
# error_smooth_trimmed /= np.median(flux_smooth_trimmed)
# flux_smooth_trimmed /= np.median(flux_smooth_trimmed)

# plt.plot(wave_trimmed, flux_trimmed, label='photon flux')
# plt.plot(wave_trimmed, flux_smooth_trimmed, label='smooth flux')
# plt.fill_between(wave_trimmed, flux_trimmed - error_trimmed, flux_trimmed + error_trimmed, color='r', alpha=0.2)
# plt.fill_between(wave_trimmed, flux_smooth_trimmed - error_smooth_trimmed, flux_smooth_trimmed + error_smooth_trimmed, color="b", alpha=0.2)
# plt.legend()
# plt.show()

# GD
# save_loc = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
#Marvin
save_loc = "/home/nius2022/2025_mcmc_ysopy/Buffer/spectra_save"

data_reduced = np.zeros((3, len(wave_trim)))
data_reduced[0] = wave_trim
data_reduced[1] = non_emission_smooth
data_reduced[2] = err_trim

np.save(f"{save_loc}/data_ex_lupi_smooth.npy", data_reduced)

