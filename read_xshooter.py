import numpy as np
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.constants as const

# Gautam
# hdu = fits.open('/Users/tusharkantidas/github/tifr_2025/xshooter_spectra/archive_data/ADP.2023-04-05T09:17:48.818.fits')
# hdu = fits.open('/Users/tusharkantidas/github/tifr_2025/xshooter_spectra/xshooter_ex_lupi_1/ADP.2014-05-15T15:38:26.910.fits')
# Marvin
# hdu = fits.open("/home/nius2022/observational_data/ex_lupi/ADP.2023-04-05T09:17:48.818.fits")
hdu = fits.open("/home/nius2022/observational_data/ex_lupi/ADP.2014-05-15T15:38:26.910.fits")

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
# plt.plot(wave_arr, flux_arr, "k")
# plt.show()
# exit(0)
####### Convolution
window_size = 21
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
# for i in range(9):
#     non_emission_smooth = np.convolve(non_emission_smooth, np.ones(window_size)/window_size, 'same')
# plt.plot(wave_trim, non_emission_smooth/np.median(non_emission_smooth), color='b', label="Smoothed")
# plt.fill_between(wave_trim, non_emission_smooth - err_trim, non_emission_smooth + err_trim, color='b', alpha=0.5)
# plt.legend()
# plt.show()
# exit(0)
# plt.plot(wave_trimmed, error_trimmed, 'r', label="Error Unfiltered")
# plt.plot(wave_trimmed, error_smooth_trimmed, "k", label="Error Smoothed")
# plt.legend()
# plt.show()
# exit(0)
#
# plt.plot(wave_arr, flux_arr, label='photon flux')
# plt.fill_between(wave_arr, flux_arr - flx_error_arr, flux_arr+flx_error_arr, alpha=0.2)
# plt.legend()
# plt.show()
# error_smooth_trimmed /= np.median(flux_smooth_trimmed)
# flux_smooth_trimmed /= np.median(flux_smooth_trimmed)

from scipy.signal import medfilt, savgol_filter

#### Below func by ChatGPT: Handle with care!


def remove_spikes_smooth_spectrum(wavelength, flux, n_sigma=5, median_kernel=11, smooth_window=15, smooth_poly=2,
                                  do_plot=False):
    """
    Removes spikes from a spectrum using n-sigma thresholding and smooths it.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength array.
    flux : np.ndarray
        Flux (intensity) array.
    n_sigma : float
        Threshold for sigma-clipping spike removal.
    median_kernel : int
        Kernel size for median filter (should be odd).
    smooth_window : int
        Window size for Savitzky-Golay filter (should be odd).
    smooth_poly : int
        Polynomial order for Savitzky-Golay filter.
    do_plot : bool
        Whether to plot the original and cleaned spectra.

    Returns
    -------
    cleaned_flux : np.ndarray
        Flux array with spikes removed and smoothed.
    """

    # Step 1: Median-filtered version of the spectrum
    median_flux = medfilt(flux, kernel_size=median_kernel)

    # Step 2: Sigma clipping to detect spikes
    residual = flux - median_flux
    std_dev = np.std(residual)
    spikes = np.abs(residual) > n_sigma * std_dev

    # Step 3: Replace spikes with median value
    cleaned_flux = flux.copy()
    cleaned_flux[spikes] = median_flux[spikes]

    # Step 4: Optional smoothing
    cleaned_flux = savgol_filter(cleaned_flux, window_length=smooth_window, polyorder=smooth_poly)

    # Plot
    if do_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(wavelength, flux, label='Original Spectrum', alpha=0.6)
        plt.plot(wavelength, cleaned_flux, label='Cleaned + Smoothed Spectrum', linewidth=2)
        plt.legend()
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        plt.title(f'Spike Removal and Smoothing (n = {n_sigma}σ)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return cleaned_flux


non_emission_smooth = np.convolve(flux_trim, np.ones(window_size)/window_size, 'smooth')
flux_gpt = remove_spikes_smooth_spectrum(wave_trim, flux_trim, n_sigma=2, median_kernel=11, smooth_window=15)
# plt.plot(wave_trimmed, flux_trimmed/np.median(flux_trimmed), label='photon flux')
# plt.plot(wave_trim, flux_trim/np.median(flux_trim), "k", label="Filtered")
# plt.plot(wave_trim, non_emission_smooth/np.median(non_emission_smooth), color="r", label="Non-emission smooth")
# plt.plot(wave_trim, flux_gpt/np.median(flux_gpt), color="b", label="GPT")
# plt.plot(wave_trimmed, flux_smooth_trimmed, label='smooth flux')
# plt.fill_between(wave_trimmed, flux_trimmed - error_trimmed, flux_trimmed + error_trimmed, color='r', alpha=0.2)
# plt.fill_between(wave_trimmed, flux_smooth_trimmed - error_smooth_trimmed, flux_smooth_trimmed + error_smooth_trimmed, color="b", alpha=0.2)
# plt.legend()
# plt.show()

# exit(0)
# GD
# save_loc = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
#Marvin
save_loc = "/home/nius2022/2025_mcmc_ysopy/Buffer/spectra_save"

data_reduced = np.zeros((3, len(wave_trim)))
data_reduced[0] = wave_trim
data_reduced[1] = flux_gpt
data_reduced[2] = err_trim

np.save(f"{save_loc}/data_ex_lupi_smooth.npy", data_reduced)

# data_less_reduced = np.zeros((3, len(wave_trim)))
# data_less_reduced[0] = wave_trim
# data_less_reduced[1] = flux_trim
# data_less_reduced[2] = err_trim
# np.save(f"{save_loc}/data_ex_lupi_high.npy", data_less_reduced)

