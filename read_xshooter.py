import numpy as np
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.constants as const

hdu = fits.open('/Users/tusharkantidas/github/tifr_2025/xshooter_spectra/archive (7)/ADP.2023-04-05T09:17:48.818.fits')
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


####### Convolution
window_size = 10
flux_smooth = np.convolve(flux_arr, np.ones(window_size)/window_size, 'same')
error_smooth = np.convolve(flx_error_arr, np.ones(window_size)/window_size, 'same')

window = [4125, 4325]  # for low mdot , Balmer jump
wave_trimmed = np.where(wave_arr > window[0], wave_arr, 0)
flux_trimmed = np.where(wave_arr > window[0], flux_arr, 0)
error_trimmed = np.where(wave_arr > window[0], flx_error_arr, 0)
flux_smooth_trimmed = np.where(wave_arr > window[0], flux_smooth, 0)
error_smooth_trimmed = np.where(wave_arr > window[0], error_smooth, 0)

flux_trimmed = np.where(wave_arr < window[1], flux_trimmed, 0)
error_trimmed = np.where(wave_arr < window[1], error_trimmed, 0)
flux_smooth_trimmed = np.where(wave_arr < window[1], flux_smooth_trimmed, 0)
error_smooth_trimmed = np.where(wave_arr < window[1], error_smooth_trimmed, 0)
wave_trimmed = np.where(wave_arr < window[1], wave_trimmed, 0)



flux_trimmed = np.trim_zeros(flux_trimmed, "fb")
wave_trimmed = np.trim_zeros(wave_trimmed, "fb")
error_trimmed = np.trim_zeros(error_trimmed, "fb")
flux_smooth_trimmed = np.trim_zeros(flux_smooth_trimmed, "fb")
error_smooth_trimmed = np.trim_zeros(error_smooth_trimmed, "fb")


# plt.plot(wave_arr, flux_arr, label='photon flux')
# plt.fill_between(wave_arr, flux_arr - flx_error_arr, flux_arr+flx_error_arr, alpha=0.2)
# plt.legend()
# plt.show()
# error_smooth_trimmed /= np.median(flux_smooth_trimmed)
# flux_smooth_trimmed /= np.median(flux_smooth_trimmed)

# plt.plot(wave_trimmed, flux_trimmed, label='photon flux')
plt.plot(wave_trimmed, flux_smooth_trimmed, label='smooth flux')
# plt.fill_between(wave_trimmed, flux_trimmed - error_trimmed, flux_trimmed + error_trimmed, color='r', alpha=0.2)
plt.fill_between(wave_trimmed, flux_smooth_trimmed - error_smooth_trimmed, flux_smooth_trimmed + error_smooth_trimmed, color="b", alpha=0.2)
plt.legend()
plt.show()

# GD
# save_loc = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
#Marvin
save_loc = "/home/nius2022/2025_mcmc_ysopy/Buffer/spectra_save"

data_reduced = np.zeros((3, len(wave_trimmed)))
data_reduced[0] = wave_trimmed
data_reduced[1] = flux_smooth_trimmed
data_reduced[2] = error_smooth_trimmed

np.save(f"f{save_loc}/data_ex_lupi.npy", data_reduced)

