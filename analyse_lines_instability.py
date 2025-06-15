import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
from ysopy.utils import config_read_bare
import time
import line_ratios as line_r
st = time.time()
wave_stable = np.load("../instability_dir/wavelength_unchanged.npy")
flux_stable = np.load("../instability_dir/flux_unchanged.npy")
config = config_read_bare("ysopy/config_file.cfg")

m_dot_changed = 1e-4 * const.M_sun.value / 31557600.0
r_in_units_of_r_sun = np.array([10, 15, 20, 25])
r_change = r_in_units_of_r_sun * const.R_sun.value
flux_arr = np.zeros((len(r_in_units_of_r_sun), len(wave_stable)))
for i in range(len(r_in_units_of_r_sun)):
    flux_arr[i] = np.load(f"../instability_dir/flux_{r_in_units_of_r_sun[i]}.npy")
et = time.time()

# plt.plot(wave_stable, flux_stable, label="stable")
# for i in range(len(r_in_units_of_r_sun)):
#     plt.plot(wave_stable, flux_arr[i], label=f"r={r_in_units_of_r_sun[i]}")
# plt.legend()
# plt.show()

##### Manipulation in flux
window = [8180, 8190]
window = [6437, 6445]
wave_trimmed, flux_stable_trimmed = line_r.trim_in_window(wavelength=wave_stable, total_flux=flux_stable, window=window)
theta = [0, 0, 1]
poly_y, wave, valid_interp_flux, interp_flux = line_r.calc_continuum_arr(theta=theta, spec_wavelength=wave_trimmed, flux=flux_stable_trimmed, threshold_l=0.98)
plt.plot(wave, interp_flux/poly_y, label=f"Stable flux")
plt.title("Continuum normalised")
# plt.plot(wave_trimmed, flux_stable_trimmed/np.median(flux_stable_trimmed), label="Stable")
for i in range(len(r_in_units_of_r_sun)):
    theta = [0, 0, 1]
    wave_trimmed, flux_trimmed = line_r.trim_in_window(wavelength=wave_stable, total_flux=flux_arr[i], window=window)
    poly_y, wave, valid_interp_flux, interp_flux = line_r.calc_continuum_arr(theta=theta, spec_wavelength=wave_trimmed, flux=flux_trimmed, threshold_l=0.98)
    # plt.plot(wave_trimmed, flux_trimmed/np.median(flux_trimmed), label=f"r = {r_in_units_of_r_sun[i]} $R_\odot$")
    # plt.plot(wave, poly_y, label=f"r = {r_in_units_of_r_sun[i]} conti")
    plt.plot(wave, interp_flux/poly_y, label=f"r = {r_in_units_of_r_sun[i]}'s flux")
plt.legend()
plt.show()