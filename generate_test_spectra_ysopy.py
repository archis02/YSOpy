import numpy as np
from scipy.interpolate import interp1d
from ysopy import base_funcs as bf
from ysopy import utils
import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import mcmc_window as mc_file

# GD
# save_loc = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
#Marvin
save_loc = "/home/nius2022/2025_mcmc_ysopy/Buffer/spectra_save"
config = utils.config_read_bare("ysopy/config_file.cfg")
# theta = np.array([6, 4.5, 2.0, 20, 9])  # test case for high accretion rate
# theta = np.array([10, 6.5, 1.5, 25, 8, 13, 1])  # test case for low accretion rate
theta = np.array([10, 6.5, 1.5, 25, 8, 13, 1, 5])  # test case for Balmer jump thing
# theta = np.array([6, 4.5, 2.0, 20, 9, 0, 1])
# generate spectra with ysopy =================
# """
wave, total_flux, flux_photon_count = mc_file.model_spec_window(theta, config)
# change the lower line to total_flux / flux_photon_count based on what do you want to save
flux = flux_photon_count
# plt.plot(wave, flux)
# plt.show()
# exit(0)
np.save(f"{save_loc}/true_flux_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy", flux)
np.save(f"{save_loc}/true_wave_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy", wave)
# """
# =================
obs_flux = np.load(f"{save_loc}/true_flux_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy")
# wavelength array for observed spectra
obs_wave = np.load(f"{save_loc}/true_wave_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy")

# trimming
# window = [4980, 5066.0]
# window = [8000, 8400]  # for low mdot , paschen jump
window = [3600, 3700]  # for low mdot , Balmer jump
wave_trimmed = np.where(obs_wave > window[0], obs_wave, 0)
flux_trimmed = np.where(obs_wave > window[0], obs_flux, 0)
flux_trimmed = np.where(obs_wave < window[1], flux_trimmed, 0)
wave_trimmed = np.where(obs_wave < window[1], wave_trimmed, 0)

flux_trimmed = np.trim_zeros(flux_trimmed, "fb")
wave_trimmed = np.trim_zeros(wave_trimmed, "fb")

obs_wave = wave_trimmed
obs_flux = flux_trimmed

snr = 100
# ############## Gaussian Noise
# noise_in_signal = obs_flux * np.random.randn(len(obs_flux)) / snr
# # noise = obs_flux * np.random.normal(0, 1/snr, len(obs_flux))
# noisy_flux = obs_flux + noise_in_signal
noise_assumed = obs_flux * np.ones(len(obs_flux)) / snr
# np.save(f"{save_loc}/trimmed_wavem_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy", wave_trimmed)
# np.save(f"{save_loc}/snr_{snr}_obs_flux_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy", noisy_flux)
# np.save(f"{save_loc}/snr_{snr}_noise_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy", noise_assumed)
# plt.plot(obs_wave, obs_flux/np.median(obs_flux))
# plt.plot(obs_wave, noisy_flux/np.median(noisy_flux))
# plt.show()
# ##################
### If using poisson noise thing
def generate_poisson_noise_for_flux(snr, flux_arr):
    flux_arr *= snr**2 / np.mean(flux_arr)  # normalise flux_arr to 1 and then scale it up by snr^2
    poisson_noisy_flux = np.random.poisson(lam=flux_arr).astype(np.float64)  # lambda = sigma2 for poisson -->
    sigma_arr = np.sqrt(flux_arr)
    return poisson_noisy_flux, sigma_arr, flux_arr

poisson_noise_flux, sigma_arr, flux_arr = generate_poisson_noise_for_flux(snr, obs_flux)

# plt.plot(obs_wave, flux_arr)#/np.median(flux_arr))
# plt.plot(obs_wave, poisson_noise_flux)#/np.median(poisson_noise_flux))
# plt.fill_between(obs_wave, flux_arr + np.sqrt(flux_arr),  flux_arr - np.sqrt(flux_arr), alpha=0.2)
# plt.show()
np.save(f"{save_loc}/trimmed_wavem_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy", wave_trimmed)
np.save(f"{save_loc}/snr_{snr}_obs_flux_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy", poisson_noise_flux)
np.save(f"{save_loc}/snr_{snr}_noise_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy",sigma_arr)
#

