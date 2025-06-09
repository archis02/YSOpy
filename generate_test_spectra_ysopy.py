import numpy as np
from scipy.interpolate import interp1d
from ysopy import base_funcs as bf
from ysopy import utils
import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt

def model_spec_window(theta, config):
    '''
    Evaluates model in the wavelength range [l_min,l_max].
    Ensure that all windows over which the chi-square is calculated lie within this range.

    Parameters
    ----------
    theta: list
        list of parameters which are varied as the least-squre fitting is done
    '''

    # config = utils.config_read_bare('ysopy/config_file.cfg')

    # overwrite the given config dictionary, after SCALING
    config['m'] = theta[0] / 10.0 * const.M_sun.value
    config['m_dot'] = 10 ** (-1.0 * theta[1]) * const.M_sun.value / 31557600.0  ## Ensure the 10** here
    config['b'] = theta[2]
    config['inclination'] = theta[3] * np.pi / 180.0  # radians
    config['t_slab'] = theta[4] * 1000.0 * u.K

    config["n_e"] = 10**theta[5] * u.cm**(-3)
    config["tau"] = theta[6]

    # get the stellar paramters from the isochrone model, Baraffe et al. 2015(?)
    m = np.array(
        [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.072, 0.075, 0.08, 0.09, 0.1, 0.11, 0.13, 0.15, 0.17, 0.2,
         0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4])
    temp_arr = np.array(
        [2345, 2504, 2598, 2710, 2779, 2824, 2864, 2897, 2896, 2897, 2898, 2908, 2936, 2955, 3012, 3078, 3142, 3226,
         3428, 3634, 3802, 3955, 4078, 4192, 4290, 4377, 4456, 4529, 4596, 4658])
    rad_arr = np.array(
        [0.271, 0.326, 0.372, 0.467, 0.536, 0.628, 0.702, 0.781, 0.803, 0.829, 0.877, 0.959, 1.002, 1.079, 1.214, 1.327,
         1.465, 1.503, 1.636, 1.753, 1.87, 1.971, 2.096, 2.2, 2.31, 2.416, 2.52, 2.612, 2.71, 2.797])
    func_temp = interp1d(m, temp_arr)
    func_rad = interp1d(m, rad_arr)

    config["t_star"] = int(func_temp(theta[0] / 10.0) / 100.0) * 100.0
    config["r_star"] = func_rad(theta[0] / 10.0) * const.R_sun.value

    # run model
    # t0 = time.time()
    bf.calculate_n_data(config)
    dr, t_max, d, r_in, r_sub = bf.generate_temp_arr(config)
    # print(config['n_data'])
    wave_ax, obs_viscous_disk_flux = bf.generate_visc_flux(config, d, t_max, dr)
    # t1 = time.time()
    obs_mag_flux = bf.magnetospheric_component_calculate(config, r_in)
    # t2 = time.time()
    obs_dust_flux = bf.generate_dusty_disk_flux(config, r_in, r_sub)
    # t3 = time.time()
    obs_star_flux = bf.generate_photosphere_flux(config)
    # t4 = time.time()
    total_flux = bf.dust_extinction_flux(config, wave_ax, obs_viscous_disk_flux, obs_star_flux, obs_mag_flux,
                                         obs_dust_flux)
    # t5 = time.time()
    # total_flux /= np.median(total_flux)

    # print(f"model run ... "
    #       f"\nvisc disk : {t1-t0:.2f}"
    #       f"\nshock: {t2-t1:.2f}"
    #       f"\ndusty disk: {t3-t2:.2f}"
    #       f"\nphotosphere: {t4-t3:.2f}")

    return wave_ax, total_flux
# GD
#save_loc = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
#Marvin
save_loc = "/home/nius2022/2025_mcmc_ysopy/Buffer/spectra_save"
config = utils.config_read_bare("ysopy/config_file.cfg")
# theta = np.array([6, 4.5, 2.0, 20, 9])  # test case for high accretion rate
theta = np.array([10, 6.5, 1.5, 25, 8, 13, 1.5])  # test case for low accretion rate

# theta = np.array([6, 4.5, 2.0, 20, 9, 0, 1])
# generate spectra with ysopy =================
# """
wave, flux = model_spec_window(theta, config)
# plt.plot(wave, flux)
# plt.show()

np.save(f"{save_loc}/true_flux_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy", flux)
np.save(f"{save_loc}/true_wave_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy", wave)
# """
# =================
obs_flux = np.load(f"{save_loc}/true_flux_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy")
# wavelength array for observed spectra
obs_wave = np.load(f"{save_loc}/true_wave_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy")

# trimming
# window = [4980, 5066.0]
window = [8225, 8375]
wave_trimmed = np.where(obs_wave > window[0], obs_wave, 0)
flux_trimmed = np.where(obs_wave > window[0], obs_flux, 0)
flux_trimmed = np.where(obs_wave < window[1], flux_trimmed, 0)
wave_trimmed = np.where(obs_wave < window[1], wave_trimmed, 0)

flux_trimmed = np.trim_zeros(flux_trimmed, "fb")
wave_trimmed = np.trim_zeros(wave_trimmed, "fb")

obs_wave = wave_trimmed
obs_flux = flux_trimmed

snr = 100
noise_in_signal = obs_flux * np.random.randn(len(obs_flux)) / snr

# noise = obs_flux * np.random.normal(0, 1/snr, len(obs_flux))
noisy_flux = obs_flux + noise_in_signal
noise_assumed = obs_flux * np.ones(len(obs_flux)) / snr

#### If using poisson noise thing
# def generate_poisson_noise_for_flux(snr, flux_arr):
#     flux_arr *= snr**2 / np.mean(flux_arr)  # normalise flux_arr to 1 and then scale it up by snr^2
#     poisson_noisy_flux = np.random.poisson(lam=flux_arr).astype(np.float64)  # lambda = sigma2 for poisson -->
#     return poisson_noisy_flux, flux_arr
np.save(f"{save_loc}/trimmed_wavem_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy", wave_trimmed)
np.save(f"{save_loc}/snr_{snr}_obs_flux_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy", noisy_flux)
np.save(f"{save_loc}/snr_{snr}_noise_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy", noise_assumed)

# plt.plot(obs_wave, obs_flux/np.median(obs_flux))
# plt.plot(obs_wave, noisy_flux/np.median(noisy_flux))
# plt.show()

