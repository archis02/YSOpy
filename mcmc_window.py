import numpy as np
from scipy.interpolate import interp1d
from ysopy import base_funcs as bf
from ysopy import utils
import astropy.constants as const
import astropy.units as u
from astropy.io import ascii
from pypeit.core import wave
import emcee
from configparser import ConfigParser
import matplotlib.pyplot as plt
import time
from multiprocessing import cpu_count

from multiprocessing import Pool
import os
import logging
import sys

# logger = logging.getLogger(__name__)
# logging.basicConfig(filename='mcmc.log', encoding='utf-8', level=logging.DEBUG)

def config_reader(filepath):
    """
    Read the config file containing the bounds for each parameter, i.e. mcmc_config.cfg
    """
    parser = ConfigParser()
    parser.read(filepath)
    config_data = dict(parser['Parameters'])

    config_data['m_u'] = float(config_data['m_u'])
    config_data['m_l'] = float(config_data['m_l'])

    config_data['log_m_dot_u'] = float(config_data['log_m_dot_u'])
    config_data['log_m_dot_l'] = float(config_data['log_m_dot_l'])

    config_data['b_u'] = float(parser['Parameters']['b_u'])
    config_data['b_l'] = float(parser['Parameters']['b_l'])
    
    config_data['cos_inclination_u'] = float(parser['Parameters']['cos_inclination_u'])
    config_data['cos_inclination_l'] = float(parser['Parameters']['cos_inclination_l'])
    
    config_data['t_0_u'] = float(parser['Parameters']['t_0_u'])
    config_data['t_0_l'] = float(parser['Parameters']['t_0_l'])
    
    config_data['t_slab_u'] = float(parser['Parameters']['t_slab_u'])
    config_data['t_slab_l'] = float(parser['Parameters']['t_slab_l'])
    
    config_data['log_n_e_u'] = float(parser['Parameters']['log_n_e_u'])
    config_data['log_n_e_l'] = float(parser['Parameters']['log_n_e_l'])
    
    config_data['tau_u'] = float(parser['Parameters']['tau_u'])
    config_data['tau_l'] = float(parser['Parameters']['tau_l'])

    config_data['const_term_l'] = float(parser['Parameters']['const_term_l'])
    config_data['const_term_u'] = float(parser['Parameters']['const_term_u'])

    config_data['other_coeff_l'] = float(parser['Parameters']['other_coeff_l'])
    config_data['other_coeff_u'] = float(parser['Parameters']['other_coeff_u'])

    config_data['av_l'] = float(parser['Parameters']['av_l'])
    config_data['av_u'] = float(parser['Parameters']['av_u'])

    return config_data

def generate_initial_conditions(params_label, config_data,n_windows,poly_order,n_walkers, n_params):
    '''
    Generates initial conditions by drawing samples from a uniform distribution.
    A flat continuum is chosen'''

    np.random.seed(123456)
    
    # params = ['m', 'log_m_dot', 'b', 'cos_inclination', 't_slab', "log_n_e", "tau", "av"]
    initial_conditions = np.zeros((n_walkers, n_params + n_windows*(poly_order+1)))
    for i, param in enumerate(params_label):
        low = config_data[param + '_l']
        high = config_data[param + '_u']

        # uniform distribution
        initial_conditions[:,i] = np.random.uniform(low,high,size=n_walkers)

    # add initial conditions for the continuum
    for j in range(n_windows):
        for k in range(poly_order):
            initial_conditions[:, n_params + j * (poly_order + 1) + k] = np.random.uniform(
                config_data['other_coeff_l'], config_data['other_coeff_u'], size=n_walkers)
        initial_conditions[:, n_params + j * (poly_order + 1) + poly_order] = np.random.uniform(
            config_data['const_term_l'], config_data['const_term_u'], size=n_walkers)
    return initial_conditions

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
    config['inclination'] = np.arccos(theta[3]/10)# * np.pi / 180.0  # radians
    # config["t_0"] = theta[4] * 1000
    # config['t_slab'] = theta[4] * 1000.0 * u.K
    # config["n_e"] = 10**theta[5] * u.cm**(-3)
    # config["tau"] = theta[6]
    # config["av"] = theta[7]
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
    func_temp = interp1d(m, temp_arr, bounds_error=False, fill_value="extrapolate")
    func_rad = interp1d(m, rad_arr, bounds_error=False, fill_value="extrapolate")

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
    flux_photon = (total_flux * 1e-7) * (wave_ax * 1e-10) / (const.h.value * const.c.value)
    # print(f"model run ... "
    #       f"\nvisc disk : {t1-t0:.2f}"
    #       f"\nshock: {t2-t1:.2f}"
    #       f"\ndusty disk: {t3-t2:.2f}"
    #       f"\nphotosphere: {t4-t3:.2f}")

    return wave_ax, total_flux, flux_photon

def convert_to_photon_counts(wave_ax, total_flux):
    flux_photo = (total_flux * 1e-7) * (wave_ax * 1e-10) / (const.h.value * const.c.value)
    return flux_photo

def log_prior(theta, config, params_label, config_mcmc):
    """
    Vectorized log prior using bounds from config_mcmc:
    - Model parameters: uniform priors from config_mcmc.
    - Continuum coefficients:
        - Constant term (x^0): bounds from config_mcmc['const_term_l'], ['const_term_u']
        - Other terms: bounds from config_mcmc['other_coeff_l'], ['other_coeff_u']
    """
    n_windows = len(config['windows'])
    poly_order = config['poly_order']
    # params = ['m', 'log_m_dot', 'b', 'cos_inclination', 't_slab', "log_n_e", "tau", "av"]
    params = params_label
    n_model_params = len(theta) - n_windows * (poly_order + 1)

    theta_model = theta[:n_model_params]
    theta_continua = theta[n_model_params:]

    # === Model parameters prior check ===
    lower_bounds = np.array([config_mcmc[p + '_l'] for p in params])
    upper_bounds = np.array([config_mcmc[p + '_u'] for p in params])
    if not np.all((theta_model > lower_bounds) & (theta_model < upper_bounds)):
        return -np.inf

    # === Continuum polynomial prior check ===
    const_l = config_mcmc['const_term_l']
    const_u = config_mcmc['const_term_u']
    other_l = config_mcmc['other_coeff_l']
    other_u = config_mcmc['other_coeff_u']

    coeffs = theta_continua.reshape(n_windows, poly_order + 1)
    constant_terms = coeffs[:, -1]
    other_terms = coeffs[:,:-1]

    if not np.all((constant_terms > const_l) & (constant_terms < const_u)):
        return -np.inf
    if not np.all((other_terms > other_l) & (other_terms < other_u)):
        return -np.inf

    return 0.0


def log_likelihood_window(theta, config, x_obs, y_obs, yerr):
    """
    Compute log-likelihood using windowed residuals and independent continuum correction per window.
    Assumes data arrays x_obs (wavelengths), y_obs (normalized flux), yerr (errors), 
    and a model_spec_window function that returns (wavelength, model_flux).
    """
    poly_order = config['poly_order']
    windows = config['windows']  # list of (lower, upper) wavelength bounds for each window
    n_windows = len(windows)
    n_model_params = len(theta) - n_windows * (poly_order + 1)
    theta_model = theta[:n_model_params]

    # Model spectrum (full)
    wave_model, total_flux, flux_photon_count = model_spec_window(theta_model, config)
    # change the below code from total_flux to photon_counts if obs_spectra is in photon counts
    model_flux = total_flux

    log_like = 0.0

    # loop over the windows
    for i, window in enumerate(windows):
        # Get the polynomial coefficients for this window
        poly_coeffs = theta[n_model_params + i * (poly_order + 1) : n_model_params + (i + 1) * (poly_order + 1)]

        # Get indices for this window
        l_idx = np.searchsorted(x_obs, window[0])
        u_idx = np.searchsorted(x_obs, window[1])
        window_obs = x_obs[l_idx:u_idx]
        flux_obs_window = y_obs[l_idx:u_idx]
        err_window = yerr[l_idx:u_idx]

        # Normalise within the window
        err_window /= np.median(flux_obs_window)
        flux_obs_window /= np.median(flux_obs_window)

        # Interpolate model to the observed window wavelengths
        model_flux_window = np.interp(window_obs, wave_model, model_flux)
        model_flux_window = model_flux_window/np.median(model_flux_window)
        # scaling the wavelength to remove degeneracy of slope and intercept
        scaled_wave = np.linspace(-1, 1, len(window_obs))
        # Apply the polynomial continuum correction
        poly_func = np.polyval(poly_coeffs, scaled_wave)
        model_corrected = model_flux_window * poly_func

        # Compute log-likelihood contribution from this window
        sigma2 = err_window ** 2
        log_like += -0.5 * np.sum((flux_obs_window - model_corrected) ** 2 / sigma2 + np.log(sigma2))

    return log_like

def log_probability_window(theta, params_label, config, config_mcmc, xobs, yobs, yerr): # gives the posterior probability

    lp = log_prior(theta,config=config,config_mcmc=config_mcmc, params_label=params_label)

    if not np.isfinite(lp): # if not finite, then probability is 0
        return -np.inf
    
    return lp + log_likelihood_window(theta,config, xobs, yobs, yerr)

def rad_vel_correction(wave_ax, vel):
    """
    Apply correction to wavelength due to the doppler shift"""
    del_wav = (vel/const.c) * wave_ax
    return wave_ax - del_wav

def main(p0, params_label, n_dim, n_walkers, n_iter, cpu_cores_used, save_filename, config_dict, config_data_mcmc, x_obs, y_obs, yerr):
    '''
    Sets the MCMC running, parallelized by multiprocessing'''

    print("Model running...")
    start = time.time()

    backend = emcee.backends.HDFBackend(save_filename)
    backend.reset(n_walkers,n_dim)
    with Pool(processes=cpu_cores_used) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability_window, args=(params_label, config_dict, config_data_mcmc, x_obs, y_obs, yerr), backend=backend, pool=pool, blobs_dtype=float)
        sampler.run_mcmc(p0, n_iter, progress=True)
    
    end = time.time()
    multi_time = end - start
    print(f"Multiprocessing took {multi_time:.2f} seconds")
    
    # get the chain
    print("getting chain ... ")
    params = sampler.get_chain()

    return params


def resume_sampling(params_label, backend_filename, niter_more, config_dict, config_data_mcmc, x_obs, y_obs, yerr, cpu_cores_used):
    start = time.time()
    # Load the existing backend
    backend = emcee.backends.HDFBackend(backend_filename)

    # Get number of walkers and ndim from existing chain
    print(backend.shape)
    nwalkers, ndim = backend.shape
    # Get last position
    last_pos = backend.get_chain()[-1]

    print(f"Starting {nwalkers} walkers and {ndim} dimensions")
    print(f"Will do {niter_more} iterations")

    # Resume sampling
    with Pool(processes=cpu_cores_used) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_window,
                                        args=(params_label, config_dict, config_data_mcmc, x_obs, y_obs, yerr), backend=backend,
                                        pool=pool, blobs_dtype=float)
        sampler.run_mcmc(last_pos, niter_more, progress=True)
    end = time.time()
    multi_time = end - start
    print("Finished resuming.")
    print(f"Multiprocessing took {multi_time:.2f} seconds")

    # get the chain
    print("getting chain ... ")
    params = sampler.get_chain()

    return params

##############################################
# This code is to run MCMC for the first time
"""
if __name__=="__main__":
    cpu_cores_used = cpu_count()
    # read data for Marvin
    # path_to_valid = "../../FU_ori_HIRES/"
    # data = ascii.read(path_to_valid+'KOA_42767/HIRES/extracted/tbl/ccd0/flux/HI.20030211.26428_0_02_flux.tbl.gz')

    #read the data, V960 Mon
    # path_to_valid = "../../../validation_files/"
    # path_to_valid = "/home/nius2022/observational_data/v960mon/"
    path_to_valid = "/Users/tusharkantidas/NIUS/ysopy_valid/"
    data = ascii.read(path_to_valid+'KOA_93088/HIRES/extracted/tbl/ccd1/flux/HI.20141209.56999_1_04_flux.tbl.gz')
    data = [data['wave'],data['Flux']/np.median(data['Flux']),data['Error']/np.median(data['Flux'])]

    # radial velocity correction, taken from header
    data[0] = rad_vel_correction(data[0]*u.AA, 40.3 * u.km / u.s)

    x_obs = data[0].value
    y_obs = data[1]
    yerr = data[2]

    # filename where the chain will be stored
    save_filename = 'mcmc_total_spec.h5'

    n_params = 5 # number of parameters that are varying
    n_walkers = 70
    n_iter = 500

    # generate initial conditions
    config_data_mcmc = config_reader('mcmc_config.cfg')
    config_dict = utils.config_read_bare("ysopy/config_file.cfg")
    n_windows = len(config_dict['windows'])
    poly_order = config_dict['poly_order']
    p0 = generate_initial_conditions(config_data_mcmc, n_windows=n_windows, poly_order=poly_order, n_walkers=n_walkers)
    n_dim = n_params + n_windows * (poly_order + 1)

    # MAIN
    params = main(p0, n_dim, n_walkers, n_iter, cpu_cores_used, config_dict, config_data_mcmc, x_obs, y_obs, yerr)

    np.save(f"trial1_v960_steps_{n_iter}_walkers_{n_walkers}.npy",params)

    print("completed")
##############################################
"""

##############################################
#  This block is to restart sampling from a pre calculated chain
"""
if __name__ == "__main__":
    cores = cpu_count()
    # read data for Marvin
    # path_to_valid = "../../FU_ori_HIRES/"
    # data = ascii.read(path_to_valid+'KOA_42767/HIRES/extracted/tbl/ccd0/flux/HI.20030211.26428_0_02_flux.tbl.gz')

    # read the data, V960 Mon
    # path_to_valid = "../../../validation_files/"
    # path_to_valid = "/home/nius2022/observational_data/v960mon/"
    path_to_valid = "/Users/tusharkantidas/NIUS/ysopy_valid/"
    data = ascii.read(path_to_valid + 'KOA_93088/HIRES/extracted/tbl/ccd1/flux/HI.20141209.56999_1_04_flux.tbl')
    data = [data['wave'], data['Flux'] / np.median(data['Flux']), data['Error'] / np.median(data['Flux'])]

    # radial velocity correction, taken from header
    data[0] = rad_vel_correction(data[0] * u.AA, 40.3 * u.km / u.s)

    x_obs = data[0].value
    y_obs = data[1]
    yerr = data[2]

    # filename where the chain will be stored
    save_filename = 'mcmc_total_spec.h5'

    n_iter_more = 20 # Define the extra number of iterations to be done

    # generate initial conditions
    config_data_mcmc = config_reader('mcmc_config.cfg')
    config_dict = utils.config_read_bare("ysopy/config_file.cfg")

    # print(log_likelihood_window(p0, config_dict))
    params = resume_sampling(save_filename, n_iter_more, config_dict, config_data_mcmc, x_obs, y_obs, yerr, cpu_cores_used=cores)

    # MAIN
    # params = main(p0, n_dim, n_walkers,config_dict,config_data_mcmc, x_obs, y_obs, yerr)
    # np.save(f"trial1_v960_steps_{n_iter}_walkers_{n_walkers}.npy",params)

    print("completed")
"""