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


from multiprocessing import Pool
import os
import logging

os.environ["OMP_NUM_THREADS"] = "1"
logger = logging.getLogger(__name__)
logging.basicConfig(filename='mcmc.log', encoding='utf-8', level=logging.DEBUG)
np.seterr(all="ignore") # IGNORE numpy warnings

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
    
    config_data['inclination_u'] = float(parser['Parameters']['inclination_u'])
    config_data['inclination_l'] = float(parser['Parameters']['inclination_l'])
    
    config_data['t_0_u'] = float(parser['Parameters']['t_0_u'])
    config_data['t_0_l'] = float(parser['Parameters']['t_0_l'])
    
    # config_data['t_slab_u'] = float(parser['Parameters']['t_slab_u'])
    # config_data['t_slab_l'] = float(parser['Parameters']['t_slab_l'])
    
    # config_data['log_n_e_u'] = float(parser['Parameters']['log_n_e_u'])
    # config_data['log_n_e_l'] = float(parser['Parameters']['log_n_e_l'])
    
    # config_data['tau_u'] = float(parser['Parameters']['tau_u'])
    # config_data['tau_l'] = float(parser['Parameters']['tau_l'])

    return config_data


def generate_initial_conditions(config_data,n_walkers):
    '''
    Generates initial conditions by drawing samples from a known distribution in the parameter space.
    If a gaussian is used, the points outside the limits mentioned in the config file must be manually excluded.'''

    np.random.seed(123456)
    
    # params = ['m', 'log_m_dot', 'b', 'inclination',  'log_n_e', 'r_star', 't_0', 't_slab', 'tau']
    params = ['m', 'log_m_dot', 'b', 'inclination', 't_0']
    initial_conditions = np.zeros((n_walkers, n_params))

    for i, param in enumerate(params):
        low = config_data[param + '_l']
        high = config_data[param + '_u']

        # Gaussian: this will generate the initial condition close to middle of the interval
        # initial_conditions[:, i] = np.random.normal(loc = 0.5*(low+high), scale = (high-low)/5, size=n_walkers)
        
        # uniform distribution
        initial_conditions[:,i] = np.random.uniform(low,high,size=n_walkers)

    return initial_conditions


def total_spec(theta,wavelength):
    """
    Generates the model spectra by running ysopy for the given parameters in theta array
    theta is the parameter array
    returns normalized flux evaluated at the passed wavelength array
    """

    t0 = time.time()
    # modify config file, to run model
    # params = ['m', 'log_m_dot', 'b', 'inclination',  'log_n_e', 'r_star', 't_0', 't_slab', 'tau']
    # config = bf.config_read('config_file.cfg')
    # config['m'] = theta[0] * const.M_sun
    # config['m_dot'] = 10**theta[1] * const.M_sun / (1 * u.year).to(u.s) ## Ensure the 10** here
    # config['b'] = theta[2] * u.kilogauss
    # config['inclination'] = theta[3] * u.degree
    # config['n_e'] = 10**theta[4] * u.cm**-3  ## Ensure the 10** here
    # config['r_star'] = theta[5] * const.R_sun
    # config['t_0'] = theta[6] * u.K
    # config['t_slab'] = theta[7] * u.K
    # config['tau'] = theta[8]

    # less params
    # params = ['m', 'log_m_dot', 'b', 'inclination', 't_0']
    config = utils.config_read('ysopy/config_file.cfg')
    config['m'] = theta[0] * const.M_sun
    config['m_dot'] = 10**theta[1] * const.M_sun / (1 * u.year).to(u.s) ## Ensure the 10** here
    config['b'] = theta[2] * u.kilogauss
    config['inclination'] = theta[3] * u.degree
    config['t_0'] = theta[4] * u.K

    # config['n_e'] = 10**theta[4] * u.cm**-3  ## Ensure the 10** here
    # config['t_slab'] = theta[7] * u.K
    # config['tau'] = theta[8]

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

    config["t_star"] = int(func_temp(config["m"]/const.M_sun)/100) * 100 * u.K
    config["r_star"] = func_rad(config["m"]/const.M_sun) * const.R_sun
    
    # print(f"Using T_photo : {config['t_star']}\nUsing R_star : {config['r_star']/const.M_sun}")
    
    #run model
    dr, t_max, d, r_in, r_sub = bf.generate_temp_arr(config)
    wave_ax, obs_viscous_disk_flux = bf.generate_visc_flux(config, d, t_max, dr)
    t1 = time.time()
    obs_mag_flux = bf.magnetospheric_component_calculate(config, r_in)
    t2 = time.time()
    obs_dust_flux = bf.generate_dusty_disk_flux(config, r_in, r_sub)
    t3 = time.time()
    obs_star_flux = bf.generate_photosphere_flux(config)
    t4 = time.time()
    total_flux = bf.dust_extinction_flux(config, wave_ax, obs_viscous_disk_flux, obs_star_flux, obs_mag_flux, obs_dust_flux)
    t5 = time.time()

    #interpolate to required wavelength
    func = interp1d(wave_ax,total_flux)## CHECK if this works, for units
    result_spec = func(wavelength)
    result_spec /= np.median(result_spec)

    logger.info(f"params {theta}")
    logger.info(f"visc disk time : {t1-t0}")
    logger.info(f"magnetosphere time : {t2-t1}")
    logger.info(f"dust disk time : {t3-t2}")
    logger.info(f"photosphere time : {t4-t3}")
    logger.info(f"model run .. time taken {t5 - t0} s,\n params {str(theta)}")

    #print(f"model run ... time taken {t5 - t0} s")

    return result_spec

def model_spec_window(theta,config):
    '''Evaluates model in the wavelength range [l_min,l_max].
    Ensure that all windows over which the chi-square is calculated lie within this range.
    
    Parameters
    ----------
    theta: list
        list of parameters which are varied as the least-squre fitting is done
    '''

    # config = utils.config_read_bare('ysopy/config_file.cfg')
    # overwrite the given config dictionary
    config['m'] = theta[0] * const.M_sun.value
    config['m_dot'] = 10**theta[1] * const.M_sun.value / 31557600.0 ## Ensure the 10** here
    # config['b'] = theta[2]
    config['inclination'] = theta[2] * np.pi / 180.0 # radians
    # config['t_0'] = theta[4] *1000.0
    config['t_slab'] = theta[3] *1000.0 * u.K

    # get the stellar paramters from the isochrone model, Baraffe et al. 2015(?)
    m = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.072, 0.075, 0.08, 0.09, 0.1, 0.11, 0.13, 0.15, 0.17, 0.2,
         0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4])
    temp_arr = np.array([2345, 2504, 2598, 2710, 2779, 2824, 2864, 2897, 2896, 2897, 2898, 2908, 2936, 2955, 3012, 3078, 3142, 3226,
         3428, 3634, 3802, 3955, 4078, 4192, 4290, 4377, 4456, 4529, 4596, 4658])
    rad_arr = np.array([0.271, 0.326, 0.372, 0.467, 0.536, 0.628, 0.702, 0.781, 0.803, 0.829, 0.877, 0.959, 1.002, 1.079, 1.214, 1.327,
         1.465, 1.503, 1.636, 1.753, 1.87, 1.971, 2.096, 2.2, 2.31, 2.416, 2.52, 2.612, 2.71, 2.797])
    func_temp = interp1d(m, temp_arr)
    func_rad = interp1d(m, rad_arr)

    config["t_star"] = int(func_temp(theta[0])/100.0) * 100.0
    config["r_star"] = func_rad(theta[0]) * const.R_sun.value

    #run model
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
    total_flux = bf.dust_extinction_flux(config, wave_ax, obs_viscous_disk_flux, obs_star_flux, obs_mag_flux, obs_dust_flux)
    # t5 = time.time()
    total_flux /= np.median(total_flux)

    # print(f"model run ... "
    #       f"\nvisc disk : {t1-t0:.2f}"
    #       f"\nshock: {t2-t1:.2f}"
    #       f"\ndusty disk: {t3-t2:.2f}"
    #       f"\nphotosphere: {t4-t3:.2f}")

    return wave_ax, total_flux


def log_prior(theta):
    """
    Define uniform priors, this can even be skipped
    """
    config_data = config_reader('mcmc_config.cfg')
    # params = ['m', 'log_m_dot', 'b', 'inclination',  'log_n_e', 'r_star', 't_0', 't_slab', 'tau']
    params = ['m', 'log_m_dot', 'b', 'inclination', 't_0']
    condition = True

    for i, param in enumerate(params):
        low = config_data[param + '_l']
        high = config_data[param + '_u']
        condition = condition and (low < theta[i] < high)

    if condition:
        return 0.0
    return -np.inf


def log_likelihood(theta):
    #y is of the form (wavelength,normalized_flux,normalized_err), where normalization is by the median flux
    wavelength = data[0]*u.AA
    model = total_spec(theta,wavelength)
    sigma2 = data[2]**2
    return -0.5 *( np.sum((data[1] - model) ** 2 / sigma2 + np.log(sigma2)) )

def log_likelihood_window(theta, config):
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
    wave_model, model_flux = model_spec_window(theta_model)

    log_like = 0.0
    for i, window in enumerate(windows):
        # Get the polynomial coefficients for this window
        poly_coeffs = theta[n_model_params + i * (poly_order + 1) : n_model_params + (i + 1) * (poly_order + 1)]

        # Get indices for this window
        l_idx = np.searchsorted(x_obs, window[0])
        u_idx = np.searchsorted(x_obs, window[1])
        window_obs = x_obs[l_idx:u_idx]
        flux_obs_window = y_obs[l_idx:u_idx]
        err_window = yerr[l_idx:u_idx]

        # Interpolate model to the observed window wavelengths
        model_flux_window = np.interp(window_obs, wave_model, model_flux)

        # Apply the polynomial continuum correction
        poly_func = np.polyval(poly_coeffs, window_obs)
        model_corrected = model_flux_window * poly_func

        # Compute log-likelihood contribution from this window
        sigma2 = err_window ** 2
        log_like += -0.5 * np.sum((flux_obs_window - model_corrected) ** 2 / sigma2 + np.log(sigma2))

    return log_like



def log_probability(theta): # gives the posterior probability
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

def rad_vel_correction(wave_ax, vel):
    """
    Apply correction to wavelength for the doppler shift due to
    radial velocity of the star.
    :param wave_ax: Quantity numpy array
    :param vel: astropy.units.Quantity
    :return: astropy.units.Quantity array
    """
    del_wav = (vel/const.c) * wave_ax
    return wave_ax - del_wav


def main(p0,nwalkers,niter,ndim,lnprob):

    print("Model running...")
    start = time.time()
    
    with Pool(processes=8) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        sampler.run_mcmc(p0, niter, progress=True)
    
    end = time.time()
    multi_time = end - start
    print("single core took {0:.1f} seconds".format(multi_time))
    print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    
    # get the chain
    print("getting chain ... ")
    params = sampler.get_chain()

    return params

if __name__=="__main__":
    # read data for Marvin
    # path_to_valid = "../../FU_ori_HIRES/"
    # data = ascii.read(path_to_valid+'KOA_42767/HIRES/extracted/tbl/ccd0/flux/HI.20030211.26428_0_02_flux.tbl.gz')

    #read the data, V960 Mon
    path_to_valid = "../../../validation_files/"
    data = ascii.read(path_to_valid+'KOA_93088/HIRES/extracted/tbl/ccd1/flux/HI.20141209.56999_1_04_flux.tbl.gz')
    data = [data['wave'],data['Flux']/np.median(data['Flux']),data['Error']/np.median(data['Flux'])]
    #vac to air correction for given data
    wavelengths_air = wave.vactoair(data[0]*u.AA)
    # data[0] = wavelengths_air

    # radial velocity correction to wavelength
    data[0] = rad_vel_correction(wavelengths_air, 40.3 * u.km / u.s)# from header file
    

    # plt.plot(data[0],data[1])
    # plt.show()

    n_params = 5 # number of parameters that are varying
    nwalkers = 10
    niter = 100

    #check time for a single run
    # theta_single = [ 5.03197142e-01, -4.03054252e+00,  9.68469043e-01 , 1.20689315e+01,
    #   1.26199606e+01,  1.81237601e+00,  3.82239928e+03 , 7.06072326e+03,
    #   1.01185058e+00]
    # logger.info("Single spec run")
    # total_spec(theta_single, data[0]*u.AA)


    # generate initial conditions
    config_data = config_reader('mcmc_config.cfg')
    p0 = generate_initial_conditions(config_data, nwalkers)

    params = main(p0,nwalkers,niter,n_params,log_probability)
    np.save("22122024_v960_1000it.npy",params)

    print("completed")
