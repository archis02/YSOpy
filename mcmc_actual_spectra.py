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
import mcmc_window as mc_file
from multiprocessing import Pool
import os
import logging
import sys

# logger = logging.getLogger(__name__)
# logging.basicConfig(filename='mcmc.log', encoding='utf-8', level=logging.DEBUG)


##############################################
# This code is to run MCMC for the first time

if __name__ == "__main__":
    cpu_cores_used = cpu_count()
    # read data for Marvin
    path_to_valid = "/home/nius2022/2025_mcmc_ysopy/Buffer/spectra_save"
    # GD
    # path_to_valid = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
    # AM
    # path_to_valid = "/home/arch/yso/results/synthetic_fit"
    # loading data for V960 Mon
    # data = np.load(f"{path_to_valid}/stitched_HIRES_data_V960.npy")
    # data = np.load(f"{path_to_valid}/data_v899_mon_cc2.npy")
    # data = np.load(f"{path_to_valid}/data_hbc722_ccd2.npy")
    # data = np.load(f"{path_to_valid}/data_ex_lupi.npy")
    data = np.load(f"{path_to_valid}/data_v960mon_round2.npy")

    # radial velocity correction, taken from header
    data[0] = mc_file.rad_vel_correction(data[0]*u.AA, 43 * u.km / u.s)  # V960 Mon
    # data[0] = mc_file.rad_vel_correction(data[0] * u.AA, 27.92 * u.km / u.s)  # V899 Mon
    # data[0] = mc_file.rad_vel_correction(data[0] * u.AA, -10.0 * u.km / u.s)  # HBC 722, Carvalho paper
    x_obs = data[0]
    y_obs = data[1]  # /np.median(data_flux)  # this is not correct --> should be done for each window separately
    yerr = data[2]  # /np.median(data_flux)

    # filename where the chain will be stored
    # save_filename = f'v899_mon_ccd2.h5'
    #save_filename = f'hbc722_ccd2.h5'
    save_filename = f"v960_mon_ccd_mass_const.h5"
    # save_filename = f"ex_lupi.h5"
    # params_label = ['m', 'log_m_dot', 'b', 'cos_inclination']  # for V960 Mon
    params_label = ['log_m_dot', 'b', 'cos_inclination']  # for V960 Mon (mass fixed)
    # params_label = ['m', 'log_m_dot', 'b', 'cos_inclination', "t_0", "t_slab", "log_n_e", "tau", "av"]  # for Ex Lupi, 899
    # params_label = ['log_m_dot', 'b', 'cos_inclination', "t_0", "t_slab", "log_n_e", "tau"]
    # params_hbc722 = ['m', 'log_m_dot', 'b', 'cos_inclination', "t_0", "t_slab", "log_n_e", "tau", "av"]
    # params_label = params_hbc722
    n_params = len(params_label)  # number of parameters that are varying
    n_walkers = 80
    n_iter = 1000

    # generate initial conditions
    config_data_mcmc = mc_file.config_reader('mcmc_config.cfg')
    config_dict = utils.config_read_bare("ysopy/config_file.cfg")
    n_windows = len(config_dict['windows'])
    poly_order = config_dict['poly_order']
    # params = ['m', 'log_m_dot', 'b', 'cos_inclination', 't_slab', "log_n_e", "tau", "av"]
    p0 = mc_file.generate_initial_conditions(params_label, config_data_mcmc, n_windows=n_windows, poly_order=poly_order,
                                             n_walkers=n_walkers, n_params=n_params)
    n_dim = n_params + n_windows * (poly_order + 1)

    # MAIN
    params = mc_file.main(p0, params_label, n_dim, n_walkers, n_iter, cpu_cores_used, save_filename, config_dict,
                          config_data_mcmc, x_obs, y_obs, yerr)

    # np.save(f"trial1_v960_steps_{n_iter}_walkers_{n_walkers}.npy",params)

    print("completed")

##############################################


##############################################
#  This block is to restart sampling from a pre calculated chain
"""
if __name__ == "__main__":
    cores = cpu_count()

    # read data for Marvin
    path_to_valid = "/home/nius2022/2025_mcmc_ysopy/Buffer/spectra_save"
    # GD
    # path_to_valid = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
    # AM
    # path_to_valid = "/home/arch/yso/results/synthetic_fit"

    # data = np.load(f"{path_to_valid}/stitched_HIRES_data_V960.npy")
    # data = np.load(f"{path_to_valid}/data_ex_lupi.npy")
    data = np.load(f"{path_to_valid}/data_v899_mon.npy")
    data = np.load(f"{path_to_valid}/data_hbc722_mon_ccd2.npy")
    
    # radial velocity correction, taken from header
    # data[0] = mc_file.rad_vel_correction(data[0]*u.AA, 40.3 * u.km / u.s)
    # data[0] = mc_file.rad_vel_correction(data[0] * u.AA, 27.92 * u.km / u.s)  # V899 Mon
    data[0] = mc_file.rad_vel_correction(data[0] * u.AA, -10.0 * u.km / u.s)  # HBC 722, Carvalho paper
    x_obs = data[0]
    y_obs = data[1]  # /np.median(data_flux)  # this is not correct --> should be done for each window separately
    yerr = data[2]  # /np.median(data_flux)

    # filename where the chain will be stored
    # save_filename = f'v960_stitched_all_windows.h5'
    save_filename = f'v899_mon_less_params.h5'
    save_filename = f'hbc722_ccd2.h5'
    n_iter_more = 1000  # Define the extra number of iterations to be done

    # generate initial conditions
    config_data_mcmc = mc_file.config_reader('mcmc_config.cfg')
    config_dict = utils.config_read_bare("ysopy/config_file.cfg")

    # print(log_likelihood_window(p0, config_dict))
    # params_label = ['m', 'log_m_dot', 'b', 'cos_inclination']
    # params_label = ['m', 'log_m_dot', 'b', 'cos_inclination', "t_slab", "log_n_e", "tau", "av"]
    # params_label = ['log_m_dot', 'b', 'cos_inclination', "t_0", "t_slab", "log_n_e", "tau"]
    params_hbc722 = ['m', 'log_m_dot', 'b', 'cos_inclination', "t_0", "t_slab", "log_n_e", "tau", "av"]
    params_label = params_hbc722
    params = mc_file.resume_sampling(params_label, save_filename, n_iter_more, config_dict, config_data_mcmc, x_obs,
                                     y_obs, yerr, cpu_cores_used=cores)

    # MAIN
    # params = main(p0, n_dim, n_walkers,config_dict,config_data_mcmc, x_obs, y_obs, yerr)
    # np.save(f"trial1_v960_steps_{n_iter}_walkers_{n_walkers}.npy",params)

    print("completed")
"""