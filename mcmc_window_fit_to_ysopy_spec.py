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

if __name__=="__main__":
    cpu_cores_used = cpu_count()
    # theta = np.array([6, 4.5, 2.0, 20, 9])  # test case for high accretion rate
    theta = np.array([10, 6.5, 1.5, 25, 8, 13, 1.5])  # test case for low accretion rate

    snr = 100
    # read data for Marvin
    path_to_valid = "/home/nius2022/2025_mcmc_ysopy/Buffer/spectra_save"
    # GD
    # path_to_valid = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
    # AM
    # path_to_valid = "/home/arch/yso/results/synthetic_fit"

    data_wave = np.load(f"{path_to_valid}/trimmed_wavem_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy")
    data_flux = np.load(f"{path_to_valid}/snr_{snr}_obs_flux_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy")
    data_error = np.load(f"{path_to_valid}/snr_{snr}_noise_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy")

    # data = [data['wave'],data['Flux']/np.median(data['Flux']),data['Error']/np.median(data['Flux'])]
    #
    # # radial velocity correction, taken from header
    # data[0] = rad_vel_correction(data[0]*u.AA, 40.3 * u.km / u.s)

    x_obs = data_wave
    y_obs = data_flux/np.median(data_flux)
    yerr = data_error/np.median(data_flux)

    # filename where the chain will be stored
    save_filename = f'mcmc_total_spec_{snr}_low_acc_gaussian_noise.h5'

    n_params = 7  # number of parameters that are varying
    n_walkers = 35
    n_iter = 500

    # generate initial conditions
    config_data_mcmc = mc_file.config_reader('mcmc_config.cfg')
    config_dict = utils.config_read_bare("ysopy/config_file.cfg")
    n_windows = len(config_dict['windows'])
    poly_order = config_dict['poly_order']
    p0 = mc_file.generate_initial_conditions(config_data_mcmc, n_windows=n_windows, poly_order=poly_order, n_walkers=n_walkers, n_params=n_params)
    n_dim = n_params + n_windows * (poly_order + 1)

    # MAIN
    params = mc_file.main(p0, n_dim, n_walkers, n_iter, cpu_cores_used, save_filename, config_dict, config_data_mcmc, x_obs, y_obs, yerr)

    # np.save(f"trial1_v960_steps_{n_iter}_walkers_{n_walkers}.npy",params)

    print("completed")

##############################################


##############################################
#  This block is to restart sampling from a pre calculated chain
"""
if __name__ == "__main__":
    cores = cpu_count()
    snr = 100
    theta = np.array([6, 4.5, 2.0, 20, 9])
    # read data for Marvin
    # path_to_valid = "../../FU_ori_HIRES/"
    # data = ascii.read(path_to_valid+'KOA_42767/HIRES/extracted/tbl/ccd0/flux/HI.20030211.26428_0_02_flux.tbl.gz')
    #
    # read the data, V960 Mon
    # path_to_valid = "../../../validation_files/"
    path_to_valid = "/Users/tusharkantidas/NIUS/ysopy_valid/"
    # path_to_valid = "/home/nius2022/2025_mcmc_ysopy/Buffer/spectra_save"

    data_wave = np.load(
        f"{path_to_valid}/trimmed_wavem_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy")
    data_flux = np.load(
        f"{path_to_valid}/snr_{snr}_obs_flux_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy")
    data_error = np.load(
        f"{path_to_valid}/snr_{snr}_noise_m_{theta[0]}_mdot_{theta[1]}_b_{theta[2]}_i_{theta[3]}_tslab_{theta[4]}.npy")

    x_obs = data_wave
    y_obs = data_flux / np.median(data_flux)
    yerr = data_error / np.median(data_flux)

    # filename where the chain will be stored
    save_filename = f'mcmc_total_spec_{snr}_low_acc_gaussian_noise.h5'

    n_iter_more = 20 # Define the extra number of iterations to be done

    # generate initial conditions
    config_data_mcmc = mc_file.config_reader('mcmc_config.cfg')
    config_dict = utils.config_read_bare("ysopy/config_file.cfg")

    # print(log_likelihood_window(p0, config_dict))
    params = mc_file.resume_sampling(save_filename, n_iter_more, config_dict, config_data_mcmc, x_obs, y_obs, yerr, cpu_cores_used=cores)

    # MAIN
    # params = main(p0, n_dim, n_walkers,config_dict,config_data_mcmc, x_obs, y_obs, yerr)
    # np.save(f"trial1_v960_steps_{n_iter}_walkers_{n_walkers}.npy",params)

    print("completed")
"""