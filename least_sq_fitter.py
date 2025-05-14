import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys
import multiprocessing as mp
from functools import partial

import astropy.units as u
from astropy.io import ascii
import astropy.constants as const
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from pypeit.core import wave
from ysopy import base_funcs as bf
from ysopy import utils

#stop at runtime warnings
import warnings

# Turn all warnings into errors
warnings.simplefilter('error', RuntimeWarning)

# Alternatively, target only RuntimeWarnings
# warnings.filterwarnings('error', category=RuntimeWarning)

def rad_vel_correction(wave_ax, vel):
    """
    Apply correction to wavelength for the doppler shift due to
    radial velocity of the star.
    """
    del_wav = (vel/const.c) * wave_ax
    return wave_ax - del_wav

#read the data, V960 Mon
path_to_valid = "../../../validation_files/"
data = ascii.read(path_to_valid+'KOA_93088/HIRES/extracted/tbl/ccd1/flux/HI.20141209.56999_1_04_flux.tbl.gz')
data = [data['wave'],data['Flux']/np.median(data['Flux']),data['Error']/np.median(data['Flux'])]    # median normalized
wavelengths_air = wave.vactoair(data[0]*u.AA)   # vac to air correction for given data
data[0] = rad_vel_correction(wavelengths_air, 40.3 * u.km / u.s)    # radial velocity correction to wavelength, from header file

OUTPUT_FILE = '/home/arch/yso/results/best_fit_params.txt'
N_PARAMS = 5
INITIAL_GUESS = np.array([0.6,-4.55,1.0,15.0,4,8])   # params = ['m', 'log_m_dot', 'b', 'inclination', 't_0'/1000, 't_slab'/1000]
BOUNDS = ([0.4,-6.5,0.8,5.0,3.0,7.5], [0.8,-4.6,1.2,30.0,4.5,8.5])

x_obs, y_obs, yerr = data[0], data[1], data[2]

def model_spec(theta,wavelength):
    # t0 = time.time()

    config = utils.config_read('ysopy/config_file.cfg')
    config['m'] = theta[0] * const.M_sun
    config['m_dot'] = 10**theta[1] * const.M_sun / (1 * u.year).to(u.s) ## Ensure the 10** here
    config['b'] = theta[2] * u.kilogauss
    config['inclination'] = theta[3] * u.degree
    config['t_0'] = theta[4] *1000.0 * u.K
    config['t_slab'] = theta[5] *1000.0 * u.K

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

    #run model
    dr, t_max, d, r_in, r_sub = bf.generate_temp_arr(config)
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

    # interpolate to required wavelength
    func = interp1d(wave_ax,total_flux)     # CHECK if this works, for units
    result_spec = func(wavelength)
    result_spec /= np.median(result_spec)

    # print(f"model run ... time taken {t5 - t0} s")

    return result_spec

def residuals(theta):
    start = time.time()
    model_flux = model_spec(theta, x_obs)
    residual = (y_obs - model_flux) / yerr**2
    print(f"Evaluated at theta={np.round(theta, 3)} | time: {time.time() - start:.2f}s")
    return residual

def run_optimization():
    print("Starting least-sq optimization...")
    start_time = time.time()
    result = least_squares(residuals, INITIAL_GUESS, method='trf', bounds=BOUNDS, verbose=2, xtol=1e-8, ftol=1e-8)
    total_time = time.time() - start_time
    print(f"Optimization finished in {total_time:.2f} seconds.")
    
    # Save best-fit parameters
    np.savetxt(OUTPUT_FILE, result.x, header="Best-fit parameters", fmt="%.6f")
    print(f"Best-fit parameters saved to: {OUTPUT_FILE}")
    return result.x

# PLOT
def plot_fit(best_params):
    model_flux = model_spec(best_params, x_obs*u.AA)
    plt.figure(figsize=(10, 5))
    plt.plot(x_obs, y_obs, label="Observed", color='black')
    plt.plot(x_obs, model_flux, label="Model Fit", linestyle='--')
    plt.fill_between(x_obs.value, y_obs.value - yerr.value, y_obs.value + yerr.value, color='gray', alpha=0.3, label="Error")
    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.legend()
    plt.title("Best-Fit Model vs Observed Spectrum")
    plt.tight_layout()
    plt.savefig("fit_result.png", dpi=300)
    plt.show()

# if __name__ == "__main__":
#     best_fit = run_optimization()
#     plot_fit(best_fit)

# sys.exit(0)

##################################################################

#### residual surface plot over m and log_m_dot

param_names = ['m', 'log_m_dot', 'b', 'inclination', 't_0/1000', 't_slab/1000']
param_i, param_j = 0, 1  # parameters to vary
n_points = 20
pi_vals = np.linspace(BOUNDS[0][param_i], BOUNDS[1][param_i], n_points)
pj_vals = np.linspace(BOUNDS[0][param_j], BOUNDS[1][param_j], n_points)
PI, PJ = np.meshgrid(pi_vals, pj_vals)

def evaluate_residual(theta_base, pi_val, pj_val, param_i, param_j):
    theta = theta_base.copy()
    theta[param_i] = pi_val
    theta[param_j] = pj_val
    model_flux = model_spec(theta,x_obs)
    residual = (y_obs - model_flux) / yerr
    return np.sum(residual**2)

# parallelize
def parallel_grid_eval():
    print("Starting parallel residual surface evaluation...")

    grid_points = [(pi, pj) for pi in pi_vals for pj in pj_vals]
    with mp.Pool(mp.cpu_count()) as pool:
        func = partial(evaluate_residual, INITIAL_GUESS, param_i=param_i, param_j=param_j)
        chi2_flat = pool.starmap(func, grid_points)

    chi2_grid = np.array(chi2_flat).reshape(n_points, n_points)

    np.savetxt("residual_grid.csv", chi2_grid, delimiter=",", fmt="%.6f")
    print("Residual grid saved to 'residual_grid.csv'")
    return chi2_grid

# plot
def plot_residual_surface(chi2_grid):
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(PI, PJ, chi2_grid.T, levels=50, cmap='viridis') ## transposing the array is important here
    plt.colorbar(label="Chi-squared")
    plt.xlabel(f"{param_names[param_i]}")
    plt.ylabel(f"{param_names[param_j]}")
    plt.title("Residual Surface")
    plt.tight_layout()
    plt.savefig("residual_surface.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    chi2_grid = parallel_grid_eval()
    plot_residual_surface(chi2_grid)
