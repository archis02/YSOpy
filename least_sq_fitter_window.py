import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import multiprocessing as mp
from functools import partial
import cProfile

import astropy.units as u
from astropy.io import ascii
import astropy.constants as const
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from pypeit.core import wave
from ysopy import base_funcs as bf
from ysopy import utils

# # stop at runtime warnings
# import warnings
# Turn all warnings into errors
# warnings.simplefilter('error', RuntimeWarning)
# Alternatively, target only RuntimeWarnings
# warnings.filterwarnings('error', category=RuntimeWarning)

def rad_vel_correction(wave_ax, vel):
    """
    Apply correction to wavelength for the doppler shift due to
    radial velocity of the star.
    """
    del_wav = (vel/const.c) * wave_ax
    return wave_ax - del_wav

# read the data, V960 Mon, initially, using file 04 ###############
# when reading a custom-stitched file, ensure that the wavelength axis is in ascending order, with no overlaps
# path_to_valid = "../../../validation_files/"
path_to_valid = "/Users/tusharkantidas/NIUS/ysopy_valid/"
data = ascii.read(path_to_valid+'KOA_93088/HIRES/extracted/tbl/ccd1/flux/HI.20141209.56999_1_04_flux.tbl')
data = [data['wave'],data['Flux']/np.median(data['Flux']),data['Error']/np.median(data['Flux'])]    # median normalized
wavelengths_air = wave.vactoair(data[0]*u.AA)   # vac to air correction for given data
data[0] = rad_vel_correction(wavelengths_air, 40.3 * u.km / u.s)    # radial velocity correction to wavelength, from header file

# # test fit to model itself, without any noise
# path_to_valid = "/home/arch/yso/results/synthetic_fit/"
# # y_obs = np.load(path_to_valid+"extinguished_spectra.npy")
# y_obs = np.load(path_to_valid+"noisy_flux_snr_100.npy")
# y_obs = y_obs/np.median(y_obs)
# x_obs = np.load(path_to_valid+"wave_arr.npy")

tstamp = time.time()
OUTPUT_FILE = f'/home/arch/yso/results/best_fit_file09_window_{tstamp}.txt'
# N_PARAMS = 5
INITIAL_GUESS = np.array([0.60,-4.5,20,8.5]) # params = ['m', 'log_m_dot', 'inclination', 't_slab'/1000]
BOUNDS = np.array([[0.4,-5.0,5.0,6.5], [1.5,-4.0,30.0,9.0]])
BOUNDS = BOUNDS.tolist()

x_obs, y_obs, yerr = data[0].value, data[1], data[2]

def model_spec_window(theta):
    '''Evaluates model in the range [l_min,l_max]. Ensure that all windows over which the shi-square is calculated lie within this range.
    Parameters
    ----------
    theta: list
        list of parameters which are varied as the least-squre fitting is done
    '''

    config = utils.config_read_bare('ysopy/config_file.cfg')
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


# stitch the different files into one file and then define the windows on which the 
# chi-square is to be calculated
def residuals_windows(theta,poly_order,residual_len):
    start=time.time()

    residual = np.zeros(residual_len)
    windows = config['windows']  # list of wavelength ranges in which residual is to be evaluated
    n_windows = len(windows)
    n_model_params = len(theta) - n_windows * (poly_order + 1)
    theta_model = theta[:n_model_params]
    wave, model_flux = model_spec_window(theta_model)
    counter = 0
    
    for i,window in enumerate(windows):
        
        poly_coeffs = theta[n_model_params + i*(poly_order+1):n_model_params+(i+1)*(poly_order+1)]

        # trim all arrays to the required window
        window_obs_l = np.searchsorted(x_obs,window[0])
        window_obs_u = np.searchsorted(x_obs,window[1])
        window_obs = x_obs[window_obs_l:window_obs_u]
        flux_obs_window = y_obs[window_obs_l:window_obs_u]
        err_window = yerr[window_obs_l:window_obs_u]

        # interpolate the model to the same axis
        model_flux_window = np.interp(window_obs, wave, model_flux)
        
        # continuum correction
        poly_func = np.polyval(poly_coeffs, window_obs)

        residual_window = (flux_obs_window - model_flux_window * poly_func)/err_window

        # assign to the subset of the residual array
        residual[counter:counter+residual_window.shape[0]:] = residual_window
        counter += residual_window.shape[0]

    # show time taken to run, and parameters
    print(f"Evaluated at theta={np.round(theta_model, 6)} | "
          f"poly={np.round(theta[n_model_params:], 6)} | "
          f"time: {time.time() - start:.3f}s")

    return residual


def run_optimization_window(config,n_windows):
    poly_order = config['poly_order']
    print(f"Starting least-sq optimization...\nThe continuum is a polynomial of order {poly_order}")
    start_time = time.time()

    # set initial guess for the polynomial coeffs, set bounds
    poly_params_guess = np.array([0.0]*poly_order+[1]) # flat initial guess
    poly_params_guess_all = np.tile(poly_params_guess,n_windows)
    INITIAL_GUESS_TOT = np.concatenate([INITIAL_GUESS,poly_params_guess_all])
    bd_lower = [-1] * poly_order + [0.5]
    bd_upper = [1] * poly_order + [1.5]
    BOUNDS[0] = BOUNDS[0] + bd_lower*n_windows
    BOUNDS[1] = BOUNDS[1] + bd_upper*n_windows

    # calculate the residual array length
    residual_len = 0
    for i, window in enumerate(config['windows']):
        window_obs_l = np.searchsorted(x_obs,window[0])
        window_obs_u = np.searchsorted(x_obs,window[1])
        residual_len += (window_obs_u-window_obs_l)
    

    # set step size when performing least-sqr minimization
    ## IMPORTANT: do not make the step size too small for m and m_dot, it should be comparable to the step size for inclination
    diff_step_model = [3e-2,1e-2,5e-3,1e-3] # for model params
    diff_step_poly = [1e-3,1e-5]*n_windows # for continua, this can be customized as required
    diff_step = diff_step_model + diff_step_poly
    result = least_squares(residuals_windows, INITIAL_GUESS_TOT, args=(poly_order,residual_len), method='trf', bounds=BOUNDS, verbose=2, xtol=1e-8, ftol=1e-8, diff_step=diff_step)

    total_time = time.time() - start_time
    print(f"Optimization finished in {total_time:.2f} seconds.")
    
    # Save best-fit parameters
    np.savetxt(OUTPUT_FILE, result.x, header="Best-fit parameters", fmt="%.6f")
    print(f"Best-fit parameters saved to: {OUTPUT_FILE}"
          f"\n Paramters: {result.x}")
    return result.x

def plot_fit_windows(config, best_params):

    poly_order = config['poly_order']
    n_windows = len(config['windows'])
    n_model_params = len(best_params) - n_windows*(poly_order + 1)
    theta_model = best_params[:n_model_params]
    wave_ax,model_flux = model_spec_window(theta_model)

    # counter = 0
    
    plt.figure(figsize=(10, 5))
    
    for i,window in enumerate(config['windows']):
        wave_obs_l = np.searchsorted(x_obs,window[0])
        wave_obs_u = np.searchsorted(x_obs,window[1])

        plt.plot(x_obs[wave_obs_l:wave_obs_u], y_obs[wave_obs_l:wave_obs_u], label="Observed", color='black')

        poly_coeffs = best_params[n_model_params+i*(poly_order+1):n_model_params+(i+1)*(poly_order+1)]
        wave_model_trimmed_l = np.searchsorted(wave_ax,window[0])
        wave_model_trimmed_u = np.searchsorted(wave_ax,window[1])
        poly_func = np.polyval(poly_coeffs, wave_ax)
        model_continuum_corrected = model_flux * poly_func
        
        plt.plot(wave_ax[wave_model_trimmed_l:wave_model_trimmed_u], model_continuum_corrected[wave_model_trimmed_l:wave_model_trimmed_u], label="Model Fit", linestyle='--')
    
    plt.fill_between(x_obs, y_obs - yerr, y_obs + yerr, color='gray', alpha=0.3, label="Error")
    
    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.legend()
    plt.title("Best-Fit Model vs Observed Spectrum")
    plt.tight_layout()
    plt.savefig(f"fit_result_file01_{tstamp}.png", dpi=300)
    plt.show()

##################################################################
# Run least-square optimization
##################################################################

if __name__ == "__main__":
    
    config = utils.config_read_bare('ysopy/config_file.cfg')
    
    #switch off plot and save
    config['plot'] = False
    config['save'] = False
    config['verbose'] = False

    # cProfile.run(run_optimization(config['poly_order']))
    
    best_fit = run_optimization_window(config,n_windows=2)

    # read manually
    # OUTPUT_FILE = f'/home/arch/yso/results/best_fit_file01_window_1747907636.1083474.txt'
    # best_fit = np.loadtxt(OUTPUT_FILE)

    plot_fit_windows(config,best_fit)

sys.exit(0)

##################################################################
#### residual surface plot over m and log_m_dot
##################################################################

config = utils.config_read_bare('ysopy/config_file.cfg')

param_names = ['m', 'log_m_dot', 'inclination', 't_slab/1000']
param_i, param_j = 0, 1  # parameters to vary
n_points = 20
pi_vals = np.linspace(BOUNDS[0][param_i], BOUNDS[1][param_i], n_points)
pj_vals = np.linspace(BOUNDS[0][param_j], BOUNDS[1][param_j], n_points)
PI, PJ = np.meshgrid(pi_vals, pj_vals)

def evaluate_residual(theta_base, pi_val, pj_val, param_i, param_j):
    theta = theta_base.copy()
    theta[param_i] = pi_val
    theta[param_j] = pj_val
    model_flux,wave_ax = model_spec_window(theta,x_obs)
    residual = (y_obs - model_flux) / yerr
    return np.sum(residual**2)

# parallelize
def parallel_grid_eval():
    st = time.time()
    print("Starting parallel residual surface evaluation...")

    grid_points = [(pi, pj) for pi in pi_vals for pj in pj_vals]
    with mp.Pool(mp.cpu_count()) as pool:
        func = partial(evaluate_residual, INITIAL_GUESS, param_i=param_i, param_j=param_j)
        chi2_flat = pool.starmap(func, grid_points)

    chi2_grid = np.array(chi2_flat).reshape(n_points, n_points)

    # !! ** !! CHANGE FILE NUMBER HERE !! ** !!
    np.savetxt("residual_grid_file_04.txt", chi2_grid, delimiter=",", fmt="%.6f")
    et = time.time()
    print(f"Residual grid saved to 'residual_grid_file_04.txt' ... time taken = {et-st}")
    return chi2_grid

# plot
def plot_residual_surface(chi2_grid):
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(PI, PJ, chi2_grid.T, levels=20, cmap='viridis') ## transposing the array is important here
    plt.colorbar(label="Chi-squared")
    plt.xlabel(f"{param_names[param_i]}")
    plt.ylabel(f"{param_names[param_j]}")
    plt.title("Residual Surface")
    plt.tight_layout()
    plt.savefig("residual_surface_file_04.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    chi2_grid = parallel_grid_eval()
    # chi2_grid = np.load
    plot_residual_surface(chi2_grid)
