import numpy as np
import matplotlib.pyplot as plt
import emcee
from ysopy import utils
import corner
import astropy.units as u
from configparser import ConfigParser
import mcmc_window as mc_file
import mcmc_emission_spectra as mcem_file


plot_dir = "/Users/tusharkantidas/github/archis/Buffer/plot_directory"
def plot_trace(flat_for_trace, labels):
    ndim = len(labels)
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(flat_for_trace[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(flat_for_trace))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    # plt.savefig(f"{plot_dir}/trace_v960_splitting_1.pdf")
    plt.show()


# def plot_corner(flat_samples, labels):
#     fig = corner.corner(
#         # flat_for_trace,
#         flat_samples,
#         figsize=(7, 5),
#         labels=labels,
#         # truths=[9, 13, 30],
#         plot_contours=True,
#         quantiles=[0.16, 0.5, 0.84],
#         # quantiles=[0.035, 0.5, 0.975],
#         show_titles=True,
#         title_kwargs={"fontsize": 12},
#         # smooth=True,
#         # bins=[100, 100, 100, 100, 100, 100, 100],
#         # range=range_corner
#     )
#     # plt.suptitle(f"{filename}\n\n\n")
#     # plt.savefig(f"../Buffer/plot_directory/mcmc_total_model_corner.pdf")
#     plt.show()

import matplotlib.pyplot as plt
import corner
import numpy as np


def plot_corner(samples, labels, save_name:str,truths=None):
    """
    Generate a corner plot showing:
    - True values (blue lines)
    - Median values from the posterior (red lines)

    Parameters:
    - flat_samples: (n_samples, n_params) array
    - labels: list of parameter names
    - truths: list of known true values (blue lines)
    """
    # Compute median from the samples

    try:
        nsteps, nwalkers, ndim = samples.shape
    except:
        total_walk, ndim = samples.shape
    fsamples = samples.reshape((-1, ndim))
    medians = np.median(fsamples, axis=0)

    # Plot with 'truths' set to the blue true values
    fig = corner.corner(
        fsamples,
        figsize=(7, 5),
        labels=labels,
        truths=truths,
        truth_color="blue",  # Blue lines for known truths
        plot_contours=True,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        smooth=1.0,
        show_grid=True,
        # smooth1D=3.0
    )

    # Add red lines for medians

    print(ndim)
    axes = np.array(fig.axes).reshape((ndim, ndim))

    for i in range(ndim):
        ax = axes[i, i]
        ax.axvline(medians[i], color="red", linestyle="--", linewidth=1)

        for j in range(i):
            axes[i, j].axvline(medians[j], color="red", linestyle="--", linewidth=1)
            axes[i, j].axhline(medians[i], color="red", linestyle="--", linewidth=1)
    # plt.savefig(f"{plot_dir}/corner_{save_name}.pdf")
    plt.show()


def filter_bad_walks_one_param(samples: np.ndarray, param_index: int,  lb: np.ndarray, ub: np.ndarray):
    """
    Filter out bad steps from any walker for given lower bound and upper bound on the parameters.
    This is NOT to remove a walker which has taken any bad step

    returns:
    Filtered samples
    """
    nsteps, nwalkers, ndim = samples.shape
    print(nsteps, nwalkers, ndim)
    # Flatten all samples across walkers and steps to compute overall mean and std
    # flat_samples = samples.reshape(-1, ndim)
    samples = samples.reshape((-1, ndim))
    print(samples.shape)
    print("Before removing samples ", len(samples))
    correct_step = []
    for i in range(nsteps * nwalkers):
        step_params = samples[i]
        # print(within_bounds)
        within_bounds = bool((step_params[param_index] >= lb) & (step_params[param_index] <= ub))
        if within_bounds:
            correct_step.append(i)
    non_lost_walks = np.zeros((len(correct_step), ndim))
    for i in range(len(correct_step)):
        non_lost_walks[i] = samples[correct_step[i]]

    samples = non_lost_walks
    print("After removing samples ", len(samples))
    return samples



def filter_walkers_within_n_sigma(n:int, samples:np.ndarray):
    """
    Given n (integer), filter out the mcmc steps from samples array to
    contain only those sets which are within the n-sigma bounds.
    returns:
    flat_samples_within_n_sigma: np.ndarray
    """
    try:
        nsteps, nwalkers, ndim = samples.shape
    except:
        total_walk, ndim = samples.shape
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    # print("mean", mean)
    # print("std", std)
    # Define 4σ bounds
    lower_bound = mean - n * std
    upper_bound = mean + n * std
    # print(lower_bound, upper_bound)
    correct_steps = []
    for i in range(len(samples)):
        step_params = samples[i]
        # print(within_bounds)
        within_bounds = np.all((step_params >= lower_bound) & (step_params <= upper_bound))
        if within_bounds:
            correct_steps.append(i)
    flat_samples_in_n_sigma = np.zeros((len(correct_steps), ndim))
    for i in range(len(correct_steps)):
        flat_samples_in_n_sigma[i] = samples[correct_steps[i]]
    # print("After extracting within sigmas", len(flat_samples_in_n_sigma))
    return flat_samples_in_n_sigma


def extract_normalised_window_for_plots(theta, config, x_obs, y_obs, y_err):
    """
    Compute log-likelihood using windowed residuals and independent continuum correction per window.
    Assumes data arrays x_obs (wavelengths), y_obs (normalized flux), yerr (errors),
    and a model_spec_window function that returns (wavelength, model_flux).
    """
    poly_order = config['poly_order']
    windows = config['windows']  # list of (lower, upper) wavelength bounds for each window
    n_windows = len(windows)
    print("n_windows", n_windows)
    print("poly order", poly_order)
    n_model_params = len(theta) - n_windows * (poly_order + 1)
    print(n_model_params)
    theta_model = theta[:n_model_params]
    print("theta_model", theta_model)
    # Model spectrum (full)
    wave_model, total_flux, flux_photon_count = mc_file.model_spec_window(theta_model, config)
    # change the below code from total_flux to photon_counts if obs_spectra is in photon counts
    model_flux = total_flux

    fluxes_list = []
    wavelenghts_list = []
    obs_wave_list = []
    obs_flux_list = []
    obs_error_list = []
    # loop over the windows
    for i, window in enumerate(windows):
        # Get the polynomial coefficients for this window
        poly_coeffs = theta[n_model_params + i * (poly_order + 1) : n_model_params + (i + 1) * (poly_order + 1)]

        # For observed data
        # Get indices for this window
        l_idx = np.searchsorted(x_obs, window[0])
        u_idx = np.searchsorted(x_obs, window[1])
        window_obs = x_obs[l_idx:u_idx]
        flux_obs_window = y_obs[l_idx:u_idx]
        err_window = y_err[l_idx:u_idx]
        flux_obs_window/=np.median(flux_obs_window)
        err_window/=np.median(flux_obs_window)
        # Appending to list
        obs_wave_list.append(window_obs)
        obs_flux_list.append(flux_obs_window)
        obs_error_list.append(err_window)


        # Get indices for this window
        l_idx = np.searchsorted(wave_model, window[0])
        u_idx = np.searchsorted(wave_model, window[1])
        window_model = wave_model[l_idx:u_idx]
        flux_model_window = model_flux[l_idx:u_idx]

        # Normalise within the window
        flux_model_window /= np.median(flux_model_window)

        # scaling the wavelength to remove degeneracy of slope and intercept
        scaled_wave = np.linspace(-1, 1, len(window_model))
        # Apply the polynomial continuum correction
        poly_func = np.polyval(poly_coeffs, scaled_wave)
        model_corrected = flux_model_window * poly_func
        # append to list
        fluxes_list.append(model_corrected)
        wavelenghts_list.append(window_model)

    return wavelenghts_list, fluxes_list, obs_wave_list, obs_flux_list, obs_error_list


# tslab_1000, log10_ne, tau_10, log_f = theta


# generate initial conditions
config_dict = utils.config_read_bare("ysopy/config_file.cfg")
saveloc = "/Users/tusharkantidas/github/archis/Buffer/store_mcmc_files/"
# filename = 'mcmc_total_spec.h5'
# snr = 100
# filename = f"mcmc_total_spec_{snr}.h5"
# filename = f'mcmc_total_spec_{snr}_gaussian_noise.h5'
# filename = f"mcmc_total_spec_100_low_acc_gaussian_noise.h5"
# filename = f"mcmc_total_spec_100_low_acc_poly_order_2.h5"

# filename = f"mcmc_total_spec_100_balmer_jump.h5"
# for v960 Mon file
# filename = "v960_stitched_all_windows.h5"
# filename = "v960_mon_changed_inclination_all_windows.h5"
# filename = "v960_mon_log_run.h5"
filename = "v960_mon_round2.h5"
# for Ex-Lupi File
# filename = "ex_lupi.h5"
# filename = "ex_lupi_balmer.h5"
# filename = "ex_lupi_balmer_smooth.h5"
# filename = "ex_lupi_balmer_smooth_5001.h5"
# filename ="ex_lupi_balmer_upto_3900A.h5"
# filename = "ex_lupi_balmer_04052010.h5"
# for V899 Mon Files
# filename = "v899_mon_all_windows.h5"
# filename = "v899_mon_less_params.h5"
# filename = "v899_mon_ccd2.h5"

# HBC 722
# filename = "hbc722_ccd2_old.h5"
# filename = "hbc722_ccd2.h5"

# Lup713
# filename = f"lup713.h5"
print(filename)
filepath = saveloc + filename
# filepath = filename
reader = emcee.backends.HDFBackend(filepath)

# flat_samples = reader.get_chain(discard=00)
# niter, nwalkers, nparams = flat_samples.shape
# flat_for_params = np.zeros((niter, nwalkers, 7))
# print(flat_samples.shape)
# flat_for_params[:, :, 0] = flat_samples[:, :, 0]
# flat_for_params[:, :, 1] = flat_samples[:, :, 1]
# # flat_for_params[:, :, 2] = flat_samples[:, :, 2]
# flat_for_params[:, :, 2] = np.arccos(flat_samples[:, :, 2]/10) * 180 / np.pi
# flat_for_params[:, :, 3] = flat_samples[:, :, 3]
# # flat_for_params[:, :, 3] = np.arccos(flat_samples[:, :, 3]/10) * 180 / np.pi
# flat_for_params[:, :, 4] = flat_samples[:, :, 4]
# flat_for_params[:, :, 5] = flat_samples[:, :, 5]
# flat_for_params[:, :, 6] = flat_samples[:, :, 6]
# flat_for_params[:, :, 7] = flat_samples[:, :, 7]
# flat_for_params[:, :, 8] = flat_samples[:, :, 8]
# flat_for_params[:, :, 9] = flat_samples[:, :, 9]
# flat_for_params[:, :, 10] = flat_samples[:, :, 10]
# ########## file = total_spec
#
# flat_samples = reader.get_chain(discard=550)
# niter, nwalkers, nparams = flat_samples.shape
# flat_for_params = np.zeros((niter, nwalkers, 7))
# print(flat_samples.shape)
# labels = ['m', 'log_m_dot', 'B', 'inclination', 't_0', "a", "b"]#, "av", "a", "b"]
# theta = np.array([6, 4.5, 2.0, 20, 9, 0, 1])
# plot_trace(flat_for_trace=flat_samples, labels=labels)
# plot_corner(samples=flat_samples, labels=labels, save_name=f"{filename}_on_model_data", truths=theta)
# exit(0)
########### mcmc_total_spec_{snr}_gaussian_noise.h5
# labels_total_spec_gaussian_noise = ['m', 'log_m_dot', 'B', 'inclination', 't_0', "a", "b"]
# labels_total_spec_gaussian_noise_except_b_t0 = ['m', 'log_m_dot', 'inclination', "a", "b"]
# theta = np.array([6, 4.5, 2.0, 20, 9, 0, 1])
# theta_except = np.array([6, 4.5, 20, 0, 1])
# # plot_trace(flat_for_trace=flat_samples, labels=labels_total_spec_gaussian_noise)
# filtered_samples = filter_bad_walks_one_param(flat_samples, 1, np.array([4.4]), np.array([4.5]))
# filter_n_sig = filter_walkers_within_n_sigma(5, filtered_samples)
# filter_sampl = np.zeros((len(filter_n_sig), 5))
# filter_sampl[:, 0] = filter_n_sig[:, 0]
# filter_sampl[:, 1] = filter_n_sig[:, 1]
# filter_sampl[:, 2] = filter_n_sig[:, 3]
# filter_sampl[:, 3] = filter_n_sig[:, 5]
# filter_sampl[:, 4] = filter_n_sig[:, 6]
# print(filter_n_sig.shape, filter_sampl.shape)
# plot_corner(samples=filter_sampl, labels=labels_total_spec_gaussian_noise_except_b_t0, save_name=f"{filename}_5_sigma_filtered_remove_b_t0", truths=theta_except)
# plot_corner(samples=filter_n_sig, labels=labels_total_spec_gaussian_noise, save_name=f"{filename}_5_sigma_filtered", truths=theta)
# exit(0)
###################

# #### file "mcmc_total_spec_100_low_acc_gaussian_noise.h5"
# labels_gaussian = ['m', 'log_m_dot', 'B', 'inclination', 't_slab', "log_n_e", "tau", "a", "b"]#, "b"]
# theta = np.array([10, 6.5, 1.5, 25, 8, 13, 1, 0, 1])
# plot_trace(flat_for_trace=flat_samples, labels=labels_gaussian)
# plot_corner(samples=flat_samples, labels=labels_gaussian, save_name=f"{filename}_on_model_data_low_mdot", truths=theta)

#### file "mcmc_total_spec_100_low_acc_gaussian_noise.h5"
# flat_samples = reader.get_chain(discard=5500)
# niter, nwalkers, nparams = flat_samples.shape
# flat_for_params = np.zeros((niter, nwalkers, 7))
# print(flat_samples.shape)
# labels_poly2 = ['m', 'log_m_dot', 'B', 'inclination', 't_slab', "log_n_e", "tau", "a", "b", "c"]#, "b"]
# labels_poly2_except = ['m', 'log_m_dot', 'B', 'inclination', 't_slab', "log_n_e", "tau"]#, "b"]
# theta = np.array([10, 6.5, 1.5, 25, 8, 13, 1, 0, 0, 1])
# theta_except_poly = np.array([10, 6.5, 1.5, 25, 8, 13, 1])
# plot_trace(flat_for_trace=flat_samples, labels=labels_poly2)
# flat_samples = flat_samples.reshape((-1, nparams))
# plot_corner(samples=flat_samples, labels=labels_poly2, save_name=f"{filename}_on_model_data_low_mdot_poly_order2", truths=theta)
# filter_samples = np.zeros((len(flat_samples), 7))
# filter_samples[:, :] = flat_samples[:, 0:7]
# plot_corner(samples=filter_samples, labels=labels_poly2_except, save_name=f"{filename}_on_model_data_low_mdot_poly_order2_without_poly_params", truths=theta_except_poly)

##### filename = f"mcmc_total_spec_100_balmer_jump.h5"
# flat_samples = reader.get_chain(discard=1000)
# niter, nwalkers, nparams = flat_samples.shape
# flat_for_params = np.zeros((niter, nwalkers, 7))
# print(flat_samples.shape)
# labels_balmer = ['m', 'log_m_dot', 'B', 'inclination', 't_slab', "log_n_e", "tau", "av", "a", "b"]#, "b"]
# labels_balmer_except = ['m', 'log_m_dot', 'B', 'inclination', 't_slab', "log_n_e", "tau", "av"]#, "b"]
# theta = np.array([10, 6.5, 1.5, 25, 8, 13, 1, 5, 0, 1])
# theta_except_poly = np.array([10, 6.5, 1.5, 25, 8, 13, 1, 5])
# plot_trace(flat_for_trace=flat_samples, labels=labels_balmer)
# flat_samples = flat_samples.reshape((-1, nparams))
# plot_corner(samples=flat_samples, labels=labels_balmer, save_name=f"{filename}_on_model_data_balmer_line", truths=theta)
# filter_samples = np.zeros((len(flat_samples), 8))
# filter_samples[:, :] = flat_samples[:, 0:8]
# plot_corner(samples=filter_samples, labels=labels_balmer_except, save_name=f"{filename}_on_model_data_balmer_without_poly_params", truths=theta_except_poly)


##### filename = f"v960_stitched_all_windows.h5"
# flat_samples = reader.get_chain(discard=240)
# niter, nwalkers, nparams = flat_samples.shape
# flat_for_params = np.zeros((niter, nwalkers, 7))
# print(flat_samples.shape)
# labels_big = ['m', 'log_m_dot', 'B', 'inclination',
#           "1", "2", "3", "4", "1", "2", "3", "4",
#           "1", "2", "3", "4", "1", "2", "3", "4",
#           "1", "2", "3", "4", "1", "2", "3", "4",
#           "1", "2", "3", "4", "1", "2", "3", "4"]
# labels_big_except = ['m', 'log_m_dot', 'B', 'inclination', "a", "b"]#, "b"]
# filter_samples = np.zeros((niter, nwalkers, len(labels_big_except)))
# filter_samples[:, :, :] = flat_samples[:, :, 0:len(labels_big_except)]
# plot_trace(flat_for_trace=filter_samples, labels=labels_big_except)
# flatten_samples = filter_bad_walks_one_param(filter_samples, 1, np.array([4.45]), np.array([4.48]))
# flatten_samples = filter_walkers_within_n_sigma(5, flatten_samples)
# # flatten_samples = filter_samples.reshape((-1, len(labels_big_except)))
# plot_corner(samples=flatten_samples, labels=labels_big_except, save_name=f"{filename}_on_v960")

###### filename = "v960_mon_log_run.h5"
# flat_samples = reader.get_chain(discard=0)
# niter, nwalkers, nparams = flat_samples.shape
# print(flat_samples.shape)
# # flat_for_params = np.zeros((niter, nwalkers, 7))
# # labels_big = ['m', 'log_m_dot', 'B', 'inclination',
# #           "1", "2", "3", "4", "1", "2", "3", "4",
# #           "1", "2", "3", "4", "1", "2", "3", "4",
# #           "1", "2", "3", "4", "1", "2", "3", "4",
# #           "1", "2", "3", "4", "1", "2", "3", "4"]
# labels_big_except = ['m', 'log_m_dot', 'B', 'cos_inclination', "a", "b"]#, "b"]
# filter_samples = np.zeros((niter, nwalkers, len(labels_big_except)))
# filter_samples[:, :, :] = flat_samples[:, :, 0:len(labels_big_except)]
# # filter_samples[:, :, 3] = np.arccos(filter_samples[:, :, 3] /10 ) * 180/np.pi
# plot_trace(flat_for_trace=filter_samples, labels=labels_big_except)
# plot_corner(samples=filter_samples, labels=labels_big_except, save_name="")
# exit(0)
###### filename = "v960_mon_round2.h5"
flat_samples = reader.get_chain(discard=300)
niter, nwalkers, nparams = flat_samples.shape
print(flat_samples.shape)
# flat_for_params = np.zeros((niter, nwalkers, 7))
# labels_big = ['m', 'log_m_dot', 'B', 'inclination',
#           "1", "2", "3", "4", "1", "2", "3", "4",
#           "1", "2", "3", "4", "1", "2", "3", "4",
#           "1", "2", "3", "4", "1", "2", "3", "4",
#           "1", "2", "3", "4", "1", "2", "3", "4"]
labels_big_except = ['m', 'log_m_dot', 'B', 'cos_inclination', "a", "b"]#, "b"]
filter_samples = np.zeros((niter, nwalkers, len(labels_big_except)))
filter_samples[:, :, :] = flat_samples[:, :, 0:len(labels_big_except)]
# filter_samples[:, :, 3] = np.arccos(filter_samples[:, :, 3] /10 ) * 180/np.pi
plot_trace(flat_for_trace=filter_samples, labels=labels_big_except)
plot_corner(samples=filter_samples, labels=labels_big_except, save_name=f"{filename}_v960_16_order_1")
exit(0)

###### filename = "v960_mon_changed_inclination_all_windows.h5"
flat_samples = reader.get_chain(discard=500)
niter, nwalkers, nparams = flat_samples.shape
print(flat_samples.shape)
# flat_for_params = np.zeros((niter, nwalkers, 7))
# labels_big = ['m', 'log_m_dot', 'B', 'inclination',
#           "1", "2", "3", "4", "1", "2", "3", "4",
#           "1", "2", "3", "4", "1", "2", "3", "4",
#           "1", "2", "3", "4", "1", "2", "3", "4",
#           "1", "2", "3", "4", "1", "2", "3", "4"]
labels_big_except = ['m', 'log_m_dot', 'B', 'cos_inclination', "a", "b"]#, "b"]
filter_samples = np.zeros((niter, nwalkers, len(labels_big_except)))
filter_samples[:, :, :] = flat_samples[:, :, 0:len(labels_big_except)]
# filter_samples[:, :, 3] = np.arccos(filter_samples[:, :, 3] /10 ) * 180/np.pi
plot_trace(flat_for_trace=filter_samples, labels=labels_big_except)
# exit(0)
# plot_corner(samples=filter_samples, labels=labels_big_except, save_name="")
flatten_samples = filter_bad_walks_one_param(filter_samples, 1, np.array([4.638]), np.array([4.644]))
# flatten_samples = filter_bad_walks_one_param(filter_samples, 1, np.array([4.68]), np.array([4.69]))
flatten_samples = filter_walkers_within_n_sigma(5, flatten_samples)
flatten_samples = filter_samples.reshape((-1, len(labels_big_except)))
plot_corner(samples=flatten_samples, labels=labels_big_except, save_name=f"{filename}_v960_16_order_1")
#
# exit(0)
####### plot the spectra ###############
# flat_sample_filter = filter_bad_walks_one_param(flat_samples, 1, np.array([4.638]), np.array([4.644]))
flat_sample_filter = filter_bad_walks_one_param(flat_samples, 1, np.array([4.68]), np.array([4.69]))
flat_sample_filter = filter_walkers_within_n_sigma(5, flat_sample_filter)
means = np.median(flat_sample_filter, axis=0)
print(means)


path_to_valid = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
# loading data for V960 Mon
# data = np.load(f"{path_to_valid}/stitched_HIRES_data_V960.npy")
data = np.load(f"{path_to_valid}/stitched_HIRES_data_V960.npy")
data[0] = mc_file.rad_vel_correction(data[0]*u.AA, 43 * u.km / u.s)
x_obs = data[0]
y_obs = data[1]/np.median(data[1])
yerr = data[2]/np.median(data[1])
plt.plot(x_obs, y_obs, color='blue')
plt.show()
plt.close()
wavelenghts_list, fluxes_list, obs_wave_list, obs_flux_list, obs_error_list = extract_normalised_window_for_plots(theta=means, config=config_dict, x_obs=x_obs, y_obs=y_obs, y_err=yerr)

for i in range(len(wavelenghts_list)):
    plt.plot(obs_wave_list[i], obs_flux_list[i], color="red", alpha=0.7)
    plt.plot(wavelenghts_list[i], fluxes_list[i], color="black")
    # plt.fill_between(obs_wave_list[i], obs_flux_list[i] - obs_error_list[i], obs_flux_list[i] + obs_error_list[i], alpha=1)
plt.legend()
plt.show()
exit(0)


###### filename = "hbc722_ccd2.h5"
flat_samples = reader.get_chain(discard=4000)
niter, nwalkers, nparams = flat_samples.shape
# exit(0)
labels_hbc722 = ['m', 'log_m_dot', 'b', 'cos_inclination', "t_0", "t_slab", "log_n_e", "tau", "av"]
filter_samples = np.zeros((niter, nwalkers, len(labels_hbc722)))
filter_samples[:, :, :] = flat_samples[:, :, 0:len(labels_hbc722)]
# filter_samples[:, :, 3] = np.arccos(filter_samples[:, :, 3] /10 ) * 180/np.pi
# plot_trace(flat_for_trace=filter_samples, labels=labels_hbc722)

filter_samples = filter_bad_walks_one_param(samples=filter_samples, param_index=0,  lb= np.array([2.45]), ub=np.array([2.7]))
# plot_trace(flat_for_trace=filter_samples, labels=labels_hbc722)
# plot_corner(samples=filter_samples, labels=labels_hbc722, save_name="")

# filter only High accreytion rate parameters
print(filter_samples.shape)
filt_high_params = np.zeros((filter_samples.shape[0], filter_samples.shape[1] - 4))
print(filt_high_params.shape)
filt_high_params[:, 0] = filter_samples[:, 0]
filt_high_params[:, 1] = filter_samples[:, 1]
filt_high_params[:, 2] = filter_samples[:, 2]
filt_high_params[:, 3] = filter_samples[:, 3]
filt_high_params[:, 4] = filter_samples[:, 8]
label_high_acc_hbc_722 = ['m', 'log_m_dot', 'b', 'cos_inclination', "av"]
plot_corner(samples=filt_high_params, labels=label_high_acc_hbc_722, save_name="")
exit(0)
flatten_samples = filter_samples.reshape((-1, len(labels_hbc722)))

print(flatten_samples.shape)
medians = np.median(flat_samples.reshape((-1, nparams)), axis=0)

small_medians = np.median(flatten_samples, axis=0)
print(small_medians)
# small_medians[8] = 1
# wave_model, total_flux, flux_photon_count = mc_file.model_spec_window(small_medians, config_dict)
# plt.plot(wave_model, total_flux)
# plt.show()
# exit(0)
path_to_valid = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
# loading data for V960 Mon
# data = np.load(f"{path_to_valid}/stitched_HIRES_data_V960.npy")
data = np.load(f"{path_to_valid}/data_hbc722_ccd2.npy")
data[0] = mc_file.rad_vel_correction(data[0]*u.AA, -10.0 * u.km / u.s)
x_obs = data[0]
y_obs = data[1]/np.median(data[1])
yerr = data[2]/np.median(data[1])
# plt.plot(x_obs, y_obs, color='blue')
wavelenghts_list, fluxes_list, obs_wave_list, obs_flux_list, obs_error_list = extract_normalised_window_for_plots(theta=medians, config=config_dict, x_obs=x_obs, y_obs=y_obs, y_err=yerr)
for i in range(len(wavelenghts_list)):
    plt.plot(obs_wave_list[i], obs_flux_list[i], color="red", alpha=0.7)
    plt.plot(wavelenghts_list[i], fluxes_list[i], color="black")
    plt.fill_between(obs_wave_list[i], obs_flux_list[i] - obs_error_list[i], obs_flux_list[i] + obs_error_list[i], alpha=0.2)
# plt.legend()
plt.show()
exit(0)

######### file = ex_lupi_balmer_04052010.h5
# flat_samples = reader.get_chain(discard=00)
# niter, nwalkers, nparams = flat_samples.shape
# print(flat_samples.shape)
# # exit(0)
# labels_exlupi_2010 = ['m', 'log_m_dot', 'b', 'cos_inclination', "t_0", "t_slab", "log_n_e", "tau"]
# filter_samples = np.zeros((niter, nwalkers, len(labels_exlupi_2010)))
# filter_samples[:, :, :] = flat_samples[:, :, 0:len(labels_exlupi_2010)]
# # filter_samples[:, :, 3] = np.arccos(filter_samples[:, :, 3] /10 ) * 180/np.pi
# plot_trace(flat_for_trace=filter_samples, labels=labels_exlupi_2010)
# plot_corner(samples=filter_samples, labels=labels_exlupi_2010, save_name="")
# flatten_samples = filter_samples.reshape((-1, len(labels_exlupi_2010)))
#
# exit(0)
# print(flatten_samples.shape)
# medians = np.median(flat_samples.reshape((-1, nparams)), axis=0)
#
# small_medians = np.median(flatten_samples, axis=0)
# print(small_medians)
# # small_medians[8] = 1
# # wave_model, total_flux, flux_photon_count = mc_file.model_spec_window(small_medians, config_dict)
# # plt.plot(wave_model, total_flux)
# # plt.show()
# # exit(0)
# path_to_valid = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
# # loading data for V960 Mon
# # data = np.load(f"{path_to_valid}/stitched_HIRES_data_V960.npy")
# data = np.load(f"{path_to_valid}/data_hbc722_ccd2.npy")
# data[0] = mc_file.rad_vel_correction(data[0]*u.AA, -10.0 * u.km / u.s)
# x_obs = data[0]
# y_obs = data[1]/np.median(data[1])
# yerr = data[2]/np.median(data[1])
# # plt.plot(x_obs, y_obs, color='blue')
# wavelenghts_list, fluxes_list, obs_wave_list, obs_flux_list, obs_error_list = extract_normalised_window_for_plots(theta=medians, config=config_dict, x_obs=x_obs, y_obs=y_obs, y_err=yerr)
# for i in range(len(wavelenghts_list)):
#     plt.plot(obs_wave_list[i], obs_flux_list[i], color="red", alpha=0.7)
#     plt.plot(wavelenghts_list[i], fluxes_list[i], color="black")
#     plt.fill_between(obs_wave_list[i], obs_flux_list[i] - obs_error_list[i], obs_flux_list[i] + obs_error_list[i], alpha=0.2)
# # plt.legend()
# plt.show()
# exit(0)



###### Lup713

flat_samples = reader.get_chain(discard=00)
niter, nwalkers, nparams = flat_samples.shape
print(flat_samples.shape)
# exit(0)
labels_lup713 = ['m', 'log_m_dot', 'b', 'cos_inclination', "t_slab", "log_n_e", "tau"]  # For Lup713
filter_samples = np.zeros((niter, nwalkers, len(labels_lup713)))
filter_samples[:, :, :] = flat_samples[:, :, 0:len(labels_lup713)]
# filter_samples[:, :, 3] = np.arccos(filter_samples[:, :, 3] /10 ) * 180/np.pi
plot_trace(flat_for_trace=filter_samples, labels=labels_lup713)
plot_corner(samples=filter_samples, labels=labels_lup713, save_name="")
