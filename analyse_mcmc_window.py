import numpy as np
import matplotlib.pyplot as plt
import emcee
from ysopy import utils
import corner
import astropy.units as u
from configparser import ConfigParser
import mcmc_window as mc_file

# tslab_1000, log10_ne, tau_10, log_f = theta


# nwalkers = 70
# mcmc_iter = 100
# n_params = 5
# params = ['m', 'log_m_dot', 'b', 'inclination', 't_slab', "log_n_e", "tau"]
params = ['m', 'log_m_dot', 'b', 'inclination']#, 't_slab', "log_n_e", "tau", "av"]

# generate initial conditions
config_dict = utils.config_read_bare("ysopy/config_file.cfg")
saveloc = "/Users/tusharkantidas/github/archis/Buffer/store_mcmc_files/"
# filename = 'mcmc_total_spec.h5'
snr = 100
# filename = f"mcmc_total_spec_{snr}.h5"
# filename = f'mcmc_total_spec_{snr}_gaussian_noise.h5'
# filename = f"mcmc_total_spec_100_low_acc_gaussian_noise.h5"
# filename = f"mcmc_total_spec_100_low_acc_poly_order_2.h5"

filename = f"mcmc_total_spec_100_balmer_jump.h5"
# for v960 Mon file
filename = "v960_stitched_all_windows.h5"
filename = "v960_mon_changed_inclination_all_windows.h5"
# for Ex-Lupi File
# filename = "ex_lupi.h5"
# filename = "ex_lupi_balmer.h5"
filename = saveloc + filename
reader = emcee.backends.HDFBackend(filename)
# flat_samples = reader.get_chain(discard=5200)
# config_dict["poly_order"] = 2
flat_samples = reader.get_chain(discard=0)
niter, nwalkers, nparams = flat_samples.shape
flat_for_params = np.zeros((niter, nwalkers, 8))
flat_for_params[:, :, 0] = flat_samples[:, :, 0]
flat_for_params[:, :, 1] = flat_samples[:, :, 1]
flat_for_params[:, :, 2] = flat_samples[:, :, 2]
flat_for_params[:, :, 3] = np.arccos(flat_samples[:, :, 3]/10) * 180 / np.pi
flat_for_params[:, :, 4] = flat_samples[:, :, 32]
flat_for_params[:, :, 5] = flat_samples[:, :, 33]
flat_for_params[:, :, 6] = flat_samples[:, :, 34]
flat_for_params[:, :, 7] = flat_samples[:, :, 35]


# exit(0)
# tau = reader.get_autocorr_time()
# print(tau)
# print(flat_samples.shape)
lb = np.array([4.45, 4.224, 1, 17.2, 7, -1e-3, 0.7])
ub = np.array([4.55, 4.25, 3, 20, 10.5, 1e-3, 1.5])

# lb = np.array([4.2, 4.47, 1.2, 18, 7.2, -2e-4, 0.9990])
# ub = np.array([9.1, 4.60, 2.8, 21, 10.9, 5e-4, 1.0010])
# lb = np.array([4.2, 4.47, 1.2, 18, 7.2, -1e-3, 0.9990])
# ub = np.array([9.1, 4.9, 2.8, 21, 10.9, 5e-4, 1.0010])
#
# cond1 = np.all(flat_samples < ub, axis=-1)
# cond2 = np.all(flat_samples > lb, axis=-1)
# cond = np.logical_and(cond1, cond2)
# print(cond.shape)
# # print(cond.shape)
flat_for_trace = flat_samples
# flat_samples = flat_samples[cond]
print(flat_samples.shape)

n_windows = len(config_dict['windows'])
poly_order = config_dict['poly_order']

labels = ['m', 'log_m_dot', 'B', 'inclination', 't_slab', "log_n_e", "tau", "a", "b", "c"]  # , "c", "d"]
labels = ['m', 'log_m_dot', 'B', 'inclination', 't_slab',"log_n_e", "tau", "av", "a", "b"]  #, "c", "d"]
labels_big = ['m', 'log_m_dot', 'B', 'inclination',
          "1", "2", "3", "4", "1", "2", "3", "4",
          "1", "2", "3", "4", "1", "2", "3", "4",
          "1", "2", "3", "4", "1", "2", "3", "4",
          "1", "2", "3", "4", "1", "2", "3", "4"]
labels = ['m', 'log_m_dot', 'B', 'inclination', "a1", "b1", "a2", "b2"]
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
    # plt.savefig(f"../Buffer/plot_directory/mcmc_total_model_trace.pdf")
    plt.show()


def plot_corner(flat_samples, labels):
    fig = corner.corner(
        # flat_for_trace,
        flat_samples,
        figsize=(7, 5),
        labels=labels,
        # truths=[9, 13, 30],
        plot_contours=True,
        quantiles=[0.16, 0.5, 0.84],
        # quantiles=[0.035, 0.5, 0.975],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        # smooth=True,
        # bins=[100, 100, 100, 100, 100, 100, 100],
        # range=range_corner
    )
    # plt.suptitle(f"{filename}\n\n\n")
    # plt.savefig(f"../Buffer/plot_directory/mcmc_total_model_corner.pdf")
    plt.show()




# exit(0)

# labels = ['m', 'log_m_dot', 'B', 'inclination', 't_slab', "log_n_e", "tau", "a", "b", "c"]
labels = ['m', 'log_m_dot', 'B', 'inclination', 't_slab', "log_n_e", "tau", "av", "a", "b"]
labels_big = ['m', 'log_m_dot', 'B', 'inclination',
          "1", "2", "3", "4", "1", "2", "3", "4",
          "1", "2", "3", "4", "1", "2", "3", "4",
          "1", "2", "3", "4", "1", "2", "3", "4",
          "1", "2", "3", "4", "1", "2", "3", "4"]
labels_v960 = ['m', 'log_m_dot', 'B', 'inclination', "a1", "b1", "a2", "b2"]
labels_exlupi =['m', 'log_m_dot', 'B', 'inclination', 't_slab', "log_n_e", "tau", "av", "a1"]#, "b1", "a2", "b2"]

# # For V 960 Mon
# plot_trace(flat_for_trace, labels=labels_big)
plot_trace(flat_for_trace=flat_for_params, labels=labels_v960)
# plot_corner(flat_samples, labels_big)
plot_corner(flat_for_params, labels=labels_v960)
exit(0)
# for Ex Lupi plotting
plot_trace(flat_for_trace, labels=labels_exlupi)
plot_corner(flat_samples, labels_exlupi)
exit(0)

def extract_normalised_window_for_plots(theta, config, x_obs, y_obs, y_err):
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

####### For V960 Mon
flat_samples = flat_samples.reshape((-1, len(labels_big)))
means = np.median(flat_samples, axis=0)
print(means)

path_to_valid = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
# loading data for V960 Mon
# data = np.load(f"{path_to_valid}/stitched_HIRES_data_V960.npy")
data = np.load(f"{path_to_valid}/stitched_HIRES_data_V960.npy")
data[0] = mc_file.rad_vel_correction(data[0]*u.AA, 40.3 * u.km / u.s)
x_obs = data[0]
y_obs = data[1]/np.median(data[1])
yerr = data[2]/np.median(data[1])
wavelenghts_list, fluxes_list, obs_wave_list, obs_flux_list, obs_error_list = extract_normalised_window_for_plots(theta=means, config=config_dict, x_obs=x_obs, y_obs=y_obs, y_err=yerr)
for i in range(len(wavelenghts_list)):
    plt.plot(obs_wave_list[i], obs_flux_list[i], color="red", alpha=0.7)
    plt.plot(wavelenghts_list[i], fluxes_list[i], color="black")
    plt.fill_between(obs_wave_list[i], obs_flux_list[i] - obs_error_list[i], obs_flux_list[i] + obs_error_list[i], alpha=0.2)
plt.legend()
plt.show()
exit(0)
########

####### For Ex-Lupi
flat_samples = flat_samples.reshape((-1, len(labels_exlupi)))
means = np.mean(flat_samples, axis=0)
print(means)
wave_model, total_flux, flux_photon_count = mc_file.model_spec_window(means, config_dict)
path_to_valid = "/Users/tusharkantidas/github/archis/Buffer/store_spectra"
# loading data for V960 Mon
# data = np.load(f"{path_to_valid}/stitched_HIRES_data_V960.npy")
data = np.load(f"{path_to_valid}/data_ex_lupi.npy")

x_obs = data[0]
y_obs = data[1]/np.median(data[1])
yerr = data[2]/np.median(data[1])
wavelenghts_list, fluxes_list, obs_wave_list, obs_flux_list, obs_error_list = extract_normalised_window_for_plots(theta=means, config=config_dict, x_obs=x_obs, y_obs=y_obs, y_err=yerr)
for i in range(len(wavelenghts_list)):
    plt.plot(obs_wave_list[i], obs_flux_list[i], color="red", alpha=0.7)
    plt.plot(wavelenghts_list[i], fluxes_list[i], color="black")
    plt.fill_between(obs_wave_list[i], obs_flux_list[i] - obs_error_list[i], obs_flux_list[i] + obs_error_list[i], alpha=0.2)
plt.legend()
plt.show()
exit(0)
########
labels_trimmed = ['m', 'log_m_dot', 'b', 'inclination', 't_slab', "log_n_e", "tau"]
# plot_corner(trimmed_flats,labels=labels_trimmed)
# tau = reader.get_autocorr_time()
# print(tau)
# print(flat_samples.shape)



# print(std_arr)

def filter_bad_walks(samples: np.ndarray, lb: np.ndarray, ub: np.ndarray):
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
        within_bounds = np.all((step_params >= lb) & (step_params <= ub))
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


# lb = np.array([4.2, 4.45, 1.2, 18, 7.2, -1e-3, 0.9990])
# ub = np.array([9.1, 4.9, 2.8, 21, 10.9, 5e-4, 1.0010])

# for high mdot
# lb = np.array([4.2, 4.45, 1.2, 18, 7, -1e-2, 0.9])
# ub = np.array([9.1, 4.9, 2.8, 21, 11, 5e-2, 1.1])

# for low m dot
lb = np.array([8.8, 6.25, 1, 20.0, 7, 11, 0.5, -1e-3, -2e-2, 0.7])
ub = np.array([12, 6.75, 3, 30.0, 10, 16, 2.5, 1e-3, 5e-2, 1.5])

good_walks = filter_bad_walks(samples=flat_for_trace, lb=lb, ub=ub)
plot_corner(good_walks, labels)
good_walks = np.array(flat_for_trace.reshape((-1, ndim)))
filter_walks = filter_walkers_within_n_sigma(n=5, samples=good_walks)

# exit(0)
trimmed_flats = np.zeros((len(filter_walks), 7))
trimmed_flats[:] = filter_walks[:, 0:7]
print(trimmed_flats.shape)
print()
plot_corner(trimmed_flats, labels_trimmed)
