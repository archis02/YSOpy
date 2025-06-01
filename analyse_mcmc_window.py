import numpy as np
import matplotlib.pyplot as plt
import emcee
from ysopy import utils
import corner
import astropy.units as u
from configparser import ConfigParser



# tslab_1000, log10_ne, tau_10, log_f = theta


nwalkers = 70
mcmc_iter = 100
n_params = 5

params = ['m', 'log_m_dot', 'b', 'inclination', 't_slab']



# generate initial conditions
config_dict = utils.config_read_bare("ysopy/config_file.cfg")
saveloc = "/Users/tusharkantidas/github/archis/Buffer/store_mcmc_files/"
# filename = 'mcmc_total_spec.h5'
snr = 100
filename = f"mcmc_total_spec_{snr}.h5"
filename = saveloc + filename
reader = emcee.backends.HDFBackend(filename)
flat_samples = reader.get_chain(discard=0)
# tau = reader.get_autocorr_time()
# print(tau)
# print(flat_samples.shape)
# lb = np.array([4.0, 4.224, 1, 17.2, 7, -1e-3, 0.7])
# ub = np.array([4.5, 4.25, 3, 20, 10.5, 1e-3, 1.5])
# cond1 = np.all(flat_samples<ub, axis=-1)
# cond2 = np.all(flat_samples>lb, axis=-1)
# cond = np.logical_and(cond1, cond2)
# print(cond.shape)
# # print(cond.shape)
# flat_for_trace = flat_samples
# flat_samples = flat_samples[cond]
# print(flat_samples.shape)


n_windows = len(config_dict['windows'])
poly_order = config_dict['poly_order']
ndim = n_params + n_windows * (poly_order + 1)
def plot_trace():
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    labels = ['m', 'log_m_dot', 'b', 'inclination', 't_slab', "a", "b"]#, "c", "d"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(flat_samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(flat_samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    plt.savefig(f"../Buffer/plot_directory/mcmc_total_model_trace.pdf")
    plt.show()


def plot_corner():
    labels = ['m', 'log_m_dot', 'b', 'inclination', 't_slab', "a", "b"]#, "c", "d"]

    fig = corner.corner(
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
        # bins=[100, 100, 100],
        # range=range_corner
    )
    # plt.suptitle(f"{filename}\n\n\n")
    # plt.savefig(f"../Buffer/plot_directory/mcmc_total_model_corner.pdf")
    plt.show()


plot_trace()
plot_corner()
