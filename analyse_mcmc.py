import numpy as np
import matplotlib.pyplot as plt
import emcee
from ysopy import utils
import corner
import astropy.units as u

config = utils.config_read_bare("ysopy/config_file.cfg")
############# Result plotting

# tslab_1000, log10_ne, tau_10, log_f = theta

inital_guess = np.array([10, 12, 20])
snr = 50
nwalkers = 30
mcmc_iter = 10000
ndim = len(inital_guess)

i, j, k = 0, 0, 0
# These are the params used to create the observed data
# Here they are used just to read the files
t_slab_arr = np.array([9000]) * u.K
log_ne_arr = np.array([13])
ne_arr = (10 ** log_ne_arr) * u.cm ** (-3)
tau_arr = np.array([3])
# filename = f"hslab_mcmc_walker_{nwalkers}_iter_{mcmc_iter}_snr_{snr}.h5"
# filename = f"hslab_mcmc_walker_{nwalkers}_iter_{mcmc_iter}.h5"
filename = f"hslab_mcmc_walker_{nwalkers}_iter_{mcmc_iter}_snr_{snr}_T{int((t_slab_arr[i]).value/1000)}_logne_{log_ne_arr[j]}_tau_{tau_arr[k]}_lmin_{int(config['l_min'])}_l_max_{int(config['l_max'])}.h5"


direct = "/Users/tusharkantidas/Downloads"
filename = f"{direct}/{filename}"
reader = emcee.backends.HDFBackend(filename)

flat_samples = reader.get_chain()
# tau = reader.get_autocorr_time()
# print(tau)
"""

burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
print("flat log prior shape: {0}".format(log_prior_samples.shape))

all_samples = np.concatenate(
    (samples, log_prob_samples[:, None], log_prior_samples[:, None]), axis=1
)

labels = list(map(r"$\theta_{{{0}}}$".format, range(1, ndim + 1)))
labels += ["log prob", "log prior"]
"""
# corner.corner(all_samples, labels=labels);
# exit(0)
# chain = reader.get_chain()[:, :, 0].T
#
# plt.hist(chain.flatten(), 100)
# plt.show()

# exit(0)
def plot_trace():
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    labels = ["tslab_1000", "log10_ne", "tau_10"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(flat_samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(flat_samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    plt.show()
def plot_corner():
    labels = ["tslab_1000", "log10_ne", "tau_10"]
    fig = corner.corner(
        flat_samples,
        labels=labels,
        truths=[9, 13, 30],
        # plot_contours=True,
        quantiles=[0.16, 0.5, 0.84],
        #
        # quantiles=[0.035, 0.5, 0.975],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        # smooth=True
    )
    plt.show()

plot_trace()
plot_corner()
