import numpy as np
import matplotlib.pyplot as plt
import emcee
from ysopy import utils
import corner

config = utils.config_read_bare("ysopy/config_file.cfg")
############# Result plotting

# tslab_1000, log10_ne, tau_10, log_f = theta

inital_guess = np.array([10, 12, 20])
snr = 50
nwalkers = 100
mcmc_iter = 5000
ndim = len(inital_guess)

filename = f"hslab_mcmc_walker_{nwalkers}_iter_{mcmc_iter}_snr_{snr}.h5"
# filename = f"hslab_mcmc_walker_{nwalkers}_iter_{mcmc_iter}.h5"

direct = "/Users/tusharkantidas/Downloads"
filename = f"{direct}/{filename}"
reader = emcee.backends.HDFBackend(filename)

flat_samples = reader.get_chain()
tau = reader.get_autocorr_time()
print(tau)
chain = reader.get_chain()[:, :, 0].T

plt.hist(chain.flatten(), 100)
plt.show()

exit(0)
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
labels = ["tslab_1000", "log10_ne", "tau_10"]
fig = corner.corner(
    flat_samples,
    labels=labels,
    truths=[8, 13, 10],
    # plot_contours=True,
    quantiles=[0.16, 0.5, 0.84],
    #
    # quantiles=[0.035, 0.5, 0.975],
    show_titles=True,
    title_kwargs={"fontsize": 12},
    # smooth=True
)
plt.show()