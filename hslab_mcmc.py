import numpy as np
from ysopy import h_emission
from ysopy import h_minus_emission
import ysopy.base_funcs as bf
import astropy.units as u
import matplotlib.pyplot as plt
from ysopy import utils
import warnings
import multiprocessing
import emcee


config = utils.config_read_bare("ysopy/config_file.cfg")
import corner

def wrap_h_slab(t_slab, n_e, tau):
    # print(t_slab)
    config["t_slab"] = t_slab
    config["n_e"] = n_e
    config["tau"] = tau

    dr, t_max, d, r_in, r_sub = bf.generate_temp_arr(config)

    h_flux = h_emission.get_h_intensity(config)
    h_minus_flux = h_minus_emission.get_h_minus_intensity(config)
    h_slab_flux = (h_flux + h_minus_flux) * u.sr
    h_slab_flux = h_slab_flux.value
    # Here we need not take the flux all the way upto the 1e6 AA. Rather till
    # l_max. Therefore, we need not take the blackbody assumption beyond l_max.
    # That calculation is needed only if we are concerned with spectra.
    return h_slab_flux

# theta arr = tslab, log10_ne, tau, log_f


def log_likelihood(theta, obs_flux, yerr=None):
    if yerr is None:
        yerr = np.zeros((len(obs_flux)))

    tslab_1000, log10_ne, tau_10 = theta
    # rescaling theta array
    tslab = 1e3*tslab_1000 * u.K
    ne = 10**log10_ne * u.cm**(-3)
    tau = tau_10/10

    h_slab_flux = wrap_h_slab(tslab, ne, tau)
    sigma2 = yerr**2
    return -0.5 * np.sum(((obs_flux - h_slab_flux) ** 2)/sigma2 + np.log(sigma2))

def log_prior(theta):
    tslab_1000, log10_ne,  tau_10 = theta
    if 7 < tslab_1000 < 12 and 11 < log10_ne < 16 and 1 < tau_10 < 50:# and 0 < log_f < 1:
        return 0.0
    return -np.inf

def log_probability(theta, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, y, yerr)

inital_guess = np.array([8.5, 14, 25])

nwalkers = 30

cpu_cores_used = multiprocessing.cpu_count() - 5

ndim = len(inital_guess)
pos = np.tile(inital_guess, (nwalkers, 1))
# print(pos)
# rand_matrix = np.random.randn(nwalkers, ndim)
rand_matrix = np.random.uniform(-1, 1, size=(nwalkers, ndim))
scale = np.array([1.2, 2, 20])
rand_matrix = np.multiply(rand_matrix, scale)

pos = pos + rand_matrix

# Choose the column index you want to mask (e.g., column 1)
col_index = 2

# Create a copy of the matrix to avoid changing the original
masked_matrix = pos.copy()

# Apply the mask: set negative values in the selected column to 0
masked_matrix[pos[:, col_index] < 0, col_index] = 0.1
print(masked_matrix)
pos = masked_matrix

# exit(0)
# pos = pos + 1 * np.random.randn(nwalkers, ndim)
# obs_flux = np.load("obs_h_slab_flux.npy")
snr = 50

# These are the params used to create the observed data
# Here they are used just to read the files
t_slab_arr = np.array([9000]) * u.K
log_ne_arr = np.array([13])
ne_arr = (10 ** log_ne_arr) * u.cm ** (-3)
tau_arr = np.array([3])


i, j, k = 0, 0, 0
# obs_flux = np.load(f"snr_{snr}_obs_h_slab_flux.npy")
obs_flux = np.load(f"snr_{snr}_obs_h_slab_flux_T{int((t_slab_arr[i]).value/1000)}_logne_{log_ne_arr[j]}_tau_{tau_arr[k]}_lmin_{int(config['l_min'])}_l_max_{int(config['l_max'])}.npy")
# yerr = np.zeros(len(obs_flux))

# yerr = np.load(f"snr_{snr}_noise.npy")
yerr = np.load(f"snr_{snr}_noise_T{int((t_slab_arr[i]).value/1000)}_logne_{log_ne_arr[j]}_tau_{tau_arr[k]}_lmin_{int(config['l_min'])}_l_max_{int(config['l_max'])}.npy")
# saving the chains
mcmc_iter = 25000
# filename = f"hslab_mcmc_walker_{nwalkers}_iter_{mcmc_iter}_snr_{snr}.h5"
filename = f"hslab_mcmc_walker_{nwalkers}_iter_{mcmc_iter}_snr_{snr}_T{int((t_slab_arr[i]).value/1000)}_logne_{log_ne_arr[j]}_tau_{tau_arr[k]}_lmin_{int(config['l_min'])}_l_max_{int(config['l_max'])}.h5"
filename = f"DE_Move_{filename}"
print(filename)
# exit(0)

backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)  # commented for rerun


"""
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(obs_flux, yerr), backend=backend
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sampler.run_mcmc(pos, mcmc_iter, progress=True, store=True);
"""
# mcmc_iter = 25000 # Only rerun
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with multiprocessing.get_context("fork").Pool(processes=cpu_cores_used) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(obs_flux, yerr), backend=backend, pool=pool, blobs_dtype=float,
            moves=[
                (emcee.moves.DEMove(), 0.8),
                (emcee.moves.DESnookerMove(), 0.2),
            ],
        )

        sampler.run_mcmc(pos, mcmc_iter, progress=True, store=True)
        # sampler.run_mcmc(None, mcmc_iter, progress=True, store=True)



