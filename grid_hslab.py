import ysopy.base_funcs as bf
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from ysopy import h_emission
from ysopy import h_minus_emission
import multiprocessing as mp
from astropy.modeling.physical_models import BlackBody
from scipy.integrate import trapezoid
import warnings
from functools import partial


from ysopy import utils
config = utils.config_read_bare("ysopy/config_file.cfg")
# print(config["t_slab"])
# print(config["n_e"])
# print(config["tau"])

t_slab_arr = np.linspace(7000, 12000, 10) * u.K
log_ne_arr = np.linspace(10, 16, 10)
ne_arr = (10 ** log_ne_arr) * u.cm ** (-3)
tau_arr = np.array([0.1, 0.2, 0.3, 0.5, 0.75, 1, 2, 3, 4, 5])


# t_slab_arr = np.array([8000]) * u.K
# log_ne_arr = np.array([13])
# ne_arr = (10 ** log_ne_arr) * u.cm ** (-3)
# tau_arr = np.array([1])
T_SLAB, NE, TAU = np.meshgrid(t_slab_arr, ne_arr, tau_arr)

wav_slab = np.logspace(np.log10(config['l_min']), np.log10(config['l_max']), config['n_h']) * u.AA
def wrap_h_slab(obs_flux, t_slab, n_e, tau):
    config["t_slab"] = t_slab
    config["n_e"] = n_e
    config["tau"] = tau

    dr, t_max, d, r_in, r_sub = bf.generate_temp_arr(config)

    h_flux = h_emission.get_h_intensity(config)
    h_minus_flux = h_minus_emission.get_h_minus_intensity(config)
    h_slab_flux = (h_flux + h_minus_flux) * u.sr
    h_slab_flux = h_slab_flux.value
    # Here we need not take the flux all the way upto the 1e6 AA. Rather till
    # l_max. Therefore we need not take the blackbody assumption beyond l_max.
    # That calculation is needed only if we are concerned with spectra.
    chi_sq = np.sum((h_slab_flux - obs_flux)**2)

    return chi_sq


# print(len(t_slab_arr), len(ne_arr), len(tau_arr))
# Use the below function if pool doesn't work

"""with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i in range(len(t_slab_arr)):
        print("******** i", i)
        for j in range(len(ne_arr)):
            print("*** j", j)
            for k in range(len(tau_arr)):
                h_slab_flux = wrap_h_slab(t_slab=t_slab_arr[i], n_e=ne_arr[j], tau=tau_arr[k])  # units erg/(AA s cm2)
                h_slab_flux = h_slab_flux.value
                np.save("obs_h_slab_flux.npy", h_slab_flux)"""




def parallel_grid_eval():
    obs_flux = np.load("obs_h_slab_flux.npy")
    grid_points = [(ts, ne, tau) for ts in t_slab_arr for ne in ne_arr for tau in tau_arr]
    with mp.Pool(mp.cpu_count()) as pool:
        func = partial(wrap_h_slab, obs_flux)
        chi2_flat = pool.starmap(func, grid_points)
    chi2_grid = np.array(chi2_flat).reshape(len(t_slab_arr), len(ne_arr), len(tau_arr))
    np.save("chi2_grid.npy", chi2_grid)
    return chi2_grid

#
# if __name__ == "__main__":
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         chi2_grid = parallel_grid_eval()

chi_sq_grid = np.load("chi2_grid.npy")
print(chi_sq_grid.shape)
plt.imshow(chi_sq_grid[5])
plt.colorbar()
plt.show()

