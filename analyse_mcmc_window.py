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

def config_reader(filepath):
    """
    Read the config file containing the bounds for each parameter, i.e. mcmc_config.cfg
    """
    parser = ConfigParser()
    parser.read(filepath)
    config_data = dict(parser['Parameters'])

    config_data['m_u'] = float(config_data['m_u'])
    config_data['m_l'] = float(config_data['m_l'])

    config_data['log_m_dot_u'] = float(config_data['log_m_dot_u'])
    config_data['log_m_dot_l'] = float(config_data['log_m_dot_l'])

    config_data['b_u'] = float(parser['Parameters']['b_u'])
    config_data['b_l'] = float(parser['Parameters']['b_l'])

    config_data['inclination_u'] = float(parser['Parameters']['inclination_u'])
    config_data['inclination_l'] = float(parser['Parameters']['inclination_l'])

    # config_data['t_0_u'] = float(parser['Parameters']['t_0_u'])
    # config_data['t_0_l'] = float(parser['Parameters']['t_0_l'])

    config_data['t_slab_u'] = float(parser['Parameters']['t_slab_u'])
    config_data['t_slab_l'] = float(parser['Parameters']['t_slab_l'])

    # config_data['log_n_e_u'] = float(parser['Parameters']['log_n_e_u'])
    # config_data['log_n_e_l'] = float(parser['Parameters']['log_n_e_l'])

    # config_data['tau_u'] = float(parser['Parameters']['tau_u'])
    # config_data['tau_l'] = float(parser['Parameters']['tau_l'])

    config_data['const_term_l'] = float(parser['Parameters']['const_term_l'])
    config_data['const_term_u'] = float(parser['Parameters']['const_term_u'])

    config_data['other_coeff_l'] = float(parser['Parameters']['other_coeff_l'])
    config_data['other_coeff_u'] = float(parser['Parameters']['other_coeff_u'])

    return config_data

# generate initial conditions
config_data_mcmc = config_reader('mcmc_config.cfg')
config_dict = utils.config_read_bare("ysopy/config_file.cfg")

filename = 'mcmc_total_spec.h5'

reader = emcee.backends.HDFBackend(filename)
flat_samples = reader.get_chain()

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
    # plt.savefig(f"{filename}_trace.pdf")
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
    # plt.savefig(f"{filename}_corner.pdf")
    plt.show()


plot_trace()
plot_corner()
