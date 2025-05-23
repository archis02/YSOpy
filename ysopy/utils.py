from configparser import ConfigParser
import numpy as np
import astropy.units as u
import astropy.constants as const
from functools import cache
import ast

@cache
def config_read(path):
    """Read data from config file and cast to expected data types

    Parameters
    ----------
    path : str
        path to the config file

    Returns
    ----------
    dict_config : dict
        dictionary containing the parameters of the system in the expected units
    """
    config = ConfigParser()
    config.read(path)
    config_data = config['Parameters']
    dict_config = dict(config_data)

    # convert to the required astropy units
    dict_config["l_min"] = float(dict_config["l_min"]) * u.AA
    dict_config["l_max"] = float(dict_config["l_max"]) * u.AA
    dict_config["n_data"] = int(dict_config["n_data"])

    dict_config["b"] = float(dict_config["b"]) * u.kilogauss
    dict_config["m"] = float(dict_config["m"]) * const.M_sun
    dict_config["m_dot"] = float(dict_config["m_dot"]) * const.M_sun / (1 * u.year).to(u.s)
    dict_config["r_star"] = float(dict_config["r_star"]) * const.R_sun
    dict_config["inclination"] = float(dict_config["inclination"]) * u.degree
    dict_config["n_disk"] = int(dict_config["n_disk"])
    dict_config["n_dust_disk"] = int(dict_config["n_dust_disk"])
    dict_config["d_star"] = float(dict_config["d_star"]) * const.pc
    dict_config["t_star"] = float(dict_config["t_star"]) * u.K
    dict_config["log_g_star"] = float(dict_config["log_g_star"])
    dict_config["t_0"] = float(dict_config["t_0"]) * u.K
    dict_config["av"] = float(dict_config["av"])
    dict_config["rv"] = float(dict_config["rv"])
    dict_config["l_0"] = float(dict_config["l_0"]) * u.AA
    dict_config["t_slab"] = float(dict_config["t_slab"]) * u.K
    dict_config["n_e"] = float(dict_config["n_e"]) * u.cm ** (-3)
    dict_config["tau"] = float(dict_config["tau"])
    dict_config["n_h"] = int(dict_config["n_h"])
    dict_config["l_l_slab"] = float(dict_config["l_l_slab"]) * u.AA
    dict_config["n_h_minus"] = int(dict_config["n_h_minus"])
    dict_config["poly_order"] = int(dict_config["poly_order"])
    
    for param in ["save", "save_each", "plot", "save_grid_data", "verbose"]:
        if dict_config[param] == "True":
            dict_config[param] = True
        elif dict_config[param] == "False":
            dict_config[param] = False

    if dict_config['save']:
        with open(f"{dict_config['save_loc']}/details.txt", 'a+') as f:
            f.write(str(dict_config))

    return dict_config

@cache
def config_read_bare(path):
    """Read data from config file and cast to expected data types
    Unit convention: 
    wavelengths - Angstrom
    flux - erg / (cm^2 s A)
    masses - kg
    distances - m
    temperatures - Kelvin

    Parameters
    ----------
    path : str
        path to the config file

    Returns
    ----------
    dict_config : dict
        dictionary containing the parameters of the system in the expected units
    """
    config = ConfigParser()
    config.read(path)
    config_data = config['Parameters']
    dict_config = dict(config_data)

    # convert to the required astropy units
    dict_config["l_min"] = float(dict_config["l_min"])
    dict_config["l_max"] = float(dict_config["l_max"])
    dict_config["n_data"] = int(dict_config["n_data"])

    dict_config["b"] = float(dict_config["b"]) # in kiloGauss
    dict_config["m"] = float(dict_config["m"]) * const.M_sun.value # kg
    dict_config["m_dot"] = float(dict_config["m_dot"]) * const.M_sun.value / 31557600.0  # converted year to seconds, unit: kg / s
    dict_config["r_star"] = float(dict_config["r_star"]) * const.R_sun.value # m
    dict_config["inclination"] = float(dict_config["inclination"]) * np.pi / 180.0 # radians
    dict_config["n_disk"] = int(dict_config["n_disk"])
    dict_config["n_dust_disk"] = int(dict_config["n_dust_disk"])
    dict_config["d_star"] = float(dict_config["d_star"]) * const.pc.value # m
    dict_config["t_star"] = float(dict_config["t_star"]) # K
    dict_config["log_g_star"] = float(dict_config["log_g_star"])
    dict_config["t_0"] = float(dict_config["t_0"])
    dict_config["av"] = float(dict_config["av"])
    dict_config["rv"] = float(dict_config["rv"])
    dict_config["l_0"] = float(dict_config["l_0"])
    dict_config["t_slab"] = float(dict_config["t_slab"]) * u.K # units retained for h-slab stuff for convenience
    dict_config["n_e"] = float(dict_config["n_e"]) * u.cm ** (-3)
    dict_config["tau"] = float(dict_config["tau"])
    dict_config["n_h"] = int(dict_config["n_h"])
    dict_config["l_l_slab"] = float(dict_config["l_l_slab"]) * u.AA
    dict_config["n_h_minus"] = int(dict_config["n_h_minus"])
    dict_config["poly_order"] = int(dict_config["poly_order"])
    dict_config['windows'] = ast.literal_eval(dict_config['windows'])

    for param in ["save", "save_each", "plot", "save_grid_data", "verbose"]:
        if dict_config[param] == "True":
            dict_config[param] = True
        elif dict_config[param] == "False":
            dict_config[param] = False

    if dict_config['save']:
        with open(f"{dict_config['save_loc']}/details.txt", 'a+') as f:
            f.write(str(dict_config))

    return dict_config