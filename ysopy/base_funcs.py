import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.integrate import trapezoid
import astropy.constants as const
import astropy.units as u
from astropy.io.votable import parse
from astropy.modeling.physical_models import BlackBody
from dust_extinction.parameter_averages import F19
from dust_extinction.averages import G21_MWAvg
from pypeit.core import wave

from . import utils
from . import h_emission
from . import h_minus_emission

import argparse
import time
from functools import cache
import logging
import warnings
import sys
import cProfile


def calculate_n_data(config):

    # the chosen temperature and logg is the BT Settl model that had highest sampling.
    # It is recommended to check your photosphere models and choose accordingly.
    temperature = 16
    logg = 3.5

    loc = config['bt_settl_path']
    l_min = config['l_min']
    l_max = config['l_max']

    if temperature >= 100:
        address = f"{loc}/lte{temperature}-{logg}-0.0a+0.0.BT-Settl.7.dat.npy"
    elif (temperature > 25) and temperature < 100:
        address = f"{loc}/lte0{temperature}-{logg}-0.0a+0.0.BT-Settl.7.dat.npy"
    elif (temperature >= 20) and temperature < 100:
        address = f"{loc}/lte0{temperature}-{logg}-0.0.BT-Settl.7.dat.npy"
    else:
        address = f"{loc}/lte0{temperature}-{logg}-0.0.BT-Settl.7.dat.npy"
    data = np.load(address)
    
    l_pad = 50 # excess padding to ensure that sampling is high enough

    l_bound = np.searchsorted(data[0],l_min - 1.5 * l_pad)
    u_bound = np.searchsorted(data[0],l_max + 1.5 * l_pad)
    trimmed_wave = data[0][l_bound:u_bound]

    # overwrite the config dictionary
    config['n_data'] = trimmed_wave.shape[0]

@cache
def load_npy_file(address):
    return np.load(address)

def read_bt_settl_npy(config, temperature: int, logg: float, r_in=None, ret_full_wavelength_ax=False):
    """Read the stored BT-Settl model spectra from locally stored .npy files. This is much faster than
    reading .xml files 

    Parameters
    ----------
    config : dict
        dictionary containing system parameters

    temperature : int
        t // 100, where t is the t_eff of the model in Kelvin

    logg : float
        log of surface gravity of the atmosphere model

    r_in : float
        if supplied, calculates the padding required, default padding is 20 A

    ret_full_wavelength : bool
        controls whether the output will be the full wavelength range

    Returns
    ----------
    trimmed_wave : numpy.ndarray
        array having the wavelength axis of the read BT-Settl data, in units of Angstrom,
        a padding of 20 A is left to avoid errors in interpolation, wavelengths in air

    trimmed_flux : numpy.ndarray
        array of flux values, in units of erg / (cm^2 s A)
    """

    loc = config['bt_settl_path']
    l_min = config['l_min'] 
    l_max = config['l_max']
    n_data = config['n_data']

    if ret_full_wavelength_ax:
        l_min = 1000.0
        l_max = 50000.0
    
    if temperature >= 100:
        address = f"{loc}/lte{temperature}-{logg}-0.0a+0.0.BT-Settl.7.dat.npy"
    elif (temperature > 25) and temperature < 100:
        address = f"{loc}/lte0{temperature}-{logg}-0.0a+0.0.BT-Settl.7.dat.npy"
    elif (temperature >= 20) and temperature < 100:
        address = f"{loc}/lte0{temperature}-{logg}-0.0.BT-Settl.7.dat.npy"
    else:
        address = f"{loc}/lte0{temperature}-{logg}-0.0.BT-Settl.7.dat.npy"
    
    data = load_npy_file(address)

    # convert to air, for trimming
    l_min_air = wave.vactoair(l_min * u.AA).value
    l_max_air = wave.vactoair(l_max * u.AA).value

    # trim data to region of interest, extra region left for re-interpolation
    l_pad = 20
    if r_in is not None:
        v_max = np.sqrt(const.G.value * config['m'] / r_in) * np.sin(config['inclination']) / const.c.value
        l_pad = l_max_air * v_max

    if ret_full_wavelength_ax:
        l_min = 1000.0
        l_max = 50000.0
    
    l_bound = np.searchsorted(data[0],l_min_air - 1.5 * l_pad)
    u_bound = np.searchsorted(data[0],l_max_air + 1.5 * l_pad)
    trimmed_wave = data[0][l_bound:u_bound] #* u.AA
    trimmed_flux = data[1][l_bound:u_bound] #* (u.erg / (u.cm * u.cm * u.s * u.AA))

    # for low-resolution data, make a linear re-interpolation
    if 20 <= temperature <= 25:
        x, y = unif_reinterpolate(config, trimmed_wave, trimmed_flux, l_pad)
        trimmed_wave = np.linspace(l_min_air - l_pad, l_max_air + l_pad, n_data, endpoint=True) #* u.AA
        trimmed_flux = np.interp(trimmed_wave, x, y) #* u.erg / (u.cm * u.cm * u.s * u.AA)

    return trimmed_wave, trimmed_flux


def read_bt_settl(config, temperature: int, logg: float, r_in=None, ret_full_wavelength_ax=False):
    """read the stored BT-Settl model spectra from VOtable format, for the supplied temperature and logg values

    Parameters
    ----------
    config : dict
        dictionary containing system parameters

    temperature : int
        t // 100, where t is the t_eff of the model in Kelvin

    logg : float
        log of surface gravity of the atmosphere model

    r_in : float
        if supplied, calculates the padding required, default padding is 20 A

    ret_full_wavelength_ax : bool
        controls whether the output will be the full wavelength range

    Returns
    ----------
    trimmed_wave : astropy.units.Quantity
        array having the wavelength axis of the read BT-Settl data, in units of Angstrom,
        a padding of 20 A is left to avoid errors in interpolation

    trimmed_flux : astropy.units.Quantity
        array of flux values, in units of erg / (cm^2 s A)
    """

    loc = config['bt_settl_path']
    m = config['m']
    inclination = config['inclination']
    l_min = config['l_min']
    l_max = config['l_max']
    n_data = config['n_data']

    if temperature >= 100:
        address = f"{loc}/lte{temperature}-{logg}-0.0a+0.0.BT-Settl.7.dat.xml"
    elif (temperature > 25) and temperature < 100:
        address = f"{loc}/lte0{temperature}-{logg}-0.0a+0.0.BT-Settl.7.dat.xml"
    elif (temperature >= 20) and temperature < 100:
        address = f"{loc}/lte0{temperature}-{logg}-0.0.BT-Settl.7.dat.xml"
    else:
        address = f"{loc}/lte0{temperature}-{logg}-0.0.BT-Settl.7.dat.xml"
    table = parse(address)
    data = table.get_first_table().array

    l_pad = 20
    if r_in is not None:
        v_max = np.sqrt(const.G.value * m.value / r_in) * np.sin(inclination) / const.c.value
        l_pad = l_max * v_max

    if ret_full_wavelength_ax:
        l_min = 1000.0
        l_max = 50000.0

    l_bound = np.searchsorted(data[0],l_min - 1.5 * l_pad)
    u_bound = np.searchsorted(data[0],l_max + 1.5 * l_pad)
    trimmed_wave = data[0][l_bound:u_bound] #* u.AA
    trimmed_flux = data[1][l_bound:u_bound] #* (u.erg / (u.cm * u.cm * u.s * u.AA))

    # trimmed_data = np.extract(data['WAVELENGTH'] > l_min.value - 1.5 * l_pad, data)
    # trimmed_data = np.extract(trimmed_data['WAVELENGTH'] < l_max.value + 1.5 * l_pad, trimmed_data)
    # trimmed_wave = trimmed_data['WAVELENGTH'].astype(np.float64) * u.AA
    # trimmed_flux = trimmed_data['FLUX'].astype(np.float64) * (u.erg / (u.cm * u.cm * u.s * u.AA))

    # for faulty data make a linear re-interpolation
    if 20 <= temperature <= 25:
        x, y = unif_reinterpolate(config, trimmed_wave, trimmed_flux, l_pad)
        trimmed_wave = np.linspace(l_min.value - l_pad, l_max.value + l_pad, n_data, endpoint=True) #* u.AA
        trimmed_flux = np.interp(trimmed_wave, x, y) #* u.erg / (u.cm * u.cm * u.s * u.AA)

    return trimmed_wave, trimmed_flux


def unif_reinterpolate(config, x, y, l_pad):
    """interpolate the datasets having very low sampling, primarily internal use

    Parameters
    ----------
    config : dict
        dictionary containing system parameters

    x : astropy.units.Quantity
        array of wavelength values, in units of Angstrom
    
    y : astropy.units.Quantity
        array of flux values, in units of erg / (cm^2 s A)
    
    l_pad : float
        padding in wavelength axis (in Angstrom)
    
    Returns
    ----------
    wav : astropy.units.Quantity
        new wavelength axis over which the flux values are interpolated
    
    f(wav) : astropy.units.Quantity
        interpolated flux values, in units of erg / (cm^2 s A)
    """
    l_min = config['l_min']
    l_max = config['l_max']
    n_data = config['n_data']

    wav = np.linspace(l_min - 1.2 * l_pad, l_max + 1.2 * l_pad, n_data, endpoint=True) #* u.AA
    return wav, np.interp(wav, x, y) #* u.erg / (u.cm * u.cm * u.s * u.AA))


def interpolate_conv(config, wavelength, flux, v_red, sampling=100):
    """Interpolate the given data to a logarithmic scale in the wavelength axis,
    to account for a variable kernel during convolution. Also determines the length of the kernel.

    Parameters
    ----------
    config : dict
        dictionary containing system parameters

    wavelength : numpy.ndarray
        full array of wavelength values, in units of Angstrom

    flux : numpy.ndarray
        array of flux values, in units of erg / (cm^2 s A)

    v_red : float
        reduced velocity of the annulus along the line of sight, i.e. v_kep * sin(i) / c
    
    sampling : int
        desired approximate number of points in the range (l0 - l_max, l0 + l_max)

    Returns
    ----------
    kernel_length : int 
        number of points in the kernel
    
    wave_log_trimmed : numpy.ndarray
        array of wavelengths in logarithmic, unit Angstrom
    
    flux_interpolated : numpy.ndarray
        array of flux values corresponding to wave_log_trimmed, unit: erg / (cm^2 s A)
    """

    l0 = config['l_0']
    l_min = config['l_min']
    l_max = config['l_max']

    # determine the number of points in interpolated axis
    # spacing of points in interpolated axis must match that of kernel
    k = (1 + v_red) / (1 - v_red)
    n_points = sampling / np.log10(k) * np.log10(wavelength[-1] / wavelength[0])

    wavelength_log = np.logspace(np.log10(wavelength[0]), np.log10(wavelength[-1]), int(n_points))
    l_lower = np.searchsorted(wavelength_log, l_min - 5)
    l_upper = np.searchsorted(wavelength_log, l_max + 5)
    wave_log_trimmed = wavelength_log[l_lower:l_upper]
    
    # interpolate to the desired wavelengths
    flux_interpolated = np.interp(wave_log_trimmed, wavelength, flux)

    # determine the exact number of points to be taken in kernel
    upper = np.searchsorted(wavelength_log, (l0 * (1 + v_red)))
    lower = np.searchsorted(wavelength_log, (l0 * (1 - v_red)))
    l_around = upper-lower

    kernel_length = (l_around // 2) * 2 + 1  # odd number to ensure  symmetry

    return kernel_length, wave_log_trimmed, flux_interpolated


def logspace_reinterp(config, wavelength, flux):
    """interpolates the given wavelength-flux data and interpolates to a logarithmic axis in the wavelength,
    used to convert all the SEDs to a common wavelength axis, so that they can be added

    Parameters
    ----------
    config : dict
        dictionary containing system parameters

    wavelength : numpy.ndarray
        wavelength array

    flux : numpy.ndarray
        flux array to be interpolated

    Returns
    ----------
    wavelength_req : numpy.ndarray
        new wavelength axis

    flux_final : numpy.ndarray
        interpolated flux array
    """
    l_min = config['l_min']
    l_max = config['l_max']
    n_data = config['n_data']

    wavelength_req = np.logspace(np.log10(l_min), np.log10(l_max), n_data)
    flux_final = np.interp(wavelength_req, wavelength, flux) #* u.erg / (u.cm * u.cm * u.s * u.AA)

    return wavelength_req, flux_final


def ker(x, l_0, l_max):
    """Defines the kernel for the convolution. A kernel for a rotating ring is taken

    Parameters
    ----------
    x : float
        value at which kernel function is to be evaluated

    l_0 : float
        central wavelength around which kernel function is evaluated

    l_max : float
        maximum deviation from l_0 up to which the kernel function is well-defined"""
    return 1 / np.sqrt(1 - ((x - l_0) / l_max) ** 2)


def generate_kernel(config: dict, sampling: int, v_red):
    """generates the kernel in the form of an array,
    to be used for convolution of the flux in subsequent steps.

    Parameters
    ----------
    config : dict
        dictionary containing system parameters

    sampling : int
        Number of points in the kernel array

    v_red : float
        Ratio of l_max to l_0, which is equal to the reduced velocity, i.e., v_kep * sin(i) / c

    Returns
    ----------
        kernel_arr : numpy.ndarray
                     numpy array of the kernel
    """
    l0 = config['l_0']

    # since data is uniformly sampled in log wavelength space, kernel has to be done similarly
    log_ax = np.logspace(np.log10(l0 * (1 - v_red)), np.log10(l0 * (1 + v_red)), sampling, endpoint=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernel_arr = ma.masked_invalid(ker(log_ax, l0, l0 * v_red))
    kernel_arr = ma.filled(kernel_arr, 0)
    # log_ax = np.delete(log_ax,0)
    kernel_arr = np.delete(kernel_arr, 0)

    # normalize kernel
    norm = np.sum(kernel_arr)
    kernel_arr = kernel_arr / norm
    return kernel_arr


def temp_visc(r, r_in, m, m_dot):
    """Define the temperature profile for the viscously heated disk

    Parameters
    ----------
    r : numpy.ndarray
        value of the radius at which the temperature is to be calculated, this is required to be a numpy array

    r_in : float
        inner truncation radius of the viscously heated disk

    Returns
    ----------
    t : float
        temperature at the given radius
    """
    # m = config['m']
    # m_dot = config['m_dot']
    # if r > 49.0 / 36.0 * r_in:
    #     t = ((3 * const.G.value * m * m_dot) * (1 - np.sqrt(r_in / r)) / (8 * np.pi * const.sigma_sb * r ** 3)) ** 0.25
    # else:
    #     t = ((3 * const.G.value * m * m_dot) * (1 / 7) / (8 * np.pi * const.sigma_sb * (49 / 36 * r_in) ** 3)) ** 0.25
    t = ((3 * const.G.value * m * m_dot) * (1 - np.sqrt(r_in / r)) / (8 * np.pi * const.sigma_sb.value * r ** 3)) ** 0.25
    t[np.where(r<=49.0/36.0 * r_in)] = ((3 * const.G.value * m * m_dot) * (1 / 7) / (8 * np.pi * const.sigma_sb.value * (49 / 36 * r_in) ** 3)) ** 0.25
    
    return t


def generate_temp_arr(config):
    """Calculate r_in, generate the temperature vs radius arrays, and bundle into a dictionary
    
    Parameters
    ----------
    config : dict
        dictionary containing system parameters

    Returns
    ----------
    dr : float
        thickness of each annulus

    t_max : int
        maximum value of the temperature // 100 which is required to be called from BT-Settl database

    d : dict
        dictionary having radii of the annuli as keys and the temperature rounded to the nearest
        available BT-Settl spectra as the values

    r_in : float
        inner truncation radius of the viscously heated disk

    r_sub : float
        radius at which t_visc = 1400 K, formal boundary of the viscously heated disk
    """

    m_sun_yr = const.M_sun.value / 31557600.0  # solar mass per yr
    plot = config['plot']
    m = config['m']
    r_star = config['r_star']
    m_dot = config['m_dot']
    n_disk = config['n_disk']
    b = config["b"]

    r_in = 7.186 * b ** (4.0 / 7.0) * (r_star / (2 * const.R_sun.value)) ** (5.0 / 7.0) / (
            (m_dot / (1e-8 * m_sun_yr)) ** (2.0 / 7.0) * (m / (0.5 * const.M_sun.value)) ** (1.0 / 7.0)) * r_star    ##### ask the 7.186??????????????????
    
    r_in = r_in / 2.0  # correction factor taken 0.5, ref Long, Romanova, Lovelace 2005
    r_in = max([r_in, r_star])

    # estimate r_sub
    r_sub_approx = ((3 * const.G.value * m * m_dot) / (8 * np.pi * const.sigma_sb.value * 1400.0 ** 4)) ** (1.0 / 3.0)
    r_visc = np.linspace(r_in, r_sub_approx, n_disk)
    t_visc = temp_visc(r_visc, r_in, m, m_dot)

    # truncate at T < 1400 K, i.e. sublimation temperature of the dust
    t_visc = ma.masked_less(t_visc, 1400.0)
    r_visc = ma.masked_where(ma.getmask(t_visc), r_visc)
    t_visc = ma.compressed(t_visc)
    r_visc = ma.compressed(r_visc)
    d = {}

    # Change to nearest available temperature in BT Settl data
    for i in range(len(r_visc)):
        t_int = int(np.round(t_visc[i] / 100))
        if t_int < 71:
            d[r_visc[i]] = int(np.round(t_visc[i] / 100))
        elif 120 >= t_int > 70 and t_int % 2 == 1:
            d[r_visc[i]] = int(np.round(t_visc[i] / 200)) * 2
        elif 120 < t_int:
            d[r_visc[i]] = int(np.round(t_visc[i] / 500)) * 5

    if len(t_visc) == 0: # no viscously heated component
        r_sub = r_in
        t_max = 14
        dr = None
    else:
        t_max = int(max(d.values()))
        r_sub = r_visc[-1]
        dr = r_visc[1] - r_visc[0]

    # if config['save']:
    #     np.save("radius_arr.npy", r_visc)
    #     np.save("temp_arr.npy", t_visc)

    if plot:
        plt.plot(r_visc / const.R_sun.value, t_visc)
        plt.ylabel('Temperature [Kelvin]')
        plt.xlabel(r'Radius $R_{\odot}$')
        plt.show()

    return dr, t_max, d, r_in, r_sub


def generate_temp_arr_planet(config, mass_p, dist_p, d):
    # mass and distance of planet fix such that it is within the viscous disk
    r_visc = np.array([radius for radius, t in d.items()]) * u.m
    t_visc = np.array([t for radius, t in d.items()]) * u.Kelvin
    mass_plnt = mass_p * u.jupiterMass
    dist_plnt = dist_p * u.AU
    m = config["m"]
    # Distance of star to L1 from star
    print("********************************************************************************")
    print(mass_plnt.unit)
    print(m.unit)
    low_plnt_lim = dist_plnt * (1 - np.sqrt(mass_plnt / (3 * m)))
    up_plnt_lim = dist_plnt * (1 + np.sqrt(mass_plnt / (3 * m)))
    print("Check this ******************")
    print('Position of L1: ', low_plnt_lim.to(u.AU))
    print('Position of L2: ', up_plnt_lim.to(u.AU))
    print('Last element of radius array: ', r_visc.to(u.AU)[-1])
    print("*********************")
    print(f"planet's influence: {low_plnt_lim} to {up_plnt_lim}\n")
    r_new = []
    for r in r_visc:
        if r > low_plnt_lim:
            if r < up_plnt_lim:
                r_new.append(r.to(u.AU).value)
    r_new = np.array(r_new) * u.AU

    terms = np.where(low_plnt_lim < r_visc)
    terms2 = np.where(up_plnt_lim > r_visc)
    terms_act = []

    for i in terms[0]:
        for j in terms2[0]:
            if i == j:
                terms_act.append(i)
    # removing this radius from r_visc
    for r in r_visc:
        if r in r_new:
            r_visc = np.delete(r_visc, np.where(r_visc == r))
    # print(r_visc.to(u.AU))
    # print(temp_arr, len(temp_arr))
    # temp_arr = list(temp_arr.value)
    t_visc = list(t_visc.value)
    for i in terms_act:
        # temp_arr.remove(temp_arr[terms_act[0]])
        t_visc.remove(t_visc[terms_act[0]])
    # temp_arr = np.array(temp_arr) * u.Kelvin
    t_visc = np.array(t_visc) * u.Kelvin
    # print(temp_arr, len(temp_arr))
    # plt.plot(r_visc.to(u.AU), t_visc)
    # plt.show()
    d_new = {}
    for i in range(len(r_visc)):
        t_int = int(np.round(t_visc[i].value))
        if t_int < 71:
            d_new[r_visc[i].value] = int(np.round(t_visc[i].value))
        elif 120 >= t_int > 70 and t_int % 2 == 1:  # As per temperatures in BT-Settl data
            d_new[r_visc[i].value] = int(
                np.round(t_visc[i].value)) * 2
        elif 120 < t_int:  # and t_int % 5 != 0:
            d_new[r_visc[i].value] = int(
                np.round(t_visc[i].value)) * 5
    return r_visc, t_visc, d_new  # , temp_arr


def generate_visc_flux(config, d: dict, t_max, dr, r_in=None):
    """Generate the flux contributed by the viscously heated disk

    Parameters
    ----------
    config : dict
        dictionary containing system parameters
    d : dict
        dictionary produced by generate_temp_arr, having the radii and their
        corresponding temperatures reduced to the integer values
    t_max : int
        maximum temperature of the viscously heated disk, reduced to nearest int BT-Settl value
    dr : 
        thickness of each annulus
    r_in : astropy.units.Quantity
        inner truncation radius, needed to estimate padding

    Returns
    ----------
    wavelength : numpy.ndarray
        wavelength array in units of Angstrom, in vacuum
    obs_viscous_disk_flux : numpy.ndarray
        observed flux from the viscous disk, in units of erg / (cm^2 s A)
    """
    plot = config['plot']
    save = config['save']
    save_each = config['save_each']
    save_loc = config['save_loc']
    d_star = config['d_star']
    inclination = config['inclination']
    m = config['m']
    
    # change air to vac, this requires astropy quantities, why am I doing this step????????
    # l_min = wave.vactoair(config['l_min'] * u.AA).value
    # l_max = wave.vactoair(config['l_max'] * u.AA).value
    l_min = config['l_min']
    l_max = config['l_max']
    n_data = config['n_data']

    viscous_disk_flux = np.zeros(n_data)

    for int_temp in range(t_max, 13, -1):

        temp_flux = np.zeros(n_data)  # to store total flux contribution from annuli of this temperature

        radii = np.array([r for r, t in d.items() if t == int_temp])
        radii = sorted(radii, reverse=True)

        if int_temp in range(14, 20):  # constrained by availability of BT-Settl models
            logg = 3.5
        else:
            logg = 1.5

        if len(radii) != 0:
            # get the wavelength (in AIR) and flux
            wavelength_air, flux = read_bt_settl_npy(config, int_temp, logg, r_in, ret_full_wavelength_ax=True)
            # convert to vacuum
            wavelength = wave.airtovac(wavelength_air*u.AA).value
            
        for r in radii:

            if inclination == 0:
                x_throw, y_final = logspace_reinterp(config, wavelength, flux)

            else:
                v_kep = np.sqrt(const.G.value * m / r)
                v_red = v_kep * np.sin(inclination) / const.c.value
                
                ## Modify the line below with sampling=[your desired sampling] to get a different size of the kernel, ref. documentation of this function
                kernel_len, wavelength_new, flux_new = interpolate_conv(config, wavelength, flux, v_red)
                kernel = generate_kernel(config, kernel_len, v_red)
                convolved_spectra = np.convolve(flux_new, kernel, mode="same")
                x_throw, y_final = logspace_reinterp(config, wavelength_new, convolved_spectra)

                if save_each: ## this is not really useful
                    np.save(f'{save_loc}/radius_{r}_flux.npy', y_final)
            
            temp_flux += y_final * np.pi * (2 * r * dr + dr ** 2)

        viscous_disk_flux += temp_flux

        if config['verbose']:
            print("completed for temperature of", int_temp, "\nnumber of rings included:", len(radii))
        # if save:
        #     np.save(f'{save_loc}/{int_temp}_flux.npy', temp_flux)
    
    # wavelength must be redefined here, because x_throw is not in scope here
    wavelength = np.logspace(np.log10(l_min), np.log10(l_max), n_data)
    obs_viscous_disk_flux = viscous_disk_flux * np.cos(inclination) / (np.pi * d_star ** 2)

    if save:
        np.save(f'{save_loc}/disk_component.npy', obs_viscous_disk_flux)
    if plot:
        plt.plot(wavelength, obs_viscous_disk_flux)
        plt.xlabel("Wavelength [Angstrom]")
        plt.ylabel(r"Flux [erg / ($cm^{2}$ s A)]")
        plt.title("Viscous Disk SED")
        plt.show()

    # #change wavelength axis to vacuum #### why???????
    # wavelength = wave.airtovac(wavelength * u.AA).value

    return wavelength, obs_viscous_disk_flux


def generate_photosphere_flux(config):
    """generate the flux from the stellar photosphere

    Parameters
    ----------
    config : dict
        dictionary containing system parameters

    Returns
    ----------
    obs_star_flux : astropy.units.Quantity
        flux array due to the stellar photosphere
    """
    l_min = config['l_min']
    l_max = config['l_max']
    # convert to air
    l_min_air = wave.vactoair(l_min * u.AA).value
    l_max_air = wave.vactoair(l_max * u.AA).value

    log_g_star = config['log_g_star']
    t_star = config["t_star"]
    r_star = config["r_star"]
    d_star = config["d_star"]

    int_star_temp = int(np.round(t_star / 100))

    address = f"{config['bt_settl_path']}/lte0{int_star_temp}-{log_g_star}-0.0a+0.0.BT-Settl.7.dat.npy"
    data = np.load(address)

    l_bound = np.searchsorted(data[0],l_min_air - 10.0)
    u_bound = np.searchsorted(data[0],l_max_air + 10.0)
    x2 = data[0][l_bound:u_bound]
    y2 = data[1][l_bound:u_bound]

    x2_vac = wave.airtovac(x2*u.AA).value

    wavelength, y_new_star = logspace_reinterp(config, x2_vac, y2)
    obs_star_flux = y_new_star * (r_star / d_star) ** 2.0

    if config['plot']:
        plt.plot(wavelength, obs_star_flux)
        plt.xlabel("Wavelength in $\AA$ ----->")
        plt.ylabel("Flux [erg / ($cm^{2}$ s $\AA$)] ----->")
        plt.title("Stellar Photosphere SED")
        plt.show()

    if config['save']:
        np.save(f"{config['save_loc']}/stellar_component.npy", obs_star_flux)

    return obs_star_flux


def cos_gamma_func(phi, theta, incl):
    """Calculate the dot product between line-of-sight unit vector and area unit vector
    This is required only for a blackbody magnetosphere,as done by Liu et al.
    """
    cos_gamma = np.sin(theta) * np.cos(phi) * np.sin(incl) + np.cos(theta) * np.cos(incl)
    if cos_gamma < 0:
        return 0
    else:
        return cos_gamma * np.sin(theta)


def magnetospheric_component_calculate(config, r_in):
    """Calculte the flux from the shock-heated region
    Parameters
    ----------
    config : dict
        dictionary containing system parameters
    
    r_in : float
        inner truncation radius of the visscously heated disk

    Returns
    ----------
    obs_mag_flux : numpy.ndarray
        array of flux values, unit: erg / (cm^2 s A)
    """

    if r_in<=config["r_star"]:
        spec = np.zeros(config["n_data"]) #* u.erg / (u.cm ** 2 * u.s * u.AA * u.sr)
        if config['verbose']:
            print("High accretion, no magnetospheric contribution")
        return spec
    else: print("Magnetospheric contribution active")

    ###########################    HYDROGEN SLAB    #############################
    if config['mag_comp']=="hslab":

        h_flux = h_emission.get_h_intensity(config)
        h_minus_flux = h_minus_emission.get_h_minus_intensity(config)
        h_slab_flux = (h_flux + h_minus_flux) * u.sr

        # two wavelength regimes are used
        wav_slab = np.logspace(np.log10(config['l_min']), np.log10(config['l_max']), config['n_h']) * u.AA
        wav2 = np.logspace(np.log10(config['l_max']), np.log10(1e6), 250) * u.AA

        # a blackbody SED is a good approximation for the Hydrogen slab beyond l_max (=50000 A)
        bb_int = BlackBody(temperature=config['t_slab'], scale=1 * u.erg / (u.cm ** 2 * u.s * u.AA * u.sr))
        bb_spec = bb_int(wav2)
        bb_spec = bb_spec * u.sr
        h_slab_flux = np.append(h_slab_flux, bb_spec[1:])
        wav_slab = np.append(wav_slab, wav2[1:])
        h_slab_flux = h_slab_flux.to(u.joule / (u.m ** 2 * u.s * u.AA)).value
        wav_slab = wav_slab.value

        # calculate the total luminosity to get the area of the shock
        # this is the approach taken by Liu et al., however  for low accretion rates, this yields a very large covering fraction
        integrated_flux = trapezoid(h_slab_flux, wav_slab) # in J / m^2 s

        # print(f"int flux: {integrated_flux} J / m^2 s")
        # print(f"G: {const.G.value}"
        #       f"\nmass :{config['m']}"
        #       f"\nm_dot : {config['m_dot']}"
        #       f"\nr star: {config['r_star']}"
        #       f"\nr_in: {r_in}")

        l_mag = const.G.value * config['m'] * config['m_dot'] * (1 / config['r_star'] - 1 / r_in)
        area_shock = l_mag / integrated_flux  # in m^2

        if config['verbose']:
            print(f"shock area : {area_shock}")

        # shock area fraction warning
        fraction = area_shock / (4 * np.pi * config['r_star'] ** 2)
        if config['verbose']:
            print(f"fraction of area {fraction:.4f}")
        if fraction > 1:
            if config['save']:
                with open(f"{config['save_loc']}/details.txt", 'a+') as f:
                    f.write("WARNING/nTotal area of shock required is larger than stellar surface area")
        
        # getting the geometry of the shocked region, if it is less than the stellar photosphere area
        # calculate corresponding theta max and min
        th_max = np.arcsin(np.sqrt(config['r_star'] / r_in))
        if fraction + np.cos(th_max) > 1:
            if config['verbose']:
                print('Theta min not well defined')
            th_min = 0.0
            if config['save']:
                with open(f"{config['save_loc']}/details.txt", 'a+') as f:
                    f.write(f"Theta_min not well defined")
        else:
            th_min = np.arccos(fraction + np.cos(th_max))

        if config['verbose']:
            print(f"The values are (in degrees)\nth_min : {th_min*180.0/np.pi:.2f}\nth_max : {th_max*180.0/np.pi:.2f}")
        
        # integrate to get the contribution along l.o.s.
        intg_val1, err = dblquad(cos_gamma_func, th_min, th_max, 0, 2 * np.pi, args=(config['inclination'],))
        intg_val2, err = dblquad(cos_gamma_func, np.pi - th_max, np.pi - th_min, 0, 2 * np.pi, args=(config['inclination'],))
        intg_val = intg_val1 + intg_val2
        if config['verbose']:
            print(f"integral val : {intg_val}, error : {err}")
        if config['save']:
            with open(f"{config['save_loc']}/details.txt", 'a+') as f:
                f.write(f"integral val : {intg_val}, error : {err}")
                f.write(f"The values are (in degrees)\nth_min : {th_min*180.0/np.pi:.2f}\nth_max : {th_max*180.0/np.pi:.2f}")

        # interpolate to required wavelength axis,same as the other components
        # func_slab = interp1d(wav_slab, h_slab_flux)
        wav_ax = np.logspace(np.log10(config['l_min']), np.log10(config['l_max']), config['n_data'])
        h_slab_flux_interp = np.interp(wav_ax, wav_slab, h_slab_flux)
        # h_slab_flux_interp = h_slab_flux_interp * u.erg / (u.cm ** 2 * u.s * u.AA)
        obs_mag_flux = h_slab_flux_interp * (config['r_star'] / config['d_star']) ** 2 * intg_val

    ###########################    BLACKBODY    ###################################
    elif config["mag_comp"]=="blackbody":

        l_mag = const.G.value * config["m"] * config["m_dot"] * (1 / config["r_star"] - 1 / r_in)
        area_shock = l_mag / (const.sigma_sb.value * config["t_slab"].value ** 4)  # allowing the effective temperature of the shock to vary
        
        # shock area fraction warning
        fraction = area_shock / (4 * np.pi * config["r_star"] ** 2)
        if config['verbose']:
            print(f"fraction of area {fraction}")
        if fraction > 1:
            if config["save"]:
                with open(f"{config['save_loc']}/details.txt", 'a+') as f:
                    f.write("WARNING/nTotal area of shock required is more than stellar surface area")
        else:
            if config["save"]:
                with open(f"{config['save_loc']}//details.txt", 'a+') as f:
                    f.write(f"ratio of area of shock to stellar surface area =  {fraction}")

        # getting the geometry of the shocked region, if it is less than the stellar photosphere area
        # calculate corresponding theta max and min
        th_max = np.arcsin(np.sqrt(config["r_star"] / r_in))
        if fraction + np.cos(th_max) > 1:
            if config['verbose']:
                print('Theta min not well defined')
            th_min = 0.0

            if config["save"]:
                with open(f"{config['save_loc']}/details.txt", 'a+') as f:
                    f.write(f"Theta_min not well defined")
        else:
            th_min = np.arccos(fraction + np.cos(th_max))

        if config['verbose']:
            print(f"The values are \nth_min : {th_min*180.0/np.pi:.2f}\nth_max : {th_max*180.0/np.pi:.2f}")

        # integrate
        intg_val1, err = dblquad(cos_gamma_func, th_min, th_max, 0, 2 * np.pi, args=(config["inclination"],))
        intg_val2, err = dblquad(cos_gamma_func, np.pi - th_max, np.pi - th_min, 0, 2 * np.pi, args=(config["inclination"],))
        intg_val = intg_val1 + intg_val2

        if config['verbose']:
            print(f"integral val : {intg_val}, error : {err}")
        if config["save"]:
            with open(f"{config['save_loc']}/details.txt", 'a+') as f:
                f.write(f"integral val : {intg_val}, error : {err}")
                f.write(f"The values are \nth_min : {th_min*180.0/np.pi:.2f}\nth_max : {th_max*180.0/np.pi:.2f}")
        
        # evaluate blackbody spectrum on the wavelength axis, at temperature t_slab
        scale_unit = u.erg / (u.cm ** 2 * u.s * u.AA * u.sr)
        bb = BlackBody(config["t_slab"], scale=1 * scale_unit)
        wavelength = np.logspace(np.log10(config["l_min"]), np.log10(config["l_max"]), config["n_data"]) * u.AA
        flux_bb = bb(wavelength)
        obs_bb_flux = flux_bb * (config["r_star"] / config["d_star"]) ** 2 * intg_val
        obs_mag_flux = obs_bb_flux.value

    else:
        raise ValueError("Only accepted magnetosphere models are \'blackbody\' and \'hslab\'")

    if config['plot']:
        wav_ax = np.logspace(np.log10(config['l_min']), np.log10(config['l_max']), config['n_data'])
        plt.plot(wav_ax, obs_mag_flux)
        plt.xlabel(r"Wavelength in $\AA$ ----->")
        plt.ylabel(r"Flux [erg / ($cm^{2}$ s $\AA$)] ----->")
        plt.title("Magnetospheric Shock Region SED")
        plt.show()
    
    return obs_mag_flux


def t_eff_dust(r, config):
    """Define the temperature profile in the passively heated dusty disk

    Parameters
    ----------
    r : astropy.units.Quantity
        radius at which temperature is to be evaluated

    config : dict
        dictionary containing system parameters

    Returns
    ------------
    t : astropy.units.Quantity
        temperature value at r

    """
    r_star = config['r_star']
    t_star = config['t_star']
    m = config['m']
    t_0 = config['t_0']
    alpha_0 = 0.003 * (r_star / (1.6 * const.R_sun.value)) / (r / const.au.value) + 0.05 * (t_star / 3400) ** (4.0 / 7.0) * (
            r_star / (1.6 * const.R_sun.value)) ** (2.0 / 7.0) * (r / const.au.value) ** (2.0 / 7.0) / (m / (0.3 * const.M_sun.value)) ** (4.0 / 7.0)
    t = (alpha_0 / 2) ** 0.25 * (r_star / r) ** 0.5 * t_0
    return t


def generate_dusty_disk_flux(config, r_in, r_sub):
    """Generates the SED of the dusty disk component, as worked out by Liu et al. 2022, assuming each annulus to emit in
    the form a blackbody, having a temperature profile that is either radiation dominated (transition layer present),
    or has region of viscously heated dust (transition layer absent).

    Parameters
    ----------
    config : dict
        dictionary containing system parameters

    r_in : astropy.units.Quantity
        inner truncation radius of the viscously heated disk

    r_sub : astropy.units.Quantity
        radius at which t_visc = 1400 K, i.e. where dust sublimation begins, sets a formal
        outer boundary of the viscously heated disk

    Returns
    ----------
    obs_dust_flux : astropy.units.Quantity
        array of observed flux from the dusty disk component
    """
    plot = config['plot']
    save = config['save']
    save_loc = config['save_loc']
    n_data = config['n_data']
    d_star = config['d_star']
    inclination = config['inclination']

    r_dust = np.logspace(np.log10(r_sub), np.log10(270 * const.au.value), config["n_dust_disk"])
    t_dust_init = t_eff_dust(r_dust, config)

    if t_eff_dust(r_sub, config) > 1400:
        t_dust[np.where(t_dust>1400.0)] = 1400
        # t_dust = ma.masked_greater(t_dust_init, 1400)
        # t_dust = t_dust.filled(1400)
        # t_dust = t_dust * u.K
    else:
        # t_visc_dust = np.zeros(len(r_dust)) * u.K
        t_visc_dust = temp_visc(r_dust, r_in, config['m'], config['m_dot'])
        t_dust = np.maximum(t_dust_init, t_visc_dust)

    if plot:
        plt.plot(r_dust / const.au, t_dust)
        plt.xlabel("Radial distance [AU]")
        plt.ylabel("Temperature [K]")
        plt.title("Passively heated radial temperature profile")
        plt.show()

    dust_flux = np.zeros(n_data) #* u.erg / (u.cm * u.cm * u.s * u.AA * u.sr) * (u.m * u.m)
    wavelength = np.logspace(np.log10(config['l_min']), np.log10(config['l_max']), config["n_data"]) #* u.AA

    for i in range(len(r_dust) - 1):
        scale_unit = u.erg / (u.cm ** 2 * u.s * u.AA * u.sr)
        dust_bb = BlackBody(t_dust[i] * u.K, scale=1 * scale_unit)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dust_bb_flux = dust_bb(wavelength)

        dust_flux += dust_bb_flux.value * np.pi * (r_dust[i + 1] ** 2 - r_dust[i] ** 2)
        if i % 100 == 0:  # print progress after every 100 annuli
            if config['verbose']:
                print(f"done temperature {i}")

    obs_dust_flux = dust_flux * np.cos(inclination) / (np.pi * d_star ** 2) #* u.sr

    if save:
        np.save(f'{save_loc}/dust_component.npy', obs_dust_flux)

    if plot:
        plt.plot(wavelength, obs_dust_flux)
        plt.xlabel("Wavelength in $\AA$ ----->")
        plt.ylabel(r"Flux [erg / ($cm^{2}$ s $\AA$)] ----->")
        plt.title("Dust Dominated Disk SED")
        plt.show()

    return obs_dust_flux


def dust_extinction_flux(config, wavelength, obs_viscous_disk_flux, obs_star_flux, obs_mag_flux, obs_dust_flux):
    """Redden the spectra with the Milky Way extinction curves. Ref. Gordon et al. 2021, Fitzpatrick et. al 2019

    Parameters
    ----------
    config : dict
             dictionary containing system parameters

    wavelength: astropy.units.Quantity
        wavelength array, in units of Angstrom

    obs_star_flux: astropy.units.Quantity
    obs_viscous_disk_flux: astropy.units.Quantity
    obs_mag_flux: astropy.units.Quantity
    obs_dust_flux: astropy.units.Quantity

    Returns
    ----------
    total_flux: astropy.units.Quantity
        spectra reddened as per the given parameters of a_v and r_v in details
    """
    r_v = config['rv']
    a_v = config['av']
    save_loc = config['save_loc']

    wavelength = wavelength * u.AA
    break_id = np.searchsorted(wavelength, 1./3. * 1e5 * u.AA)
    wav1 = wavelength[:break_id]
    wav2 = wavelength[break_id:]
    # wav1 = np.extract(wavelength < 33e3 * u.AA, wavelength)
    # wav2 = np.extract(wavelength >= 33e3 * u.AA, wavelength)

    total = obs_star_flux + obs_viscous_disk_flux + obs_dust_flux + obs_mag_flux

    total_flux_1 = total[:len(wav1)]
    total_flux_2 = total[len(wav1):]
    ext1 = F19(Rv=r_v)
    ext2 = G21_MWAvg()  # Gordon et al. 2021, milky way average curve

    exting_spec_1 = total_flux_1 * ext1.extinguish(wav1, Av=a_v)
    exting_spec_2 = total_flux_2 * ext2.extinguish(wav2, Av=a_v)

    total_flux = np.append(exting_spec_1, exting_spec_2)

    if config['save']:
        np.save(f'{save_loc}/extinguished_spectra.npy', total_flux)
        np.save(f'{save_loc}/wave_arr.npy', wavelength.value)
    if config['plot']:
        plt.plot(wavelength, total_flux, label='extinguished spectrum')
        plt.xlabel("Wavelength in $\AA$ ----->")
        plt.ylabel("Flux [erg / ($cm^{2}$ s $\AA$)] ----->")
        plt.title("Extinguished Spectra")
        plt.legend()
        plt.show()

    return total_flux


def parse_args(raw_args=None):
    """Take config file location from the command line"""
    parser = argparse.ArgumentParser(description="YSO Spectrum generator")
    parser.add_argument('ConfigfileLocation', type=str, help="Path to config file")
    args = parser.parse_args(raw_args)
    return args


def total_spec(dict_config):

    # logger = logging.getLogger(__name__)
    # logging.basicConfig(filename='timer_bf.log', encoding='utf-8', level=logging.DEBUG)
    # logger.info(str(dict_config))
    
    t1 = time.time()
    calculate_n_data(dict_config) # set n_data
    dr, t_max, d, r_in, r_sub = generate_temp_arr(dict_config)
    wavelength, obs_viscous_disk_flux = generate_visc_flux(dict_config, d, t_max, dr)
    t2 = time.time()
    # logger.info(f'Viscous disk done, time taken : {t2 - t1}')
    print(f'Viscous disk done, time taken : {t2 - t1}')

    obs_mag_flux = magnetospheric_component_calculate(dict_config, r_in)
    t3 = time.time()
    # logger.info(f"Magnetic component done, time taken :{t3-t2}")
    print(f"Magnetic component done, time taken :{t3-t2}")

    obs_dust_flux = generate_dusty_disk_flux(dict_config, r_in, r_sub)
    t4 = time.time()
    # logger.info(f"Dust component done, time taken : {t4-t3}")
    print(f"Dust component done, time taken : {t4-t3}")

    obs_star_flux = generate_photosphere_flux(dict_config)
    t5 = time.time()
    # logger.info(f"Photospheric component done, time taken {t5-t4}")
    print(f"Photospheric component done, time taken {t5-t4}")
    total_flux = dust_extinction_flux(dict_config, wavelength, obs_viscous_disk_flux, 
                                      obs_star_flux, obs_mag_flux,obs_dust_flux)
    total_flux /= np.median(total_flux)

    return wavelength, total_flux


def main(raw_args=None):
    """Calls the base functions sequentially, and finally generates extinguished spectra for the given system"""

    st = time.time()

    args = parse_args(raw_args)
    dict_config = utils.config_read_bare(args.ConfigfileLocation)
    
    # save the data
    dict_config['save'] = True
    dict_config['plot'] = False
    dict_config['verbose'] = True

    calculate_n_data(dict_config) # set n_data
    dr, t_max, d, r_in, r_sub = generate_temp_arr(dict_config)
    wavelength, obs_viscous_disk_flux = generate_visc_flux(dict_config, d, t_max, dr)
    if dict_config["verbose"]:
        print('Viscous disk done')

    t1 = time.time()
    obs_mag_flux = magnetospheric_component_calculate(dict_config, r_in)
    t2 = time.time()
    if dict_config["verbose"]:
        print(f"Magnetospheric component done, time taken:{t2-t1:.5f}")

    dust_st = time.time()
    obs_dust_flux = generate_dusty_disk_flux(dict_config, r_in, r_sub)
    print(f"time taken for dust = {time.time() -dust_st}")

    if dict_config["verbose"]:
        print("Dust component done")
    obs_star_flux = generate_photosphere_flux(dict_config)
    if dict_config["verbose"]:
        print("Photospheric component done")
    total_flux = obs_viscous_disk_flux + obs_star_flux + obs_mag_flux + obs_dust_flux
    total_flux = dust_extinction_flux(dict_config, wavelength, obs_viscous_disk_flux, 
                                      obs_star_flux, obs_mag_flux,obs_dust_flux)
    
    #save with noise
    snr = 100
    noise = np.random.normal(0,1/snr,total_flux.shape[0])*total_flux
    noisy_spec = total_flux + noise
    if dict_config['save']:
        np.save(f"{dict_config['save_loc']}/noisy_flux_snr_{snr}.npy",noisy_spec)

    et = time.time()
    print(f"Total time taken : {et - st}")
    if dict_config['plot']:
        plt.plot(wavelength, obs_star_flux, label="Stellar photosphere")
        plt.plot(wavelength, total_flux, label="Total")
        plt.plot(wavelength, obs_viscous_disk_flux, label="Viscous Disk")
        plt.plot(wavelength, obs_mag_flux, label="Magnetosphere")
        plt.plot(wavelength, obs_dust_flux, label="Dusty disk")
        plt.legend()
        plt.xlabel("Wavelength [Angstrom]")
        plt.ylabel("Flux [erg / cm^2 s A]")
        plt.show()
    print("done")

# if __name__ == "__main__":
    
    # cProfile.run("main()")
    
    # config = utils.config_read_bare('config_file.cfg')
    # h_flux = h_emission.get_h_intensity(config)
    # plt.plot(h_flux)
    # plt.show()

    # main()

'''from pypeit.core import wave
from astropy.io import ascii

def rad_vel_correction(wave, vel):
    """
    Apply correction to wavelength for the doppler shift due to
    radial velocity of the star.
    :param wave: Quantity numpy array
    :param vel: astropy.units.Quantity
    :return: astropy.units.Quantity array
    """
    del_wav = (vel/const.c) * wave
    return wave - del_wav

#read the data, V960 Mon
path_to_valid = "/home/arch/yso/validation_files/"
data = ascii.read(path_to_valid+'KOA_93088/HIRES/extracted/tbl/ccd1/flux/HI.20141209.56999_1_04_flux.tbl.gz')
data = [data['wave'],data['Flux']/np.median(data['Flux']),data['Error']/np.median(data['Flux'])]
# print(f" data[2] {data[2].unit}")

# vac to air correction for given data
# wavelengths_air = wave.vactoair(data[0]*u.AA)
data[0] = rad_vel_correction(data[0]*u.AA, 40.3 * u.km / u.s).value  # from header file
plt.plot(data[0],data[1],label = "Obseved flux")
plt.fill_between(data[0], data[1] - data[2], data[1] + data[2], color='gray', alpha=0.4, label='Error')

# get model spectra
dict_config = utils.config_read_bare("config_file.cfg")
wave_model, flux_model = total_spec(dict_config)
plt.plot(wave_model, flux_model, label="model")
plt.legend()
plt.show()'''