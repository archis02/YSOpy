import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
import astropy.constants as const
import astropy.units as u
from astropy.io.votable import parse
from astropy.modeling.physical_models import BlackBody
from dust_extinction.parameter_averages import F19
from dust_extinction.averages import G21_MWAvg

from . import utils
from . import h_emission
from . import h_minus_emission
from . import base_funcs

import argparse
import time
from functools import cache
import logging

def new_contribution(): # check what this function does. ideally, move it to a different file
    config = config_read("config_file.das")
    dr, t_max, d, r_in, r_sub = generate_temp_arr(config)

    save_loc = config['save_loc']
    l_min = config['l_min']
    l_max = config['l_max']
    n_data = config['n_data']
    r_star = config['r_star']
    print(d)
    r_visc = np.array([r for r, t in d.items()])
    r_visc = sorted(r_visc) * u.m
    d_star = config['d_star']
    inclination = config['inclination']

    fig, ax = plt.subplots()
    arr = []
    radii = []

    # trim wavelength axis to region of interest
    wavelength = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data)
    wav_new = ma.masked_where(wavelength < 10000, wavelength)
    wav_new = ma.masked_where(wav_new > 25000, wav_new)
    wav_new = wav_new.compressed()
    print(wav_new)
    print(r_in)
    print((r_sub / r_in).si)

    for i in range(0, len(r_visc), 7):
        r = r_visc[i]
        flux = np.load(f"{save_loc}/radius_{r}_flux.npy")
        flux = flux * (u.erg / (u.cm ** 2 * u.s * u.AA)) * r * dr
        # trim to region of interest
        wavelength = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data)
        flux = np.extract(wavelength < 25000, flux)
        wavelength = np.extract(wavelength < 25000, wavelength)
        flux = np.extract(wavelength > 10000, flux)
        wavelength = np.extract(wavelength > 10000, wavelength)
        # correct for distance
        flux *= np.cos(inclination) / (np.pi * d_star ** 2)
        flux = flux.to(u.erg / (u.cm ** 2 * u.s * u.AA))

        ax.plot(wav_new, np.log10(flux.value) - 0.05 * i,
                label=f"r={np.round(r / const.R_sun, 2)} R_sun, T={d[r.value] * 100} K")
        arr.append(flux.value)

    plt.xlabel("Wavelength [Angstrom]")
    plt.ylabel("log_10 Flux (+ offset) [erg / cm^2 s A]")
    plt.legend()
    plt.show()


def contribution(raw_args=None): ## check what this function does, move it to a different file.
    """find the contribution of the various annuli towards a particular line/group of lines
    """
    args = parse_args(raw_args)
    config = config_read(args.ConfigfileLocation)
    dr, t_max, d, r_in, r_sub = generate_temp_arr(config)
    inclination = config['inclination']
    m = config['m']
    n_data = config['n_data']
    l_min = config['l_min']
    l_max = config['l_max']

    arr = []  # to store the cumulative flux arrays
    t_max = max(d.values())
    wav = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data)
    wav = (wav - 22930) / 22930 * 3e5
    wav = np.extract(wav > -3000, wav)
    wav = np.extract(wav < 3000, wav)

    viscous_disk_flux = np.zeros(len(wav)) * (u.erg * u.m ** 2 / (u.cm ** 2 * u.s * u.AA))
    cumulative_flux = np.zeros(len(wav)) * (u.erg / (u.cm ** 2 * u.s * u.AA))
    # annulus_flux = np.zeros(len(wav)) * (u.erg / (u.cm ** 2 * u.s * u.AA))
    flag = 0
    z_val = []
    # loop over the temperatures
    for int_temp in range(t_max, 14, -1):

        # to store total flux contribution from annuli of this temperature
        temp_flux = np.zeros(len(wav)) * (u.erg / (u.s * u.AA))
        radii = np.array([r for r, t in d.items() if t == int_temp])
        if len(radii) == 0:
            continue
        radii = radii * u.m
        if int_temp in range(14, 20):  # constrained by availability of BT-Settl models
            logg = 3.5
        else:
            logg = 1.5

        wavelength, flux = read_bt_settl_npy(config, int_temp, logg)
        radii = sorted(radii)
        if len(radii) == 0:
            if config['verbose']:
                print("no radii at this temp")
        for r in radii:
            if inclination.value == 0:
                x_throw, y_final = logspace_reinterp(config, wavelength, flux)
            else:
                v_kep = np.sqrt(const.G * m / r)
                v_red = v_kep * np.sin(inclination) / const.c
                interp_samp, wavelength_new, flux_new = interpolate_conv(config, wavelength, flux, 100, v_red)
                kernel = generate_kernel(config, interp_samp, v_red)
                convolved_spectra = np.convolve(flux_new, kernel, mode="same")
                x_throw, y_final = logspace_reinterp(config, wavelength_new, convolved_spectra)

                # convert to velocity space and trim to required region
                x_throw = (x_throw - 22930) / 22930 * 3e5
                y_final = np.extract(x_throw > -3000, y_final)
                x_throw = np.extract(x_throw > -3000, x_throw)
                y_final = np.extract(x_throw < 3000, y_final)
                cumulative_flux += y_final

                if flag % 50 == 0:
                    arr.append(np.log10(cumulative_flux.copy().value * 2 * np.pi * r.value))
                    z_val.append(r.value.copy())
                flag += 1
            temp_flux += y_final * np.pi * (2 * r * dr + dr ** 2)
        viscous_disk_flux += temp_flux
        if config['verbose']:
            print(f"done temp {int_temp}")
    # trim to CO line
    wavelength = np.logspace(np.log10(l_min.value), np.log10(l_max.value), n_data)
    wavelength = (wavelength - 22930) / 22930 * 3e5
    wavelength = np.extract(wavelength > -3000, wavelength)
    wavelength = np.extract(wavelength < 3000, wavelength)

    # plot the heat map
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.imshow(arr, aspect='auto')
    plt.show()

    for i in range(len(arr)):
        fl = arr[i]
        z = np.ones(len(fl)) * z_val[i]
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Flux (erg / cm^2 s A)")
        ax.set_zlabel("Extent of integration")
        ax.plot(wavelength, fl, z, label=f'i = {i}')
    plt.show()
