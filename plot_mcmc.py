import numpy as np
from scipy.interpolate import interp1d
from ysopy import base_funcs as bf
from ysopy import utils
import astropy.constants as const
import astropy.units as u
from astropy.io import ascii
from pypeit.core import wave
import emcee
from configparser import ConfigParser
import matplotlib.pyplot as plt

from mcmc_window import model_spec_window

def log_likelihood_window(theta, config):
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
    wave_model, total_flux, flux_photon_count = model_spec_window(theta_model, config)
    # change the below code from total_flux to photon_counts if obs_spectra is in photon counts
    model_flux = total_flux

    fluxes_list = []
    wavelenghts_list = []

    # loop over the windows
    for i, window in enumerate(windows):
        # Get the polynomial coefficients for this window
        poly_coeffs = theta[n_model_params + i * (poly_order + 1) : n_model_params + (i + 1) * (poly_order + 1)]

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

    return wavelenghts_list,fluxes_list