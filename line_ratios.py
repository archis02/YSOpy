import numpy as np
import matplotlib.pyplot as plt
from ysopy import utils
import astropy.constants as const
import ysopy.base_funcs as bf
from scipy.optimize import least_squares
import astropy.units as u
# import smplotlib
import os


def func(dict_config):
    dr, t_max, d, r_in, r_sub = bf.generate_temp_arr(dict_config)
    wavelength, obs_viscous_disk_flux = bf.generate_visc_flux(dict_config, d, t_max, dr)
    print("visc done")
    obs_mag_flux = bf.magnetospheric_component_calculate(dict_config, r_in)
    print("mag done")
    obs_dust_flux = bf.generate_dusty_disk_flux(dict_config, r_in, r_sub)
    print("dust done")
    obs_star_flux = bf.generate_photosphere_flux(dict_config)
    total_flux = obs_viscous_disk_flux + obs_star_flux + obs_dust_flux + obs_mag_flux
    print("phot done")
    total_ext_flux = bf.dust_extinction_flux(dict_config, wavelength, obs_viscous_disk_flux,
                                         obs_star_flux, obs_mag_flux, obs_dust_flux)
    print("flux done")
    return wavelength, total_flux, total_ext_flux


def trim_in_window(wavelength, total_flux, window):
    """
    This trims the wavelength and flux in a given window asked as [max, min].
    Parameters
    ----------
    wavelength: np.ndarray
    total_flux: np.ndarray
    window: list

    Returns
    -------
    trimmed_wavelength: np.ndarray
    total_flux_trimmed: np.ndarray
    """
    wave_trimmed = np.where(wavelength > window[0], wavelength, 0)
    total_flux_trimmed = np.where(wavelength > window[0], total_flux, 0)
    total_flux_trimmed = np.where(wavelength < window[1], total_flux_trimmed, 0)
    wave_trimmed = np.where(wavelength < window[1], wave_trimmed, 0)
    total_flux_trimmed = np.trim_zeros(total_flux_trimmed, "fb")
    wave_trimmed = np.trim_zeros(wave_trimmed, "fb")
    return wave_trimmed, total_flux_trimmed


def residual(theta, x_axis, y_spectra):
    continuum = np.polyval(theta, x_axis)
    return np.sum((y_spectra - continuum) ** 2)


def calc_continuum_arr(theta: np.ndarray, spec_wavelength: np.ndarray, flux: np.ndarray, threshold_l = 0.9, threshold_u=None, plot=False):
    """
    This does a least squares fit on the normalised spectra by scaling the
    wavelength axis to [-1, 1] in a uniformly sampled axis.
    Parameters
    ----------
    theta: poly coefficients for the continuum fit
    wave: uniformly sampled wavelength axis
    flux: un-normalized flux array

    Returns
    -------
    poly_y: continuum fit of the spectra
    valid_interp_flux: Normalised spectra in uniformly sampled wavelength axis --> [-1, 1, len(spec_wavelength)]
    """

    flux = flux / np.median(flux)  # we have to normalize to 1
    # plt.plot(spec_wavelength, flux)
    # plt.show()
    flux = np.ma.masked_where(flux <= threshold_l, flux)
    if threshold_u is not None:
        flux = np.ma.masked_where(flux >= threshold_u, flux)
    # plt.plot(spec_wavelength, flux)
    # plt.show()
    wave = np.linspace(spec_wavelength[0], spec_wavelength[-1], len(spec_wavelength))
    # valid indices for rest of the calc
    valid = ~flux.mask
    
    interp_flux = np.interp(wave, spec_wavelength, flux)
    valid_interp_flux = np.interp(wave, spec_wavelength[valid], flux[valid])
    # rescaling the wavelength axis
    wave_window_eval_poly = np.linspace(-1, 1, len(spec_wavelength))
    # plt.plot(wave_window_eval_poly, valid_interp_flux)
    # plt.show()
    # continuum correction
    best_fit_coeffs_continuum = least_squares(residual, theta, args=(wave_window_eval_poly, valid_interp_flux))
    best_fit_coeffs_continuum = best_fit_coeffs_continuum.x

    print("Best fit parameters: ", best_fit_coeffs_continuum)
    poly_y = np.polyval(best_fit_coeffs_continuum, wave_window_eval_poly)
    if plot:
        plt.title("Continuum fit")
        plt.plot(wave_window_eval_poly, valid_interp_flux, label="Interpolated Flux")
        plt.plot(wave_window_eval_poly, poly_y, label="Polynomial Fit")
        plt.xlabel("Scaled Wavelength")
        plt.ylabel("Normalised Flux")
        plt.legend()
        plt.show()
    return poly_y, wave, valid_interp_flux, interp_flux

if __name__ == '__main__':
    """
    # model params, used in fit
    b = 2
    m = 0.4
    m_dot = 7e-7
    inclination = 45
    # For exlupi from https://www.aanda.org/articles/aa/pdf/2014/01/aa22428-13.pdf
    #t_0 = 4000
    t_0 = 3750
    
    # H-slab parameters, will not affect spectra
    t_slab = 8000
    n_e = 10000000000000.0
    tau = 1.0
    # known from stellar model, Baraffe
    r_star = 2.11
    t_star = 3400
    log_g_star = 3.5
    #scaling parameters, will not affect spectra
    d_star = 10
    av = 10
    rv = 3.1
    # General case
    l_min = 1260
    l_max = 56000
    """
    save_dir = "data/line_ratios"
    config = utils.config_read_bare("ysopy/config_file.cfg")
    dict_config = config
    plot_dir = "../plots/line_width"

    # wave1, fl1, fl1_ext = func(dict_config)
    # dict_config["mag_comp"] = "blackbody"
    # wave2, fl2, fl2_ext = func(dict_config)
    # #
    # np.save(f"{save_dir}/wavelength.npy", wave1)
    # np.save(f"{save_dir}/f_hslab.npy", fl1)
    # np.save(f"{save_dir}/f_bbody.npy", fl2)
    # np.save(f"{save_dir}/f_hslab_ext.npy", fl1_ext)
    # np.save(f"{save_dir}/f_bbody_ext.npy", fl2_ext)
    # plt.plot(wave1, fl1)
    # plt.plot(wave2, fl2)
    # plt.show()

    wave = np.load(f"{save_dir}/wavelength.npy")
    hslab_flux = np.load(f"{save_dir}/f_hslab.npy")
    bbody_flux = np.load(f"{save_dir}/f_bbody.npy")
    hslab_flux_ext = np.load(f"{save_dir}/f_hslab_ext.npy")
    bbody_flux_ext = np.load(f"{save_dir}/f_bbody_ext.npy")
    # plt.plot(wave, hslab_flux)
    # plt.plot(wave, bbody_flux)
    # plt.plot(wave, hslab_flux_ext)
    # plt.plot(wave, bbody_flux_ext)
    # plt.show()
    # exit(0)
    jump = "balmer"#"paschen"
    if jump == "balmer":
        window = [3550, 3750]
        wv_left_line, hslab_flux_left_line = trim_in_window(wavelength=wave, total_flux=hslab_flux, window=window)
        wv_left_line, bbody_flux_left_line = trim_in_window(wavelength=wave, total_flux=bbody_flux, window=window)
        plt.plot(wv_left_line, hslab_flux_left_line, label="H-Slab")
        # plt.plot(wv_left_line, bbody_flux_left_line, label="B-body")
        plt.show()

    if jump == "paschen":
        # line width calculation
        window = [8183, 8188]  # this is on left side of paschen jump
        # window = [8170, 8193]
        # Na I 8185.5055 Å (log gf = 0.24)
        # hslab_flux_ext = hslab_flux
        # bbody_flux_ext = bbody_flux
        wv_left_line, hslab_flux_left_line = trim_in_window(wavelength=wave, total_flux=hslab_flux_ext, window=window)
        wv_left_line, bbody_flux_left_line = trim_in_window(wavelength=wave, total_flux=bbody_flux_ext, window=window)


        window = [8327, 8332]  # this is on right side
        # window = [8320, 8340]
        #Fe I 8329.3436 Å (log gf = -1.52)
        wv_right_line, hslab_flux_right_line = trim_in_window(wavelength=wave, total_flux=hslab_flux_ext, window=window)
        wv_right_line, bbody_flux_right_line = trim_in_window(wavelength=wave, total_flux=bbody_flux_ext, window=window)



        poly_coeffs = [0, 0, 1]
        plot_bool = False

        conti_hslab_right, wv_hslab_right, valid_hslab_flux_right, hslab_flux_right = calc_continuum_arr(poly_coeffs, threshold_l=0.975, spec_wavelength=wv_right_line,
                                                                                 flux=hslab_flux_right_line, plot=plot_bool)
        conti_bbody_right, wv_bbody_right, valid_bbody_flux_right, bbody_flux_right = calc_continuum_arr(poly_coeffs, threshold_l=0.975, spec_wavelength=wv_right_line,
                                                                                 flux=bbody_flux_right_line, plot=plot_bool)
        conti_hslab_left, wv_hslab_left, valid_hslab_flux_left, hslab_flux_left = calc_continuum_arr(poly_coeffs, threshold_l=0.995, spec_wavelength=wv_left_line,
                                                                              flux=hslab_flux_left_line, plot=plot_bool)

        conti_bbody_left, wv_bbody_left, valid_bbody_flux_left, bbody_flux_left = calc_continuum_arr(poly_coeffs, threshold_l=0.995, spec_wavelength=wv_left_line,
                                                                              flux=bbody_flux_left_line, plot=plot_bool)

        # exit(0)
        window_left = [8183, 8188]
        # trimmed wavelength and flux arrays from the uniformly interpolated arrays
        wave_left, flux_l_hs = trim_in_window(wv_hslab_left, hslab_flux_left, window_left)
        wave_left, flux_l_bb = trim_in_window(wv_bbody_left, bbody_flux_left, window_left)

        # getting the continuum also in that window
        var, conti_l_hs = trim_in_window(wv_hslab_left, conti_hslab_left, window_left)
        var, conti_l_bb = trim_in_window(wv_bbody_left, conti_bbody_left, window_left)

        window_right = [8327, 8334]
        wave_right, flux_r_hs = trim_in_window(wv_hslab_right, hslab_flux_right, window_right)
        wave_right, flux_r_bb = trim_in_window(wv_bbody_right, bbody_flux_right, window_right)
        # getting the continuum also in that window
        var, conti_r_hs = trim_in_window(wv_hslab_right, conti_hslab_right, window_right)
        var, conti_r_bb = trim_in_window(wv_bbody_right, conti_bbody_right, window_right)



        print("Equivalent line widths")
        eq_width_l_hs = np.trapezoid((1 - flux_l_hs/conti_l_hs), window_left)*u.AA
        eq_width_l_bb = np.trapezoid((1 - flux_l_bb/conti_l_bb), window_left)*u.AA
        eq_width_r_hs = np.trapezoid((1 - flux_r_hs/conti_r_hs), window_right)*u.AA
        eq_width_r_bb = np.trapezoid((1 - flux_r_bb/conti_r_bb), window_right)*u.AA
        print(f"Eq. line width: (left)-->  Hslab: {eq_width_l_hs} \t Bbody {eq_width_l_bb}")
        print(f"Eq. line width: (right)-->  Hslab: {eq_width_r_hs} \t Bbody {eq_width_r_bb}")


        print("ratio left_line_width/right_line_width")
        ratio_hs = eq_width_l_hs/eq_width_r_hs
        ratio_bb = eq_width_l_bb/eq_width_r_bb
        print(f"Ratios ---> Hslab: {ratio_hs} \t Bbody {ratio_bb}")



        plot_bool = True
        if plot_bool:
            plt.title("Normalised Line-depth variation\nNa I 8185.5055 Å")
            plt.plot(wave_left, flux_l_hs, label="Spectrum with H-Slab Model")
            plt.plot(wave_left, flux_l_bb, label="Spectrum with BlackBody Model")
            plt.plot(wave_left, conti_l_hs, label="Continuum Fit for H-Slab")
            plt.plot(wave_left, conti_l_bb, label="Continuum Fit for Blackbody")
            plt.xlabel("Wavelength in Angstrom")
            plt.ylabel("Normalised Flux")
            plt.ylim((0.1, 1.3))
            plt.text(8184, 0.20, f"Eq width (Hslab): {(eq_width_l_hs.value):.3f} Ang")
            plt.text(8184, 0.15, f"Eq width (Blackbody): {(eq_width_l_bb.value):.3f} Ang")
            plt.grid()
            plt.legend(loc="upper left", fontsize="12")
            plt.savefig(f"{plot_dir}/na_8185.pdf")
            plt.show()
        # exit(0)
        if plot_bool:
            plt.title("Normalised Line-depth variation\nFe I 8329.3436 Å")
            plt.plot(wave_right, flux_r_hs, label="Spectrum with H-Slab Model")
            plt.plot(wave_right, flux_r_bb, label="Spectrum with BlackBody Model")
            plt.plot(wave_right, conti_r_hs, label="Fitted-Continuum for H-Slab")
            plt.plot(wave_right, conti_r_bb, label="Fitted-Continuum for Blackbody")
            plt.xlabel("Wavelength in Angstrom")
            plt.ylabel("Normalised Flux")
            plt.ylim((0.2, 1.35))
            plt.text(8328, 0.33, f"Eq width (Hslab): {(eq_width_r_hs.value):.3f} Ang")
            plt.text(8328, 0.27, f"Eq width (Blackbody): {(eq_width_r_bb.value):.3f} Ang")
            plt.legend(loc="upper left", fontsize="12")
            plt.grid()
            plt.savefig(f"{plot_dir}/fe_8329.pdf")
            plt.show()



