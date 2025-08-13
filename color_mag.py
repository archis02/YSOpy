import numpy as np
import matplotlib.pyplot as plt
from ysopy import utils
import astropy.constants as const
import ysopy.base_funcs as bf
from astropy.io.votable import parse
from scipy.interpolate import interp1d
import astropy.units as u
import smplotlib
import os
from tqdm import tqdm
from dust_extinction.parameter_averages import F19
from dust_extinction.averages import G21_MWAvg

save_dir = "data/color_mag_diagram"
plot_dir = "../plots/color_mag_diagram"
config = utils.config_read_bare("ysopy/config_file.cfg")
log_m_dot_arr = np.linspace(-10, -4, 50)
m_arr = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.3, 1.5])
# m_arr = np.array([1.3])
np.save(f"{save_dir}/log_m_dot_arr", log_m_dot_arr)


def update_photo_temp(config, generate_temp=False, generate_radius=False):
    """
    This function is an interpolator for finding the photospheric temperature
    and the radius of the star. THis is the data from Baraffe model, Gaia server
    1 myr isochrone.
    :param config: Dictionary of config parameters
    :param generate_temp: Switch to control if use interpolated temperature
    :param generate_radius: Switch to control if use interpolated radius
    :return: Updated r_star, and photospheric temperatures
    """
    m = np.array(
        [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.072, 0.075, 0.08, 0.09, 0.1, 0.11, 0.13, 0.15, 0.17, 0.2,
         0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4])
    temp_arr = np.array(
        [2345, 2504, 2598, 2710, 2779, 2824, 2864, 2897, 2896, 2897, 2898, 2908, 2936, 2955, 3012, 3078, 3142, 3226,
         3428, 3634, 3802, 3955, 4078, 4192, 4290, 4377, 4456, 4529, 4596, 4658])
    rad_arr = np.array(
        [0.271, 0.326, 0.372, 0.467, 0.536, 0.628, 0.702, 0.781, 0.803, 0.829, 0.877, 0.959, 1.002, 1.079, 1.214, 1.327,
         1.465, 1.503, 1.636, 1.753, 1.87, 1.971, 2.096, 2.2, 2.31, 2.416, 2.52, 2.612, 2.71, 2.797])
    # func_temp = interp1d(m, temp_arr)
    # func_rad = interp1d(m, rad_arr)
    mass_ratio = config["m"]/const.M_sun.value
    if generate_temp:
        # config["t_star"] = int(func_temp(config["m"]/const.M_sun)/100) * 100 * u.K
        config["t_star"] = int(np.interp(mass_ratio, m, temp_arr) / 100) * 100 #* u.K
    if generate_radius:
        # config["r_star"] = func_rad(config["m"]/const.M_sun) * const.R_sun
        config["r_star"] = np.interp(mass_ratio, m, rad_arr) * const.R_sun.value
    # print(f"Using T_photo : {config['t_star']}\nUsing R_star : {config['r_star']/const.M_sun}")


def generate_total_flux(config, log_m_dot_arr, save_loc):
    for i in range(len(log_m_dot_arr)):
    # for i in range(22, 31):
        print(i+1)
        config["m_dot"] = 10**log_m_dot_arr[i] * const.M_sun.value / 31557600.0
        wavelength, ext_total_flux, total_flux, obs_viscous_disk_flux, obs_dust_flux, obs_mag_flux, obs_star_flux = bf.total_spec(config)
        # total_flux = total_flux
        np.save(f"{save_loc}/ext_total_flux_{i}", ext_total_flux)
        np.save(f"{save_loc}/total_flux_{i}.npy", total_flux)
        np.save(f"{save_loc}/obs_viscous_disk_flux_{i}.npy", obs_viscous_disk_flux)
        np.save(f"{save_loc}/obs_dust_flux_{i}.npy", obs_dust_flux)
        np.save(f"{save_loc}/obs_mag_flux_{i}.npy", obs_mag_flux)
        np.save(f"{save_loc}/obs_star_flux_{i}.npy", obs_star_flux)
    np.save(f"{save_loc}/wavelength.npy", wavelength)






### Generating the magnitudes

# Gaia bands
address_G = "../filter_profiles/GAIA.GAIA3.G.xml"
address_BP = "../filter_profiles/GAIA.GAIA3.Gbp.xml"
address_RP = "../filter_profiles/GAIA.GAIA3.Grp.xml"
# address = "../GAIA.GAIA3.Grvs.xml"
# wise bands
address_W2 = "../filter_profiles/WISE.WISE.W2.xml"
address_W1 = "../filter_profiles/WISE.WISE.W1.xml"
# 2Mass Systems
address_J = "../filter_profiles/2MASS.2MASS.J.xml"
address_Ks = "../filter_profiles/2MASS.2MASS.Ks.xml"
address_H = "../filter_profiles/2MASS.2MASS.H.xml"
# Sloan Systems
address_u = "../filter_profiles/SLOAN.SDSS.u.xml"
address_u_prime = "../filter_profiles/SLOAN.SDSS.uprime_filter.xml"

# lambda effective
# l_eff = 5850.88 * u.AA  # Gaia G Band

# Zero point energy  # Vega Systems are included here
e_zero_G = 2.49769e-9 * (u.erg / (u.cm ** 2 * u.s * u.AA))
e_zero_RP = 1.23742e-9 * (u.erg / (u.cm ** 2 * u.s * u.AA))
e_zero_BP = 4.10926e-9 * (u.erg / (u.cm ** 2 * u.s * u.AA))
e_zero_W2 = 2.36824e-12 * (u.erg / (u.cm ** 2 * u.s * u.AA))
e_zero_W1 = 8.02178e-12 * (u.erg / (u.cm ** 2 * u.s * u.AA))
e_zero_J = 3.09069e-10 * (u.erg / (u.cm ** 2 * u.s * u.AA))
e_zero_H = 1.11933e-10 * (u.erg / (u.cm ** 2 * u.s * u.AA))
e_zero_Ks = 4.20615e-11 * (u.erg / (u.cm ** 2 * u.s * u.AA))
e_zero_u = 3.75079e-9 * (u.erg / (u.cm ** 2 * u.s * u.AA))
e_zero_u_prime = 3.56266e-9 * (u.erg / (u.cm ** 2 * u.s * u.AA))


# for wise data we have to trim the wavelength regime or generate the data in larger domain


# CHECK FOR THE NORMALISATION HERE
# NEED A NORMALISED TRANSMISSION FUNCTION
def get_photometry(address, e_zero, wave, flx):
    table = parse(address)
    data = table.get_first_table().array
    l_data = np.zeros(len(data))
    trans_data = np.zeros(len(data))
    for i in range(len(data)):
        l_data[i] = data[i][0]
        trans_data[i] = data[i][1]
    # print("Min wave :", l_data[0], "\nMax wave :", l_data[-1])
    # print("Min wave :", wave[0], "\nMax wave :", wave[-1])
    # plt.title("Transmission function for Gaia Filter")
    # plt.plot(l_data, trans_data)
    # plt.show()

    # determining the widths
    l_width_arr = np.zeros(len(l_data))
    # w_o
    l_width_arr[0] = (l_data[1] - l_data[0])
    for i in range(1, len(l_data) - 1):
        l_width_arr[i] = 0.5 * (l_data[i + 1] - l_data[i - 1])
    l_width_arr[-1] = (l_data[-1] - l_data[-2])
    # print(l_width_arr)
    # Area under transmission curve
    tot_area = sum(l_width_arr * trans_data)

    # normalised transmission function
    # norm_trans_data = (l_width_arr * trans_data / tot_area)
    norm_trans_data = trans_data
    # trimming the wavelengths within the range of filters

    # HERE WE ARE TAKING EXTRA BECAUSE WE WANT INTERPOLATION TO BE VALID WITHIN THE WAVELENGTH SPECIFIED BY L_DATA
    # ERROR WOULD COME UP IN THE LINE WHERE INTERPOLATED FUNCTION IS CALLED.
    # print(wave, min(l_data))
    valid_index = np.where(wave > min(l_data) * 9 / 10, 1, 0)
    valid_index = np.where(wave < max(l_data) * 11 / 10, valid_index, 0)
    trimmed_wave = []
    trimmed_flx = []

    for i in range(len(wave)):
        if valid_index[i] == 1:
            trimmed_wave.append(wave[i])
            trimmed_flx.append(flx[i])
    trimmed_wave = np.array(trimmed_wave)
    trimmed_flx = np.array(trimmed_flx) * u.erg / (u.cm * u.cm * u.s * u.AA)

    f = interp1d(trimmed_wave, trimmed_flx)
    new_flux = f(l_data) * u.erg / (u.cm * u.cm * u.s * u.AA)
    flux_mean = np.sum(new_flux * norm_trans_data * l_width_arr) / np.sum(norm_trans_data * l_width_arr)
    magnitude = -2.5 * np.log10((flux_mean / e_zero))
    # print("Magnitude :", magnitude)
    return magnitude

def lightcurve(log_m_dot_arr, save_loc):
    wv = np.load(f"{save_loc}/wavelength.npy")
    # print(wv)
    time_arr = np.linspace(0, len(log_m_dot_arr), len(log_m_dot_arr))
    mag_arr_G = np.zeros(len(log_m_dot_arr))
    mag_arr_RP = np.zeros(len(log_m_dot_arr))
    mag_arr_BP = np.zeros(len(log_m_dot_arr))
    mag_arr_W2 = np.zeros(len(log_m_dot_arr))
    mag_arr_W1 = np.zeros(len(log_m_dot_arr))
    mag_arr_J = np.zeros(len(log_m_dot_arr))
    mag_arr_H = np.zeros(len(log_m_dot_arr))
    mag_arr_Ks = np.zeros(len(log_m_dot_arr))
    mag_arr_u = np.zeros(len(log_m_dot_arr))
    mag_arr_u_prime = np.zeros(len(log_m_dot_arr))
    
    for i in range(len(log_m_dot_arr)):
        print(f"----------- {i} -----------")
        total_flux = np.load(f"{save_loc}/ext_total_flux_{i}.npy")
        mag_G = get_photometry(address_G, e_zero_G, wv, total_flux)
        mag_RP = get_photometry(address_RP, e_zero_RP, wv, total_flux)
        mag_BP = get_photometry(address_BP, e_zero_BP, wv, total_flux)
        mag_W1 = get_photometry(address_W1, e_zero_W1, wv, total_flux)
        mag_W2 = get_photometry(address_W2, e_zero_W2, wv, total_flux)
        mag_J = get_photometry(address_J, e_zero_J, wv, total_flux)
        mag_H = get_photometry(address_H, e_zero_H, wv, total_flux)
        mag_Ks = get_photometry(address_Ks, e_zero_Ks, wv, total_flux)
        mag_u = get_photometry(address_u, e_zero_u, wv, total_flux)
        mag_u_prime = get_photometry(address_u_prime, e_zero_u_prime, wv, total_flux)

        mag_arr_G[i] = mag_G
        mag_arr_RP[i] = mag_RP
        mag_arr_BP[i] = mag_BP
        mag_arr_W1[i] = mag_W1
        mag_arr_W2[i] = mag_W2
        mag_arr_J[i] = mag_J
        mag_arr_H[i] = mag_H
        mag_arr_Ks[i] = mag_Ks
        mag_arr_u[i] = mag_u
        mag_arr_u_prime[i] = mag_u_prime

    np.save(f"{save_loc}/time_arr.npy", time_arr)
    np.save(f"{save_loc}/mag_G.npy", mag_arr_G)
    np.save(f"{save_loc}/mag_RP.npy", mag_arr_RP)
    np.save(f"{save_loc}/mag_BP.npy", mag_arr_BP)
    np.save(f"{save_loc}/mag_W1.npy", mag_arr_W1)
    np.save(f"{save_loc}/mag_W2.npy", mag_arr_W2)
    np.save(f"{save_loc}/mag_J.npy", mag_arr_J)
    np.save(f"{save_loc}/mag_H.npy", mag_arr_H)
    np.save(f"{save_loc}/mag_Ks.npy", mag_arr_Ks)
    np.save(f"{save_loc}/mag_u.npy", mag_arr_u)
    np.save(f"{save_loc}/mag_u_prime.npy", mag_arr_u_prime)


# dictionary = dict(g ="mag_arr_G", rp="mag_arr_RP", bp="mag_arr_BP", w1="mag_arr_W1", w2="mag_arr_W2",
#                   j="mag_arr_J", h="mag_arr_H", k="mag_arr_Ks", u="mag_arr_u", u_prime="mag_arr_u_prime")
def color_mag_diagram(save_loc, magy:str, magx1:str, magx2:str):
    dictionary = dict(g="mag_G", rp="mag_RP", bp="mag_BP", w1="mag_W1", w2="mag_W2",
                      j="mag_J", h="mag_H", k="mag_Ks", u="mag_u", u_prime="mag_u_prime")
    magy1 = np.load(f"{save_loc}/{dictionary[f"{magy}"]}.npy")
    magx1_val = np.load(f"{save_loc}/{dictionary[f"{magx1}"]}.npy")
    magx2_val = np.load(f"{save_loc}/{dictionary[f"{magx2}"]}.npy")

    color_x = magx1_val - magx2_val
    plt.plot(color_x, magy1)
    plt.xlabel(f"{magx1} - {magx2}")
    plt.ylabel(f"{magy}")
    plt.legend()
    plt.show()


def generate_grid_m_mdot(config, log_m_dot_arr, m_arr, save_dir):
    for i in range(len(m_arr)):
        print(f"\n\n\ni = {i}/{len(m_arr)}\n\n\n")
        m = m_arr[i]
        config["m"] = m * const.M_sun.value  # kg
        update_photo_temp(config=config, generate_temp=True, generate_radius=True)

        dir_name = f"m_{m}"
        if not os.path.exists(f"{save_dir}/{dir_name}"):
            os.mkdir(f"{save_dir}/{dir_name}")

        save_loc = f"{save_dir}/{dir_name}"

        generate_total_flux(config=config, log_m_dot_arr=log_m_dot_arr, save_loc=save_loc)
        lightcurve(log_m_dot_arr=log_m_dot_arr, save_loc=save_loc)


# generate_total_flux(config, log_m_dot_arr, save_loc)
# m = m_arr[0]
# dir_name = f"m_{m}"
# save_loc = f"{save_dir}/{dir_name}"
# lightcurve(log_m_dot_arr, save_loc)
# generate_grid_m_mdot(config=config, log_m_dot_arr=log_m_dot_arr, m_arr=m_arr, save_dir=save_dir)
# exit(0)
def generate_grid_color_mag_diagram(m_arr, save_dir, magy:str, magx1:str, magx2:str, plot=False, plot_dir=None):
    fig, ax = plt.subplots()
    color_arr = ["red", "pink", "blue", "purple", "green", "magenta", "black", "indigo", "orange", "gray"]
    for i in range(0, len(m_arr), 2):
        m = m_arr[i]
        dir_name = f"m_{m}"
        save_loc = f"{save_dir}/{dir_name}"
        dictionary = dict(g="mag_G", rp="mag_RP", bp="mag_BP", w1="mag_W1", w2="mag_W2",
                          j="mag_J", h="mag_H", k="mag_Ks", u="mag_u", u_prime="mag_u_prime")
        magy1 = np.load(f"{save_loc}/{dictionary[f"{magy}"]}.npy")
        magx1_val = np.load(f"{save_loc}/{dictionary[f"{magx1}"]}.npy")
        magx2_val = np.load(f"{save_loc}/{dictionary[f"{magx2}"]}.npy")
        print(f"\t\tM_dot\tmagy\tmagx1\t\tmagx2\t\tcolor")
        for j in range(len(log_m_dot_arr)):
            print(f"Frame: {j+1}\t{log_m_dot_arr[j]:.3f}\t{magy1[j]:.3f}\t{magx1_val[j]:.5f}\t{magx2_val[j]:.5f}\t{(magx1_val[j] - magx2_val[j]):.3f}")
        color_x = magx1_val - magx2_val
        label_m = "$M_{\odot}$"
        label_m_left = "$M_{*}$"
        ax.plot(color_x, magy1, "*-", label=f"{label_m_left}: {m} {label_m}", color=color_arr[i])
    ax.grid(True)
    ax.invert_yaxis()
    ax.legend()
    plt.xlabel(f"{magx1} - {magx2} (color)")
    plt.ylabel(f"{magy} (absolute magnitude)")
    # plt.savefig(f"{plot_dir}/color_mag_{magx1}-{magx2}_vs_{magy}.pdf")
    if plot:
        plt.show()
    else:
        plt.close()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def generate_grid_color_mag_diagram_gpt(
        m_arr, save_dir, magy: str, magx1: str, magx2: str,
        plot=False, plot_dir=None,
        xlim_inset=None, ylim_inset=None  # new optional zoom params
):
    fig, ax = plt.subplots(figsize=(7, 6))
    color_arr = ["red", "pink", "blue", "purple", "green", "magenta", "black", "indigo", "orange", "gray"]

    dictionary = dict(g="mag_G", rp="mag_RP", bp="mag_BP", w1="mag_W1", w2="mag_W2",
                      j="mag_J", h="mag_H", k="mag_Ks", u="mag_u", u_prime="mag_u_prime")

    for i in range(0, len(m_arr), 2):
        m = m_arr[i]
        dir_name = f"m_{m}"
        save_loc = f"{save_dir}/{dir_name}"

        magy1 = np.load(f"{save_loc}/{dictionary[magy]}.npy")
        magx1_val = np.load(f"{save_loc}/{dictionary[magx1]}.npy")
        magx2_val = np.load(f"{save_loc}/{dictionary[magx2]}.npy")

        color_x = magx1_val - magx2_val
        label_m = "$M_{\odot}$"
        label_m_left = "$M_{*}$"

        ax.plot(color_x, magy1, "*-", label=f"{label_m_left}: {m} {label_m}", color=color_arr[i])

    ax.grid(True)
    ax.invert_yaxis()
    ax.legend()
    ax.set_xlabel(f"{magx1} - {magx2} (color)")
    ax.set_ylabel(f"{magy} (absolute magnitude)")
    # ---- ZOOMED INSET ----
    if xlim_inset and ylim_inset:
        axins = inset_axes(ax, width="35%", height="35%", borderpad=2)
        for i in range(0, len(m_arr), 2):
            m = m_arr[i]
            dir_name = f"m_{m}"
            save_loc = f"{save_dir}/{dir_name}"

            magy1 = np.load(f"{save_loc}/{dictionary[magy]}.npy")
            magx1_val = np.load(f"{save_loc}/{dictionary[magx1]}.npy")
            magx2_val = np.load(f"{save_loc}/{dictionary[magx2]}.npy")

            color_x = magx1_val - magx2_val
            axins.plot(color_x, magy1, "*-", color=color_arr[i])

        axins.set_xlim(*xlim_inset)
        axins.set_ylim(*ylim_inset)
        axins.invert_yaxis()
        axins.grid(True)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # ---- Output or Save ----
    if plot:
        plt.show()
    else:
        if plot_dir:
            out_file = f"{plot_dir}/color_mag_{magx1}-{magx2}_vs_{magy}_inset.pdf"
            plt.savefig(out_file, bbox_inches='tight')
        plt.close()


# generate_grid_color_mag_diagram(m_arr=m_arr, save_dir=save_dir, magy="u", magx1="w1", magx2="w2", plot_dir=plot_dir, plot=True)
# generate_grid_color_mag_diagram(m_arr=m_arr, save_dir=save_dir, magy="u", magx1="bp", magx2="rp", plot_dir=plot_dir, plot=True)

# generate_grid_color_mag_diagram(m_arr=m_arr, save_dir=save_dir, magy="g", magx1="bp", magx2="rp", plot_dir=plot_dir, plot=True)
# generate_grid_color_mag_diagram_gpt(xlim_inset=[6.60, 6.63], ylim_inset=[13.90, 13.94], m_arr=m_arr, save_dir=save_dir, magy="g", magx1="bp", magx2="rp", plot_dir=plot_dir, plot=True)
generate_grid_color_mag_diagram_gpt(m_arr=m_arr, save_dir=save_dir, magy="w1", magx1="w1", magx2="w2", plot_dir=plot_dir, plot=True)

# generate_grid_color_mag_diagram(m_arr=m_arr, save_dir=save_dir, magy="w1", magx1="w1", magx2="w2", plot_dir=plot_dir, plot=True)

exit(0)
def generate_flux_except_hslab(config, log_m_dot_arr, save_loc):
    for i in range(len(log_m_dot_arr)):
        config["m_dot"] = 10**log_m_dot_arr[i] * const.M_sun.value / 31557600.0
        bf.calculate_n_data(config)
        dr, t_max, d, r_in, r_sub = bf.generate_temp_arr(config)
        wavelength, obs_viscous_disk_flux = bf.generate_visc_flux(config, d, t_max, dr)
        # obs_mag_flux = bf.magnetospheric_component_calculate(config, r_in)
        obs_dust_flux = bf.generate_dusty_disk_flux(config, r_in, r_sub)
        obs_star_flux = bf.generate_photosphere_flux(config)
        total_flux = obs_star_flux + obs_viscous_disk_flux + obs_dust_flux# + obs_mag_flux
        # total_flux = bf.dust_extinction_flux(config, wavelength, obs_viscous_disk_flux,
        #                                      obs_star_flux, obs_mag_flux, obs_dust_flux)
        np.save(f"{save_loc}/total_flux_without_hslab_and_extinction{i}.npy", total_flux)
    np.save(f"{save_loc}/wavelength.npy", wavelength)






def dust_extinction_flux(config, wavelength, total_unextincted_flux):
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

    total = total_unextincted_flux

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


tslab_arr = np.array([3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000])
log10_ne_arr = np.array([11, 12, 13, 14, 15, 16])
tau_arr = np.array([0.1, 0.5, 1, 5])

######################
# THE BELOW CODE IS TO GENERATE THE TOTAL FLUX CORRESPONDING TO
# EACH M AND M_DOT WITHOUT THE H_SLAB MODEL AND DUST EXTINCTION

# for i in range(len(m_arr)):
#     print(f"\n\n\ni = {i}/{len(m_arr)}\n\n\n")
#     m = m_arr[i]
#     config["m"] = m * const.M_sun.value  # kg
#     update_photo_temp(config=config, generate_temp=True, generate_radius=True)
#
#     dir_name = f"m_{m}"
#     if not os.path.exists(f"{save_dir}/{dir_name}"):
#         os.mkdir(f"{save_dir}/{dir_name}")
#
#     save_loc = f"{save_dir}/{dir_name}"
#
#     generate_flux_except_hslab(config, log_m_dot_arr, save_loc=save_loc)


# exit(0)
#######################


# exit(0)
# #######################
# # Below code is to run the model over M, M_dot, T_slab, Tau, Ne and compute the Hslab flux,
# # Add the total unextinguished flux to it and finally run the dust extinction and save the spectra.
#
"""for n in range(6, 7):  # just m=1.0 case is being done # others will be done when necessary
    m = m_arr[n]
    for l in range(len(log_m_dot_arr)):
        config["m_dot"] = 10**log_m_dot_arr[l] * const.M_sun.value / 31557600.0
        bf.calculate_n_data(config)
        dr, t_max, d, r_in, r_sub = bf.generate_temp_arr(config)

        dir_name_wave = f"m_{m}"
        dir_name = f"m_{m}/tslab_ne_tau"
        if not os.path.exists(f"{save_dir}/{dir_name}"):
            os.mkdir(f"{save_dir}/{dir_name}")

        load_wave_loc = f"{save_dir}/{dir_name_wave}"
        save_loc = f"{save_dir}/{dir_name}"

        wavelength = np.load(f"{load_wave_loc}/wavelength.npy")

        for i in range(len(tslab_arr)):
            config["t_slab"] = tslab_arr[i] * u.K
            for j in range(len(log10_ne_arr)):
                config["n_e"] = 10**(log10_ne_arr[j]) * u.cm**(-3)
                for k in range(len(tau_arr)):
                    config["tau"] = tau_arr[k]
                    # print(m_arr[n], log_m_dot_arr[l], tslab_arr[i], log10_ne_arr[j], tau_arr[k])
                    total_flux = np.load(f"{load_wave_loc}/total_flux_without_hslab_and_extinction{l}.npy")
                    mag_flux = bf.magnetospheric_component_calculate(config, r_in)
                    total_flux = total_flux + mag_flux
                    total_extincted_flux = dust_extinction_flux(config, wavelength=wavelength, total_unextincted_flux=total_flux)
                    np.save(f"{save_loc}/log_m_dot_{log_m_dot_arr[l]:.2f}_t_slab_{tslab_arr[i]}_ne_{log10_ne_arr[j]}_tau_{tau_arr[k]}.npy", total_flux)
"""

def lightcurve_grid(log_m_dot_arr, load_wave_loc, load_flux_name, save_loc, ts, log10ne, tau):
    wv = np.load(f"{load_wave_loc}/wavelength.npy")
    # print(wv)
    time_arr = np.linspace(0, len(log_m_dot_arr), len(log_m_dot_arr))
    mag_arr_G = np.zeros(len(log_m_dot_arr))
    mag_arr_RP = np.zeros(len(log_m_dot_arr))
    mag_arr_BP = np.zeros(len(log_m_dot_arr))
    mag_arr_W2 = np.zeros(len(log_m_dot_arr))
    mag_arr_W1 = np.zeros(len(log_m_dot_arr))
    mag_arr_J = np.zeros(len(log_m_dot_arr))
    mag_arr_H = np.zeros(len(log_m_dot_arr))
    mag_arr_Ks = np.zeros(len(log_m_dot_arr))
    mag_arr_u = np.zeros(len(log_m_dot_arr))
    mag_arr_u_prime = np.zeros(len(log_m_dot_arr))

    for i in range(len(log_m_dot_arr)):
        print(f"----------- {i} -----------")
        total_flux = np.load(
            f"{load_flux_name}/log_m_dot_{log_m_dot_arr[i]:.2f}_t_slab_{ts}_ne_{log10ne}_tau_{tau}.npy")
        mag_G = get_photometry(address_G, e_zero_G, wv, total_flux)
        mag_RP = get_photometry(address_RP, e_zero_RP, wv, total_flux)
        mag_BP = get_photometry(address_BP, e_zero_BP, wv, total_flux)
        mag_W1 = get_photometry(address_W1, e_zero_W1, wv, total_flux)
        mag_W2 = get_photometry(address_W2, e_zero_W2, wv, total_flux)
        mag_J = get_photometry(address_J, e_zero_J, wv, total_flux)
        mag_H = get_photometry(address_H, e_zero_H, wv, total_flux)
        mag_Ks = get_photometry(address_Ks, e_zero_Ks, wv, total_flux)
        mag_u = get_photometry(address_u, e_zero_u, wv, total_flux)
        mag_u_prime = get_photometry(address_u_prime, e_zero_u_prime, wv, total_flux)

        mag_arr_G[i] = mag_G
        mag_arr_RP[i] = mag_RP
        mag_arr_BP[i] = mag_BP
        mag_arr_W1[i] = mag_W1
        mag_arr_W2[i] = mag_W2
        mag_arr_J[i] = mag_J
        mag_arr_H[i] = mag_H
        mag_arr_Ks[i] = mag_Ks
        mag_arr_u[i] = mag_u
        mag_arr_u_prime[i] = mag_u_prime

    np.save(f"{save_loc}/{ts}_{log10ne}_{tau}_time_arr.npy", time_arr)
    np.save(f"{save_loc}/{ts}_{log10ne}_{tau}_mag_G.npy", mag_arr_G)
    np.save(f"{save_loc}/{ts}_{log10ne}_{tau}_mag_RP.npy", mag_arr_RP)
    np.save(f"{save_loc}/{ts}_{log10ne}_{tau}_mag_BP.npy", mag_arr_BP)
    np.save(f"{save_loc}/{ts}_{log10ne}_{tau}_mag_W1.npy", mag_arr_W1)
    np.save(f"{save_loc}/{ts}_{log10ne}_{tau}_mag_W2.npy", mag_arr_W2)
    np.save(f"{save_loc}/{ts}_{log10ne}_{tau}_mag_J.npy", mag_arr_J)
    np.save(f"{save_loc}/{ts}_{log10ne}_{tau}_mag_H.npy", mag_arr_H)
    np.save(f"{save_loc}/{ts}_{log10ne}_{tau}_mag_Ks.npy", mag_arr_Ks)
    np.save(f"{save_loc}/{ts}_{log10ne}_{tau}_mag_u.npy", mag_arr_u)
    np.save(f"{save_loc}/{ts}_{log10ne}_{tau}_mag_u_prime.npy", mag_arr_u_prime)

#####
# Using the below code you can generate a distribution of spectra by varying one of the slab parameters at a time
# Say want to vary t_slab, comment tslab line below and run the for loop in "i" for changing the t_slab params.
# Change the saving location, and np.save format aswell. Likewise for the other two parameters
#####
"""m = m_arr[6]
config["t_slab"] = tslab_arr[5] * u.K
# config["n_e"] = 10**(log10_ne_arr[2]) * u.cm**(-3)
config["tau"] = tau_arr[2]
dir_name_wave = f"m_{m}"
load_wave_loc = f"{save_dir}/{dir_name_wave}"
wavelength = np.load(f"{load_wave_loc}/wavelength.npy")
dir_name = f"m_{m}/ne"
save_loc = f"{save_dir}/{dir_name}"
print("m", m)
# print("t_slab", config["t_slab"])
# print("n_e", config["n_e"])
print(dir_name_wave)
print(load_wave_loc)
print(dir_name)
print(save_loc)

for l in range(len(log_m_dot_arr)):
    config["m_dot"] = 10 ** log_m_dot_arr[l] * const.M_sun.value / 31557600.0
    total_flux = np.load(f"{load_wave_loc}/total_flux_without_hslab_and_extinction{l}.npy")
    for i in range(len(log10_ne_arr)):
        # config["t_slab"] = tslab_arr[i] * u.K
        # config["tau"] = tau_arr[i]
        config["n_e"] = 10 ** (log10_ne_arr[i]) * u.cm ** (-3)

        bf.calculate_n_data(config)
        dr, t_max, d, r_in, r_sub = bf.generate_temp_arr(config)
        mag_flux = bf.magnetospheric_component_calculate(config, r_in)
        total_flux = total_flux + mag_flux
        total_extincted_flux = dust_extinction_flux(config, wavelength=wavelength, total_unextincted_flux=total_flux)
        np.save(
            f"{save_loc}/log_m_dot_{log_m_dot_arr[l]:.2f}_t_slab_{tslab_arr[5]}_ne_{log10_ne_arr[i]}_tau_{tau_arr[2]}.npy",
            total_flux)"""
# exit(0)


##########
# Generate Lightcurve
##########
"""m = m_arr[6]
## among the next 3 lines comment that parameter which you want to vary
config["t_slab"] = tslab_arr[5] * u.K
config["n_e"] = 10**(log10_ne_arr[2]) * u.cm**(-3)
# config["tau"] = tau_arr[2]

dir_name_wave = f"m_{m}"
load_wave_loc = f"{save_dir}/{dir_name_wave}"
wavelength = np.load(f"{load_wave_loc}/wavelength.npy")
dir_name = f"m_{m}/tau"  # change the directory here
save_loc = f"{save_dir}/{dir_name}"
# print("m", m)
# print("t_slab", config["t_slab"])
# print("n_e", config["n_e"])
# print(dir_name_wave)
# print(load_wave_loc)
# print(dir_name)
# print(save_loc)
for i in range(len(tau_arr)):  # change btween tau/ne/tslab array which ever you are varying
    print("tau", tau_arr[i])
    if not os.path.exists(f"{save_loc}/{tau_arr[i]}_mag"):
        os.mkdir(f"{save_loc}/{tau_arr[i]}_mag")
    save_mag_path = f"{save_loc}/{tau_arr[i]}_mag"
    lightcurve_grid(log_m_dot_arr=log_m_dot_arr, load_wave_loc=load_wave_loc, load_flux_name=save_loc,
                    save_loc=save_mag_path, ts=tslab_arr[5], log10ne=log10_ne_arr[2], tau=tau_arr[i])  # change this aswell
    

exit(0)"""













# ###################################


#
# for n in range(6, 7):  # just m=1.0 case is being done # others will be done when necessary
#     m = m_arr[n]
#     dir_name_wave = f"m_{m}"
#     dir_flux_name = f"m_{m}/tslab_ne_tau"
#     dir_name = f"m_{m}/save_mag"
#     if not os.path.exists(f"{save_dir}/{dir_name}"):
#         os.mkdir(f"{save_dir}/{dir_name}")
#     load_wave_loc = f"{save_dir}/{dir_name_wave}"
#     save_loc = f"{save_dir}/{dir_name}"
#     load_flux_name = f"{save_dir}/{dir_flux_name}"
#     for i in range(1):#len(tslab_arr)):
#         config["t_slab"] = tslab_arr[i] * u.K
#         for j in range(1):#len(log10_ne_arr)):
#             config["n_e"] = 10**(log10_ne_arr[j]) * u.cm**(-3)
#             for k in range(len(tau_arr)):
#                 config["tau"] = tau_arr[k]
#                 lightcurve_grid(log_m_dot_arr, load_wave_loc, load_flux_name, save_loc, ts=tslab_arr[i],
#                                 log10ne=log10_ne_arr[j], tau=tau_arr[k])


def grid_color_mag_diagram(m_arr, save_dir, magy:str, magx1:str, magx2:str, plot=False, plot_dir=None):
    fig, ax = plt.subplots()
    color_arr = ["red", "pink", "blue", "purple", "green", "magenta", "black", "indigo", "orange", "gray"]
    for i in range(6,7):#0, len(m_arr), 2):
        m = m_arr[i]
        dir_name = f"m_{m}"
        save_loc = f"{save_dir}/{dir_name}"
        dictionary = dict(g="mag_G", rp="mag_RP", bp="mag_BP", w1="mag_W1", w2="mag_W2",
                          j="mag_J", h="mag_H", k="mag_Ks", u="mag_u", u_prime="mag_u_prime")
        dir_flux_name = f"m_{m}/tau"
        load_flux_name = f"{save_dir}/{dir_flux_name}"
        ######
        # variable_arr = log10_ne_arr
        # variable_arr = tslab_arr
        variable_arr = tau_arr

        for j in range(len(variable_arr)):
            t = 8000
            # tau = tau_arr[2]
            ne = 13
            # t = variable_arr[j]
            # ne = variable_arr[j]
            tau = variable_arr[j]
            magy1_ = np.load(f"{load_flux_name}/{variable_arr[j]}_mag/{t}_{ne}_{tau}_{dictionary[f"{magy}"]}.npy")
            magx1_val_ = np.load(f"{load_flux_name}/{variable_arr[j]}_mag/{t}_{ne}_{tau}_{dictionary[f"{magx1}"]}.npy")
            magx2_val_ = np.load(
                f"{load_flux_name}/{variable_arr[j]}_mag/{t}_{ne}_{tau}_{dictionary[f"{magx2}"]}.npy")
            color_x_ = magx1_val_ - magx2_val_ #+ 0.01 * j
            # ax.plot(total_flux, label=f"tau = {tau}")
            ax.plot(color_x_, magy1_, "*-", alpha=0.3, color=color_arr[j], label=f"$\\tau$ = {variable_arr[j]}")
    ax.grid(True)
    ax.invert_yaxis()
    ax.legend()
    plt.xlabel(f"{magx1} - {magx2}")
    plt.ylabel(f"{magy}")
    plt.savefig(f"{plot_dir}/color_mag_{magx1}-{magx2}_vs_{magy}_tau.pdf")
    if plot:
        plt.show()
    else:
        plt.close()
grid_color_mag_diagram(m_arr=m_arr, save_dir=save_dir, magy="g", magx1="bp", magx2="rp", plot_dir=plot_dir, plot=True)
grid_color_mag_diagram(m_arr=m_arr, save_dir=save_dir, magy="w1", magx1="w1", magx2="w2", plot_dir=plot_dir, plot=True)
grid_color_mag_diagram(m_arr=m_arr, save_dir=save_dir, magy="u", magx1="bp", magx2="rp", plot_dir=plot_dir, plot=True)
grid_color_mag_diagram(m_arr=m_arr, save_dir=save_dir, magy="u", magx1="j", magx2="k", plot_dir=plot_dir, plot=True)