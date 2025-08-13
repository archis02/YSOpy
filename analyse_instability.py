import numpy as np
import matplotlib.pyplot as plt
import unstable_temperature_profile as unstable
from ysopy.utils import config_read_bare
import astropy.units as u
import astropy.constants as const
import line_ratios as line_r
# import smplotlib
config = config_read_bare("ysopy/config_file.cfg")
m_dot_changed = 1e-4 * const.M_sun.value / 31557600.0
r_in_units_of_r_sun = np.array([10, 15, 20, 25])
color_arr = ["orange", "black", "red", "green", "blue"]
r_change = r_in_units_of_r_sun * const.R_sun.value

fig, ax = plt.subplots()

wavelength_unchanged = np.load("../instability_dir/wavelength_unchanged.npy")
ext_total_flux_unchanged = np.load("../instability_dir/flux_unchanged.npy")
wave_arr = np.zeros((5, len(wavelength_unchanged)))
flux_arr = np.zeros((5, len(wavelength_unchanged)))
# Oth row is flux with no instability
wave_arr[0, :] = wavelength_unchanged
flux_arr[0, :] = ext_total_flux_unchanged
for r_ch in range(len(r_change)):
    wavelength = np.load(f"../instability_dir/wave_{r_in_units_of_r_sun[r_ch]}.npy")
    flux = np.load(f"../instability_dir/flux_{r_in_units_of_r_sun[r_ch]}.npy")
    wave_arr[r_ch+1, :] = wavelength
    flux_arr[r_ch+1, :] = flux


### plotting the entire SED for the instability propagation
def plot_entire_SED_instability(wave_arr, flux_arr, r_in_units_f_r_sun):
    # 0th index is the one without any instability
    ax.plot(wave_arr[0], flux_arr[0], label="without instability")

    for r_ch in range(len(r_in_units_f_r_sun)):
        ax.plot(wave_arr[r_ch + 1], flux_arr[r_ch + 1], label=f"r = {r_in_units_of_r_sun[r_ch]} $R_\odot$")

    # Beautify the plot
    ax.legend()
    ax.set_xlabel("Wavelength [Angstrom]")
    ax.set_ylabel("Flux [erg / cm^2 s A]")
    ax.grid(True)
    plt.show()
    plt.close()

# plot_entire_SED_instability(wave_arr, flux_arr, r_in_units_of_r_sun)
# window = [3095, 3106]
# window = [22900, 23100]
window = [5010, 5030]
wave_trimmed, total_flux_trimmed_no_instab = line_r.trim_in_window(wavelength=wave_arr[0], total_flux=flux_arr[0], window=window)

trim_flux_arr = np.zeros((5, len(wave_trimmed)))
for i in range(5):
    extracted_data = line_r.trim_in_window(wavelength=wave_arr[0], total_flux=flux_arr[i],
                                                                       window=window)
    trim_flux_arr[i, :] = extracted_data[1]
    if i != 0:
        plt.plot(extracted_data[0], extracted_data[1], label=f"{r_in_units_of_r_sun[i-1]}", color=color_arr[i])
plt.plot(wave_trimmed, total_flux_trimmed_no_instab, color=color_arr[0])
plt.show()
plt.close()

#### Continuum normalisations
poly_coeffs = [0, 0, 1]
conti_norm_arr = np.zeros(trim_flux_arr.shape)
conti_flux_arr = np.zeros(trim_flux_arr.shape)
conti_interp_flux = np.zeros(trim_flux_arr.shape)
plot_bool = False
for i in range(5):
    # for window = 3095, 3106
    # extracted_data = line_r.calc_continuum_arr(poly_coeffs, threshold_l=0.9, spec_wavelength=wave_trimmed,
    #                                                                              flux=trim_flux_arr[i], plot=plot_bool)
    # for window = 22900, 23100
    # extracted_data = line_r.calc_continuum_arr(poly_coeffs, threshold_l=1, threshold_u=1.1,
    #                                            spec_wavelength=wave_trimmed,
    #                                            flux=trim_flux_arr[i], plot=plot_bool)
    # for window = 22900, 23100
    extracted_data = line_r.calc_continuum_arr(poly_coeffs, threshold_l=0.0, threshold_u=1.1,
                                               spec_wavelength=wave_trimmed,
                                               flux=trim_flux_arr[i], plot=plot_bool)

    conti_norm_arr[i, :] = extracted_data[0]
    conti_flux_arr[i, :] = extracted_data[2]
    conti_interp_flux[i, :] = extracted_data[3]
fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
for i in range(5):
    # ax.plot(wave_trimmed, conti_norm_arr[i], "--")
    # plt.plot(wave_trimmed, conti_flux_arr[i])
    if i == 0:
        ax.plot(wave_trimmed, conti_interp_flux[i], label="without instability", color=color_arr[i])
    else:
        ax.plot(wave_trimmed, conti_interp_flux[i], label=f"Instability at R = {r_in_units_of_r_sun[i-1]}$R_\odot$", color=color_arr[i])

ax.grid(True)
ax.legend()
ax.set_xlabel("Wavelength [Angstrom] ----->")
ax.set_ylabel("Continuum normalised flux ----->")
ax.set_title("CO Line (22930$\AA$)")
# plt.savefig("../plots/instability_plots/CO.pdf")
plt.show()