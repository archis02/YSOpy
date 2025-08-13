# import numpy as np
# import matplotlib.pyplot as plt
# from ysopy import utils
# import astropy.constants as const
# # import color_mag as cmd
#
# # from astropy.io.votable import parse
# # from scipy.interpolate import interp1d
# # import astropy.units as u
# # import smplotlib
# # import os
#
#
# save_dir = "data/color_mag_diagram"
# plot_dir = "../plots/color_mag_diagram"
# config = utils.config_read_bare("ysopy/config_file.cfg")
# log_m_dot_arr = np.linspace(-10, -4, 50)
# # m_arr = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.3, 1.5])
# m_arr = np.array([0.2])
# for i in range(len(m_arr)):
#     print(f"\n\n\ni = {i}/{len(m_arr)}\n\n\n")
#     m = m_arr[i]
#     config["m"] = m * const.M_sun.value  # kg
#     dir_name = f"m_{m}"
#     save_loc = f"{save_dir}/{dir_name}"
#
#     wavelength = np.load(f"{save_loc}/wavelength.npy")
#
#     ##### plotting the first one
#     j = 0
#     config["m_dot"] = 10 ** log_m_dot_arr[j] * const.M_sun.value / 31557600.0
#     # total_flux = total_flux
#     ext_total_flux = np.load(f"{save_loc}/ext_total_flux_{j}.npy")
#     total_flux = np.load(f"{save_loc}/total_flux_{j}.npy")
#     obs_viscous_disk_flux = np.load(f"{save_loc}/obs_viscous_disk_flux_{j}.npy")
#     obs_dust_flux = np.load(f"{save_loc}/obs_dust_flux_{j}.npy")
#     obs_mag_flux = np.load(f"{save_loc}/obs_mag_flux_{j}.npy")
#     obs_star_flux = np.load(f"{save_loc}/obs_star_flux_{j}.npy")
#     line_photo, = plt.plot(wavelength, np.log10(obs_star_flux), color="blue")
#     line_visc, = plt.plot(wavelength, np.log10(obs_viscous_disk_flux), color="orange")
#     line_dust, = plt.plot(wavelength, np.log10(obs_dust_flux), color="green")
#     line_mag, = plt.plot(wavelength, np.log10(obs_mag_flux), color="red")
#     plt.ylim((-20, -10))
#     ###### Updating the plot for the rest of the plots
#     for j in range(len(log_m_dot_arr)):
#         print(j)
#         config["m_dot"] = 10 ** log_m_dot_arr[j] * const.M_sun.value / 31557600.0
#         # total_flux = total_flux
#         ext_total_flux = np.load(f"{save_loc}/ext_total_flux_{j}.npy")
#         total_flux = np.load(f"{save_loc}/total_flux_{j}.npy")
#         obs_viscous_disk_flux = np.load(f"{save_loc}/obs_viscous_disk_flux_{j}.npy")
#         obs_dust_flux = np.load(f"{save_loc}/obs_dust_flux_{j}.npy")
#         obs_mag_flux = np.load(f"{save_loc}/obs_mag_flux_{j}.npy")
#         obs_star_flux = np.load(f"{save_loc}/obs_star_flux_{j}.npy")
#         line_photo.set_ydata(np.log10(obs_star_flux))
#         line_visc.set_ydata(np.log10(obs_viscous_disk_flux))
#         line_dust.set_ydata(np.log10(obs_dust_flux))
#         line_mag.set_ydata(np.log10(obs_mag_flux))
#         plt.title(f"{j}\t\tM_dot = {log_m_dot_arr[j]:.2f}")
#         plt.pause(0.4)
#     plt.show()
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from ysopy import utils
import astropy.constants as const
from astropy.io.votable import parse

# === Setup ===
save_dir = "data/color_mag_diagram"
plot_dir = "../plots/color_mag_diagram"
config = utils.config_read_bare("ysopy/config_file.cfg")
log_m_dot_arr = np.linspace(-10, -4, 50)
m_arr = np.array([1.3])

dictionary = dict(g="mag_G", rp="mag_RP", bp="mag_BP", w1="mag_W1", w2="mag_W2",
                  j="mag_J", h="mag_H", k="mag_Ks", u="mag_u", u_prime="mag_u_prime")
def extents_of_filter(key:str):
    """Given a filter name, it returns the extent of that filter."""
    address_dict = dict(g="../filter_profiles/GAIA.GAIA3.G.xml",
                        bp="../filter_profiles/GAIA.GAIA3.Gbp.xml",
                        rp="../filter_profiles/GAIA.GAIA3.Grp.xml",
                        w2="../filter_profiles/WISE.WISE.W2.xml",
                        w1="../filter_profiles/WISE.WISE.W1.xml",
                        j="../filter_profiles/2MASS.2MASS.J.xml",
                        h="../filter_profiles/2MASS.2MASS.H.xml",
                        k="../filter_profiles/2MASS.2MASS.Ks.xml",
                        u="../filter_profiles/SLOAN.SDSS.u.xml",
                        u_prime="../filter_profiles/SLOAN.SDSS.uprime_filter.xml")
    address = address_dict[key]
    table = parse(address)
    data = table.get_first_table().array
    l_data = np.zeros(len(data))
    for i in range(len(data)):
        l_data[i] = data[i][0]
    print("Key: ", key, "\nMin wave :", l_data[0], "\nMax wave :", l_data[-1])
    return l_data[0], l_data[-1]
# define the bands to plot
# Magnitude (y axis)
magy = "g"
y_ll, y_ul = extents_of_filter(magy) # y axis's filter's lower and upper limits in wave
# Color (x axis)
magx1 = "bp"
magx2 = "rp"
x1_ll, x1_ul = extents_of_filter(magx1)
x2_ll, x2_ul = extents_of_filter(magx2)
# extents_of_filter("g")
# Wavelength zoom windows in meters
zoom_windows = [
    (y_ll, y_ul),
    (x1_ll, x1_ul),
    (x2_ll, x2_ul),
]
# print(zoom_windows)
# exit(0)
for i in range(len(m_arr)):
    m = m_arr[i]
    config["m"] = m * const.M_sun.value
    dir_name = f"m_{m}"
    save_loc = f"{save_dir}/{dir_name}"

    wavelength = np.load(f"{save_loc}/wavelength.npy")

    # Compute zoom indices once
    zoom_inds = [
        np.where((wavelength >= wmin) & (wavelength <= wmax))[0]
        for (wmin, wmax) in zoom_windows
    ]

    fig, axs = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    axs = axs.flatten()

    # Initialize lines
    lines = []
    for ax_idx, ax in enumerate(axs):
        if ax_idx == 0:
            wl = wavelength
            inds = slice(None)
        else:
            wl = wavelength[zoom_inds[ax_idx - 1]]
            inds = zoom_inds[ax_idx - 1]

        # Dummy frame
        obs_star_flux = np.load(f"{save_loc}/obs_star_flux_0.npy")
        obs_viscous_disk_flux = np.load(f"{save_loc}/obs_viscous_disk_flux_0.npy")
        obs_dust_flux = np.load(f"{save_loc}/obs_dust_flux_0.npy")
        obs_mag_flux = np.load(f"{save_loc}/obs_mag_flux_0.npy")
        total_flux = np.load(f"{save_loc}/total_flux_0.npy")
        ext_total_flux = np.load(f"{save_loc}/ext_total_flux_0.npy")
        l6, = ax.plot(wl, np.log10(ext_total_flux[inds]), color="magenta", label="Ext", alpha=0.6)
        l1, = ax.plot(wl, np.log10(obs_star_flux[inds]), color="blue", label="Star")
        l2, = ax.plot(wl, np.log10(obs_viscous_disk_flux[inds]), color="orange", label="Viscous")
        l3, = ax.plot(wl, np.log10(obs_dust_flux[inds]), color="green", label="Dust")
        l4, = ax.plot(wl, np.log10(obs_mag_flux[inds]), color="red", label="Mag")
        l5, = ax.plot(wl, np.log10(total_flux[inds]), color="black", label="Total")
        lines.append((l1, l2, l3, l4, l5, l6))

        ax.set_xlim(wl[0], wl[-1])
        ax.set_ylim(-20, -8)
        ax.set_title("Initializing...")
        ax.set_xlabel("Wavelength (m)")
        ax.set_ylabel("log10(Flux)")
        ax.legend(fontsize=8)

    # Add progress bar to top subplot
    # progress_bar, = axs[0].plot([wavelength[0], wavelength[0]], [-20, -10], color='black', lw=4, alpha=0.4)

    plt.tight_layout()

    # === Update function ===
    def update(frame):
        j = frame
        config["m_dot"] = 10 ** log_m_dot_arr[j] * const.M_sun.value / 31557600.0
        obs_star_flux = np.load(f"{save_loc}/obs_star_flux_{j}.npy")
        obs_viscous_disk_flux = np.load(f"{save_loc}/obs_viscous_disk_flux_{j}.npy")
        obs_dust_flux = np.load(f"{save_loc}/obs_dust_flux_{j}.npy")
        obs_mag_flux = np.load(f"{save_loc}/obs_mag_flux_{j}.npy")
        total_flux = np.load(f"{save_loc}/total_flux_{j}.npy")
        ext_total_flux = np.load(f"{save_loc}/ext_total_flux_{j}.npy")
        for ax_idx, ax in enumerate(axs):
            if ax_idx == 0:
                inds = slice(None)
                wl = wavelength
            else:
                inds = zoom_inds[ax_idx - 1]
                wl = wavelength[inds]

            lines[ax_idx][0].set_ydata(np.log10(obs_star_flux[inds]))
            lines[ax_idx][1].set_ydata(np.log10(obs_viscous_disk_flux[inds]))
            lines[ax_idx][2].set_ydata(np.log10(obs_dust_flux[inds]))
            lines[ax_idx][3].set_ydata(np.log10(obs_mag_flux[inds]))
            lines[ax_idx][4].set_ydata(np.log10(total_flux[inds]))
            lines[ax_idx][5].set_ydata(np.log10(ext_total_flux[inds]))
            ax.set_title(f"Frame {j+1}/{len(log_m_dot_arr)}: log(Mdot) = {log_m_dot_arr[j]:.2f}")

        # Update progress bar position on top plot
        bar_x = wavelength[0] + (wavelength[-1] - wavelength[0]) * (j + 1) / len(log_m_dot_arr)
        # progress_bar.set_xdata([wavelength[0], bar_x])
        return [line for group in lines for line in group] #+ [progress_bar]

    # === Animation object ===
    anim = FuncAnimation(
        fig,
        update,
        frames=len(log_m_dot_arr),
        interval=300,
        blit=False
    )

    # === Save the video ===
    video_path = f"{plot_dir}/color_mag_animation_m_{m}_{magy}_{magx1}-{magx2}.mp4"
    writer = FFMpegWriter(fps=5, metadata=dict(artist='ysopy'), bitrate=1800)
    anim.save(video_path, writer=writer)
    print(f"🎥 Video saved to: {video_path}")

    plt.close()
