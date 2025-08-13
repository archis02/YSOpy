from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
import astropy.constants as const
from pypeit.core import wave  # For air->vacuum conversion
import warnings

# === USER INPUT ===
file_path = "/Users/tusharkantidas/github/tifr_2025/xshooter_spectra/ADP.2014-05-17T100341.430.fits"

hdr = fits.getheader(file_path, 0)  # 0 = primary HDU

prog_id = hdr["HIERARCH ESO OBS PROG ID"]
print("Program ID:", prog_id)
# exit(0)
# === LOAD FILE ===
with fits.open(file_path) as hdul:
    hdr0 = hdul[0].header
    tab = hdul[1].data

    # Read wavelength, flux, error, and quality mask
    wave_nm_air = np.array(tab["WAVE"]).ravel()
    flux = np.array(tab["FLUX"]).ravel()
    err = np.array(tab["ERR"]).ravel()
    qual = np.array(tab["QUAL"]).ravel().astype(np.uint32)

# === EXTRACT RA/DEC FROM HEADER ===
def get_coord_from_header(header):
    possible_ra_keys = ["RA", "RA_TARG", "OBJRA", "RA_OBJ"]
    possible_dec_keys = ["DEC", "DEC_TARG", "OBJDEC", "DEC_OBJ"]

    ra_val, dec_val = None, None
    for key in possible_ra_keys:
        if key in header:
            ra_val = header[key]
            break
    for key in possible_dec_keys:
        if key in header:
            dec_val = header[key]
            break

    if ra_val is None or dec_val is None:
        raise ValueError("Could not find RA/DEC in FITS header.")

    # ESO stores RA/DEC sometimes in sexagesimal, sometimes decimal degrees
    try:
        coord = SkyCoord(ra=ra_val, dec=dec_val, unit=(u.hourangle, u.deg))
    except Exception:
        coord = SkyCoord(ra=ra_val, dec=dec_val, unit=(u.deg, u.deg))
    return coord

coord = get_coord_from_header(hdr0)
print(coord)
# === MASK BAD PIXELS ===
good = (qual == 0) & np.isfinite(flux) & np.isfinite(err) & np.isfinite(wave_nm_air)

# === AIR -> VACUUM CONVERSION ===
wave_A_air = wave_nm_air[good] * 10.0 * u.AA  # nm → Å
wave_A_vac = wave.airtovac(wave_A_air).to(u.nm).value
print(wave_A_vac)
# === BARYCENTRIC CORRECTION ===
obstime = Time(hdr0["DATE-OBS"], format="isot", scale="utc")
print("obstime", obstime)
loc = EarthLocation.of_site("Paranal Observatory")
print("loc", loc)
# exit(0)
vcorr = coord.radial_velocity_correction(obstime=obstime, location=loc)  # in m/s
print("vcorr", vcorr)
beta = (vcorr.to(u.km/u.s) / const.c).decompose().value
print("beta", beta)
wave_A_vac_bary = wave_A_vac * (1.0 + beta)
print("wave_A_vac_bary", wave_A_vac_bary)
# exit(0)
# === PLOT RESULT ===
plt.figure(figsize=(10, 4))
# plt.plot(wave_nm_air, flux)
plt.plot(wave_A_vac_bary, flux[good])
plt.xlabel("Wavelength (nm) [vacuum, barycentric]")
plt.ylabel(r"Flux (erg cm$^{-2}$ s$^{-1}$ Å$^{-1}$)")
plt.title(f"X-shooter spectrum: {hdr0.get('OBJECT','Unknown')}")
plt.tight_layout()
plt.show()
exit(0)
# === SAVE CLEANED FITS ===
coldefs = fits.ColDefs([
    fits.Column(name='WAVE_VAC_BARY', format='D', unit='nanometer',
                array=wave_A_vac_bary),
    fits.Column(name='FLUX', format='D', unit='erg cm**(-2) s**(-1) angstrom**(-1)',
                array=flux[good]),
    fits.Column(name='ERR', format='D', unit='erg cm**(-2) s**(-1) angstrom**(-1)',
                array=err[good])
])

hdu = fits.BinTableHDU.from_columns(coldefs)
# Keep primary header for metadata
phdu = fits.PrimaryHDU(header=hdr0)
hdulist = fits.HDUList([phdu, hdu])

out_name = file_path.split("/")[-1].replace(".fits", "_vac_bary_clean.fits")
hdulist.writeto(out_name, overwrite=True)
print(f"Saved: {out_name}")
