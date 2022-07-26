import matplotlib.pyplot as plt
import sys
sys.path.append('/mnt/zfsusers/susanna/Software/fgbuster/fgbuster/')
import numpy as np
import pylab
pylab.rcParams['figure.figsize'] = 12, 16
import healpy as hp
import pysm3
import pysm3.units as u
from fgbuster import get_instrument, get_sky, get_observation  # Predefined instrumental and sky-creation configurations
from fgbuster.visualization import corner_norm
# Imports needed for component separation
from fgbuster import (FreeFree, CMB, Dust, Synchrotron,  # sky-fitting model
                      basic_comp_sep)  # separation routine

plot = 0

NSIDE = 256
sky_simple = get_sky(NSIDE, 'c1d1s1') #get_sky(NSIDE, 'c1d0s0f1a1') 
instrument = get_instrument('LiteBIRD')
freq_maps_simple = get_observation(instrument, sky_simple)

#freq_maps_simple = freq_maps_simple[:, 1:]  # Select polarization
print(freq_maps_simple.shape) #(15, 2, 786432)

nfreq, npol, npix = freq_maps_simple.shape
nmaps = nfreq*npol

# Rotate coordinates (galactic to ecliptic)
rot_gal2eq = hp.Rotator(coord="GE")
freq_maps_ecliptic = freq_maps_simple*0
for i in range(nfreq):
    for j in range(npol):
        freq_maps_ecliptic[i,j] = rot_gal2eq.rotate_map_pixel(freq_maps_simple[i,j])

# Save sky maps
# galactic coord
#hp.write_map("maps_sky_signal_gal_c1d0s0.fits", freq_maps_simple.reshape([nmaps,npix]), overwrite=True)
hp.write_map("maps_sky_signal_gal_c1d1s1.fits", freq_maps_simple.reshape([nmaps,npix]), overwrite=True)
# ecliptic coord
#hp.write_map("maps_sky_signal_ecl.fits", freq_maps_ecliptic.reshape([nmaps,npix]), overwrite=True)

if plot:
    hp.mollview(freq_maps_simple[10,0,:]) #I
    hp.mollview(freq_maps_simple[10,1,:]) #Q
    hp.mollview(freq_maps_simple[10,2,:]) #U
    hp.mollview(freq_maps_ecliptic[10,0,:]) #I
    hp.mollview(freq_maps_ecliptic[10,1,:]) #Q
    hp.mollview(freq_maps_ecliptic[10,2,:]) #U
    plt.show()
