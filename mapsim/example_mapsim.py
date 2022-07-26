import healpy as hp
import numpy as np
from scipy.special import zeta
import os
import matplotlib.pyplot as plt
import mapsimulation #as ut
#from mapsimulation import Mapsim #*

ut = mapsimulation.Mapsim()

import sys
sys.path.append('/mnt/zfsusers/susanna/Software/PySM_public/pysm/')
#import pysm
#from pysm.nominal import models

# Define parameters
nside = 64
seed = 1005
np.random.seed(seed) 
# To change params:
mean_p, moment_p = ut.get_default_params()
mean_p['include_CMB'] = True
mean_p['include_dust'] = False
mean_p['include_sync'] = False
mean_p['include_T'] = True
mean_p['r_tensor'] = 0.001

# Create output directory
dirname = "111sim_LiteBIRD_sd%d_ns%d"%(seed,nside)
if mean_p['include_CMB']:
    dirname += '_CMB'
if mean_p['include_dust']:
    dirname += '_dust'
if mean_p['include_sync']:
    dirname += '_sync'
os.system('mkdir -p ' + dirname)
print(dirname)

# Use class to get frequency maps
mp = mapsimulation.Mapsim(mean_p, moment_p)
sim = mp.get_sky_realization(nside, seed)
mps_signal = sim['freq_maps']

# Save sky maps
nu = mp.get_freqs()
nfreq = len(nu)
npol = 3
nmaps = nfreq*npol
npix = hp.nside2npix(nside)
hp.write_map(dirname+"/maps_sky_signal.fits", mps_signal.reshape([nmaps,npix]),
             overwrite=True)




