import healpy as hp
import numpy as np
from scipy.special import zeta
import sys
sys.path.append('/mnt/zfsusers/susanna/Software/PySM_public/pysm/')
#import pysm
#from pysm.nominal import models
import os
import matplotlib.pyplot as plt
from mapsim import Mapsim

# Define parameters
nside = 256
seed = 1005
np.random.seed(seed) 

# Use class 
mp = Mapsim()
#print(mp.mean_pars)

# Change parameters:
mean_p, moment_p = mp.get_default_params()
mean_p['include_CMB'] = True
mean_p['include_dust'] = True
mean_p['include_sync'] = True
mean_p['include_T'] = True
mean_p['r_tensor'] = 0.001
mean_p['A_lens'] = 1.
mp = Mapsim(mean_p, moment_p)
#print(mp.mean_pars)

# Create output directory
dirname = "sim_LiteBIRD_sd%d_ns%d"%(seed,nside)
if mean_p['include_CMB']:
    dirname += '_CMB'
if mean_p['include_dust']:
    dirname += '_dust'
if mean_p['include_sync']:
    dirname += '_sync'
dirname += '_Alens%d'%(mean_p['A_lens'])
dirname += '_r%.3f'%(mean_p['r_tensor'])
os.system('mkdir -p ' + dirname)
print(dirname)

# Get frequency maps
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




