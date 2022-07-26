import numpy as np
import healpy as hp
import scipy as sc
import math as mt
import array
import csv
import sys
import matplotlib.pyplot as plt
#from .hwp import HalfWavePlate
#import hwp as h
from todsim import TODsim

# Set nside
nside=64

# Run simulation.py nmin, nt, pair 
nmin = int(str(sys.argv[1]))
nt = int(str(sys.argv[2])) + nmin
pair = str(sys.argv[3])
print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

# Input files directory
dirscr='/home/cmb/susanna/TODs2Maps/dirscr/'

#h = HalfWavePlate
#ut = TODsim(HalfWavePlate)
ut = TODsim(nside, nmin, nt, dirscr)

plot = 0

# First, let's generate a set of parameters
params, sim_params, noise_params = ut.get_default_params()
# By default, these will contain noise, but no dipole, no monopole
# To change parameters simulation e.g. remove noise
sim_params['noise'] = False
ut = TODsim(nside, nmin, nt, dirscr, params, sim_params, noise_params)

# Plot input maps, check coordinates
# Coordinates: equatorial?
# Units: uK_cmb
mapI_in, mapQ_in, mapU_in = ut.get_sky_maps()

if plot:
    hp.mollview(mapI_in, title = 'Input map I', unit='$\mu$K$_{CMB}$')
    hp.mollview(mapQ_in, title = 'Input map Q', unit='$\mu$K$_{CMB}$')
    hp.mollview(mapU_in, title = 'Input map U', unit='$\mu$K$_{CMB}$')
    plt.show()

# Read bolometer file
if (len(sys.argv) > 4):
    bolonames_=[str(sys.argv[4])]
else:
    bfile = dirscr+f'20190715_LFT_pointing.txt'
    bolonames_ = ut.readbolofile(bfile)
(theta_fp,phi_fp,psib,bolonames) = ut.ReadBoloInfo(dirscr+'20190715_LFT_pointing.txt',bolonames_,pair)
#
print('!!!!!!!!!!!!!!!!!!Bolo infos !!!!!!!!!!!!!!!!!!!!!!!!')
print(bolonames_)
print(theta_fp,phi_fp,psib)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

# Output file suffix
terminS = 'simulation'
# Output file prefix
if params['fsamp'] == 19:
    terminP = 'LB_45_50_LB_v28_19Hz'
print('POINTING FILE',terminP)

maph = np.zeros(12*nside*nside)

# Pointing files
dirp = '/home/cmb/susanna/TODs2Maps/data/'
filenameRa = dirp+'ra_'+terminP+'.bi'
filenameDec = dirp+'dec_'+terminP+'.bi'
filenamePsi = dirp+'psi_'+terminP+'.bi'

fileRa = open(filenameRa, mode='rb')
fileDec = open(filenameDec, mode='rb')
filePsi = open(filenamePsi, mode='rb')
filePsiTemp = open(filenamePsi, mode='rb')

if sim_params['idealHWP']:
    terminS = terminS + '_idealHWP'

A, B, C = h.get_MuellerMatrix(typ='RCWA_10_deg')
A0, B0, C0, phiB, phiC = h.get_MuellerMatrix(typ='ideal_HWP')

print('B_orig = ', B)
print('C_orig = ', C)

sss = params['sss']
nnn = params['nnn']
nn = params['nn']

samp = np.arange(nmin*sss,nnn,sss)

####### Including 1/f noise
fknee = noise_params['fknee']
fsamp = params['fsamp']
beta = noise_params['beta']

if sim_params['noise']:
    print("Including 1/f noise")
    terminN = str(fknee)+'_'+str(int(fsamp))+'_index'+str(beta)+'_'
    sign, sp, isp = ut.get_noise()
else:
    terminN = ''

listdet= range(np.size(bolonames))
if (pair == 'b'):
    listdet = np.arange(1,np.size(bolonames),2)
elif (pair == 'a'):
    listdet = np.arange(0,np.size(bolonames),2)

sha = sim_params['sha']

dirout='/home/cmb/susanna/TODs2Maps/simple_sim/'
####### Loop over detectors 
print("Looping over detectors")
for idet in listdet:
    print(bolonames[idet]+'_'+terminS+'.bi')
    filenameraout = '/home/cmb/susanna/TODs2Maps/data/ra_'+bolonames[idet]+'_'+terminS+'.bi'
    filenamedecout = '/home/cmb/susanna/TODs2Maps/data/dec_'+bolonames[idet]+'_'+terminS+'.bi'
    filenamepsiout = '/home/cmb/susanna/TODs2Maps/data/psi_'+bolonames[idet]+'_'+terminS+'.bi'
    fdat = open('/home/cmb/susanna/TODs2Maps/simple_sim/data_idealHWP_noNoise_noMonDip_'+bolonames[idet]+'_'+terminS+'_HWP_Mueller%i_.bi'%nmin,'w')

    if sim_params['noise']:
        if (nmin == 0):
            isp.tofile('/home/cmb/susanna/TODs2Maps/Noise/iSpf'+terminN+'_'+bolonames[idet]) 
        fdatn = open('/home/cmb/susanna/TODs2Maps/simple_sim/data_idealHWP_noNoise_noMonDip_'+bolonames[idet]+'_'+terminS+'_'+terminN+'HWP_Mueller%i_.bi'%nmin,'w')
        
    indsamp = nmin*sss
    
    print("Looping over samples")
    for isamp in samp:
        theta_pt, phi_pt, thetatmp, phitmp, psi = ut.pointing(isamp, idet, fileRa, fileDec, filePsi, filePsiTemp, phi_fp, theta_fp)

        ra = phitmp.copy()
        dec = np.pi/2.0-thetatmp
        dpsi = 0.0*dec
        
        #Mueller Matrix multiplication
        print("Multiplying Mueller Matrix")
        Ad, Bd, Cd = h.transform_HWP_coeff(theta_pt, A, A0, C, C0, B, phiB, phiC, params, sim_params)
        print(ra.shape, dec.shape, psi.shape, phi_pt.shape)
        
        SignalHWP, indpix = h.get_signal_HWP(mapI_in, mapQ_in, mapU_in, psib[idet], ra, dec, indsamp, psi, dpsi, phi_pt, nside, Ad, Bd, Cd, pars=params, sim_pars=sim_params)
        
        indsamp = indsamp + np.size(ra)
        
        # Write signal to file
        SignalHWP.tofile(fdat)

        # Add noise part
        if sim_params['noise']:
            print("Adding noise to signal HWP")
            signoi = ut.add_noise(SignalHWP, ra, sign, sp)
            # Write signal + noise to file
            signoi.tofile(fdatn)
        
        print('Segment %i written'%(isamp))
        
        for ii in indpix:
            maph[ii] += 1

#maph.tofile('/group/cmb/litebird/usr/patanch/Maps/mapinhist.bi')
print("End simulation")
