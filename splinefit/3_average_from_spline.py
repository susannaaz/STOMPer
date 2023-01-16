import numpy as np
import sys
import glob 
import os
from natsort import natsorted
from scipy.interpolate import interp1d
import time

import time as tm
tstart = tm.time() 

######### CHECK
# 1) from generate_phase_drift_rndmwalk_segmented.py
# Last phase TOD time: 2520000000
# 2) from spline_interpolation_segmented.py
# Last 2 spline segment:
# In /group/cmb/litebird/usr/susanna/SPLINE/spline_fit/sigmaphi_0.0001/
#SPLINE_nholes10_fsamp76.66666666666667_fk7.666666666666667_phi0.000100___2519000000.0-2519999999.npz
#SPLINE_nholes10_fsamp76.66666666666667_fk7.666666666666667_phi0.000100___2519500000.0-2520499999.npz
# Last time: 2520499999 (in fsamp = fk*10 units)
# Each segment is in total 1000000 elements
# Each segment start after 500000
# ---> each segment has last 500000 elements overlapping with the first 500000 of the next segment

checkarray = 0

### Info original spline data
f_HWP = 46./60.              #Hz, rotation frequency HWP (rad/sec) corresponding to spin 46rpm
n_holes = 10                 #HWP holes in encoder (test with 10 or 128)
t_meas = (1/f_HWP)/n_holes   #s, time between encoder measurements
fk = 1/t_meas                #Hz, freq of encoder measurements
fsamp = fk*10                #encoder sampling freq
tsamp = 1/fsamp              #s, time bewteen data samples

### Input arguments
init = int(sys.argv[1]) #initial segment
finit = int(sys.argv[2]) #final segment
print('spline segments: ',init,finit)
seed = int(sys.argv[3])

cifre = int(4)
phi=np.round(0.0001,cifre)

### Import spline data
fdir = f'/group/cmb/litebird/usr/susanna/SPLINE/'
fnames = glob.glob(f'{fdir}spline_fit_seed{seed}/sigmaphi_%.{cifre}f/SPLINE_nholes{n_holes}_fsamp{fsamp}_fk{fk}_phi%.{cifre}f___*-*.npz'%(phi,phi)) 
fnames = natsorted(fnames) 
print(len(fnames)) #~5039

if init > len(fnames):
    print(f'Initial segment {init} > total spline segments {len(fnames)}')
    print('Exceeded spline data!')
    exit(1)

if checkarray:
    print('CHECKING ARRAYS:')
    print('check that time is in seconds')
    a0 = np.load(fnames[0])
    a1 = np.load(fnames[-1])
    myx0 = a0['x_spline']
    myx1 = a1['x_spline']
    print(2520999999*tsamp) # 1.04 yr in seconds
    print(myx1[-1]) # 1.04 yr in seconds

    print('check that segments overlap')
    a0 = np.load(fnames[2])
    a2 = np.load(fnames[3])
    myx0 = a0['x_spline']
    myx2 = a2['x_spline']
    myy0 = a0['y_spline']
    myy2 = a2['y_spline']
    print('Times should be the same')
    print(myx0[500000:])
    print(myx2[:500000])
    print('Phase should be ~ the same')
    print(myy0[500000:])
    print(myy2[:500000])
    exit(1)

### New time array resampled at lower freq 
fsamp_new = 19.0 # Hz, bolometer sampling freq
tsamp_new = 1/fsamp_new #s, time bewteen data samples

# Segments to include for linear interp
fnames = fnames[init:finit] 

### Cut each segment up to half of overlapping region 
### i.e. index:
indx = 250000

### Import segment and do linear interpolation to downsample
allArrays = np.array([])
allx = np.array([])
for fi, fnm in enumerate(fnames):
    print(fi)
    # Import spline data
    a = np.load(fnm)
    xsp = a['x_spline'] #time in tsamp sec
    ysp = a['res_phase']
    nb_points = len(xsp)
    # Select first element of segment
    if xsp[0]==0.:
        tin = xsp[0] 
        y_spline = ysp[:-indx]
    else:
        tin = xsp[indx] 
        y_spline = ysp[(indx):(-indx)]
    # Select last element of segment
    tfin = xsp[-indx] 
    
    # Spline data without overlap
    tin_spline = tin # t[sec]
    tfin_spline = tfin #/tsamp #t[sec]
    time_spline = np.arange(tin_spline, tfin_spline, tsamp) ## in t[sec], == xsp[:-indx]

    if len(time_spline) != len(y_spline):
        try:
            y_spline = ysp[(indx):(-indx-1)]
            tfin = xsp[-indx-1]
            time_spline = np.arange(tin, tfin, tsamp)
        except:
            pass
    if not len(time_spline) == len(y_spline):
        try:
            y_spline = ysp[(indx):(-indx)]
            tfin = xsp[-indx-1]
            time_spline = np.arange(tin, tfin, tsamp)
        except:
            pass

    # Downsampled data without overlap    
    time_new = np.arange(tin_spline, tfin_spline, tsamp_new) ## in seconds
    f_interpol_linear = interp1d(time_spline, y_spline, kind='linear', fill_value='extrapolate')(time_new)   
    
    allArrays = np.concatenate([allArrays, f_interpol_linear])
    allx = np.concatenate([allx, time_new])

print(len(allArrays)) 


# Write combined file of downsampled data
fdir = f'/group/cmb/litebird/usr/susanna/SPLINE/spline_fit_seed{seed}/downsampled_fsamp{fsamp_new}_phi{phi}/'
print(fdir)
if not os.path.exists(fdir):
    os.mkdir(fdir)
filenameout = f'{fdir}data_fsamp{fsamp_new}_segs{init}-{finit}.bi'
fdat = open(filenameout,'w')
allArrays.tofile(fdat) 

# Write combined times in tsamp units
filenameout = f'{fdir}time_fsamp{fsamp_new}_segs{init}-{finit}.bi'
fdat = open(filenameout,'w')
allx.tofile(fdat) # units of seconds

tend = tm.time()
print('')
print('Running time:')
print((tend-tstart)/60,' minutes')
