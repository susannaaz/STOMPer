import numpy as np
from math import sqrt
import scipy as sc
from spline_utils import *
import sys
import os 

import time as tm
tstart = tm.time()

'''
2) 
   Import data (phase TOD) created with generate_phase_drift.py,
   introduce encoder error at each data point and spline fit these
   measurements.

'''

# Set params
fk_sim = 0.00000001          #Hz, drift measured after 3 years
f_HWP = 46./60.              #Hz, rotation frequency HWP (rad/sec) corresponding to spin 46rpm
n_holes = 10                 #HWP holes in encoder (test with 10 or 128)
t_meas = (1/f_HWP)/n_holes   #s, time between encoder measurements
fk = 1/t_meas                #Hz, freq of encoder measurements
fsamp = fk*10                #encoder sampling freq
t_samp = 1/fsamp             #s, time bewteen data samples
Omega_HWP = (f_HWP * 2.0 * np.pi / fsamp) # HWP rotating

tott = 2520000000 #~1 yr in t_samp; Full phase time.

deltatimes = 1000000 #segment lenght for interpolation (1/10 of each TOD segment)
jobs = tott/deltatimes

print('Total jobs to run: ',jobs)
print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

inum = int(str(sys.argv[1])) # Initial time segment
num = inum + deltatimes # Final time segment
seed = str(sys.argv[2]) # Seed 1/f^2 drift

phi=10**-4 ###### NB TO CHANGE DEPENDING ON VARIATION
cifre = int(4)
phi=np.round(phi,cifre)

print('(Init time spline segment) inum: ',inum)
input_segment_phase = np.arange(0, tott, 10000000) # Range of starting time for phase TOD files
in_a = np.nan
for ni, el in enumerate(input_segment_phase[:-1]):
    if inum >= el and inum <= input_segment_phase[ni+1]: 
        in_a = el #choose TOD segment where spline initial time lies in
        in_b = in_a #choose TOD segment where spline final time lies in
        if num >= input_segment_phase[ni+1]:
            in_b = input_segment_phase[ni+1]
in_a = int(in_a)
in_b = int(in_b)

# Import data (phase TOD )
# created with generate_phase_drift.py
fdir = f'/group/cmb/litebird/usr/susanna/SPLINE/'
indir = f'{fdir}phase_TOD_seed{seed}/'
if not os.path.exists(f'{fdir}phase_TOD_seed{seed}/DRIFT_nholes10_randwalk_{in_a}.npz'):
    print('Input phase TOD file does not exist')
    exit(1)
a = np.load(f'{fdir}phase_TOD/DRIFT_nholes10_randwalk_{in_a}.npz')
b = np.load(f'{fdir}phase_TOD/DRIFT_nholes10_randwalk_{in_b}.npz')
print('Phase TOD segments needed:')
print('in_a: ',in_a)
print('in_b: ',in_b)

# Test data
if in_a == in_b: # spline data is all in same TOD segment
    print('a=b')
    phase = a['dphi_time'][(inum-in_a):(num-in_a)]   #phase    len(phase) = 2521359361
    time = a['time'][(inum-in_a):(num-in_a)]  #in data sampling time (2521359361 --> 380 days)
else:
    print('a!=b')
    # Import test data from previous segment for overlap
    phase_a = a['dphi_time'][(inum-in_a):-1]
    time_a = a['time'][(inum-in_a):-1]
    # Import test data from next segment 
    phase_b = b['dphi_time'][0:(num-in_b)]
    time_b = b['time'][0:(num-in_b)]
    #
    phase = np.concatenate((phase_a, phase_b), axis=None)
    time = np.concatenate((time_a, time_b), axis=None)
#print((a['time'][1]-a['time'][0])) = 1 --> time in t_samp

rms_tod = a['rms_tod']

timem = time * t_meas             #in units of encoder measurements
times = time * t_samp             #in units of sampling time
timetot = times[-1] - times[0]    #t_samp seconds
nsamp = int(timetot/t_samp)+1     #number of samples
nmeas = int(nsamp/t_meas*t_samp)  #number of encoder measurements

print('length segment: ',len(phase))

# Check quantities
print('N_holes encoder: ',n_holes)
print('Lenght sampled time: ',(times[-1]+1)/3600,'tsamp hours')
print('time between sampled data points: ',times[1]-times[0],'tsamp seconds') #=t_samp
print('Lenght encoder measurement data (time): ',(timem[-1]+1)/3600,'tmeas hours')
print('time between encoder data points: ',timem[1]-timem[0],'tmeas seconds') #=t_meas
print('Tot time: ',int(timetot/t_samp)+1,'seconds = ',(int(timetot/t_samp)+1)/3600,' hours')
print('fsamp: ',fsamp)
print('# sampled data (fsamp): ',nsamp,'=',len(times))
print('fknee: ',fk)
print('# encoder data (fknee): ',nmeas)

# Check if file already exists
amp_enc_err = ([phi]) 
ooodir = f'{fdir}spline_fit_seed{seed}/sigmaphi_%.{cifre}f/'%(phi)
nmfile = f'{ooodir}/SPLINE_nholes{n_holes}_fsamp{fsamp}_fk{fk}_phi%.{cifre}f___{time[0]}-{time[-1]}.npz'%(phi)

if os.path.exists(nmfile):
    print(f'spline already run:')
    print(nmfile+' exists')
else:
    os.system('mkdir -p ' + ooodir)
    print('Running spline')
    # Get encoder data points:
    # x = time + error encoder, y = phase
    ns = int(nmeas) #number encoder measurements
    xo = np.linspace(times[0]/t_samp, times[-1]/t_samp, nmeas) #times encoder gets data (1/fk)
    
    phase_encoder = phase[0::10] #encoder data points (every 10 sampled data points)

    # Values of amplitude encoder error
    for phi in amp_enc_err:
        phi_sigma = phi
        x = xo + sc.randn(xo.size)*phi_sigma # time + error encoder
        y = phase_encoder 
        nknots = nsamp
        x_new = np.linspace(times[0], times[-1], nknots) 
    
        # Spline Interpolation
        y_spline = cubic_interp1d(x_new, x, y)
        
        # Residual (data - spline)
        tlow=inum
        thigh=num-1
        indres = thigh-tlow
        ress = y_spline[:indres] - phase[:indres]
        res = ress
        
        ## Check that sigma_tod = rms(tod)
        rms = np.sqrt(np.sum(ress**2)/len(ress)) 
        print('rms: ',rms)
        
        np.savez(nmfile,
                 input_data=phase[:indres], time_input_data=times[:indres], 
                 y_spline=y_spline[:indres], x_spline=x_new[:indres],
                 res_phase=ress,
                 sigma_tod=rms)

tend = tm.time()
print('')
print('Running time:')
print((tend-tstart)/60,' minutes')
