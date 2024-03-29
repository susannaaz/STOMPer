import numpy as np
import scipy as sc
import sys
import os 

import time

tinit = time.time()

'''
1) Generate timestream of HWP phase data with no correction 
   from the encoder (i.e. drawing samples from a random 
   distribution with sigma=1/f^2)
Note: 1/f^2 spectrum in Fourier space
      corresponds to random walk in real space.
'''

print_info = 0

# Model
f_hwp = 46/60 #Hz, corresponding to spin 46rpm
n_holes = 10 #HWP holes in encoder
T_meas = (1/f_hwp)/n_holes #s, time between measurements
fkk = 1/T_meas #encoder angle data samplinf
fsamp = fkk*10 #data sampling freq
fk = 0.00000001 #1e-8 = measured after 3 years
Totalyr = 1.04 #slightly more that 1 yr
nn = int(3600*24*366.0*Totalyr*fsamp)  #1.04 yr in sampling units
phi_sigma = 1
nfrac = 10000000 #sec in units of sampling time, segment length 

tsec = int(nfrac/fsamp)
tdays = int(nfrac/(86400*fsamp))

# Seed
seed = int(str(sys.argv[1])) 

if print_info:
    print(f'n_holes ',n_holes)
    print(f'T_meas ',T_meas)
    print(f'fsamp ',fsamp)
    print(f'nn ',nn)
    print(f'time in fsamp units (tsamp sec) ',nfrac) 
    print(f'tot time segment (sec) ',tsec)
    print(f'tot time segment (days) ',tdays)

step_n = nfrac
sign = phi_sigma #* np.sqrt(fk)

def randomwalk1D(x=0, y=0, n=100, sign=1., seed=None):
    '''
    Generates random walk.
    Input:
    - x : initial point on x axis.
    - y : initial point on y axis.
    - n : number of steps.
    - sign : maximum amplitude of each step.
    Returns:
    - timepoints : time in tsamp units
    - np.array(positions) : random walk 
    '''
    if seed is not None:
        np.random.seed(seed)
    timepoints = np.linspace(x, x+n, n+1)
    positions = [y]
    directions = ["UP", "DOWN"]
    for i in range(1, n+1):
        # Randomly select either UP or DOWN
        step = np.random.choice(directions)        
        # Move the object up or down
        if step == "UP":
            y += sign
        elif step == "DOWN":
            y -= sign
        # Keep track of the positions
        positions.append(y)
    return timepoints, np.array(positions)


dname=f'/group/cmb/litebird/usr/susanna/SPLINE/phase_TOD_seed{seed}/'
os.system('mkdir -p ' + dname)
for o in np.arange(0, nn, nfrac): #(t0, tot1yr, nsteps)
    i = o-nfrac
    iname = f'{dname}DRIFT_nholes{n_holes}_randwalk_{i}' #input
    oname = f'{dname}DRIFT_nholes{n_holes}_randwalk_{o}' #output
    if o == 0:
        print('initial zero')
        time, path = randomwalk1D(x=0, y=0, n=step_n, sign=sign, seed=seed)
        rms = np.sqrt((np.sum(path**2)/len(path)))
        np.savez(oname, time=time, dphi_time=path, rms_tod=rms)
    else:
        # Import previous segment for initial point 
        previous_segment = np.load(f'{iname}.npz')
        tin = previous_segment['time']; pin = previous_segment['dphi_time']
        time, path = randomwalk1D(x=tin[-1], y=pin[-1], n=step_n, sign=sign, seed=seed)
        rms = np.sqrt((np.sum(path**2)/len(path)))
        np.savez(oname, time=time, dphi_time=path, rms_tod=rms) # time is in t_samp-seconds
print(dname)

