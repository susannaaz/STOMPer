import glob
import numpy as np
import sys
import time as tm
import os
import scipy as sc

start = tm.process_time()

seed = int(sys.argv[1])

fk_sim = 0.00000001          #Hz, drift measured after 3 years
f_HWP = 46./60.              #Hz, rotation frequency HWP (rad/sec) corresponding to spin 46rpm
n_holes = 10                 #HWP holes in encoder (test with 10 or 128)
t_meas = (1/f_HWP)/n_holes   #s, time between encoder measurements
fk = 1/t_meas                #Hz, freq of encoder measurements
fsamp = fk*10                #encoder sampling freq
t_samp = 1/fsamp             #s, time bewteen data samples
Omega_HWP = (f_HWP * 2.0 * np.pi / fsamp) # HWP rotating

phi=0.0001 ####NB CHANGE HERE

# Read files
fdir = f'/group/cmb/litebird/usr/susanna/SPLINE/spline_fit_seed{seed}/downsampled_fsamp19.0_phi{phi}/'
print(fdir)
fnames = glob.glob(f'{fdir}data_fsamp19.0_segs*-*.bi')
fnames.sort()
tnames = glob.glob(f'{fdir}time_fsamp19.0_segs*-*.bi')
tnames.sort()
print(len(fnames)) #266 

# Output files
filenameout = f'{fdir}/data_fsamp19.0_phi{phi}__TOTAL.bi'
tilenameout = f'{fdir}/time_fsamp19.0_phi{phi}__TOTAL.bi'

if not os.path.exists(filenameout):
    fdat = open(filenameout,'w')
    tdat = open(tilenameout,'w')

    isamp = 0
    for i, (fn, tn) in enumerate(zip(fnames, tnames)):
        myArray = np.fromfile(fn, dtype=np.double) #dOmegat
        myx = np.fromfile(tn, dtype=np.double) #time
        fdat.seek(isamp*8)
        tdat.seek(isamp*8)
        myArray.tofile(fdat) # Write to file
        myx.tofile(tdat)
        isamp += np.size(myx)
        print(i)

data = np.fromfile(filenameout, dtype=np.double)
rms = np.sqrt((np.sum(data**2))/len(data))
print('rms TOD: %.40f'%rms)

# Check that sigma_tod = rms(tod)
phi_sigma=phi**2
if phi_sigma != 0 and fk!=0:
    phi_sigma = np.pi/phi_sigma
    sig_tod_int = phi_sigma * np.sqrt(fk * np.arctan(fsamp/fk))
elif phi_sigma==0 and fk==0:
    sig_tod_int = 0
elif phi_sigma==0 and fk!=0:
    sig_tod_int = np.sqrt(fk * np.arctan(fsamp/fk))
elif phi_sigma!=0 and fk==0:
    sig_tod_int = 0
print('sig_tod (analytic): ',sig_tod_int)

doublecheck_sigtod=0
if doublecheck_sigtod:
    # sigma_tod 
    # calculated analytically from integral(P(f) df)
    #calculate P(f)/integral(P(f) df) multiplied by sigma_tod^2
    def sigma_oof_varying(nfrac, fsamp, fknee, beta, NET, NET_glob=1.):
        #np.random.seed(1005)
        tt = nfrac #npix 
        ff = np.fft.fftfreq(tt)*fsamp
        sp = np.zeros(nfrac) 
        ff[0] = ff[1] 
        sp = (fknee/abs(ff))**beta # + 1
        if NET > 0. and fknee > 0.:
            sp += 1
            sign = NET * np.sqrt(fknee * np.arctan(fsamp/fknee)) #* np.sqrt(fsamp) #NET corresponds to sigma_fk^2, (mu=0)
        elif NET > 0. and fknee==0.:
            sp += 1
            sign = NET 
        elif NET==0 and fknee != 0:
            sign = NET_glob * np.sqrt(fknee * np.arctan(fsamp/fknee)) 
        else:
            sign=0
        #isp = 1.0/abs(sp)
        dataw = sc.randn(tt)*(sign)
        fdata = (np.fft.fft(dataw))
        #if fknee > 0.:
        #    fdata *= np.sqrt(abs(sp)) #np.sqrt(abs(isp))
        dataf = np.fft.ifft(fdata)
        datan = dataf.real #in time units 
        if fknee == 0.:
            datan = dataw
        ttt = np.arange(tt)
        if fknee==0 and NET==0:
            print("delta-phase=0!")
            datan=np.zeros_like(datan)
            fdata=np.zeros_like(fdata)
        return ttt, datan, ff, fdata

    ttt, delta_Omegathwp, fff, spec_domegat = sigma_oof_varying(nfrac=len(data), fsamp=fsamp, fknee=fk, beta=2.0, NET=phi_sigma)

    rms = np.sqrt((np.sum(delta_Omegathwp**2))/len(delta_Omegathwp))
    print('sig_TOD: ',rms) #equivalent to sig_tod_int 

print(tm.process_time() - start)
