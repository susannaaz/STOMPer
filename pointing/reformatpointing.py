import numpy as np
import healpy as hp
import scipy as sc
import matplotlib.pyplot as plt
import math as mt
import array
import os

fsamp = 19.0 #sampling frequency of data
sss = int(86400 * fsamp) #1 day in seconds of sampling time (24hr = 86400 seconds)
nn = 366 #days in 1 year
nnn = nn*sss #1 year in tsamp [s] unit

print('TOTAL NB OF SAMPLES: ', nnn)

termin='LB_45_50_LB_v28_19Hz'

dirp = "/mnt/zfsusers/susanna/Map2TOD/pointing/scans/"
dirout = "/mnt/zfsusers/susanna/Map2TOD/pointing/fileout/"
os.system('mkdir -p ' + dirout)
print(dirout)

samp = np.arange(0,nnn,sss)

filenameraout = dirout+'ra_'+termin+'.bi'
filenamedecout = dirout+'dec_'+termin+'.bi'
filenamepsiout = dirout+'psi_'+termin+'.bi'
fra = open(filenameraout,'w')
fdec = open(filenamedecout,'w')
fpsi = open(filenamepsiout,'w')

ii=0
for isamp in samp:
    filenameTheta = dirp+'theta'+termin+'%i.npy'%(ii)
    filenamePhi = dirp+'phi'+termin+'%i.npy'%(ii)
    filenamePsi = dirp+'psi'+termin+'%i.npy'%(ii)
    
    thetap = np.load(filenameTheta)
    rap = np.load(filenamePhi)
    psip = np.load(filenamePsi)
    decp = np.pi/2.0 - thetap
    print(isamp,thetap[0])
    
    fra.seek(isamp*8)
    fdec.seek(isamp*8)
    fpsi.seek(isamp*8)
    rap.tofile(fra)
    decp.tofile(fdec)
    psip.tofile(fpsi)

    ii+=1



