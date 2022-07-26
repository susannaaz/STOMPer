import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import math as mt
import os

## LiteBIRD specs 
alpha = 45.0/180.0*np.pi       # [rad]
beta = 50.0/180.0*np.pi        # [rad]
fsamp = 19.0                   # sampling frequency [Hz]
tprec = 1.51*60*60             # precession rate [sec]
wprec = 2.0*np.pi/tprec        # angular speed scan [rad/sec]
wspin = 0.1 * 2.0*np.pi/60.0   # spin rate [rad/s]
wearth = 2.0*np.pi/31556925.0  # angular speed earth [rad/s] (1yr=31556925sec) 
ysh=0.0                        # ysh = mt.asin(10.0/180.0*np.pi)

# Total spin
wspin = wspin + wprec

# Initial conditions
Phi0 = 0.0
wt0 = 0.0
Ot0 = 0.0
sigg = 1
### to match with Tomo choice
Phi0 =  -1.1096165695798597
wt0  =  -1.109611527255728
Ot0  =  -3.359251334665409
sigg = 1

dirout = "/mnt/zfsusers/susanna/Map2TOD/pointing/scans/"
os.system('mkdir -p ' + dirout)
print(dirout)

termin = 'LB_45_50_LB_v28_19Hz'

## Number of samples for 1 day (to modify to be more general)
nn = int(86400 * fsamp) # 24hr=86400sec

## Time vector
t = np.arange(nn)/fsamp

x = np.zeros(nn)
y = np.zeros(nn)
z = np.zeros(nn)

theta = np.zeros(nn)
phi = np.zeros(nn)
psi = np.zeros(nn)

# Number of days in 1 yr
idmin = 0
idmax = 366

for id in np.arange(0,idmax):
    print ("Day ",id)
    if (id > idmin-1):
        print (wt0)
        for i in range(nn):
            wt = sigg*wspin*t[i] + wt0
            Phi = sigg*wprec*t[i] + Phi0
            Ot = wearth*t[i] + Ot0
            
            ### coord vector
            X0 = np.zeros(3)
            X0[0] = np.sin(beta+ysh)*np.cos(wt)
            X0[1] = np.sin(beta+ysh)*np.sin(wt)
            X0[2] = np.cos(beta+ysh+0*wt)
            
            ### diff vector
            dX0 = np.zeros(3)
            dX0[0] = -np.sin(wt)
            dX0[1] = np.cos(wt)
            dX0[2] = 0.0    
            
            ### 1st rotation
            R1 = np.zeros((3,3))
            R1[0,0]=np.sin(Phi)*np.sin(Phi)*(1-np.cos(alpha)) + np.cos(alpha)
            R1[0,1]=-np.sin(Phi)*np.cos(Phi)*(1-np.cos(alpha))
            R1[0,2]=-np.cos(Phi)*np.sin(alpha)
            R1[1,0]=-np.sin(Phi)*np.cos(Phi)*(1-np.cos(alpha))
            R1[1,1]=np.cos(Phi)*np.cos(Phi)*(1-np.cos(alpha)) + np.cos(alpha)
            R1[1,2]=-np.sin(Phi)*np.sin(alpha)
            R1[2,0]=np.cos(Phi)*np.sin(alpha)
            R1[2,1]=np.sin(Phi)*np.sin(alpha)
            R1[2,2]=np.cos(alpha)
           
            X1 = R1.dot(X0)
            dX1 = R1.dot(dX0)
            
            ### rotate coord by 90 degrees
            R90 = np.zeros((3,3))
            R90[0,0] = 1.0
            R90[1,2] = -1.0
            R90[2,1] = 1.0
            
            X2 = R90.dot(X1)
            dX2 = R90.dot(dX1)
            
            ### motion of L2
            R2 = np.zeros((3,3))
            R2[0,0] = np.cos(Ot)
            R2[0,1] = -np.sin(Ot)
            R2[1,0] = np.sin(Ot)
            R2[1,1] = np.cos(Ot)
            R2[2,2] = 1.0
            
            X3 = R2.dot(X2)
            dX3 = R2.dot(dX2)
            
            x[i] = X3[0]
            y[i] = X3[1]
            z[i] = X3[2]
            
            ### compute sky coordinates 
            theta[i] = np.arccos(z[i])
            phi[i] = (np.arctan2(y[i],x[i]) + 4.0*np.pi)%(2.0*np.pi)
            if (dX3[2] > 0):
                psi[i] = np.arccos(dX3.dot(np.array([-np.sin(phi[i]),np.cos(phi[i]),0]))) - np.pi/2.0
            else:
                psi[i] = -np.arccos(dX3.dot(np.array([-np.sin(phi[i]),np.cos(phi[i]),0]))) - np.pi/2.0
            psi[i] = (psi[i] + 4.0*np.pi ) % (2.0*np.pi) - np.pi
        
        #plt.plot(phi[0:10],theta[0:10])
        #plt.show()
        
        np.save(dirout+'theta'+termin+'%s'%id,theta)
        np.save(dirout+'phi'+termin+'%s'%id,phi)
        np.save(dirout+'psi'+termin+'%s'%id,psi)

        if (0): #ecliptic
            dirp = '/home/cmb/tmatsumu/file2others/Guillaume/LB_L2_20180118_P2G_case7_samplerate_SCANSPEC_1440min_45.0degs_90.6min_50.0degs_0.1rpm_365day_nside256_10Hz/'
            filenameRa = dirp+'sample_boresight_365days_ra.bi'
            filenameDec = dirp+'sample_boresight_365days_dec.bi'
            filenamePsi = dirp+'sample_boresight_365days_psi.bi'
            fileRa = open(filenameRa, mode='rb')
            fileDec = open(filenameDec, mode='rb')
            filePsi = open(filenamePsi, mode='rb')
            rap = np.fromfile(fileRa,dtype=np.float64,count=nn)
            decp = np.fromfile(fileDec,dtype=np.float64,count=nn)
            psip = np.fromfile(filePsi,dtype=np.float64,count=nn)
            rot=hp.Rotator(coord=['C','E'])
            (thetaE,phiE)=rot(np.pi/2.0-decp,rap)
            np.save('ThetaE.npy',thetaE)
            np.save('phiE.npy',phiE)
            np.save('psiE.npy',psip)
            print(thetaE[0:3],theta[0:3])
            print(phiE[0:3],phi[0:3])
            print(psip[0:3],psi[0:3])
            exit(1)
    
    wt0 = wspin*t[nn-1] + wt0
    Phi0 = wprec*t[nn-1] + Phi0
    Ot0 = wearth*t[nn-1] + Ot0    




