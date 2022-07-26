import numpy as np
import healpy as hp
import scipy as sc
import math as mt
import array
import csv
#from mapsim import Mapsim
#from .hwp import HalfWavePlate

#class TODsim(HalfWavePlate):
class TODsim():
    def __init__(self, nside, nmin, nt, dirscr, pars=None, sim_pars=None, noise_pars=None):
        self.T_cmb = 2.725 # Kelvin
        self.nside = nside
        self.nmin = nmin
        self.nt = nt
        self.dirscr = dirscr        
        self.pars, self.sim_pars, self.noise_pars = self.prepare_params(pars, sim_pars, noise_pars)

    def prepare_params(self, pars=None, sim_pars=None, noise_pars=None):
        if pars is None:
            pars, _, _ = self.get_default_params()
        if sim_pars is None:
            _, sim_pars, _ = self.get_default_params()
        if noise_pars is None:
            _, _, noise_pars = self.get_default_params()
        return pars, sim_pars, noise_pars
    
    def get_default_params(self):
        """ Returns default set of parameters describing a
        time ordered data (TOD) simulation. The parameters 
        are distributed into 2 dictionaries, corresponding 
        to "params" (i.e. specific to the experiment)
        and "sim_params" (i.e. specific to the simulation.)
        
        BS = 0 # BoreSight pointing (LEAVE 0=NO)
        idealHWP = 1 # If 1: assume no HWP imperfections
        Conly = 0 # Only use if idealHWP = 0
        dipole = 0 # If 1: include dipole
        inoutSky = 0 # LEAVE 0
        CMB = 0 # If 0: do not include CMB
        sky = 1 # If 1: include already CMB + foregrounds
        noPol = 0 # if 0: include polarization
        noise = 0 # if 1: include noise (LEAVE 1)
        sha = 0 # Tilt HWP (e.g. 5) (LEAVE 0)
        interpS = 0 # If 0: do not interpolate results (LEAVE 0 TO COMPARE WITH INPUT)
        convertEG = 0 # Only use is interpS=1 (LEAVE 0)
        shIP = 0 # Shifting of HWP non-idealities (LEAVE 0)

        monopole = 0 # If 1: add monopole
        #  if added to OmegatHWP I get a fixed line, if not a broad line 

        nside = 256
        
        #NET = 59.95  #### for 89 GHz
        #NET = 50.69  #### for 140 GHz 
        NET = 38.44  #### for 140 GHz #detector's NET from table
        """
        nmin = self.nmin
        nt = self.nt
        dirscr = self.dirscr
        fsamp = 19.0 # Sampling frequency (Hz),
                     # must be the same as input pointing files
        f_HWP = 46.0/60.0 # HWP rotation frequency HWP [rad/sec]
        Totalyr = 1.0 # years of observation
        sss = 2**22 # Segments per second, parameter passed in mapmaking
                    # must be the same as parameter input in mapmaking
        nn = int(3600*24*366.0*Totalyr*fsamp/sss) # 1 year in sampling time / segment = 143
        nwhole=nn
        if (nt < nn):
            nn = nt
        nnn = nn*sss
        Omega_HWP = f_HWP * 2.0 * np.pi / fsamp # HWP rotation speed (0.25) [rad/sec^2]
                                                # must be the same as parameter input in mapmaking
    
        ####### If including 1/f noise
        fknee = 0.05 # Hz
        beta = 1.0

        ##### Data chunk sample list 
        dd1 = np.arange(nwhole)*nnn/nn
        dd2 = dd1 + nnn/nn - 1
    
        dd = np.zeros((nwhole*2))
        dd[0:2*nwhole:2] = dd1
        dd[1:2*nwhole+11:2] = dd2
        ddd = dd.astype(np.int64)

        if (nmin == 0):
            ddd.tofile(dirscr+'samplistLB_%i.bi'%nwhole)

        pars = {'fsamp': fsamp,
                'f_HWP': f_HWP,
                'Totalyr': Totalyr,
                'sss': sss,
                'nn': nn,
                'nnn': nnn,
                'Omega_HWP': Omega_HWP,
        }
    
        choice_sim = {'BS': False,
                      'idealHWP': True,
                      'Conly': False,
                      'dipole': False,
                      'inoutSky': False,
                      'CMB': False,
                      'sky': True,
                      'noPol': False,
                      'noise': True,
                      'sha': 0,
                      'interpS': False,
                      'convertEG': False,
                      'shIP': False,
                      'monopole': False,
        }

        noi_pars = {'BS': False,
                      'fknee': fknee,
                      'beta': beta,
                      'NET': 38.44,  #### for 140 GHz #detector's NET from table
                      #'NET': 59.95  #### for 89 GHz
        }
        
        return pars, choice_sim, noi_pars

    def ReadBoloInfo(self, bolofile,bololist,pair):
        nlist = np.size(bololist)
        theta_ = np.zeros(nlist)
        phi_ = np.zeros(nlist)
        psib_ = np.zeros(nlist)
        ilist = 0
        while (ilist < nlist):
            ilist_ = ilist
            with open(bolofile,'r') as bfile:
                for line in bfile:
                    row = line.split()
                    if (ilist < nlist):
                        if (row[0] != '#'):
                            namecomp = 'LFT'+row[1]+'_'+row[2]+'_'+row[4]
                            if (namecomp == bololist[ilist]):
                                theta_[ilist] = float(row[6])*np.pi/180.0
                                phi_[ilist] = float(row[7])*np.pi/180.0
                                psib_[ilist] = (float(row[2]) % 2.0)*45.0*np.pi/180.0
                                ilist += 1
                if (ilist == ilist_):
                    print('Bolometer name not found in the list')
                    exit(1)

        if (pair):
            bolonames_=bololist
            bolonames=['']
            for ip in range(np.size(bolonames_)):
                bolonames.append(bolonames_[ip]+'a')
                bolonames.append(bolonames_[ip]+'b')
            bolonames.pop(0)
            theta = np.zeros(2*nlist)
            phi = np.zeros(2*nlist)
            psib = np.zeros(2*nlist)
            theta[0:2*nlist:2] = theta_
            theta[1:2*nlist:2] = theta_
            phi[0:2*nlist:2] = phi_
            phi[1:2*nlist:2] = phi_
            psib[0:2*nlist:2] = psib_
            psib[1:2*nlist:2] = psib_+np.pi/2.0
        else:
            bolonames = bolonames_
            theta=theta_
            phi_=phi_
            psib=psib_

        return theta,phi,psib,bolonames

    def readbolofile(self,bfile):
        bolonames_ = ['']
        with open(bfile,'r') as bfile:
            for line in bfile:
                row = line.split()
                if (row[0] != '#'):
                    namecomp = 'LFT'+row[1]+'_'+row[2]+'_'+row[4]
                    bolonames_.append(namecomp)
        bolonames_=bolonames_[2:]
        idstr = 0
        bolonames_=bolonames_[idstr:]
        return bolonames_

    def pointingshift(self, x,y,thetap,phip,psip):
        theta_ = mt.asin(np.sqrt(x*x + y*y))
        phi_   = mt.atan2(y,x)
        ##psi_   = phi_  ### check the frame
        theta  = np.arccos(np.sin(thetap)*np.sin(theta_)*np.cos(phi_+psip) + np.cos(thetap)*np.cos(theta_))
        phi    = np.arctan2(-np.sin(theta_)*np.sin(phi_+psip) , (-np.cos(thetap)*np.sin(theta_)*np.cos(phi_+psip) + np.sin(thetap)*np.cos(theta_))) + phip
        ## Rotation of angles
        psi = np.zeros(np.size(thetap))
    
        drp = np.zeros(3)
        R = np.zeros((3,3))
        eph = np.zeros(3)
    
        sinthetap = np.sin(thetap)
        costhetap = np.cos(thetap)
        sinpsip = np.sin(psip)*np.cos(theta_)
        cospsip = np.cos(psip)*np.cos(theta_)
        drp[2] = -np.sin(theta_)*np.cos(phi_)
        R[1,1] = -1.0
        sinphi = np.sin(phi-phip)
        cosphi = np.cos(phi-phip)
        eph[2] = 0.0

        for ii in np.arange(0,np.size(thetap)):
            drp[0] = cospsip[ii]
            drp[1] = sinpsip[ii]
            #drp[2] = -np.sin(theta_)*np.cos(phi_+psip[ii])
            R[0,0] = -costhetap[ii]
            R[0,2] = sinthetap[ii]
            R[2,0] = sinthetap[ii]
            R[2,2] = costhetap[ii]
            dr = np.dot(R,drp)
            eph[0] = -sinphi[ii]
            eph[1] = cosphi[ii]
            #print(dr,np.dot(dr,np.transpose(eph))/np.sqrt(np.dot(dr,np.transpose(dr))))
            aaa = np.dot(dr,np.transpose(eph))/np.sqrt(np.dot(dr,np.transpose(dr)))
            if (aaa > 1.0):
                print(aaa)
                aaa = 1.0
            if (aaa < -1.0):
                print(aaa)
                aaa = -1.0
            psitmp = mt.acos(aaa)
            if (dr[2] < 0):
                psitmp = -psitmp
                #print(dr[2])
            psitmp = (psitmp +np.pi/2.0 + 4.0*np.pi ) % (2.0*np.pi) - np.pi
            #if (ii % 10000000) == 0:
            #    print(ii)
            #    print(np.double(ii)/nnn)
            psi[ii] = psitmp
        
        return (theta,phi,psi)


    def gen_monopole(self, unit):
        """
        Add monopole to CMB temperature map
        i.e. add amplitude in deltaT (here things in cmb T)
        """
        nside = self.nside
        amp_K = self.T_cmb
        if unit == 'uK':
            amp_unit = amp_K * 1.e6
        if unit == 'mK':
            amp_unit = amp_K * 1.e3
        if unit == 'K':
            amp_unit = amp_K
        npix = hp.nside2npix(nside)
        monopole_out = amp_unit*np.ones(npix)
        mapmon = hp.ud_grade(monopole_out,nside)
        return mapmon
    
    def gen_dipole(self, radip, decdip, ampldip):
        """
        Generate CMB dipole map from Planck
        """
        nside = self.nside
        pars = self.pars
        sim_pars = self.sim_pars

        indpix = np.arange(12*nside*nside)
        [thetapix, phipix] = hp.pix2ang(nside,indpix)
        x=np.sin(thetapix)*np.cos(phipix)
        y=np.sin(thetapix)*np.sin(phipix)
        z=np.cos(thetapix)
        xdip=np.sin(np.pi/2.0-decdip)*np.cos(radip)
        ydip=np.sin(np.pi/2.0-decdip)*np.sin(radip)
        zdip=np.cos(np.pi/2.0-decdip)
        mapdip = ampldip *(x*xdip + y*ydip + z*zdip) 
        mapdip = hp.ud_grade(mapdip,nside)
        return mapdip

    def Ec2Gal(self):
        """
        Rotate from ecliptic to galactic coordinates
        """
        rot=hp.Rotator(coord=['E','G'])
        (thetaE,phiE) = hp.pix2ang(512,np.arange(12*512*512))
        (thetaG,phiG)=rot(thetaE,phiE)
        (thetaGsh,phiGsh)=rot(thetaE-1e-8,phiE)
        dtheta = thetaGsh - thetaG
        dphi = phiGsh - phiG
        psi = np.arctan2(-dphi*np.sin(thetaG),dtheta)
        return thetaG, phiG, psi, dtheta

    def Ecliptic2Galactic_maps(self, mapI, mapQ, mapU):
        """
        Convert maps from ecliptic to galactic coordinates
        """
        nside = self.nside
        pars = self.pars
        sim_pars = self.sim_pars

        ##rot=hp.Rotator(coord=['E','G'])
        ##(thetaE,phiE) = hp.pix2ang(512,np.arange(12*512*512))
        ##(thetaG,phiG)=rot(thetaE,phiE)
        ##(thetaGsh,phiGsh)=rot(thetaE-1e-8,phiE)
        ##dtheta = thetaGsh - thetaG
        ##dphi = phiGsh - phiG
        ##psi = np.arctan2(-dphi*np.sin(thetaG),dtheta)
        thetaG, phiG, psi, dtheta = self.Ec2Gal()
        ipixGE = hp.ang2pix(512,thetaG,phiG)
        mapIE = hp.pixelfunc.get_interp_val(mapI, thetaG, phiG)
        mapQE = np.cos(2*psi)*hp.pixelfunc.get_interp_val(mapQ, thetaG, phiG) - np.sin(2*psi)*hp.pixelfunc.get_interp_val(mapU, thetaG, phiG)
        mapUE = np.sin(2*psi)*hp.pixelfunc.get_interp_val(mapQ, thetaG, phiG) + np.cos(2*psi)*hp.pixelfunc.get_interp_val(mapU, thetaG, phiG)
        mapI = hp.ud_grade(mapIE,nside)
        mapQ = hp.ud_grade(mapQE,nside)
        mapU = hp.ud_grade(mapUE,nside)    
        return mapI, mapQ, mapU


    def get_sky_maps(self):
        """
        Read input maps (generated w/ PySM).
        No interpolation made on purpose, only used hp.udgrade
        Realistic maps should go from interpolated maps to pixelised maps
        But here we use same pixelation scheme for input and output 
        to check that simulated maps with no noise reproduce input 
        simulation maps up to numerical value
        """
        nside = self.nside
        pars = self.pars
        sim_pars = self.sim_pars
        
        mapI = np.zeros(12*nside*nside)
        mapQ = np.zeros(12*nside*nside)
        mapU = np.zeros(12*nside*nside)
    
        terminS='signal'

        ### Read input maps    
        # Include only CMB
        if sim_pars['CMB']:
            terminS = terminS + '_onlyCMB'
            [mapI512, mapQ512, mapU512] = hp.read_map('/home/cmb/susanna/TODs2Maps/Input/LB_LFT_140_cmb_0032_FGjsg_20200420.fits',field=(0,1,2))
            mapI = hp.ud_grade(mapI512,nside)
            mapQ = hp.ud_grade(mapQ512,nside)
            mapU = hp.ud_grade(mapU512,nside)
        # Include CMB + foregrounds
        elif sim_pars['sky']:
            terminS = terminS + '_sky'
            # Include dust
            [mapI512, mapQ512, mapU512] = hp.read_map('/home/cmb/susanna/TODs2Maps/Input/LB_LFT_140_dust_FGjsg_20200420.fits',field=(0,1,2))
            mapQ = hp.ud_grade(mapQ512,nside)
            mapU = hp.ud_grade(mapU512,nside)
            # Include CMB
            [mapI512, mapQ512, mapU512] = hp.read_map('/home/cmb/susanna/TODs2Maps/Input/LB_LFT_140_cmb_0032_FGjsg_20200420.fits',field=(0,1,2))
            mapIns = hp.ud_grade(mapI512,nside)
            mapQns = hp.ud_grade(mapQ512,nside)
            mapUns = hp.ud_grade(mapU512,nside)
            mapI += mapIns
            mapQ += mapQns
            mapU += mapUns
            # Include synchrotron
            [mapI512, mapQ512, mapU512] = hp.read_map('/home/cmb/susanna/TODs2Maps/Input/LB_LFT_140_synch_FGjsg_20200420.fits',field=(0,1,2))
            mapIns = hp.ud_grade(mapI512,nside)
            mapQns = hp.ud_grade(mapQ512,nside)
            mapUns = hp.ud_grade(mapU512,nside)
            mapI += mapIns
            mapQ += mapQns
            mapU += mapUns
            # Include free-free emission
            [mapI512, mapQ512, mapU512] = hp.read_map('/home/cmb/susanna/TODs2Maps/Input/LB_LFT_140_freefree_FGjsg_20200420.fits',field=(0,1,2))
            mapIns = hp.ud_grade(mapI512,nside)
            mapQns = hp.ud_grade(mapQ512,nside)
            mapUns = hp.ud_grade(mapU512,nside)
            mapI += mapIns
            mapQ += mapQns
            mapU += mapUns
            # Include AME
            [mapI512, mapQ512, mapU512] = hp.read_map('/home/cmb/susanna/TODs2Maps/Input/LB_LFT_140_ame_FGjsg_20200420.fits',field=(0,1,2))
            mapIns = hp.ud_grade(mapI512,nside)
            mapQns = hp.ud_grade(mapQ512,nside)
            mapUns = hp.ud_grade(mapU512,nside)
            mapI += mapIns
            mapQ += mapQns
            mapU += mapUns
        
        if sim_pars['noPol']:
            mapQ = np.zeros(12*nside*nside)
            mapU = np.zeros(12*nside*nside)
            terminS = terminS + '_noPol'

        if sim_pars['monopole']:
            # Generate monopole map in K_cmb units
            monopole_K_CMB = self.gen_monopole(self.T_cmb, nside, 'uK')
            terminS = terminS + '_monopole'
            mapI += monopole_K_CMB            
        
        print('No EG')
        #terminS = terminS + '_convEG'
        mapI, mapQ, mapU = self.Ecliptic2Galactic_maps(mapI, mapQ, mapU)

        if sim_pars['dipole']:
            print("Yes dipole")
            terminS = terminS + '_dipole'
            ampldip = 3362.08 #in muK
            #print("No convertEG")
            radip = 264.021/180.0*np.pi ## Galactic
            decdip = 48.253/180.0*np.pi
            rotd=hp.Rotator(coord=['G','E']) ##to ecliptic
            (thetaEd,phiEd)=rotd(np.pi/2.0-decdip,radip)
            radip = phiEd.copy()
            decdip = np.pi/2.0-thetaEd
            #print('Dipole coord:',radip*180.0/np.pi,decdip*180.0/np.pi)
        
            mapdip = gen_dipole(nside=nside, radip=radip, decdip=decdip, ampldip=ampldip, pars=pars, sim_pars=sim_pars)

            mapI += mapdip

        return mapI, mapQ, mapU


    def get_noise(self):
        """
        Additive noise
        """
        nside = self.nside
        pars = self.pars
        noi_pars = self.noise_pars
        fsamp = pars['fsamp']
        nnn = pars['nnn']
        NET = noi_pars['NET']
        fknee = noi_pars['fknee']
        beta = noi_pars['beta']
        
        sign = NET * np.sqrt(fsamp)
        nfrac = int(nnn/nn)
        ff = np.fft.fftfreq(nfrac)*fsamp
        sp = np.zeros(nfrac)
        ff[0] = ff[1]
        sp = (fknee/abs(ff))**beta + 1
        isp = 1.0/abs(sp)
        return sign, sp, isp


    def pointing(self, isamp, idet, fileRa, fileDec, filePsi, filePsiTemp, phi_fp, theta_fp):
        nside = self.nside
        pars = self.pars
        sim_pars = self.sim_pars
    
        fileRa.seek(isamp*8)
        fileDec.seek(isamp*8)
        filePsi.seek(isamp*8)
        filePsiTemp.seek(isamp*8)

        sss = pars['sss']
        rap = np.fromfile(fileRa,dtype=np.float64,count=sss)
        decp = np.fromfile(fileDec,dtype=np.float64,count=sss)
        psip = np.fromfile(filePsi,dtype=np.float64,count=sss)
        psipTemp = np.fromfile(filePsiTemp,dtype=np.float64,count=sss)
    
        xx = np.sin(theta_fp[idet])*np.cos(phi_fp[idet])
        yy = np.sin(theta_fp[idet])*np.sin(phi_fp[idet])
        (thetatmp,phitmp,psi) = self.pointingshift(xx,yy,np.pi/2.0-decp,rap,psip)
    
        for ip in range(np.size(psip)):
            if (not(psip[ip] < 10.0)):
                psip[ip] = psip[ip-1]
                psipTemp[ip] = psipTemp[ip-1]
        
        print('Check:  ',np.std(psip-psipTemp))
    
        ### Location wrt to boresight
        sha = sim_pars['sha']
        x_bs = np.sin(theta_fp[idet])*np.cos(phi_fp[idet])
        y_bs = np.sin(theta_fp[idet])*np.sin(phi_fp[idet]) + sha/180.0*np.pi
    
        theta_bs = mt.asin(np.sqrt(x_bs*x_bs + y_bs*y_bs))
        phi_bs = np.arctan2(y_bs,x_bs) - np.pi/2.0  ##### 
    
        print('Location in the focal plane:','theta=',theta_bs,'phi=',phi_bs)
    
        return theta_bs, phi_bs, thetatmp, phitmp, psi


    def add_noise(self, SignalHWP, ra, sign, sp):
        nside = self.nside
        pars = self.pars
        sim_pars = self.sim_pars
        
        dataw = sc.randn(np.size(ra))*sign
        fdata = np.fft.fft(dataw)
        # 1/f:
        fdata *= np.sqrt(abs(sp))
        dataf = np.fft.ifft(fdata)
        #indsamp = indsamp + np.size(ra)
        #SignalHWP.tofile(fdat)
        #if (noise):
        signoi = SignalHWP + dataf.real
        noi = dataf.real
        return signoi, noi

    def fcmb(self, nu):
        """ CMB SED (in antenna temperature units).
        """
        x=0.017608676067552197*nu
        ex=np.exp(x)
        return ex*(x/(ex-1))**2

    def RJ2CMB_units(self, map_freq, nu):
        """
        Frequency map shape npix
        From RJ to T_cmb units
        """
        return map_freq*fcmb(nu)
