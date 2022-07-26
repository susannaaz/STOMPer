import numpy as np
import healpy as hp
import scipy as sc
import math as mt
import array
import csv
from mapsim import Mapsim
from .todsimulation import TODsim

class HalfWavePlate(TODsim):
    def __init__(self, mean_pars=None, moment_pars=None):
        self.mean_pars, self.moment_pars = self.prepare_params(mean_pars, moment_pars)

    def prepare_params(self, mean_pars=None, moment_pars=None):
        mp = Mapsim(mean_p=None, moment_p=None)
        if mean_pars is None:
            mean_pars, _ = mp.get_default_params()
        if moment_pars is None:
            _, moment_pars = mp.get_default_params()
        return mean_pars, moment_pars


    def get_MuellerMatrix(self,typ):
        """
        Mueller matrix model
        """
        if typ=='RCWA':
            # HWP Mueller Matrix parameters from RCWA simulations
            A = np.array([[0.978,3.075e-5,1.242e-5,6.353e-6],[2.724e-5,-2.536e-4,-1.117e-3,-4.039e-3],[-3.965e-5,-1.328e-3,-4.65e-5,-1.675e-2],[2.082e-5,-6.224e-4,1.978e-2,-0.9751]])
            B = np.array([[1.584e-4,1.825e-3,1.824e-3,3.127e-5],[1.823e-3,3.335e-3,4.446e-3,7.141e-2],[1.823e-3,2.564e-3,3.736e-3,7.139e-2],[5.658e-5,7.152e-2,7.149e-2,1.436e-3]])
            C = np.array([[1.374e-6,4.104e-5,4.117e-5,1.174e-6],[3.967e-5,0.9766,0.9764,0.01998],[3.960e-5,0.9765,0.9763,0.01998],[1.006e-6,1.699e-2,1.699e-2,3.478e-4]])
            phiB = np.array([[-1.267,2.433,0.8615,1.763],[-2.432,1.816,-0.07809,2.479],[2.281,-0.1527,-1.870,0.9089],[-0.4335,-2.479,2.233,3.008]])
            phiC = np.array([[-1.293,1.026,-0.545047,-0.92531],[-0.4935,9.621e-4,-1.57,-1.609],[-2.06216,-1.57,3.143,3.103],[1.328,1.803,0.2322,0.1961]])
            return A, B, C, phiB, phiC

        elif typ=='RCWA_10_deg':
            #### RCWA simulations 10 degrees
            A = np.array([[9.607068e-01, 8.830989e-05, -7.870052e-06, 9.168043e-05],[9.595263e-05, 1.876917e-04, 4.871667e-04, -3.450810e-03],[4.388989e-06, -4.628583e-04, 7.476250e-04, -2.115083e-02],[-9.343053e-05, -1.286971e-03, 2.416139e-02, -9.587372e-01]])
            B = np.array([[4.886619e-06, 5.152905e-04, 5.158310e-04, 2.642446e-05],[5.429647e-04, 3.104719e-03, 3.281772e-03, 2.309142e-02],[5.422577e-04, 2.960065e-03, 3.235058e-03, 2.304434e-02],[4.608402e-05, 2.310672e-02, 2.308983e-02, 1.037730e-03]])
            C = np.array([[1.092906e-07, 9.257189e-05, 9.249520e-05, 1.967281e-06],[8.863102e-05, 9.590444e-01, 9.587381e-01, 2.415395e-02],[8.865251e-05, 9.588118e-01, 9.585058e-01, 2.414760e-02],[1.557940e-06, 2.139254e-02, 2.138538e-02, 5.549560e-04]])
            return A, B, C

        elif typ=='ideal_HWP':
            #### Ideal HWP
            A0 = np.array([[1.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, -1.0]])
            B0 = np.array([[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0]])
            C0 = np.array([[0.0, 0.0, 0.0, 0.0],[0.0, 1.0, 1.0, 0.0],[0.0, 1.0, 1.0, 0.0],[0.0, 0.0, 0.0, 0.0]])
            phiB = np.array([[-2.315508, -0.485212, -2.055914, 1.201617],[2.855673, -0.251553, -1.995359, 0.115144],[1.285187, -2.007741, 2.539638, -1.454225],[-1.942084, 2.297805, 0.728116, 1.157157]])
            phiC = np.array([[-0.8422613, -3.7885e-02, -1.60835, -1.466364],[0.1421380, -6.13e-04, -1.571364, -1.631821],[-1.429282, -1.571492, 3.140942, 3.080486],[1.910318, 1.724201, 0.15346, 8.858e-02]])
            return A0, B0, C0, phiB, phiC        
        return None

    def transform_HWP_coeff(self, theta_bs, A, A0, C, C0, B, phiB, phiC, pars, sim_pars):
        """ Transform HWP coef.
        """
        #if pars is None:
        #    pars, _ = get_default_params()
        #if sim_pars is None:
        #    _, sim_pars = get_default_params()
        pars = self.mean_pars
        pars = self.mean_pars
        
        factHWP = np.sin(4.44*theta_bs)*np.sin(4.44*theta_bs)/np.sin(4.44*10.0/180.0*np.pi)/np.sin(4.44*10.0/180.0*np.pi) # Change that for a better function
        Ad = (A-A0)*factHWP + A0
        Cd = (C-C0)*factHWP + C0
        Bd = B*factHWP
    
        print('HWP imperfection boost factor:',factHWP)
        print('INSTRUMENTAL POLARIZATION:',Cd[1][0],Cd[2][0],' for Theta=', theta_bs*180.0/np.pi ,'    (Original:',C[1][0],C[2][0],', 10 degrees)')
    
        Bd = Bd*np.exp(1j*phiB)
        Cd = Cd*np.exp(1j*phiC)
        
        ## ideal HWP
        if sim_pars['idealHWP']:
            print("ideal HWP")
            Ad = np.array([[1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,-1.0]])
            Bd = np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])
            Cd = np.array([[0.0,0.0,0.0,0.0],[0.0,1.0,1.0,0.0],[0.0,1.0,1.0,0.0],[0.0,0.0,0.0,0.0]])
            phiB = np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])
            phiC = np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,-np.pi/2.0,0.0],[0.0,-np.pi/2.0,np.pi,0.0],[0.0,0.0,0.0,0.0]])
            Bd = Bd*np.exp(1j*phiB)
            Cd = Cd*np.exp(1j*phiC)
        elif sim_pars['Conly']:
            #print("wrong logic")
            Ad = np.array([[1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,-1.0]])
            Bd = np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])
            phiB = np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])
            ##Temporary
            phiC = np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,-np.pi/2.0,0.0],[0.0,-np.pi/2.0,np.pi,0.0],[0.0,0.0,0.0,0.0]])
            Cd = np.zeros_like(Ad)
            ##
            Bd = Bd*np.exp(1j*phiB)
        return Ad, Bd, Cd 


    def get_signal_HWP(self, psib, ra, dec, indsamp, psi, dpsi, phi_bs, nside, Ad, Bd, Cd, 
                       pars, sim_pars):
        """
        HWP signal per detector
        """
        ##if pars is None:
        ##    pars, _ = get_default_params()
        ##if sim_pars is None:
        ##    _, sim_pars = get_default_params()
        
        MM = np.zeros((4,4))
        MMhwp = np.zeros((4,4))
        RRM = np.zeros((4,4))
        RRM_ = np.zeros((4,4))
        PP = np.zeros((4,4))
        RRP = np.zeros((4,4))
        RRP_ = np.zeros((4,4))
    
        ## Check the origin of phases for the HWP and check the rotation direction  
        PP[0,0] = 1.0
        PP[1,0] = 1.0
        PP[0,1] = 1.0
        PP[1,1] = 1.0
        
        RRM[0,0] = 1.0
        RRP[0,0] = 1.0
        RRM_[0,0] = 1.0
        RRP_[0,0] = 1.0
    
        SignalHWP = np.zeros(np.size(ra))
        
        Omegathwp = pars['Omega_HWP'] * (np.arange(np.size(ra)) + indsamp) + dpsi# * factrotHWP

        # Flat sky approximation
        ksi = psi.copy() + phi_bs
        Phi = psi.copy() + phi_bs
        beta = Omegathwp - Phi - phi_bs
        
        indpix = hp.ang2pix(nside,np.pi/2.0-dec,ra)
        Cos2Psi = np.cos(2*(psi+psib)) #psib[idet]))
        Sin2Psi = np.sin(2*(psi+psib))
        Cos2Ksi = np.cos(2*ksi)
        Sin2Ksi = np.sin(2*ksi)
        rho = 2*Omegathwp
    
        mapI, mapQ, mapU = get_sky_maps(nside, pars, sim_pars)

        mapIinterp = hp.pixelfunc.get_interp_val(mapI, np.pi/2.0-dec, ra)
        mapQinterp = hp.pixelfunc.get_interp_val(mapQ, np.pi/2.0-dec, ra)
        mapUinterp = hp.pixelfunc.get_interp_val(mapU, np.pi/2.0-dec, ra)      
        for it in np.arange(np.size(ra)):
            #if ((it % 100000) == 0):
            #    print it
            RRP[1,1] = Cos2Psi[it]
            RRP_[1,1] = Cos2Psi[it]
            RRP[2,2] = Cos2Psi[it]
            RRP_[2,2] = Cos2Psi[it]
            RRP[1,2] = -Sin2Psi[it]
            RRP_[1,2] = Sin2Psi[it]
            RRP[2,1] = Sin2Psi[it]
            RRP_[2,1] = -Sin2Psi[it]
            RRM[1,1] = Cos2Ksi[it]
            RRM_[1,1] = Cos2Ksi[it]
            RRM[2,2] = Cos2Ksi[it]
            RRM_[2,2] = Cos2Ksi[it]
            RRM[1,2] = -Sin2Ksi[it]
            RRM_[1,2] = Sin2Ksi[it]
            RRM[2,1] = Sin2Ksi[it]
            RRM_[2,1] = -Sin2Ksi[it]
            
            # Mueller Matrix in the focal plane coordinates
            MMhwp = np.real(Ad + Bd*np.exp(1j*2.0*beta[it]) + Cd*np.exp(1j*4.0*beta[it])) 
            
            MMt = np.dot(RRM,np.dot(MMhwp,RRM_))    
            PPt = np.dot(RRP,np.dot(PP,RRP_))
            MM = np.dot(PPt,MMt)
            
            if (0):
                if (it < 2):
                    print("Matrix",MMhwp)
                    print(mapI[indpix[it]]*MM[0,0] + mapQ[indpix[it]]*MM[0,1] + mapU[indpix[it]]*MM[0,2])
                    print(it, MM[0,0], MM[0,1], MM[0,2], 'Angles: ', psi[0], beta[0])
                    print (np.cos(2.0*(-psi[it]-psib[idet] + 2.0*Omegathwp[it])), np.sin(2.0*(-psi[it]-psib[idet] + 2.0*Omegathwp[it])))
                    print('#')

            if sim_pars['interpS']:
                SignalHWP[it] = mapIinterp[it]*MM[0,0] + mapQinterp[it]*MM[0,1] + mapUinterp[it]*MM[0,2]
            else:
                SignalHWP[it] = mapI[indpix[it]]*MM[0,0] + mapQ[indpix[it]]*MM[0,1] + mapU[indpix[it]]*MM[0,2]
                #SignalHWP[it] = MM[0,0]  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                ##print(SignalHWP[0:10])

        return SignalHWP, indpix
