import healpy as hp
import numpy as np
from scipy.special import zeta
import pysm
from pysm.nominal import models
import os

import sys
sys.path.append('/mnt/zfsusers/susanna/Software/PySM_public/pysm/')
sys.path.append('/mnt/zfsusers/susanna/Software/fgbuster/fgbuster/')

class Mapsim:
    """
    Simulation of input maps
    This stage generates input maps for the LiteBIRD CMB experiment
    Uses PySM (Thorne et al)
    """
    def __init__(self, mean_pars=None, moment_pars=None):
        self.mean_pars, self.moment_pars = self.prepare_params(mean_pars, moment_pars)

    def prepare_params(self, mean_pars=None, moment_pars=None):
        if mean_pars is None:
            mean_pars, _ = self.get_default_params()
        if moment_pars is None:
            _, moment_pars = self.get_default_params()
        return mean_pars, moment_pars

    def get_default_params(self):
        """ Returns default set of parameters describing a
        given sky realization. The parameters are distributed
        into 2 dictionaries, corresponding to "mean"
        (i.e. non-moment-related) parameters and "moment" parameters.
        The mean parameters are:
        - 'A_dust_BB': B-mode dust power spectrum amplitude at
          the frequency `nu0_dust_def'.
        - 'EB_dust': EE to BB ratio.
        - 'alpha_dust_EE': tilt in D_l^EE for dust
        - 'alpha_dust_BB': tilt in D_l^BB for dust
        - 'nu0_dust_def': frequency at which 'A_dust_BB' is defined.
        - 'nu0_dust': frequency at which amplitude maps should be
          generated. At this frequency spectral index variations
          are irrelevant.
        - 'beta_dust': mean dust spectral index.
        - A copy of all the above for synchrotron (called `sync`
          instead of `dust`).
        - 'temp_dust': dust temperature.
        - 'include_XXX': whether to include component XXX, where
          XXX is CMB, sync or dust.
        - 'include_Y': whether to include Y polarization, where 
          Y is E or B. 
        The moment parameters are:
        - 'amp_beta_dust': delta_beta power spectrum amplitude
          for dust.
        - 'gamma_beta_dust': delta_beta power spectrum tilt
          for dust.
        - 'l0_beta_dust': pivot scale for delta_beta (80).
        - 'l_cutoff_beta_dust': minimum ell for which the delta
          beta power spectrum is non-zero (2).
        - A copy of the above for synchrotron.
        """
        mean_pars = {'r_tensor': 0,
                     'A_dust_BB': 5,
                     'EB_dust': 2.,
                     'TB_dust': 3.,
                     'TE_dust': 1.5,
                     'alpha_dust_EE': -0.42,
                     'alpha_dust_BB': -0.42,
                     'nu0_dust': 353.,
                     'nu0_dust_def': 353.,
                     'beta_dust': 1.6,
                     'temp_dust': 19.6,
                     'A_sync_BB': 2,
                     'EB_sync': 2.,
                     'TB_sync': 3.,
                     'TE_sync': 1.5,
                     'alpha_sync_EE': -0.6,
                     'alpha_sync_BB': -0.6,
                     'nu0_sync': 23.,
                     'nu0_sync_def': 23.,
                     'beta_sync': -3.,
                     'A_lens' : 1,
                     'include_CMB': True,
                     'include_dust': True,
                     'include_sync': True,
                     'include_T': True,
                     'include_E': True,
                     'include_B': True,
                     'dust_SED': 'mbb',
        }
        moment_pars = {'amp_beta_sync': 0.,
                       'gamma_beta_sync': -3.,
                       'l0_beta_sync': 80.,
                       'l_cutoff_beta_sync': 2,
                       'amp_beta_dust': 0.,
                       'gamma_beta_dust': -3.,
                       'l0_beta_dust': 80.,
                       'l_cutoff_beta_dust': 2}
        return mean_pars, moment_pars

        
    def get_delta_beta_cl(self, amp, gamma, ls, l0=80., l_cutoff=2):
        """
        Returns power spectrum for spectral index fluctuations.
        Args:
        - amp: amplitude
        - gamma: tilt
        - ls: array of ells
        - l0: pivot scale (default: 80)
        - l_cutoff: ell below which the power spectrum will be zero.
            (default: 2).
        Returns:
        - Array of Cls
        """
        ind_above = np.where(ls > l_cutoff)[0]
        cls = np.zeros(len(ls))
        cls[ind_above] = amp * (ls[ind_above] / l0)**gamma
        return cls

    def get_beta_map(self, nside, beta0, amp, gamma, l0=80, l_cutoff=2, seed=None, gaussian=True):
        """
        Returns realization of the spectral index map.
        Args:
        - nside: HEALPix resolution parameter.
        - beta0: mean spectral index.
        - amp: amplitude
        - gamma: tilt
        - l0: pivot scale (default: 80)
        - l_cutoff: ell below which the power spectrum will be zero.
            (default: 2).
        - seed: seed (if None, a random seed will be used).
        - gaussian: beta map from power law spectrum (if False, a spectral 
            index map obtained from the Planck data using the Commander code 
            is used for dust, and ... for sync)  
        Returns:
        - Spectral index map
        """
        if gaussian:
            if seed is not None:
                np.random.seed(seed)
            ls = np.arange(3*nside)
            cls = self.get_delta_beta_cl(amp, gamma, ls, l0, l_cutoff)
            mp = hp.synfast(cls, nside, verbose=False)
            mp += beta0
            return mp
        else:
            beta_sync = hp.ud_grade(hp.read_map('./data/beta_sync_NK_equatorial.fits', verbose=False), nside_out=nside)
            beta_dust = hp.ud_grade(hp.read_map('./data/beta_dust_pysm_equatorial.fits', verbose=False), nside_out=nside)
            return beta_sync, beta_dust
        return None

    def fcmb(self, nu):
        """ CMB SED (in antenna temperature units).
        """
        x=0.017608676067552197*nu
        ex=np.exp(x)
        return ex*(x/(ex-1))**2

    def comp_sed(self, nu,nu0,beta,temp,typ):
        """ Component SEDs (in antenna temperature units).
        """
        if typ=='cmb':
            return self.fcmb(nu)
        elif typ=='dust':
            x_to=0.04799244662211351*nu/temp
            x_from=0.04799244662211351*nu0/temp
            return (nu/nu0)**(1+beta)*(np.exp(x_from)-1)/(np.exp(x_to)-1)
        elif typ=='sync':
            return (nu/nu0)**beta
        return None

    def get_mean_spectra(self, lmax): #, mean_pars=None):
        """ Computes amplitude power spectra for all components
        """
        #if mean_pars is None:
        #    mean_pars=self.mean_pars
        mean_pars=self.mean_pars
        
        ells = np.arange(lmax+1)
        dl2cl = np.ones(len(ells))
        dl2cl[1:] = 2*np.pi/(ells[1:]*(ells[1:]+1.))
        cl2dl = (ells*(ells+1.))/(2*np.pi)

        # Translate amplitudes to reference frequencies
        A_dust_BB = mean_pars['A_dust_BB'] * (self.comp_sed(mean_pars['nu0_dust'],
                                                            mean_pars['nu0_dust_def'],
                                                            mean_pars['beta_dust'],
                                                            mean_pars['temp_dust'],
                                                            'dust'))**2
        A_sync_BB = mean_pars['A_sync_BB'] * (self.comp_sed(mean_pars['nu0_sync'],
                                                            mean_pars['nu0_sync_def'],
                                                            mean_pars['beta_sync'],
                                                            None, 'sync'))**2
          
        # Dust amplitudes
        A_dust_BB = A_dust_BB * self.fcmb(mean_pars['nu0_dust_def'])**2
        dl_dust_bb = A_dust_BB * ((ells+1E-5) / 80.)**mean_pars['alpha_dust_BB']
        dl_dust_ee = mean_pars['EB_dust'] * A_dust_BB * \
                     ((ells+1E-5) / 80.)**mean_pars['alpha_dust_EE']
        dl_dust_tt = mean_pars['TB_dust'] * A_dust_BB * \
                     ((ells+1E-5) / 80.)**mean_pars['alpha_dust_BB']
        dl_dust_te = mean_pars['EB_dust'] * A_dust_BB * \
                     ((ells+1E-5) / 80.)**mean_pars['alpha_dust_BB']
        cl_dust_bb = dl_dust_bb * dl2cl
        cl_dust_ee = dl_dust_ee * dl2cl
        cl_dust_tt = dl_dust_ee * dl2cl
        cl_dust_te = dl_dust_ee * dl2cl
        if not mean_pars['include_E']:
            cl_dust_ee *= 0 
        if not mean_pars['include_B']:
            cl_dust_bb *= 0
        if not mean_pars['include_dust']:
            cl_dust_bb *= 0
            cl_dust_ee *= 0

        # Sync amplitudes
        A_sync_BB = A_sync_BB * self.fcmb(mean_pars['nu0_sync_def'])**2
        dl_sync_bb = A_sync_BB * ((ells+1E-5) / 80.)**mean_pars['alpha_sync_BB']
        dl_sync_ee = mean_pars['EB_sync'] * A_sync_BB * \
                     ((ells+1E-5) / 80.)**mean_pars['alpha_sync_EE']
        dl_sync_tt = mean_pars['TB_sync'] * A_sync_BB * \
                     ((ells+1E-5) / 80.)**mean_pars['alpha_sync_BB']
        dl_sync_te = mean_pars['EB_sync'] * A_sync_BB * \
                     ((ells+1E-5) / 80.)**mean_pars['alpha_sync_BB']
        cl_sync_bb = dl_sync_bb * dl2cl
        cl_sync_ee = dl_sync_ee * dl2cl
        cl_sync_tt = dl_sync_tt * dl2cl
        cl_sync_te = dl_sync_te * dl2cl
        if not mean_pars['include_E']:
            cl_sync_ee *= 0 
        if not mean_pars['include_B']:
            cl_sync_bb *= 0
        if not mean_pars['include_sync']:
            cl_sync_bb *= 0
            cl_sync_ee *= 0

        # CMB amplitude
        # Lensing
        l,dtt,dee,dbb,dte=np.loadtxt("/mnt/zfsusers/susanna/Map2TOD/mapsim/data/camb_lens_nobb.dat",unpack=True)
        l = l.astype(int)
        msk = l <= lmax
        l = l[msk]
        dltt=np.zeros(len(ells)); dltt[l]=dtt[msk]
        dlee=np.zeros(len(ells)); dlee[l]=dee[msk]
        dlbb=np.zeros(len(ells)); dlbb[l]=dbb[msk]
        dlte=np.zeros(len(ells)); dlte[l]=dte[msk]  
        cl_cmb_bb_lens=dlbb * dl2cl
        cl_cmb_ee_lens=dlee * dl2cl
        if not mean_pars['include_E']:
            cl_cmb_ee_lens *= 0 
        if not mean_pars['include_B']:
            cl_cmb_bb_lens *= 0
        if not mean_pars['include_CMB']:
            cl_cmb_bb_lens *= 0
            cl_cmb_ee_lens *= 0

        # Lensing + r=1
        l,dtt,dee,dbb,dte=np.loadtxt("/mnt/zfsusers/susanna/Map2TOD/mapsim/data/camb_lens_r1.dat",unpack=True)
        l = l.astype(int)
        msk = l <= lmax
        l = l[msk]
        dltt=np.zeros(len(ells)); dltt[l]=dtt[msk]
        dlee=np.zeros(len(ells)); dlee[l]=dee[msk]
        dlbb=np.zeros(len(ells)); dlbb[l]=dbb[msk]
        dlte=np.zeros(len(ells)); dlte[l]=dte[msk]  
        cl_cmb_tt=dltt * dl2cl
        cl_cmb_bb_r1=dlbb * dl2cl
        cl_cmb_ee_r1=dlee * dl2cl
        cl_cmb_te=dlte * dl2cl
        if not mean_pars['include_E']:
            cl_cmb_ee_r1 *= 0 
        if not mean_pars['include_B']:
            cl_cmb_bb_r1 *= 0
        if not mean_pars['include_CMB']:
            cl_cmb_bb_r1 *= 0
            cl_cmb_ee_r1 *= 0

        cl_cmb_ee = mean_pars['A_lens'] * cl_cmb_ee_lens + mean_pars['r_tensor'] * (cl_cmb_ee_r1-cl_cmb_ee_lens)
        cl_cmb_bb = mean_pars['A_lens'] * cl_cmb_bb_lens + mean_pars['r_tensor'] * (cl_cmb_bb_r1-cl_cmb_bb_lens)
        return(#A_dust_BB/fcmb(mean_pars['nu0_dust_def'])**2,
            #A_sync_BB/fcmb(mean_pars['nu0_sync_def'])**2,
            #fcmb(mean_pars['nu0_dust_def'])**2,
            ells, dl2cl, cl2dl,
            cl_dust_tt, cl_dust_ee, cl_dust_bb, cl_dust_te,
            cl_sync_tt, cl_sync_ee, cl_sync_bb, cl_sync_te,
            cl_cmb_tt, cl_cmb_ee, cl_cmb_bb, cl_cmb_te)


    def get_freqs(self):
        """ Return LB frequencies.
        """
        #return np.array([27., 39., 93., 145., 225., 280.])
        return np.array([40., 50., 60., 68., 78., 89., 100., 119., 140., 166., 195., 235., 280., 337., 402.])


    def get_sky_realization(self, nside, plaw_amps=True, gaussian_betas=True, seed=None,
                            compute_cls=False, delta_ell=10):
        """ CHANGEDDDDDDD
        Generate a sky realization for a set of input sky parameters.
        Args:
        - nside: HEALPix resolution parameter.
        - seed: seed to be used (if `None`, then a random seed will
            be used).
        - mean_pars: mean parameters (see `get_default_params`).
            If `None`, then a default set will be used.
        - moment_pars: mean parameters (see `get_default_params`).
            If `None`, then a default set will be used.
        - compute_cls: return also the power spectra? Default: False.
        - delta_ell: bandpower size to use if compute_cls is True.
        - gaussian_betas: gaussian spectral index maps, see 'get_beta_map'.
            Default: True.
        - plaw_amps: dust and synchrotron amplitude maps modelled as power
            laws. If false, returns realistic amplitude maps in equatorial
            coordinates. Default: True. 
        Returns:
        - A dictionary containing the different component maps,
          spectral index maps and frequency maps.
        - If `compute_cls=True`, then the dictionary will also
          contain information of the signal, noise and total 
          (i.e. signal + noise) power spectra. 
        """
        nu = self.get_freqs()
        npix = hp.nside2npix(nside)
        if seed is not None:
            np.random.seed(seed)
        #if mean_pars is None:
        mean_pars= self.mean_pars #get_default_params()
        #if moment_pars is None:
        moment_pars = self.moment_pars #get_default_params()
        lmax = 3*nside-1
        ells, dl2cl, cl2dl, cl_dust_tt, cl_dust_ee, cl_dust_bb, cl_dust_te, cl_sync_tt, cl_sync_ee, cl_sync_bb, cl_sync_te, cl_cmb_tt, cl_cmb_ee, cl_cmb_bb, cl_cmb_te = self.get_mean_spectra(lmax)
        cl0 = 0 * cl_dust_bb
        
        # Sync amplitudes  TT, EE, BB, TE, EB, TB
        I_sync, Q_sync, U_sync = hp.synfast([cl_sync_tt, cl_sync_ee, cl_sync_bb, cl_sync_te,
                                             cl0, cl0],nside, new=True, verbose=False)#[1:]

        # Dust amplitudes  TT, EE, BB, TE, EB, TB
        I_dust, Q_dust, U_dust = hp.synfast([cl_dust_tt, cl_dust_ee, cl_dust_bb, cl_dust_te,
                                             cl0, cl0], nside, new=True, verbose=False)#[1:]

        # CMB amplitudes  TT, EE, BB, TE, EB, TB
        I_cmb, Q_cmb, U_cmb = hp.synfast([cl_cmb_tt, cl_cmb_ee, cl_cmb_bb, cl_cmb_te, cl0, cl0],
                                         nside, new=True, verbose=False)#[1:]
        
        if not mean_pars['include_dust']:
            I_dust *= 0
            Q_dust *= 0
            U_dust *= 0
        if not mean_pars['include_sync']:
            I_sync *= 0
            Q_sync *= 0
            U_sync *= 0
        if not mean_pars['include_CMB']:
            I_cmb *= 0
            Q_cmb *= 0
            U_cmb *= 0
    
        # Dust and Synchrotron spectral indices
        beta_dust = self.get_beta_map(nside,
                                      mean_pars['beta_dust'],
                                      moment_pars['amp_beta_dust'],
                                      moment_pars['gamma_beta_dust'],
                                      moment_pars['l0_beta_dust'],
                                      moment_pars['l_cutoff_beta_dust'],
                                      gaussian=True)
        beta_sync = self.get_beta_map(nside, 
                                      mean_pars['beta_sync'],
                                      moment_pars['amp_beta_sync'],
                                      moment_pars['gamma_beta_sync'],
                                      moment_pars['l0_beta_sync'],
                                      moment_pars['l_cutoff_beta_sync'],
                                      gaussian=True)

        # Dust temperature
        temp_dust = np.ones(npix) * mean_pars['temp_dust']

        # Create PySM simulation
        zeromap = np.zeros(npix)
        # Dust
        d2 = models("d1", nside)
        d2[0]['spectral_index'] = beta_dust
        d2[0]['temp'] = temp_dust
        d2[0]['nu_0_I'] = mean_pars['nu0_dust']
        d2[0]['nu_0_P'] = mean_pars['nu0_dust']
        d2[0]['A_I'] = I_dust #zeromap
        d2[0]['A_Q'] = Q_dust
        d2[0]['A_U'] = U_dust
        # Sync
        s1 = models("s1", nside)
        s1[0]['nu_0_I'] = mean_pars['nu0_sync']
        s1[0]['nu_0_P'] = mean_pars['nu0_sync']
        s1[0]['A_I'] = I_sync #zeromap
        s1[0]['A_Q'] = Q_sync
        s1[0]['A_U'] = U_sync
        s1[0]['spectral_index'] = beta_sync
        # CMB
        c1 = models("c1", nside)
        c1[0]['model'] = 'pre_computed' #different output maps at different seeds 
        c1[0]['A_I'] = I_cmb #zeromap
        c1[0]['A_Q'] = Q_cmb
        c1[0]['A_U'] = U_cmb

        sky_config = {'dust' : d2, 'synchrotron' : s1, 'cmb' : c1}
        sky = pysm.Sky(sky_config)
        instrument_config = {
            'frequencies' : np.array([40., 50., 60., 68., 78., 89., 100., 119., 140., 166., 195., 235., 280., 337., 402.]),
            'beams' : np.array([69.3, 56.8, 49.0, 44.5, 40.0, 36.7, 37.8, 33.6, 30.8, 28.9, 28.6, 24.7, 22.5, 20.9, 17.9]),
            'sens_P' : np.array([59.29, 32.78, 25.76, 15.91, 13.10, 11.25, 7.74, 5.37, 5.65, 5.81, 6.48, 15.16, 17.98, 24.99, 49.9]),
            'sens_I' : np.array([59.29, 32.78, 25.76, 15.91, 13.10, 11.25, 7.74, 5.37, 5.65, 5.81, 6.48, 15.16, 17.98, 24.99, 49.9])/np.sqrt(2),
            'nside' : nside,
            'noise_seed' : 12521,
            'use_bandpass' : False,
            'add_noise' : False,
            'output_units' : 'uK_RJ',
            'use_smoothing' : False,
            'output_directory' : 'none',
            'output_prefix' : 'liteBIRD_v28'
        }

        sky = pysm.Sky(sky_config)
        instrument = pysm.Instrument(instrument_config)
        maps_signal, _ = instrument.observe(sky, write_outputs=False)
        if not mean_pars['include_T']:
            maps_signal = maps_signal[:,1:,:]
        # Change to CMB units
        maps_signal = maps_signal/self.fcmb(nu)[:,None,None]

        dict_out = {'maps_dust': np.array([I_dust, Q_dust, U_dust]),
                    'maps_sync': np.array([I_sync, Q_sync, U_sync]),
                    'maps_cmb': np.array([I_cmb, Q_cmb, U_cmb]),
                    'beta_dust': beta_dust,
                    'beta_sync': beta_sync,
                    'freq_maps': maps_signal}

        return dict_out

