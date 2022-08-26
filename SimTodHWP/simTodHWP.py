import os
import sys
import numpy as np
import healpy as hp
import scipy as sp
from tqdm import tqdm
import argparse 
from itertools import product

def sens_LB_V28(freq, diameter, NET='NETdet'):
    
    freq_list = np.array([ 40.,  50.,  60.,  68.,  78.,  89., 100., 119., 140., 166., 195., 235., 280., 337., 402.])
    if NET == 'NETarr':
        net_ukrs_comb = np.array([ 18.50, 16.54, 10.54, 8.34, 5.97, 5.58, 3.24, 2.26, 2.37, 2.75, 2.89, 5.34, 6.82, 10.85, 23.45 ])
        net_ukrs = np.array([ [18.50, 18.50], [16.54, 16.54], [10.54, 10.54], [9.84, 15.70], [7.69, 9.46], [6.07, 14.22], [5.11, 4.19], [3.8,2.82], [3.58,3.16], [2.75, 2.75], [3.48,5.19], [5.34, 5.34], [6.82, 6.82], [10.85, 10.85], [23.45, 23.45] ])
    elif NET == 'NETdet':
        net_ukrs = np.array([ [114.63, 114.63], [72.48, 72.48], [65.28, 65.28], [105.64, 68.81], [82.51, 58.61], [65.18, 62.33], [54.88,71.70], [40.78,55.65], [38.44, 54.00], [54.37, 54.37], [59.61, 73.96], [76.06, 76.06], [97.26, 97.26], [154.64, 154.64], [385.69, 385.69] ])
        
    if freq not in freq_list:
        if freq == -1: return net_ukrs_comb
        if freq != -1: 
            print("ERROR: unexpected frequency!!")
            return 0
    else: freq_idx = np.where(freq_list==freq)[0][0]

    diameter_idx = 0
    if freq in [68, 78, 89] and diameter == 32: diameter_idx = 1
    if freq in [100, 119, 140] and diameter == 12: diameter_idx = 1
    if freq in [195] and diameter == 6.6: diameter_idx = 1

    return net_ukrs[freq_idx][diameter_idx]

def pointingshift(x,y,thetap,phip,psip):
    theta_ = np.arcsin(np.sqrt(x*x + y*y))
    phi_   = np.arctan2(y,x)
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
        R[0,0] = -costhetap[ii]; 
        R[0,2] = sinthetap[ii]
        R[2,0] = sinthetap[ii]
        R[2,2] = costhetap[ii]
        dr = np.dot(R,drp)
        eph[0] = -sinphi[ii]
        eph[1] = cosphi[ii]
        aaa = np.dot(dr,np.transpose(eph))/np.sqrt(np.dot(dr,np.transpose(dr)))
        if (aaa > 1.0): aaa = 1.0
        if (aaa < -1.0): aaa = -1.0
        psitmp = np.arccos(aaa)
        if (dr[2] < 0): psitmp = -psitmp
        psitmp = (psitmp +np.pi/2.0 + 4.0*np.pi ) % (2.0*np.pi) - np.pi
        psi[ii] = psitmp
       
    return (theta,phi,psi)

def mueller(freq, inc_ang, mueller_database):
    
    file_mueller_db = np.load(mueller_database,allow_pickle=True)
    freq_ave = file_mueller_db['freq']
    param = file_mueller_db['param']
    fact = file_mueller_db['fact']
    
    f = np.argwhere(freq_ave == freq).flatten()[0]
    A0 = np.zeros((4,4))
    A2 = np.zeros((4,4)); phi2 = np.zeros((4,4))
    A4 = np.zeros((4,4)); phi4 = np.zeros((4,4))
    A6 = np.zeros((4,4)); phi6 = np.zeros((4,4))
    A8 = np.zeros((4,4)); phi8 = np.zeros((4,4))
    for i, j in product(range(4),range(4)):
        A0[i][j] = np.poly1d(fact[f][0][i][j])(inc_ang)
        A2[i][j] = np.poly1d(fact[f][1][i][j])(inc_ang) 
        A4[i][j] = np.poly1d(fact[f][2][i][j])(inc_ang)
        A6[i][j] = np.poly1d(fact[f][3][i][j])(inc_ang)
        A8[i][j] = np.poly1d(fact[f][4][i][j])(inc_ang)
        phi2[i][j] = np.poly1d(fact[f][5][i][j])(inc_ang)
        phi4[i][j] = np.poly1d(fact[f][6][i][j])(inc_ang)
        phi6[i][j] = np.poly1d(fact[f][7][i][j])(inc_ang)
        phi8[i][j] = np.poly1d(fact[f][8][i][j])(inc_ang)
    return A0, A2, A4, A6, A8, phi2, phi4, phi6, phi8

def main():

    parser = argparse.ArgumentParser(description='LiteBIRD TOD Simulation with AHWP Mueller matrix')
    parser.add_argument('-b', '--bolo', default=['000_001_000_QA_140_T'], nargs="*", type=str, help='a list of bolometer names') 
    parser.add_argument('-n', '--nmin', default=0, type=int, help='start index of data chunk')
    parser.add_argument('-s', '--nseg', default=1, type=int, help='number of segment per jobs')
    parser.add_argument('--boresight', action='store_true', help='bit to use boresight pointing or not')
    parser.add_argument('--fg', default=1, type=int,  help='bit to include foreground or not')
    parser.add_argument('--dipole', default=1, type=int, help='bit to include dipole or not')
    parser.add_argument('--monopole', default=1, type=int, help='bit to include monopole or not')
    parser.add_argument('--idealHWP', action='store_true', help='use ideal HWP Mueller matrix')
    parser.add_argument('--convertEG', action='store_true', help='the pointing is converted from galactic to ecliptic coords')
    parser.add_argument('--shwp', default=5., type=float, help='bit to include monopole or not')
    parser.add_argument('--shIP', action='store_true', help='the M_QI (4f) and M_UI (4f) phase are shifted by pi/2 (quadrature of phase IP terms)')
    parser.add_argument('--noPol', action='store_true', help='bit to not include polarization signal, only intensity')
    parser.add_argument('--only4f', action='store_true', help='bit to not include nf signal (n=2,6,8)')
    parser.add_argument('--nside', default=256, type=int, help='nside (less than 512)')
    parser.add_argument('--inoutSky', default='', type=int, help='suffix of input map files')
    parser.add_argument('-d', '--work_dir', default='/group/cmb/litebird/usr/ysakurai/HWP_IP/', help='path to working directory')
    args = parser.parse_args()
    
    ## Arguments ##
    # boloList=['000_001_000_QA_140_T', '000_001_000_QA_140_B', '000_006_035_QA_140_T', '000_006_035_QA_140_B']
    boresight = args.boresight
    boloList=args.bolo
    nmin = args.nmin
    nseg = args.nseg
    fg = args.fg
    dipole = args.dipole
    monopole = args.monopole
    idealHWP = args.idealHWP
    shwp = args.shwp
    convertEG = args.convertEG
    shIP = args.shIP
    noPol = args.noPol
    only4f = args.only4f
    nside = args.nside
    work_dir=args.work_dir

    print('TOD simulation')
    print('-----------------------')
    print('parameters')
    ## Parameters ##
    fsamp = 19.
    fknee = 0.05
    alpha = 1.
    f_HWP = 46./60.
    Omega_HWP = f_HWP * 2.0 * np.pi / fsamp
    
    terminS = 'v28_nside' + str(nside) # signal suffix
    if boresight: terminS += '_boresight'
    terminN = 'fknee_' + str(fknee)+'Hz_fsamp_'+str(int(fsamp))+'Hz_alpha_' + str(int(alpha)) # noise suffix
    terminP = 'LB_45_50_LB_v28_19Hz' # pointing file suffix
    
    ## Call Bolometer Info ##
    file_fp_db = work_dir + '/Database/bolometer_database_IMo.npy'
    fp_db = np.load(file_fp_db, allow_pickle=True)
    boloNames = np.array([fp_db[i]['boloName'] for i in range(len(fp_db))])
    boloIdx = np.array([np.where(boloNames==boloList[i])[0] for i in range(len(boloList))]).flatten()

    print('nside: ', nside)
    print('Bolometer List: ', boloNames[boloIdx])
    ## Read Input Maps ##
    mapI = np.zeros(12*nside*nside)
    mapQ = np.zeros(12*nside*nside)
    mapU = np.zeros(12*nside*nside)
    
    terminS += '_CMB'
    map_path = work_dir + '/Maps/PTEP_20200915_compsep/'
    [mapI512, mapQ512, mapU512] = hp.read_map(map_path + '/cmb/0000/LB_LFT_140_cmb_0000_PTEP_20200915_compsep.fits',field=(0,1,2))
    mapI += hp.ud_grade(mapI512,nside); mapQ += hp.ud_grade(mapQ512,nside); mapU += hp.ud_grade(mapU512,nside)

    if fg:
        terminS = terminS + '_fg'
        [mapI512, mapQ512, mapU512] = hp.read_map(map_path + '/foregrounds/dust/LB_LFT_140_dust_PTEP_20200915_compsep.fits',field=(0,1,2))
        mapI += hp.ud_grade(mapI512,nside); mapQ += hp.ud_grade(mapQ512,nside); mapU += hp.ud_grade(mapU512,nside)
        [mapI512, mapQ512, mapU512] = hp.read_map(map_path + '/foregrounds/synch/LB_LFT_140_synch_PTEP_20200915_compsep.fits',field=(0,1,2))
        mapI += hp.ud_grade(mapI512,nside); mapQ += hp.ud_grade(mapQ512,nside); mapU += hp.ud_grade(mapU512,nside)
        [mapI512, mapQ512, mapU512] = hp.read_map(map_path + '/foregrounds/ame/LB_LFT_140_ame_PTEP_20200915_compsep.fits',field=(0,1,2))
        mapI += hp.ud_grade(mapI512,nside); mapQ += hp.ud_grade(mapQ512,nside); mapU += hp.ud_grade(mapU512,nside)
        [mapI512, mapQ512, mapU512] = hp.read_map(map_path + '/foregrounds/freefree/LB_LFT_140_freefree_PTEP_20200915_compsep.fits',field=(0,1,2))
        mapI += hp.ud_grade(mapI512,nside); mapQ += hp.ud_grade(mapQ512,nside); mapU += hp.ud_grade(mapU512,nside)
   
    if dipole:
        terminS += '_dipole'
        radip = 264.021/180.0*np.pi
        decdip = 48.253/180.0*np.pi
        if not convertEG:
            rotd=hp.Rotator(coord=['G','E'])
            (thetaEd,phiEd)=rotd(np.pi/2.0-decdip,radip)
            radip = phiEd.copy()
            decdip = np.pi/2.0-thetaEd
            # print('Dipole coord:',radip*180.0/np.pi,decdip*180.0/np.pi)
        ampldip = 3362.08
        indpix = np.arange(12*nside*nside)
        [thetapix, phipix] = hp.pix2ang(nside,indpix)
        x=np.sin(thetapix)*np.cos(phipix)
        y=np.sin(thetapix)*np.sin(phipix)
        z=np.cos(thetapix)
        xdip=np.sin(np.pi/2.0-decdip)*np.cos(radip)
        ydip=np.sin(np.pi/2.0-decdip)*np.sin(radip)
        zdip=np.cos(np.pi/2.0-decdip)
        mapdip = ampldip *(x*xdip + y*ydip + z*zdip) 
        mapI += mapdip

    if monopole == 1:
        terminS = terminS + '_monopole'
        Tcmb = 2.7255*1e6 #[uK]
        npix = hp.nside2npix(nside)
        monopole = np.array([Tcmb for i in range(npix)])
        mapI += monopole

    if not convertEG:
        rot=hp.Rotator(coord=['E','G'])
        (thetaE,phiE) = hp.pix2ang(512,np.arange(12*512*512))
        (thetaG,phiG)=rot(thetaE,phiE)
        ipixGE = hp.ang2pix(512,thetaG,phiG)
        (thetaGsh,phiGsh)=rot(thetaE-1e-8,phiE)
        dtheta = thetaGsh - thetaG
        dphi = phiGsh - phiG
        psi = np.arctan2(-dphi*np.sin(thetaG),dtheta)
        mapIE = hp.pixelfunc.get_interp_val(mapI, thetaG, phiG)
        mapQE = np.cos(2*psi)*hp.pixelfunc.get_interp_val(mapQ, thetaG, phiG) - np.sin(2*psi)*hp.pixelfunc.get_interp_val(mapU, thetaG, phiG)
        mapUE = np.sin(2*psi)*hp.pixelfunc.get_interp_val(mapQ, thetaG, phiG) + np.cos(2*psi)*hp.pixelfunc.get_interp_val(mapU, thetaG, phiG)
        mapI = hp.ud_grade(mapIE,nside)
        mapQ = hp.ud_grade(mapQE,nside)
        mapU = hp.ud_grade(mapUE,nside) 
    
    if idealHWP: terminS = terminS + '_idealHWP'
    if noPol:
        terminS += '_noPol'
        mapQ = np.zeros(12*nside*nside)
        mapU = np.zeros(12*nside*nside)   
    if only4f:   terminS += '_only4f'
    if shIP:     terminS = terminS + '_shIP'
    if shwp:     terminS = terminS + '_shwp' + str(int(shwp))

    if nmin == 0:
        mapI.tofile(work_dir + '/Maps/InputMap_I_' + terminS + '.bi')
        mapQ.tofile(work_dir + '/Maps/InputMap_Q_' + terminS + '.bi')
        mapU.tofile(work_dir + '/Maps/InputMap_U_' + terminS + '.bi')
        print('Input I Map        : ', work_dir + '/Maps/InputMap_I_' + terminS + '.bi')
        print('Input Q Map        : ', work_dir + '/Maps/InputMap_Q_' + terminS + '.bi')
        print('Input U Map        : ', work_dir + '/Maps/InputMap_U_' + terminS + '.bi')

    ## Scan Parameters ##
    nt = nmin + nseg
    Totalyr = 1.0 # year observation
    sss = 2**22 # ?? Probably number of pointing per segment? Arbitary?
    nmax = int(3600*24*366.0*Totalyr*fsamp/sss) 
    if nt > nmax: nt = nmax
    samp = np.arange(nmin*sss,nt*sss,sss)
    indsamp = nmin*sss
    print('nmin: ', nmin, ' nmax: ', nmax, ' nseg: ', nseg, )
    
    ##### Data chunk sample list  #######
    dd1 = np.arange(nmax)*sss
    dd2 = dd1 + sss - 1
    dd = np.zeros((nmax*2))
    dd[0:2*nmax:2] = dd1
    dd[1:2*nmax+11:2] = dd2
    ddd = dd.astype(np.int64)
    if (nmin == 0):
        print('Output sample list : ', work_dir+'/Database/samplistLB_%i.bi'%nmax)
        ddd.tofile(work_dir+'/Database/samplistLB_%i.bi'%nmax)
    
    ## Read Pointing ##
    filenameRa = work_dir+'/Pointing/ra_'+terminP+'.bi'
    filenameDec = work_dir+'/Pointing/dec_'+terminP+'.bi'
    filenamePsi = work_dir+'/Pointing/psi_'+terminP+'.bi'
    fileRa = open(filenameRa, mode='rb')
    fileDec = open(filenameDec, mode='rb')
    filePsi = open(filenamePsi, mode='rb')
    
    
    print('Loop over bolos')
    print('-----------------------')
    for ndet, idet in enumerate(boloIdx):
        
        print('boloName: ', boloNames[idet])
        
        theta_fp = np.deg2rad(fp_db[idet]['theta'])
        phi_fp = np.deg2rad(fp_db[idet]['phi'])
        psib = np.deg2rad(fp_db[idet]['polang'])
        diameter = fp_db[idet]['diameter'] 
        freq = fp_db[idet]['freq']
        NET = sens_LB_V28(freq, diameter)
        x = fp_db[idet]['x'] 
        if boresight:
            theta_fp = 0
            phi_fp = 0

        ##### additive noiseNET
        white_noise_level = NET * np.sqrt(fsamp)
        one_over_f_freq = np.fft.fftfreq(sss)*fsamp
        one_over_f_fft = np.zeros(sss)
        one_over_f_fft[1:] = (fknee/abs(one_over_f_freq[1:]))**alpha + 1
        one_over_f_fft[0] = one_over_f_fft[1]
        inv_one_over_f = 1.0/abs(one_over_f_fft)
        inv_one_over_f.tofile(work_dir + '/Noise/iSpf_'+terminN+'_'+boloNames[idet]+'.bi')
        
        xx = round(np.sin(theta_fp)*np.cos(phi_fp),10)
        yy = round(np.sin(theta_fp)*np.sin(phi_fp),10)

        ## HWP tilt treatment ##
        x_bs = np.sin(theta_fp)*np.cos(phi_fp)
        y_bs = np.sin(theta_fp)*np.sin(phi_fp) + shwp/180.0*np.pi
        theta_bs = np.arcsin(np.sqrt(x_bs*x_bs + y_bs*y_bs))
        phi_bs = np.arctan2(y_bs,x_bs) - np.pi/2.0  

        ## HWP Mueller matrix ##    
        if idealHWP:
            A0, A2, A4, A6, A8 = np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)) 
            phi2, phi4, phi6, phi8 = np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4))
            A0[0][0]=1.0; A0[3][3]=-1.0
            A4[1][1]=1.0; A4[2][2]=1.0
            A4[1][2]=1.0; A4[2][1]=1.0
            phi4[1][2]=-np.pi/2.; phi4[2][1]=-np.pi/2.
            phi4[2][2]= np.pi
        else: 
            # A0, A2, A4, A6, A8, phi2, phi4, phi6, phi8 = mueller(freq,theta_bs,work_dir + '/Database/HWP_MuellerMatrix_database_carbide_19_19.npz')
            A0, A2, A4, A6, A8 = np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)) 
            phi2, phi4, phi6, phi8 = np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4)), np.zeros((4,4))
            A0[0][0]=1.0; A4[1][1]=1.0
            A4[2][2]=1.0; A0[3][3]=-1.0
            phi4[1][1]=0;         phi4[1][2]=-np.pi/2.; 
            phi4[2][1]=-np.pi/2.; phi4[2][2]= np.pi; 
            # I to 4f 
            A4[0][1]=1.0e-3; A4[0][2]=1.0e-3;
            A4[1][0]=1.0e-3; A4[2][0]=1.0e-3;
            phi4[0][1]=np.deg2rad(60.);  phi4[0][2]=np.deg2rad(60.);
            phi4[1][0]=np.deg2rad(60.);  phi4[2][0]=np.deg2rad(60.);
            # I to 2f
            # A2[0][1]=2.0e-3; A2[0][2]=1.0e-3
            # A2[1][0]=2.0e-3; A2[2][0]=1.0e-3
            # phi2[0][1]=np.deg2rad(140.);   phi2[0][2]=np.deg2rad(50.)
            # phi2[1][0]=np.deg2rad(-140.);  phi2[2][0]=np.deg2rad(140.)
        if shIP:
            phi4[1][0] += -np.pi/2.0
            phi4[2][0] += -np.pi/2.0
        
        A0d = A0
        A2d = A2*np.exp(1j*phi2)
        A4d = A4*np.exp(1j*phi4)
        A6d = A6*np.exp(1j*phi6)
        A8d = A8*np.exp(1j*phi8)
        
        ## FOR TEST
        ###############################
        n = sss
        samp_tmp = samp
        ###############################
        
        fdat = open(work_dir + '/data/data_' + boloNames[idet] + '_' + terminS + '_nmin%i_max%i_per%i.bi'%(nmin,nmax,nseg),'w')
        fdatn = open(work_dir + '/data/data_' + boloNames[idet] + '_' + terminS + '_' + terminN + '_nmin%i_max%i_per%i.bi'%(nmin,nmax,nseg),'w')
        if ndet == 0: ftime = open(work_dir + '/data/time_' + terminS + '_nmin%i_max%i_per%i.bi'%(nmin,nmax,nseg),'w')
        print('Output TOD         : ', work_dir + '/data/data_' + boloNames[idet] + '_' + terminS + '_nmin%i_max%i_per%i.bi'%(nmin,nmax,nseg))
        print('Output TOD+noise   : ', work_dir + '/data/data_' + boloNames[idet] + '_' + terminS + '_' + terminN + '_nmin%i_max%i_per%i.bi'%(nmin,nmax,nseg))
        print('Output Time        : ', work_dir + '/data/time_' + terminS + '_nmin%i_max%i_per%i.bi'%(nmin,nmax,nseg))

        for nsamp,isamp in enumerate(samp_tmp):
        
            print('nsamp: ', nsamp)
            
            fileRa.seek(isamp*8)
            fileDec.seek(isamp*8)
            filePsi.seek(isamp*8)
            ra_p = np.fromfile(fileRa,dtype=np.float64,count=sss)
            dec_p = np.fromfile(fileDec,dtype=np.float64,count=sss)
            psi_p = np.fromfile(filePsi,dtype=np.float64,count=sss)
            theta_p = np.pi/2.0 - dec_p
            phi_p = ra_p
            print('')
            print('Calculating Pointing shift ...')
            if boresight:
                (theta,phi,psi) = (theta_p[:n],ra_p[:n],psi_p[:n])
            else:
                (theta,phi,psi) = pointingshift(xx,yy,theta_p[:n],ra_p[:n],psi_p[:n])
            print(theta,phi,psi)

            if convertEG:
                rot=hp.Rotator(coord=['E','G'])
                (thetaG,phiG)=rot(theta,phi)
                (thetaGsh,phiGsh)=rot(theta-1e-8,phi)
                dtheta = thetaGsh - thetaG
                dphi = phiGsh - phiG
                dpsi = -np.arctan2(-dphi*np.sin(thetaG),dtheta)
                psi += dpsi
                ra = phiG.copy()
                dec = np.pi/2.0-thetaG
            else:
                ra = phi.copy()
                dec = np.pi/2.0-theta
                dpsi = 0.0
 
   
            Omegathwp = Omega_HWP * (np.arange(np.size(ra)) + indsamp) + dpsi
            rho = Omegathwp - (psi + phi_bs)
            Ksi = psi + phi_bs # = boresight rotation + HWP tilted angle
            Psi = psi + psib # = boresight rotation + detector anttena angle 
            
            ipix = hp.ang2pix(nside,theta,phi)

            white_noise = np.random.randn(np.size(ipix[:n]))*white_noise_level
            Noise_fft = np.fft.fft(white_noise) * np.sqrt(abs(one_over_f_fft[:n]))
            Noise = np.fft.ifft(Noise_fft)
            
            tod = np.zeros(np.size(ipix))
            tod_noise = np.zeros(np.size(ipix))

            for i in tqdm(range(len(ipix[:n]))):
                PP = np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]])
                RRP  = np.array([[1,0,0,0],[0,np.cos(2*Psi[i]),-np.sin(2*Psi[i]),0],[0,np.sin(2*Psi[i]),np.cos(2*Psi[i]),0],[0,0,0,0]])
                RRP_ = np.array([[1,0,0,0],[0,np.cos(2*Psi[i]),np.sin(2*Psi[i]),0],[0,-np.sin(2*Psi[i]),np.cos(2*Psi[i]),0],[0,0,0,0]])
                RRM  = np.array([[1,0,0,0],[0,np.cos(2*Ksi[i]),-np.sin(2*Ksi[i]),0],[0,np.sin(2*Ksi[i]),np.cos(2*Ksi[i]),0],[0,0,0,0]])
                RRM_ = np.array([[1,0,0,0],[0,np.cos(2*Ksi[i]),np.sin(2*Ksi[i]),0],[0,-np.sin(2*Ksi[i]),np.cos(2*Ksi[i]),0],[0,0,0,0]])
                MMhwp = np.real(A0d + A2d*np.exp(1j*2.0*rho[i]) + A4d*np.exp(1j*4.0*rho[i]) + A6d*np.exp(1j*6.0*rho[i]) + A8d*np.exp(1j*8.0*rho[i]))
                if only4f: 
                    MMhwp = np.real(A0d + A4d*np.exp(1j*4.0*rho[i]))
                MMt = np.dot(RRM,np.dot(MMhwp,RRM_))    
                PPt = np.dot(RRP,np.dot(PP,RRP_))
                MM = np.dot(PPt,MMt)
                
                tod[i] = mapI[ipix[i]]*MM[0,0] + mapQ[ipix[i]]*MM[0,1] + mapU[ipix[i]]*MM[0,2]
                tod_noise[i] = mapI[ipix[i]]*MM[0,0] + mapQ[ipix[i]]*MM[0,1] + mapU[ipix[i]]*MM[0,2] + Noise[i].real
                
            indsamp += np.size(ipix)
            np.array(tod).tofile(fdat)
            np.array(tod_noise).tofile(fdatn)
            if ndet == 0: 
                time = np.arange(isamp,isamp+n,1)/fsamp
                time.tofile(ftime)

            print('Segment %i written'%(isamp))
        fdat.close()
        fdatn.close()
        ftime.close()
        
if __name__ == "__main__":
    main()
