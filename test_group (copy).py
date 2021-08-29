from NGC_SGC import *
import numpy as np
import healpy as hp
from scipy import stats
import fitsio as fio 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import time
import os
from NPTFit import create_mask as cm
import pandas as pd
import treecorr as tc
from pixell import enmap, enplot, reproject, utils, curvedsky 

# t0 = time.time()
fontsize=20
alpha=0.7
nside=512
#z_arr = np.array([0.1, 0.33, 0.67, 1.0])
z_arr = np.array([0.1,0.3,0.5,0.7])
#mass_cut = np.array([12.7,13.0,13.6])
mass_cut = np.array([13.1, 13.4, 13.9])
color = np.array(['r','orange','b'])
Q=12
N_lens_sim=100

tSZ=True
threshold=0.5

class LoadData(object):

    def __init__(self, zmin, zmax, mass):
        self.zmin = zmin
        self.zmax = zmax
        self.mass = mass


    def load_SELECTION_less(self):
        t0 = time.time()
        print ('======================== %s<z<%s, mass<%s ========================'%(self.zmin,self.zmax,self.mass))
        filename = '../desi_data/SELECTION/groups_%sto%s_less_M%s.fits'%(self.zmin,self.zmax,self.mass)
        f = fio.FITS(filename)[-1].read()
        ra = f['ra']
        dec = f['dec']
        M = f['M']
        photo_z = f['photo_z']
        Nmember = f['Nmember']
        L = f['L']
        id = f['id']
        print ('===> time to load SELECTION = ', time.time()-t0)
        return ra,dec,photo_z,M,Nmember,L,id


    def load_SELECTION_moreeq(self):
        t0 = time.time()
        print ('======================== %s<z<%s, mass>=%s ========================'%(self.zmin,self.zmax,self.mass))
        filename = '../desi_data/SELECTION/groups_%sto%s_moreeq_M%s.fits'%(self.zmin,self.zmax,self.mass)
        f = fio.FITS(filename)[-1].read()
        ra = f['ra']
        dec = f['dec']
        M = f['M']
        photo_z = f['photo_z']
        Nmember = f['Nmember']
        L = f['L']
        id = f['id']
        print ('===> time to load SELECTION = ', time.time()-t0)
        return ra,dec,photo_z,M,Nmember,L,id


class Mask(object):

    def __init__(self, threshold):
        self.threshold = threshold

    def fi_mask(self):
        from NPTFit import create_mask as cm
        nothrh = np.load('../desi_data/survey_mask/ud512_mask_nothrh.npy')
        mask_thrh = np.load('../desi_data/survey_mask/ud512_mask_thrh%s.npy'%self.threshold)
        fi = nothrh * mask_thrh
        return fi, mask_thrh


    def total_mask(self):
        K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB(tSZ=tSZ)
        g_fi, g_mask = fi_mask(self.threshold)
        mask = g_mask * K_mask
        return g_fi, g_mask, mask


class AfterMask(object):

    def __init__(self, id,Nmember,ra,dec,photo_z,M,L,threshold):
        self.id = id
        self.Nmember = Nmember
        self.ra = ra
        self.dec = dec
        self.photo_z = photo_z
        self.M = M
        self.L = L
        self.threshold = threshold

    def after_masking(self):
        t0=time.time()
        g_fi, g_mask, mask = total_mask(self.threshold)
        mask_pxl = np.array(np.where(mask==0))[0] #masked pixel index    
        ra,dec = fk5_2_galactic(ra,dec)
        pxl = hp.ang2pix(nside, np.pi/2.-np.deg2rad(dec), np.deg2rad(ra))
        arr1 = np.array([pxl,self.id,self.Nmember,self.ra,self.dec,self.photo_z,self.M,self.L])
        print ('****************** START MASKING *****************')
        slt = np.in1d(pxl,mask_pxl)
        index = np.where(slt==True)
        arr2 = np.delete(arr1,index,axis=1)
        id,Nmember,ra,dec,photo_z,M,L = arr2[1],arr2[2],arr2[3],arr2[4],arr2[5],arr2[6],arr2[7]
        print ('---> the time after masking=',time.time()-t0)
        print ('****************** END MASKING *******************')
        return arr2  # use this when run after masking
    #    return id,Nmember,ra,dec,photo_z,M,L


    def after_MASK(self):
        t0=time.time()
        g_fi, g_mask, mask = total_mask(threshold)
        mask_pxl = np.array(np.where(mask==0))[0] #masked pixel index    
        ra,dec = fk5_2_galactic(ra,dec)
        print ('---> number of groups before mask=',len(ra))
        pxl = hp.ang2pix(nside, np.pi/2.-np.deg2rad(dec), np.deg2rad(ra))
        print ('---> len(pxl) before mask=',len(pxl))
        arr1 = np.array([pxl,self.id,self.Nmember,self.ra,self.dec,self.photo_z,self.M,self.L])
        print ('****************** START MASKING *****************')
        slt = np.in1d(pxl,mask_pxl)
        index = np.where(slt==True)
        arr2 = np.delete(arr1,index,axis=1)
        id,Nmember,ra,dec,photo_z,M,L = arr2[1],arr2[2],arr2[3],arr2[4],arr2[5],arr2[6],arr2[7]
        print ('---> arr2.shape = ',arr2.shape)
        print ('===> number of groups after masking = ',len(ra))
        print ('---> the time after masking=',time.time()-t0)
        print ('****************** END MASKING *******************')
        return id,Nmember,ra,dec,photo_z,M,L



class LoadAfterMask(object):
    def __init__(self,loc,lab,z):
        self.loc = loc
        self.lab = lab
        self.z = z

    def load_AFTER_MASK(self):
        print ('==================%s, %s =================='%(self.lab,self.z))
        t0 = time.time()
        filename = '../desi_data/SELECTION/%s/arr_%s_z%s.fits'%(self.loc,self.lab,self.z)
        f = fio.FITS(filename)[-1].read()
        ra = f['ra']
        dec = f['dec']
        M = f['M']
        photo_z = f['photo_z']
        Nmember = f['Nmember']
        L = f['L']
        id = f['id']
        print ('===> len(ra) = ',len(ra))
        print ('===> time to load AFTER MASK = ', time.time()-t0)
        return id,Nmember,ra,dec,photo_z,M,L


    def load_AFTER_MASK_z(self):
        print ('==================z%s==================='%self.z) 
        t0 = time.time()
        filename='../desi_data/SELECTION/%s/arr_z%s.fits'%(self.loc, self.z)
        f = fio.FITS(filename)[-1].read()
        print ('===> time to load AFTER MASK =', time.time()-t0)
        return f


    def load_AFTER_MASK_array(self):
        print ('==================array ==================')
        t0 = time.time()
        filename = '../desi_data/SELECTION/%s/arr.fits'%(self.loc)
        f = fio.FITS(filename)[-1].read()
        ra = f['ra']
        dec = f['dec']
        M = f['M']
        photo_z = f['photo_z']
        Nmember = f['Nmember']
        L = f['L']
        id = f['id']
        print ('===> len(ra) = ',len(ra))
        print ('===> time to load AFTER MASK = ', time.time()-t0)
        return id,Nmember,ra,dec,photo_z,M,L


class CountGroup(object):
    def __init__(self, ra, dec):
        self.ra = ra
        self.dec = dec

    def counts_group(self,nside=512):
        t0=time.time()
    #    ra,dec = fk5_2_galactic(ra, dec)  # RUN AFTER MASK has changed coor
        pxl = hp.ang2pix(nside, np.pi/2.-np.deg2rad(self.dec), np.deg2rad(self.ra) )
        print ('===>pxl.shape=',pxl.shape)
        pixel = pd.DataFrame({'pixel':pxl})
        f = pd.DataFrame({'pixel': pxl,
                          'counts':pxl})
        F = f.groupby('pixel',as_index=False)['counts'].agg('count')

        pixel = F.pop('pixel')
        counts = F.pop('counts')
        meanfield = np.mean(counts)
        colorpix = np.zeros(hp.nside2npix(nside))
        colorpix[pixel] = counts
        print ('===> time to count group number =', time.time()-t0)
        return colorpix, meanfield


class ModifiedMap(object):

    def __init__(self,colorpix, threshold):
        self.colorpix = colorpix
        self.threshold = threshold
    # new delta
    def modified_map(self):
        t0=time.time()
        g_fi, g_mask, mask = total_mask(self.threshold)
        fi = g_fi * mask
        N_pxl = np.sum(mask)  # rest pxls after masking
        f_sky = N_pxl / len(mask) 
        print ('---> number of pixel = ', N_pxl)
        print ('---> f_sky = ', f_sky)
        ans = np.where(fi!=0)
        density = self.colorpix*mask
        density[ans] = density[ans] / fi[ans]
        print ('===> effective density, effective density shape = ', density, density.shape)
        mod_mean = np.sum(density) / N_pxl
        print ('mod_mean=',mod_mean)
        Modified = density / mod_mean -1
        print ('===> time to modify map = ',time.time()-t0)
        return Modified

class ClkgTest(object):
    def __init__(self, Modified, threshold):
        self.Modified = Modified
        self.threshold = threshold

    def clkg_test(self):
        t0=time.time()
        K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB(tSZ=tSZ)
        g_fi, g_mask, mask = total_mask(self.threshold)
        kg = hp.anafast(K*mask, self.Modified*mask)
        kg_bin = C_new(kg,Q=12)[1]
        gg = hp.anafast(Modified*mask)
        gg_bin = C_new(gg,Q=12)[1]
        print ('kg_bin=',kg_bin)
        print ('gg_bin=',gg_bin)
        print ('===> time to get clkg =', time.time()-t0)
        return kg,kg_bin,gg,gg_bin
        print ('gg_bin=',gg_bin)
        print ('===> time to get clkg =', time.time()-t0)
        return kg,kg_bin,gg,gg_bin


    def kgerr(self):
        t0=time.time()
        Nsim=N_lens_sim
        K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB(tSZ=tSZ)
        g_fi, g_mask, mask = total_mask(threshold)
        clskg = np.ones((Nsim, 3*nside))
        print ('======= error Kg ==========')
        gsam_ma = Modified*mask  
        for n in range(Nsim):
            print ('Nsam=',n)
            ksam = load_CMB_sim(n,tSZ)      # load sim from 60-299 
            ksam_ma = ksam * mask     # K_mask is obs mask, not sim mask
            clskg[n] = hp.anafast(gsam_ma, ksam_ma)
        print ('===> clskg = ',clskg)
        print ('===> time to get kgerr =', time.time()-t0)
        print ('======= End of error kg ==========')
        return clskg




    def get_cov(clskg):
        t0 = time.time()
        Nsam = N_lens_sim     
        clskg_bin = np.ones((Nsam,Q))
        for n in range(Nsam):
            clskg_bin[n] = C_new(clskg[n],Q)[1]
        err_kg = np.std(clskg_bin, axis=0)
        cov = np.cov(clskg_bin,rowvar=False)
        r = np.corrcoef(clskg_bin,rowvar=False)
        print ('err_kg=',err_kg)
        print ('===> time to get covariance = ',time.time()-t0)
        return err_kg,cov,r


    def Cl_kmu(filename,i, bias=1.0):
        import numpy as np
        import pyccl as ccl 
        import fitsio as fio 
        import matplotlib.pyplot as plt 
        f = fio.FITS(filename)[-1].read()
        z = f['z']
        nz = f['n_%s'%(i+1)]   # i-th redshift bin in redshift distribution
        h = 0.6736
        cosmo = ccl.Cosmology(Omega_c = 0.120/h**2, Omega_b = 0.0493, h = h, sigma8 = 0.8111, n_s = 0.9649 )  # planck18-only, table2 col5
        lens1 = ccl.WeakLensingTracer(cosmo, dndz=(z, nz)) #CCL automatically normalizes dNdz
        cmbl = ccl.CMBLensingTracer(cosmo, 1090.)
        ell = np.arange(3*512)
        cl_kmu = ccl.angular_cl(cosmo, lens1, cmbl, ell)
        return cl_kmu


    def kmu_th(filename,i):
        g_fi, g_mask, mask = total_mask(threshold)
        t0 = time.time()
        Nnoise=100
        wn_flt = wiener_filter()
        norm = (12*512**2/4./np.pi)
        cl_kmu = Cl_kmu(filename, i=i)
        kg = np.ones((Nnoise, nside*3))
        for n in range(Nnoise):
            print ('noise=',n)
            noise = np.load('../cmb-cross-correlation/noise/noise_No.%s.npy'%(n+1))
            Nlm = hp.map2alm(noise)
            Nlm1 = hp.almxfl(Nlm, np.sqrt(cl_kmu))
            Nlm2 = hp.almxfl(Nlm1, np.sqrt(wn_flt))
            maps = hp.alm2map(Nlm2,nside=nside,verbose=False)
            maps_ma = maps * mask
            kg[n] = hp.anafast(maps_ma)*norm
        kg_mean = np.mean(kg, axis=0)
        print ('kg_th_mean = ',kg_mean)
        print ('===> time to get kg_th = ',time.time()-t0)
        return kg_mean



    def Cl_th_pyccl(filename,i, bias=1.0):
        import numpy as np
        import pyccl as ccl 
        import fitsio as fio 
        import matplotlib.pyplot as plt 
        f = fio.FITS(filename)[-1].read()
        z = f['z']
        nz = f['n_%s'%(i+1)]   # i-th redshift bin in redshift distribution
        h = 0.6736
        cosmo = ccl.Cosmology(Omega_c = 0.120/h**2, Omega_b = 0.0493, h = h, sigma8 = 0.8111, n_s = 0.9649 )  # planck18-only, table2 col5    
        b = bias * np.ones(len(z))
        cmbl = ccl.CMBLensingTracer(cosmo, 1090.) 
        clu = ccl.NumberCountsTracer(cosmo, has_rsd = False, dndz = (z, nz), bias=(z,b)) 
        ell = np.arange(3*512)
        cls_cmbl_cross_clu = ccl.angular_cl(cosmo, cmbl, clu, ell)
        cls_cmbl = ccl.angular_cl(cosmo, cmbl, cmbl, ell)
        cls_clu = ccl.angular_cl(cosmo, clu, clu, ell)
        return cls_cmbl_cross_clu, cls_clu




    def kg_th(filename,i):
        g_fi, g_mask, mask = total_mask(threshold)
        t0 = time.time()
        Nnoise=100
        wn_flt = wiener_filter()
        norm = (12*512**2/4./np.pi)
        cls_cmbl_cross_clu, cls_clu = Cl_th_pyccl(filename, i=i)
        kg = np.ones((Nnoise, nside*3))
        for n in range(Nnoise):
            print ('noise=',n)
            noise = np.load('../cmb-cross-correlation/noise/noise_No.%s.npy'%(n+1))
            Nlm = hp.map2alm(noise)
            Nlm1 = hp.almxfl(Nlm, np.sqrt(cls_cmbl_cross_clu))
            Nlm2 = hp.almxfl(Nlm1, np.sqrt(wn_flt))
            maps = hp.alm2map(Nlm2,nside=nside,verbose=False)
            maps_ma = maps * mask
            kg[n] = hp.anafast(maps_ma)*norm
        kg_mean = np.mean(kg, axis=0)
        print ('kg_th_mean = ',kg_mean)
        print ('===> time to get kg_th = ',time.time()-t0)
        return kg_mean





    def W_k(z):
        from astropy.cosmology import Planck18_arXiv_v2 as cosmo
        from astropy import constants as const
        import astropy.units as u

        c = const.c.to(u.km/u.s)
        H0 = cosmo.H(0)
        
        comoving_d = cosmo.comoving_distance(z)
        comoving_CMB = cosmo.comoving_distance(z=1090)
        Omega_m = cosmo.Odm(z)
        Om0 = cosmo.Odm0
        Hz = cosmo.H(z)
    #    W = 3./2. * Omega_m * (1+z) * comoving_d * (1. - comoving_d / comoving_CMB)
        W_L = 3./2./(c) * Om0 * H0**2 / Hz * (1+z) * comoving_d * (1. - comoving_d / comoving_CMB)
        return W_L, comoving_d

    #W_L, comoving_d = W_k(np.arange(0,1.1,0.01))
    #print (np.array(W_L))
    #np.save('../desi_data/kappa_data/W_L.npy',W_L.value)


    def M_h_bar(x,richness):
        if richness==5:
            a = np.array([-2.85643310e-02,  1.11747493e+00, -1.35532938e+01,  6.29229504e+01])
        else:
            a = np.array([-2.11132240e-02,  8.08597402e-01, -9.28435307e+00,  4.32453463e+01])
        y_poly = a[0]*x**3 + a[1]*x**2 + a[2]*x**1 + a[3]*x**0
        return y_poly



    def sigmaM(x,richness):
        if richness==5:
            b = np.array([ 1.15540205e-02, -5.89402759e-01,  1.11940448e+01, -9.37742573e+01, 2.92540411e+02])
        else:
            b = np.array([ 2.20191357e-02, -1.15361071e+00,  2.25747280e+01, -1.95565950e+02, 6.33175042e+02])
        y_poly = b[0]*x**4 + b[1]*x**3 + b[2]*x**2 + b[3]*x**1 + b[4]*x**0
        return y_poly



    def M_true(x,richness):
        random = np.random.normal(0,1,len(x))    
        M_t = random * sigmaM(x,richness) + M_h_bar(x,richness)
        print ('M = ',x)
        print ('M_t = ',M_t)
        return M_t


    def weighted_avg_and_std(values, weights):
        """
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values-average)**2, weights=weights)
        return np.sqrt(variance)



    def b1_b2(photo_z,M,model):
        t0 = time.time()
        from colossus.lss import bias
        from colossus.cosmology import cosmology
        cosmology.setCosmology('planck18-only')
        cosmo = cosmology.getCurrent()
        print ('===> you are calculate b_Wk......')
        M = 10**M
        b = bias.haloBias(M, model = model, z = photo_z, mdef = '180c')
        W_L, comoving_d = W_k(z = photo_z)
        b_Wk1 = sum(b * W_L * comoving_d) / sum(W_L * comoving_d)

        sigma_b = weighted_avg_and_std(b,W_L * comoving_d)
        print ('===> b_Wk1 = ',b_Wk1)
        print ('===> sigma_b = ',sigma_b)
        print ('===> time to get bias = ',time.time()-t0)
        return b_Wk1, sigma_b


    def App_mag(z,L):
        # input luminosity unit is 10**10L_sun
        from astropy import constants as c
        from astropy import units as u
        from astropy.cosmology import Planck18_arXiv_v2 as cosmo
        L_0 = 3.0128*10**28  # unit: W
        L_sun = ((L_0*u.W). to(u.L_sun) ).value
        abs_mag = -2.5*np.log10(L*10**10/L_sun)
        comoving_d = cosmo.comoving_distance(z) # unit:Mpc
        app_mag = abs_mag -5 +5*np.log10(comoving_d.value*1e6) #D unit: pc
        print ('apparent magnitude=',app_mag)
        return app_mag


    def slope(z,L):
        app_mag = App_mag(z,L)
        print ('---> min app_mag, max app_mag, len app_mag=',np.min(app_mag),np.max(app_mag),len(app_mag))

        # magnitude limit
    #    abs_mag = abs_mag[abs_mag<=-21]
    #    print ('abs_mag max value=', max(abs_mag))
        delta_m = 0.01 #+-
        N, bins, patched = plt.hist(app_mag,bins=int((max(app_mag)-min(app_mag))/(2*delta_m)),cumulative=True,alpha=0.5,log=True)
        s = (np.log10(N[-1])-np.log10(N[-2])) / (bins[-1]-bins[-2])
        N_group = len(app_mag)
        err_s = (np.log10(N_group) - np.log10(N_group - np.sqrt(N_group))) / delta_m
        plt.xlabel('m')
        plt.ylabel('N')
        plt.title('$s = %.4f, \Delta_s = %.4f$'%(s,err_s))    
        plt.savefig('../desi_data/SELECTION/s_plot/s_'+time.strftime('%Y-%m-%d-%H%M%S',time.localtime(time.time()))+'.png',dpi=300)
        print ('---> s, err_s=',s, err_s)
        return s, err_s,


    def phi_z_prime(z):
        mu, sigma = 0.39355101, 0.20027535
        return 1./np.sqrt(2*np.pi) / sigma * np.exp(-(z-mu)**2 / 2/sigma**2)


    def integral_g_z_prime(z):
        from astropy.cosmology import Planck18_arXiv_v2 as cosmo
        from astropy import constants as const
        import astropy.units as u
        import numpy as np
        c = const.c.to(u.km/u.s)
        H0 = cosmo.H(0)
        comoving_d = cosmo.comoving_distance(z)
        comoving_CMB = cosmo.comoving_distance(z=1090)

        dz_prime=0.01
        z_prime = np.arange(z,1090,dz_prime)
        g_z_prime = np.zeros(len(z_prime))
        for i in range(len(z_prime)):
            comoving_d_prime = cosmo.comoving_distance(z_prime[i])
            g_z_prime[i] = comoving_d.value * (comoving_d_prime.value - comoving_d.value) / comoving_d_prime.value * phi_z_prime(z_prime[i])
        Integral_gz_prime = np.sum(g_z_prime) * dz_prime
        
        return Integral_gz_prime



#def W_mu(L, z):
#    from astropy.cosmology import Planck18_arXiv_v2 as cosmo
#    from astropy import constants as const
#    import astropy.units as u

#    c = const.c.to(u.km/u.s)
#    H0 = cosmo.H(0)
    
#    comoving_d = cosmo.comoving_distance(z)
#    comoving_CMB = cosmo.comoving_distance(z=1090)
#    Omega_m = cosmo.Odm(z)
#    Om0 = cosmo.Odm0
#    Hz = cosmo.H(z)

#    s,err_s = slope(L)     
#    w_mu = (5*s-2) * 3./2./(c) *Om0 * H0**2 / Hz * (1+z) * integral_g_z_prime(z)
#    return w_mu.value

#def cl_kmu(z,l, w_mu, w_k):
#    from astropy.cosmology import Planck18_arXiv_v2 as cosmo
#    from astropy import constants as const
#    import astropy.units as u

#    c = const.c.to(u.km/u.s)
#    comoving_d = cosmo.comoving_distance(z)
#    Hz = cosmo.H(z)
    
#    dz = 0.01
#    Cl_kmu = ( dz * Hz / c * w_k * w_mu / comoving_d**2 * P_XY )    
#    return Cl_XY


def namaster(i,j,cut,w):
    import pymaster as nmt
    mask=np.load('../desi_data/survey_mask/total_mask.npy')
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap=load_CMB()
    f_1 = nmt.NmtField(mask, [hp.read_map('../planck/Kmap.fits', field=0, verbose=False)])
    f_0 = nmt.NmtField(mask, [hp.read_map('../desi_data/%scut_map/fits%s/overdensity_z%s_%s.fits'%(cut,w,i,j),field=0,verbose=False)])
    b = nmt.NmtBin.from_nside_linear(nside, 1,is_Dell=False)
    cl_01 = nmt.compute_full_master(f_0, f_1, b)
    cl=C_new(cl_01[0],Q)[1]
    return cl

def plot_namaster(cut,w):
    fontsize=20
    alpha=0.6
    Nsam=240
    Q=12
    msize=8
    z_range = np.linspace(0.1,0.9,5)
    member_cut = np.array([0,2,4,6])
    color = np.array(['g','r','orange','b'])
    ell=np.arange(lmax+1)
    fig,axs = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(17,12),sharey=False)
    CL = np.ones((4,4,Q))
    for ax,i in zip(axs.flat,range(4)):
        zmin=round(z_range[i],1)
        zmax=round(z_range[i+1],1)
        for j in range(4):
            if cut=='mass':
                nn=mass_cut[j]
                label = '$log$M>%s %s'%(nn,w)
            if cut=='member':
                nn=member_cut[j]
                label = '$N_{member} \geqslant %s$ %s'%(nn+1,w)
            print ('===========z=%s, %s=%s============='%(i,cut,nn))
            cl_01=namaster(i,j,cut=cut,w=w)
            CL[i,j] = cl_01
            ax.loglog(ell_new, cl_01,label=label,c=color[j],marker='s',lw=3,markersize=msize,alpha=alpha)
        ax.legend(ncol=1,loc=8,fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(10,1e3)
        ax.set_title('%s<z<%s,Namaster'%(zmin,zmax),fontsize=fontsize)
        ax.set_xlabel(r'$\ell$',fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.set_ylabel(r'$\frac{\ell(\ell+1)} {2\pi}\mathcal{C}_\ell^{\kappa g}$', fontsize = fontsize)    
    plt.tight_layout()
    np.save('../desi_data/namaster/CLkg_%s%s.npy'%(cut,w),CL)
    plt.savefig('figure/Clkg_nmt_%s_'%(cut)+time.strftime('%Y-%m-%d-%H%M%S',time.localtime(time.time()))+'.png',dpi=300)

#plot_namaster(cut='mass',w='')






















