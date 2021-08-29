import numpy as np
import healpy as hp
#import matplotlib.pyplot as plt
from pandas import value_counts
from astropy.io import fits
import fitsio as fio 
import time
nside = 512
lmax = 1024

#-------------select the magnitude cut of r-band
def load_galaxy_mag_r(autoplot=True, autosave=True):
    import time 
    t0 = time.time()

    import os 
    import numpy as np
    import fitsio as fio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    filename = '/mnt/d2/public/zysun/zouhu/LS_DR8_total_csp.fits'
    f = fio.FITS(filename)[-1].read()
    z = f['photo_z']
    mag_r = f['MAG_ABS_R']
    print '===>the number of galaxies=',len(z)
    
    if autoplot:
        plt.figure(figsize=(10,7))
        plt.scatter(z, mag_r, c='b', marker='.',s=5)
        fontsize=22
        plt.xlabel('z',fontsize=fontsize)
        plt.ylabel(r'$mag_{r}$',fontsize=fontsize)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        figname = '../figure/mag_r_vs_z_'+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.png'
        plt.savefig(figname)
    if autosave:
        filename='/mnt/d2/public/zysun/cmb-cross-correlation/mag_r_vs_z_'+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.fits'
        try:
            os.remove(filename)
        except OSError:
            pass
        f = fio.FITS(filename,'rw')
        f.write([z, mag_r], names=['photo_z', 'MAG_ABS_R'])
        f.close()
    print '===>time to load galaxy mag_r =',time.time()-t0
    return z, mag_r

#-------load m_r cut galaxy
def cut_mag_r(zmin, zmax, cut):
    import time 
    t0 = time.time
    import os 
    import numpy as np
    import fitsio as fio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    #filename = '../cmb-cross-correlation/mag_r_vs_z_2020-09-15-215758.fits'
    filename = '../zouhu/LS_DR8_total_csp.fits'
    f = fio.FITS(filename)[-1].read()
    z = f['photo_z']
    mag_r = f['MAG_R']
    RA = f['RA']
    DEC = f['DEC']

    print '===> you are selecting ......'
    slt = (zmin<z) & (z<zmax) & (mag_r<cut)
    z_ph = z[slt]
    m_r = mag_r[slt]
    ra = RA[slt]
    dec = DEC[slt]

    filename = '../cmb-cross-correlation/galaxy_%sto%s_mag_r_cut%s.fits'%(zmin, zmax, cut)
    try:
        os.remove(filename)
    except OSError:
        pass
    f = fio.FITS(filename,'rw')
    f.write([z_ph, m_r, ra, dec], names=['photo_z','MAG_R','RA','DEC'])
    f.close()
    print '===>you have saved photo_z and mag_r fits file at %s<z<%s!'%(zmin, zmax)
    if autoplot:
        plt.figure(figsize=(10,7))
        plt.scatter(z, mag_r, c='b', marker='.',s=5, alpha=0.5)
        fontsize=22
        plt.xlabel('z',fontsize=fontsize)
        plt.ylabel(r'$m_{r}$',fontsize=fontsize)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        figname = '../figure/m_r_vs_z_%sto%s_cut%s_'%(zmin,zmax,cut)+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.png'
        plt.savefig(figname)
        print '===>you have save the mag_r vs z figure at %s<z<%s!'%(zmin, zmax)

    print '===> time to load galaxies = ', time.time()-t0
    print '===> number of selected galaxies =', len(z_ph)
    print '===> number of selected galaxies =', len(ra)
    return z_ph, m_r, ra, dec

cut_mag_r(zmin=0.1, zmax=0.9, cut=21)
cut_mag_r(zmin=0.1, zmax=0.3, cut=21)
cut_mag_r(zmin=0.3, zmax=0.5, cut=21)
cut_mag_r(zmin=0.5, zmax=0.7, cut=21)
cut_mag_r(zmin=0.7, zmax=0.9, cut=21)


def run_Cl_cut(zmin,zmax,cut,autoplot=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import healpy as hp
    import time
    import fitsio as fio

    # read galaxy and target selection
    filename = '../cmb-cross-correlation/galaxy_%sto%s_mag_r_cut%s.fits'%(zmin, zmax, cut)
    f = fio.FITS(filename)[-1].read()
    galaxy = f[['RA','DEC']]
    g_l, g_b = fk5_2_galactic( galaxy['RA'], galaxy['DEC'] )
    print '===> you are pixelizing galaxy map ......'
    g_ra, g_dec, g_c, g_mean, colorpix, g_mask = pix_gal(g_l, g_b)
    colorpix = colorpix/g_mean
    if autoplot:
        hp.mollview(colorpix, title='%s<z<%s, m_r<%s'%(zmin, zmax, cut))
        plt.savefig('../figure/maps_cut205/overdensity_%sto%s_cut%s.png'%(zmin,zmax,cut))
        print '===> you have saved overdensity map at %s<z<%s!'%(zmin,zmax)
    # read CMB kappa
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB()
    # combine mask
    mask = comb_mask( K_mask, g_mask )
    if autoplot:
        hp.mollview(mask, title='%s<z<%s, m_r<%s'%(zmin, zmax, cut))
        plt.savefig('../figure/maps_cut205/mask_%sto%s_cut%s.png'%(zmin, zmax, cut))
        print '===> you have saved mask map at %s<z<%s!'%(zmin,zmax)
    K_ma = hp.ma(K)
    K_ma.mask = np.logical_not(mask)
    K_lm = hp.map2alm(K_ma,lmax=lmax)

    g_ma = hp.ma(colorpix)
    g_ma.mask = np.logical_not(mask)
    g_lm = hp.map2alm(g_ma,lmax=lmax)
    print('---> len of g_lm = ', len(g_lm))
    loc = '/mnt/d2/public/zysun/cmb-cross-correlation/cls'

    C_Kg = hp.alm2cl(K_lm,g_lm)
    C_gg = hp.alm2cl(g_lm)
    C_KK = hp.alm2cl(K_lm)
    print '===> you have done the calculation of power spectrum!!!'
    loc = '../cmb-cross-correlation/cls_cut21'
    np.save('%s/C_Kg_obs_%sto%s_cut%s_'%(loc, zmin, zmax, cut)+time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))+'.npy', C_Kg)
    np.save('%s/C_gg_obs_%sto%s_cut%s_'%(loc, zmin, zmax, cut)+time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))+'.npy', C_gg)
    np.save('%s/C_KK_obs_%sto%s_cut%s_'%(loc, zmin, zmax, cut)+time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))+'.npy', C_KK)
    return C_Kg


def eff_bias():
    bias = np.array([1.0, 1.2, 1.3, 1.2])
    # bias = np.array([1.0, 1.0, 1.0, 1.0])
    cut = ['21.5','22','22.5']
    wn_flt = wiener_filter()
    norm = (12*512**2/4./np.pi)
    noise = np.load('../cmb-cross-correlation/noise/noise_No.1.npy')
    Nlm = hp.map2alm(noise, lmax=lmax)
    clgg_th = np.ones((4,len(cut),lmax+1))
    clkg_th = np.ones((4,len(cut),lmax+1))
    for i in range(4):
        zmin = 0.1 + 0.2 *i
        zmax = 0.3 + 0.2 *i
        loc = '../cmb-cross-correlation/Cl_th_pyccl'
        cls_cmbl_cross_clu = np.load('%s/Cl_kg_th_%sto%s_b%s.npy'%(loc, zmin, zmax, bias[i]))
        cls_clu= np.load('%s/Cl_gg_th_%sto%s_b%s.npy'%(loc, zmin, zmax, bias[i]))
        Nlm_gg1 = hp.almxfl(Nlm, np.sqrt(cls_clu))
        Nlm_gg2 = hp.almxfl(Nlm_gg1, np.sqrt(wn_flt))
        T_gg = hp.alm2map(Nlm_gg2, nside=nside, verbose=False)
        print '===> you have done T_gg! '

        Nlm_kg1 = hp.almxfl(Nlm, np.sqrt(cls_cmbl_cross_clu))
        Nlm_kg2 = hp.almxfl(Nlm_kg1, np.sqrt(wn_flt))
        T_kg = hp.alm2map(Nlm_kg2, nside=nside, verbose=False)
        print '===> you have done T_kg!'

        for j,c in zip(range(len(cut)),cut):
            mask = np.load('../cmb-cross-correlation/mask_cut/mask_%sto%s_cut%s.npy'%(zmin,zmax,cut[j]))
            T_ggma = T_gg * mask
            T_kgma = T_kg * mask
            clgg_th[i,j] = hp.anafast(T_ggma, lmax=lmax)*norm
            clkg_th[i,j] = hp.anafast(T_kgma, lmax=lmax)*norm
            print '===> you have done the power spectrum at %s<z<%s cut=%s!!'%(zmin,zmax,cut[j])

    np.save('../cmb-cross-correlation/cl_th_cut/clgg_th_allz_cut21.5_22_22.5.npy',clgg_th)
    np.save('../cmb-cross-correlation/cl_th_cut/clkg_th_allz_cut21.5_22_22.5.npy',clkg_th)

eff_bias()

# ------------- get n(z)
def get_nz(cut, zmin=0.0, zmax=1.0):
    import time
    t0 = time.time()

    import os
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    import numpy as np
    from scipy import stats
    import fitsio as fio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    filename = '../zouhu/LS_DR8_total_csp.fits'
    print '===> you are reading file ......'
    f = fio.FITS(filename)[-1].read()
    z = f['photo_z']
    mag_r = f['MAG_R']
    print '===> you are selecting cut=%s......'%cut
    slt = (0.1<z) & (z<0.9) & (mag_r<cut)   # photo-z range
    z = z[slt]
    print 'len of z = ', len(z)
    f = f[slt]
    print 'len of f = ', len(f)
    dz = f['photo_zerr']
    print 'len of photoz err = ', len(dz)
    print '===> you are selecting photo_z and photo_zerr ... '

    z_tr = np.linspace(zmin,zmax,int(zmax-zmin)*100+1)   # true redshift
    print 'z_tr shape = ', z_tr.shape
    nz = np.zeros((5,len(z_tr)))
    count = np.zeros(5)

    PDF = stats.norm.pdf(z_tr, loc=z.reshape(-1, 1), scale=dz.reshape(-1, 1))
    PDF_sum = np.diff(stats.norm.cdf(z_tr[[0, -1]], loc=z.reshape(-1, 1), scale=dz.reshape(-1, 1)), axis=-1)
    PDF /= PDF_sum

    for i in range(4):
        z_min = i*0.2 + 0.1
        z_max = (i+1)*0.2 + 0.1
        slt = (z_min<z) & (z<z_max)
        nz[i+1][:] = np.sum(PDF[slt],0)
        nz[0][:] += nz[i+1][:]   # nz[0] is the true redshift distribution at all z 
    nz /= len(z)

    print '===> you are plotting ......'
    plt.figure()
    N, bins, patches = plt.hist(z, bins=40, density=True,edgecolor='white',linewidth=1,alpha=0.4)
    for i in range(0,10):
        patches[i].set_facecolor('g')
    for i in range(10,20):    
        patches[i].set_facecolor('orange')
    for i in range(20,30):
        patches[i].set_facecolor('c')
    for i in range(30,40):
        patches[i].set_facecolor('r')

    plt.plot(z_tr,nz[0],label='0.1<z<0.9',c='k')
    colors=['g','orange','c','r']
    for i in range(4):
        zmin = 0.1+0.2*i
        zmax = 0.3+0.2*i
        plt.plot(z_tr,nz[i+1],label='%s<z<%s'%(zmin,zmax),c='%s'%colors[i])
    plt.xlabel('z')
    plt.ylabel(r'$n(z)$')
    plt.legend()
    filename = '../cmb-cross-correlation/nz_cut_plot/nz_cut%s_'%cut+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.png'
    plt.savefig(filename,dpi=500)

    filename = '../cmb-cross-correlation/nz_cut/nz_cut%s_'%cut+time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))+'.fits'
    try:
        os.remove(filename)
    except OSError:
        pass
    f = fio.FITS(filename,'rw')
    f.write( [z_tr, nz[0], nz[1], nz[2], nz[3], nz[4]], names=['z','n_all','n_1','n_2','n_3','n_4'] )
    f.close()
    print '===> time to get nz = ', time.time()-t0
    print '===> you have saved the plots and fits file cut=%s!!!'%cut

#cuts=['21.5','22','22.5']
get_nz(cut=23)
get_nz(cut=21.5)
get_nz(cut=22)
get_nz(cut=22.5)




# ----------- load DESI DR7.1 data
def load_saved_galaxy(filename):

    from time import time
    t0 = time()

    import fitsio as fio
    galaxy = fio.FITS(filename)[-1].read()

    #print '===> time to load galaxies = ', time()-t0
    return galaxy

# ----------- convert sky coordinates from FK5 to galactic
def fk5_2_galactic( ra, dec, autoplot=False ):
    import time
    t0=time.time()

    import astropy.units as u
    from astropy.coordinates import SkyCoord
    import healpy as hp
    import numpy as np

    gc = SkyCoord(ra=ra*u.degree,dec=dec*u.degree,frame='fk5')
    l = gc.galactic.l.deg
    b = gc.galactic.b.deg

    #print '===> time to convert fk5 to galactic coordinates = ', time.time()-t0
    return l,b

# -------------- pixelize the galaxy position
def pix_gal( ra, dec, nside=nside):

    import time
    t0 = time.time()
    
    from pandas import value_counts
    import healpy as hp
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    #print '===> size of pixel in unit of arcmin = ',np.sqrt( 41253.*60.*60./hp.nside2npix(nside) )

    pxl = hp.ang2pix(nside, np.pi/2.-np.deg2rad(dec), np.deg2rad(ra) )
    pxlc = value_counts(pxl)
    meanfield = np.mean(pxlc.values)
    pxlc = pxlc - meanfield

    [pxl_dec, pxl_ra] = np.rad2deg(hp.pix2ang(nside, pxlc.index))
    pxl_dec = 90.-pxl_dec

    g_mask = np.zeros(hp.nside2npix(nside))
    g_mask[pxlc.index] = 1

#     if autoplot: # plot the galaxy number over density per pixel
    colorpix = np.ones(hp.nside2npix(nside)) * (-meanfield)
    colorpix[pxlc.index] = pxlc.values

#    loc = '../cmb-cross-correlation/galaxy_data'
#    np.save('%s/colorpix_nside%s_%sto%s.npy'%(loc,nside, zmin, zmax), colorpix)
#    np.save('%s/g_mask_nside%s_%sto%s.npy'%(loc, nside, zmin, zmax), g_mask)

#    hp.mollview(colorpix, title='galaxy count')
#    filename = '%s/healpix_map_galactic_nside%s_'%(loc, nside)+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.pdf'
#   plt.savefig(filename)

    #print '===> time to pixelize the galaxies = ', time.time()-t0
    return pxl_ra, pxl_dec, pxlc.values, meanfield, colorpix, g_mask

#load Planck kappa map, resolution default nside=2048, but generated mask map is 512 to fit the galaxy mask size
def load_CMB(nside_dg=nside, nside=nside):

    from time import time
    t0 = time()
    
    import healpy as hp
    import numpy as np

    mask = hp.read_map('../planck/mask.fits', verbose=False)

    # downgrade CMB mask to fit galaxy mask
    K_mask = hp.ud_grade(mask, nside_dg)
    K_mask[K_mask<1] = 0

    K_lm, mmax = hp.read_alm('../planck/dat_klm.fits', return_mmax=True)
    tmp = np.loadtxt('../planck/nlkk.dat')
    
    ell = np.array(tmp[:,0])
    N = np.array(tmp[:,1])
    C = np.array(tmp[:,2]-tmp[:,1])
    N_lm = hp.almxfl(K_lm, N/(C+N))
    K_lm = hp.almxfl(K_lm, C/(C+N))
    Kmap = hp.alm2map(K_lm, nside=nside, verbose=False)
    Nmap = hp.alm2map(N_lm, nside=nside, verbose=False)

    K_dec, K_ra = np.rad2deg( hp.pix2ang( nside, range(hp.nside2npix(nside)) ) )
    K_dec = 90.-K_dec
# here was changed, mask -> K_mask 
    K_ra = K_ra[K_mask>0]
    K_dec = K_dec[K_mask>0]
    K_val = Kmap[K_mask>0]

#    loc = '../cmb-cross-correlation/lensing_data'
#    np.save('%s/Kappa_map_nside%s.npy'%(loc, nside), K)
#    np.save('%s/K_mask_nside%s.npy'%(loc, nside), K_mask)

#    hp.mollview(K, title='lensing map')
#    plt.savefig('%s/K_map_nside%s_'%(loc, nside)+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.pdf')

    #print '===> time to load CMB kappa map = ', time()-t0
    return K_ra, K_dec, K_val, K_mask, K_lm, N_lm, Kmap, Nmap


# --------------- combine CMB and galaxy masks, plot and save
def comb_mask( K_mask, g_mask,):

    import healpy as hp
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    mask = K_mask*g_mask


    return mask

# --------------- select targets in the mask
def in_mask(nside, mask, ra, dec, field):
    import time
    t0 = time.time()

    import numpy as np
    import healpy as hp
   
    # fit into mask
    hparr = hp.ma(mask)
    m = np.arange(hp.nside2npix(nside))
    m = m[hparr>0]
    pix = hp.ang2pix(nside, np.pi/2.-np.deg2rad(dec), np.deg2rad(ra), nest=False)
    slt = np.in1d(pix, m, assume_unique=False)

    ra = ra[slt]
    dec = dec[slt]
    field = field[slt]

    print( '===> time to fit in mask = ', time.time()-t0)
    return ra,dec,field


# --------------- data loading, randoms, masks, ready for correlation
def Nl_gg():
    import time
    t0 = time.time()
    import numpy as np
    import healpy as hp
    import fitsio as fio
    Nl_gg_allz = []
    for i in range(4):
        print( '---> you redshift you are in is = ', i+1)
        zmin = 0.1 + 0.2 *i
        zmax = 0.3 + 0.2 *i

        filename = '/mnt/d2/public/zysun/cmb-cross-correlation/galaxy_%sto%s.fits'%(zmin, zmax)
        galaxy = load_saved_galaxy(filename)

        Ng = len(galaxy['RA'])
        print( '===> the number of galaxies in %s<z<%s = '%(zmin, zmax), Ng)
        g_l, g_b = fk5_2_galactic(galaxy['RA'], galaxy['DEC'])
        g_ra, g_dec, g_c, g_mean, overdensity_map, density_map, g_mask = pix_gal(g_l, g_b)
        g_c = g_c / g_mean

        # read CMB kappa
        K_ra, K_dec, K_val, K_mask, K_lm, N_lm, Kmap, Nmap = load_CMB()

        # combine mask
        mask = comb_mask( K_mask, g_mask )

        density_map_ma = hp.ma(density_map)
        density_map_ma.mask = np.logical_not(mask)
        Ng_ma = np.sum(density_map_ma)
        print( '===> the number of galaxies in %s<z<%s after combined mask = '%(zmin, zmax), Ng_ma)

        f_sky = np.sum(mask) / len(mask)
        n_bar = Ng_ma / (4*np.pi * f_sky)
        Nl_gg = 1./n_bar
        print( '===> Nl_gg = ', Nl_gg)
        Nl_gg_allz.append(Nl_gg)
    np.save('/mnt/d2/public/zysun/cmb-cross-correlation/Nl_gg_allz.npy', Nl_gg_allz)

    print ('===> time to get Nl_gg_allz = ', time.time()-t0)
    return Nl_gg_allz


# --------------- run power spectra with healpy
def run_Cl():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import healpy as hp
    import time
    import fitsio as fio

    # read galaxy and target selection
    filename = '/mnt/d2/public/zysun/cmb-cross-correlation/galaxy_%sto%s.fits'%(zmin, zmax)

    galaxy = load_saved_galaxy(filename)
    g_l, g_b = fk5_2_galactic( galaxy['RA'], galaxy['DEC'] )
    g_ra, g_dec, g_c, g_mean, colorpix, g_mask = pix_gal(g_l, g_b)
    colorpix = colorpix/g_mean

    # read CMB kappa
    K_ra, K_dec, K_val, K_mask, K_lm, K = load_CMB()

    # combine mask
    #mask = comb_mask( K_mask, g_mask )
        
    K_ma = hp.ma(K)
    K_ma.mask = np.logical_not(mask)
    K_lm = hp.map2alm(K_ma,lmax=lmax)

    g_ma = hp.ma(colorpix)
    g_ma.mask = np.logical_not(mask)
    g_lm = hp.map2alm(g_ma,lmax=lmax)
    print('---> len of g_lm = ', len(g_lm))
    loc = '/mnt/d2/public/zysun/cmb-cross-correlation/cls'
#    np.save('%s/K_lm_ma_nside%s_%sto%s.npy'%(loc, nside, zmin, zmax), K_lm)
#    np.save('%s/g_lm_ma_nside%s_%sto%s.npy'%(loc, nside, zmin, zmax), g_lm)
    
    C_Kg = hp.alm2cl(K_lm,g_lm)
    #C_gg = hp.alm2cl(g_lm)
    #C_KK = hp.alm2cl(K_lm)

#    loc = '/mnt/d2/public/zysun/cmb-cross-correlation/cls'
#    np.save('%s/C_Kg_obs_%sto%s_nside%s_'%(loc, zmin, zmax, nside)+time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))+'.npy', C_Kg)
#    np.save('%s/C_gg_obs_%sto%s_nside%s_'%(loc, zmin, zmax, nside)+time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))+'.npy', C_gg)
#    np.save('%s/C_KK_obs_%sto%s_nside%s_'%(loc, zmin, zmax, nside)+time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))+'.npy', C_KK)
    
    return C_Kg

#run_Cl()

def run_Clkg_jk():
    for i in range(4):
        zmin = 0.1 + 0.2 *i
        zmax = 0.3 + 0.2 *i
        mask = np.load('%s/mask/comb_mask_%sto%s_%s.npy'%(loc, zmin, zmax, nside))

# =====================binning part=======================
Q=15
cut=9
def ell_new():
    ell = np.arange(0.,lmax+1)
    smallell = ell[ell<=cut]; largeell=ell[ell>cut]
    largex=np.logspace(np.log10(largeell[0]), np.log10(largeell[-1]), Q+1)
    largex_int = np.ones(len(largex))
    for i in range(len(largex_int)):
        largex_int[i] = round(largex[i],0)
    new_ell=[]
    for i in range(len(largex)-1):
        new_ell.append( round((largex[i]+largex[i+1]) / 2, 0) )
    ell_new = np.hstack((smallell, new_ell))
    Nbin = len(ell_new)
    print( 'largex_int = ', largex_int)
    print( 'ell_new = ', ell_new)
    print( '===> Number of bins =',Nbin)
    return smallell, largex_int, ell_new, Nbin
#smallell, largex_int, ell_new, Nbin = ell_new()
# ------------------bin l, cl has been multiplied by coefficient
def C_new(cl):
    ell = np.arange(0.,lmax+1)
    bin_averages=[]
    for i in range(len(largex_int)-1):
        bin_averages.append(np.mean(cl[int(largex_int[i]):int(largex_int[i+1])]))
    C_new = np.hstack((cl[ell<=cut], bin_averages)) * ell_new*(ell_new+1)/(2.*np.pi) 
    return C_new
#------------------ no coefficient
def C_new_noco(cl):
    ell = np.arange(0.,lmax+1)
    bin_averages=[]
    for i in range(len(largex_int)-1):
        bin_averages.append(np.mean(cl[int(largex_int[i]):int(largex_int[i+1])]))
    C_new = np.hstack((cl[ell<=cut], bin_averages))  
    return C_new
# -------------------the delta ell
def delta_ell():
    delta_largex_int=[]
    for i in range(len(largex_int)-1):
        delta_largex_int.append( largex_int[i+1] - largex_int[i] )
        delta_ell = np.hstack((np.ones(len(smallell)), delta_largex_int))
    return delta_ell

# -------------------here the clkg_err didn't divided by sqrt(delta_ell*(2ell+1))
def C_err(clkg_err):
    delta_l = delta_ell()
    err = C_new_noco(clkg_err) / np.sqrt( delta_l *(2.*ell_new+1) )
    return err

# ---------------wiener filter, 4096
def wiener_filter():
    import numpy as np
    import healpy as hp

    tmp = np.loadtxt('/mnt/d2/public/zysun/planck/nlkk.dat')  # wiener filter  
    ell = np.array(tmp[:,0])
    N = np.array(tmp[:,1])
    C = np.array(tmp[:,2]-tmp[:,1])
    wn_flt = C / (C+N)
    return wn_flt



def mask():    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import healpy as hp
    import time
    import fitsio as fio
    loc = '../cmb-cross-correlation'
    print( '===============mask&overdensity map calculation===========')
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, Kmap, Nmap = load_CMB()
    for i in range(4):  # i-th redshift bin
        zmin = 0.1 + 0.2 *i
        zmax = 0.3 + 0.2 *i
        print( '===> you are in %s-th redshift bin'%(i+1))
         # read galaxy and target selection
        filename = '/mnt/d2/public/zysun/cmb-cross-correlation/galaxy_%sto%s.fits'%(zmin, zmax)
        galaxy = load_saved_galaxy(filename)
        g_l, g_b = fk5_2_galactic( galaxy['RA'], galaxy['DEC'] )
        g_ra, g_dec, g_c, g_mean, colorpix, g_mask = pix_gal(g_l, g_b)
        colorpix = colorpix/g_mean

    for i in range(4):  # i-th redshift bin
        zmin = 0.1 + 0.2 *i
        zmax = 0.3 + 0.2 *i
        print( '===> you are in %s-th redshift bin'%(i+1))
         # read galaxy and target selection
        filename = '/mnt/d2/public/zysun/cmb-cross-correlation/galaxy_%sto%s.fits'%(zmin, zmax)
        galaxy = load_saved_galaxy(filename)
        g_l, g_b = fk5_2_galactic( galaxy['RA'], galaxy['DEC'] )
        g_ra, g_dec, g_c, g_mean, colorpix, g_mask = pix_gal(g_l, g_b)
        colorpix = colorpix/g_mean

        # combine mask
#        mask = comb_mask( K_mask, g_mask )
#        
#        colorpix_ma = hp.ma(colorpix)
#        colorpix_ma.mask = np.logical_not(mask)
#
#        Kmap_ma = hp.mask(Kmap)
#        Kmap_ma.mask = np.logical_not(mask)
#
        #np.save('%s/mask/comb_mask_%sto%s_%s.npy'%(loc, zmin, zmax, nside), mask)
        np.save('%s/overdensity_map/overdensity_map_%sto%s_%s.npy'%(loc, zmin, zmax, nside), colorpix)
        plt.figure()
        hp.mollview(colorpix, title='galaxy overdensity map, %sto%s'%(zmin,zmax))
        filename = '%s/overdensity_map/healpix_map_%sto%s_%s.pdf'%(loc, zmin, zmax, nside)
        plt.savefig(filename)
    print( '============END OF COMBINED MASK==============')
#mask()


# =================== generate jackknife numbers================
def jk_numbers(mask, nside=512, regN=200, autoplot=True):
    import time
    t0 = time.time()
   
#    import os
#    os.environ['OMP_NUM_THREADS']='1'
#    os.environ['MKL_NUM_THREADS'] = '1'
#    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    loc = '../cmb-cross-correlation'
    import kmeans_radec
    from kmeans_radec import KMeans, kmeans_sample
    from kmeans_radec.test import plot_centers
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.pyplot import draw, figure, show
    import matplotlib.pyplot as plt
    import healpy as hp
    import numpy as np
    loc = '../cmb-cross-correlation'
    theta,phi = np.rad2deg( hp.pix2ang( nside, range(hp.nside2npix(nside)) ) )
    ra = phi[mask==1]
    dec = 90.-theta[mask==1]

    X = np.vstack((ra, dec)).T
    ncen = regN
    km = kmeans_sample(X, ncen, maxiter=200, tol=1.0e-5)

    print("found centers:",km.centers)
    print("converged?",km.converged)
    print("labels size:",km.labels.size)
    print("cluster sizes:", np.bincount(km.labels))
    print("shape of distances:",km.distances.shape)

    cen_guess=km.centers
    km=KMeans(cen_guess)

    km.run(X, maxiter=200)

    if not km.converged:
        cen_guess=km.centers
        km=KMeans(cen_guess)
        km.run(X, maxiter=500)

    centers=km.centers
    km=KMeans(centers)
    labels=km.find_nearest(X)

    jk_mask = mask
    jk_mask[mask==1] = labels+1
    if autoplot:
        plt.figure()
        hp.mollview(jk_mask,title='jackknife regions')
        filename = '%s/jk/figure/jk_regN%s_'%(loc, regN)+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.pdf'
        plt.savefig(filename)

        plt.figure()
        plt.hist(jk_mask[jk_mask>0], regN)
        filename = '%s/jk/figure/jk_hist_regN%s_'%(loc, regN)+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.pdf'
        plt.savefig(filename)

    print( '===> time to generate jackknife number = ', time.time()-t0)
    return jk_mask
def run_jk_number():
    import time
    t0=time.time()
    from multiprocessing import Pool
    from itertools import product
    pool3 = Pool(processes=3)
    jk_mask = np.ones((4, hp.nside2npix(nside)))
    for i in range(4):
        zmin = 0.1 + 0.2 *i
        zmax = 0.3 + 0.2 *i
        print '===> you are in redshift = ',i+1
        loc = '../cmb-cross-correlation'
        m = np.load('%s/mask/comb_mask_%sto%s_%s.npy'%(loc, zmin, zmax, nside))
        regN = 200
        jk_mask = jk_numbers(mask=m, nside=512, regN=regN, autoplot=True)
#        jk_mask[i] = pool3.apply_async(jk_numbers, (m, nside, regN))
        np.save('%s/jk/jk_mask_%sto%s_%s_regN%s.npy'%(loc, zmin, zmax, nside, regN), jk_mask)
#    np.save('%s/jk/jk_mask_%s_regN%s.npy'%(loc, nside, regN), jk_mask)
    print '==================time to run jk number is =', time.time()-t0
    return jk_mask

#run_jk_number()

# ======================= C_kg error estimation ==========================
def C_kg_err():	
    import time
    t0 = time.time()

    print( ' ======================= C_kg error estimation ==========================')
    Nnoise = 1000
    bias = np.array([1.0, 1.2, 1.3, 1.2])
    loc = '../cmb-cross-correlation'
    wn_flt = wiener_filter()   
#    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, Kmap, Nmap = load_CMB()
    #Nl_gg_allz = Nl_gg()
    Nl_gg_allz = np.load('/mnt/d2/public/zysun/cmb-cross-correlation/Nl_gg_allz.npy')
    clkg_th = np.ones((4, Nnoise, lmax+1))
    clkg_th_nomask = np.ones((4, Nnoise, lmax+1))
    clkg_th_noflt_nomask = np.ones((4, Nnoise, lmax+1))
    cls_clu_gg = np.ones((4, Nnoise, lmax+1))
    clgg_th = np.ones((4, Nnoise, lmax+1))
    C_KK = C_KN = np.ones((4, lmax+1))
    f_sky = np.ones(4)
    norm = (12*512**2/4./np.pi)
    for i in range(4):  # i-th redshift bin
        zmin = 0.1 + 0.2 *i
        zmax = 0.3 + 0.2 *i
        print( '===> you are in %s-th redshift bin'%(i+1))
        # load C_kg at i-th redshift bin with certain bias by pyccl package
        cls_cmbl_cross_clu = np.load('%s/Cl_th_pyccl/Cl_kg_th_%sto%s_b%s.npy'%(loc, zmin, zmax, bias[i]))
        cls_clu = np.load('%s/Cl_th_pyccl/Cl_gg_th_%sto%s_b%s.npy'%(loc, zmin, zmax, bias[i]))

        # combine mask
        mask = np.load('%s/mask/comb_mask_%sto%s_%s.npy'%(loc, zmin, zmax, nside))
        g_mask = np.load('/mnt/d2/public/zysun/cmb-cross-correlation/galaxy_data/g_mask_nside%s_%sto%s.npy'%(nside, zmin, zmax))
#        Kmap_ma = Kmap * mask
#        Nmap_ma = Nmap * mask
#        C_KK[i] = hp.anafast(Kmap_ma, lmax=lmax)
#        C_KN[i] = hp.anafast(Nmap_ma, lmax=lmax)
        f_sky[i] = np.sum(mask) / len(mask)
        print( '===> f_sky=',f_sky)

        for n in range(Nnoise):  # noise realization
            # ------------- generate the theoretical C_kg with filter and mask 
            noise = np.load('%s/noise/noise_No.%s.npy'%(loc, n+1))
            Nlm = hp.map2alm(noise, lmax=lmax)
            Nlm_1 = hp.almxfl(Nlm, np.sqrt(cls_cmbl_cross_clu))  # sqrt C_kg * Nlm
            T_n1 = hp.alm2map(Nlm_1, nside=nside, verbose=False)
            Nlm_2 = hp.almxfl(Nlm_1, np.sqrt(wn_flt))  # Nlm * wiener filter
            T_n = hp.alm2map(Nlm_2, nside = nside, verbose=False)
            T_ma = T_n * mask
#            # ------------- kappa-galaxy power spectrum
            clkg_th_noflt_nomask[i,n] = hp.anafast(T_n1, lmax=lmax)*norm
            clkg_th_nomask[i,n] = hp.anafast(T_n, lmax=lmax) * norm
            clkg_th[i,n] = hp.anafast(T_ma, lmax=lmax) * norm

#            # ----------- generate the theoretical C_gg only with galaxy mask
            noise_ma = hp.ma(noise)
            noise_ma.mask = np.logical_not(g_mask)
            Nlm_ma = hp.map2alm(noise_ma, lmax=lmax)
            cls_clu_alm = hp.almxfl(Nlm_ma, np.sqrt(cls_clu)) # theory C_gg times g_masked noise map
            cls_clu_gg[i,n] = hp.alm2cl(cls_clu_alm)*norm

            # C_kg err = sqrt( C_kg^2 + (C_k+C_k_N)*(C_g + C_g_N) / ((2l+1) * delta ell * f_sky) ) 
            # --------------- generate the theoretical C_gg with comb mask
            Nlm_3 = hp.almxfl(Nlm, np.sqrt(cls_clu))
            G_n = hp.alm2map(Nlm_3, nside=nside, verbose=False)  # galaxy map with Nlm and filter
            G_ma = G_n * mask
            clgg_th[i,n] = hp.anafast(G_ma, lmax=lmax) * norm
#    print 'Clkg_th_nomask',clkg_th_nomask
#    print 'Clkg_th_noflt_nomask',clkg_th_noflt_nomask
#    print 'Clkg_th',clkg_th
#    print 'Cls_clu_gg',cls_clu_gg
#    print 'clgg_th', clgg_th
    np.save('%s/Ckg_error/clkg_th_%s_'%(loc, Nnoise)+time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))+'.npy', clkg_th)
    np.save('%s/Ckg_error/clkg_th_nomask_%s_'%(loc, Nnoise)+time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))+'.npy', clkg_th_nomask)
    np.save('%s/Ckg_error/clkg_th_noflt_nomask_%s_'%(loc, Nnoise)+time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))+'.npy', clkg_th_noflt_nomask)
    np.save('%s/Ckg_error/cls_clu_gg_%s_'%(loc, Nnoise)+time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))+'.npy', cls_clu_gg)
    np.save('%s/Ckg_error/clgg_th_%s_'%(loc, Nnoise)+time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))+'.npy', clgg_th)
    #np.save('%s/Ckg_error/C_KK_'%(loc)+time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))+'.npy', C_KK)
    #np.save('%s/Ckg_error/C_KN_'%(loc)+time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))+'.npy', C_KN)

#    clkg_th = np.load('%s/Ckg_error/clkg_th_1000_2020-07-08-0523.npy'%loc)
    C_KK = np.load('%s/Ckg_error/C_KK_2020-07-08-0523.npy'%loc)
    C_KN = np.load('%s/Ckg_error/C_KN_2020-07-08-0523.npy'%loc) 

    # 1 realization error
    Ckg_err_1 = np.ones((4, Nbin))
    for i in range(4):
        err_1 = np.sqrt( clkg_th[i,0]**2 + (C_KK[i]+C_KN[i]) * (clgg_th[i,0]+Nl_gg_allz[i])) / np.sqrt(f_sky[i])
        Ckg_err_1[i] = C_err(err_1)
    print( 'Ckg_err_1', Ckg_err_1)
    np.save('%s/Ckg_error/Ckg_error_1_'%(loc)+time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))+'.npy',Ckg_err_1)  

    # 1000 realizations err
    Ckg_err_1000 = np.ones((4, Nbin))
    C_KK_1000 = C_KN_1000 = np.ones((4, Nnoise, lmax+1))
    for i in range(4):
        for n in range(Nnoise):
            C_KK_1000[i,n] = C_KK[i]
            C_KN_1000[i,n] = C_KN[i]
        err_1000 = np.std( np.sqrt(clkg_th[i]**2 + (C_KK_1000[i]+C_KN_1000[i])*(clgg_th[i]+ Nl_gg_allz[i])), axis=0 )
        Ckg_err_1000[i] = C_new_noco(err_1000)
    print ('Ckg_err_1000 = ', Ckg_err_1000)

    # jackknife err


    np.save('%s/Ckg_error/Ckg_error_%s_'%(loc, Nnoise)+time.strftime('%Y-%m-%d-%H%M', time.localtime(time.time()))+'.npy',Ckg_err_1000)
   
    print( '===> time to calculate Ckg error = ', time.time()-t0)
    print( '========================END Ckg ERROR PART=========================')
#C_kg_err()


#====================== METHOD 2: STANDARD ERROR ESTIMATION==========================
def Mask(Amap, m): 
    Amap_ma = hp.ma(Amap)
    Amap_ma.mask = np.logical_not(m)
    return Amap_ma
Nsam=1
def Method2_gal():
    import os
#    os.environ['OMP_NUM_THREADS']='8'
#    os.environ['MKL_NUM_THREADS'] = '1' 
#    os.environ['OPENBLAS_NUM_THREADS'] = '1' 

    import numpy as np
    import healpy as hp
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import time
    t0 =time.time()

    loc = '../cmb-cross-correlation'
    
    bias = np.array([1.0, 1.2, 1.3, 1.2])
    norm = (12*512**2/4./np.pi)
    print( '--------START GENERATE GALAXY AND GALAXY NOISE SAMPLES-------')
    galaxy_map = np.ones((4, Nsam, hp.nside2npix(nside)))
    galaxy_noise_map = np.ones((4, Nsam, hp.nside2npix(nside)))
    for i in range(4):  # i-th redshift bin
        zmin = 0.1 + 0.2 *i
        zmax = 0.3 + 0.2 *i
        print( '===> you are in redshift = ',i+1)
        cls_clu = np.load('%s/pyccl/Cl_gg_th_%sto%s_b%s.npy'%(loc, zmin, zmax, bias[i]))
        Nl_gg_allz = np.load('%s/galaxy_data/galaxy_noise_samples/Nl_gg_allz.npy'%loc)
	print 'cls_clu=', cls_clu
	print 'Nl_gg_allz=', Nl_gg_allz
        for n in range(Nsam):
            noise = np.load('%s/noise/noise_No.%s.npy'%(loc, n+1))
            Nlm = hp.map2alm(noise)
            alm_gg = hp.almxfl(Nlm, np.sqrt(cls_clu))    
            cgg = hp.alm2cl(alm_gg)*norm
	    print 'cgg=',cgg
            #galaxy_map[i,n] = hp.synfast(cgg, nside=nside, verbose=False)
            #galaxy_noise_map[i,n] = hp.synfast(np.ones(lmax+1)*Nl_gg_allz[i], nside=nside, verbose=False)

#    hp.mollview(galaxy_map[0,0], title='galaxy_map')
#    plt.savefig('%s/cov_matrix/gal_sam_'%loc+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.pdf')
#    hp.mollview(galaxy_noise_map[0,0], title='galaxy_noise_map')
#    plt.savefig('%s/cov_matrix/gal_noise_sam_'%loc+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.pdf')

    print( '------------END OF GALAXY SAMPLES------------')
    print( '===> time to generate galaxy samples is = ', time.time()-t0 )
    return galaxy_map + galaxy_noise_map
Method2_gal()


def Method2_len():
    import os
    import numpy as np
    import healpy as hp
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import time
    t0 =time.time()
    
    loc = '../cmb-cross-correlation'
    print( '---------START GENERATE KAPPA AND KAPPA NOISE SAMPLES-----------')
#    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, Kmap, Nmap = load_CMB()
#    ckk = hp.alm2cl(K_lm)
#    ckn = hp.alm2cl(N_lm)
    tmp = np.loadtxt('../planck/nlkk.dat')
    ell = np.array(tmp[:,0])
    N = np.array(tmp[:,1])
    C = np.array(tmp[:,2]-tmp[:,1])
   
    ksam = np.ones((4, Nsam, hp.nside2npix(nside)))
    knsam = np.ones((4, Nsam, hp.nside2npix(nside)))
    simfile = '/mnt/d2/public/zysun/planck/MV_sim_klm_000_029/MV/'
    for i in range(4):  # i-th redshift bin
        zmin = 0.1 + 0.2 *i
        zmax = 0.3 + 0.2 *i
        print( '===> you are in redshift = ',i+1)
        #mask = np.load('%s/mask/comb_mask_%sto%s_%s.npy'%(loc, zmin, zmax, nside))
        for n in range(Nsam):
            sim_klm = hp.read_alm('%s/sim_klm_%03d.fits'%(simfile, n))
	    #N_lm = hp.almxfl(sim_klm, N/(C+N))
	    K_lm = hp.almxfl(sim_klm, C/(C+N))

            ksam[i,n] = hp.alm2map(K_lm, nside=nside, verbose=None)
            #knsam[i,n] = hp.alm2map(N_lm, nside=nside, verbose=None)
            #ksam[i,n] = hp.synfast(ckk, nside=nside, verbose=False)
            #knsam[i,n] = hp.synfast(ckn, nside=nside, verbose=False)     
 
         #   ksam_ma[i,n] = Mask(ksam, mask)
         #   knsam_ma[i,n] = Mask(knsam, mask)
         
    print('---------END OF GENERATING KAPPA SAMPLES-----------')
    print( '===> time to generate kappa samples = ', time.time()-t0)
    return ksam #+ knsam
#Method2_len()

def Method2():
    import os
    #os.environ['OMP_NUM_THREADS']='6'
    #os.environ['MKL_NUM_THREADS'] = '6'
    #os.environ['OPENBLAS_NUM_THREADS'] = '6'
    import numpy as np
    import healpy as hp
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import time
    t0 =time.time()
    
    loc = '../cmb-cross-correlation'
    print( '===============START METHOD2============')
#    gsam = Method2_gal()
#    ksam = Method2_len()
#    print 'galaxy_map_ma.shape=',gsam.shape
#    print 'ksam_ma.shape=',ksam.shape
    Cov_Matrix = []; Err_Method2=[]
    err = np.ones((4, Nsam, lmax+1))
    for i in range(4):
        zmin = 0.1 + 0.2 *i
        zmax = 0.3 + 0.2 *i
        print( '===> you are in redshift = ',i+1)
        g_mask = np.load('%s/galaxy_data/g_mask/g_mask_nside%s_%sto%s.npy'%(loc, nside, zmin, zmax))
        mask = np.load('%s/mask/comb_mask_%sto%s_%s.npy'%(loc, zmin, zmax, nside))
        for n in range(Nsam):
            
            gsam_ma = Mask(gsam[i,n], g_mask)
            ksam_ma = Mask(ksam[i,n], mask)
            err[i,n] = np.sqrt(hp.anafast(gsam_ma, ksam_ma, lmax=lmax))
            
    print 'err=', err
    
    for i in range(4):
        cov_matrix = np.cov( err[i], rowvar=False)  
        print 'np.array(cov_matrix).shape=', cov_matrix.shape
        Cov_Matrix.append(cov_matrix)
        err_method2 = np.diag(cov_matrix)
        Err_Method2.append(err_method2)
    print 'Err_Method2=',Err_Method2
    np.save('%s/cov_matrix/Cov_Matrix_method2_'%loc+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.npy', Cov_Matrix)    
    np.save('%s/cov_matrix/Err_Method2_'%loc+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.npy', Err_Method2)
    print( '===============END OF START METHOD2 PROCESS============')
    print( '===> time to do method2 = ', time.time()-t0)
#Method2()

def preprocess():
    print('===========PREPROCESS==============')
    import time
    t0=time.time()
    import os
#    os.environ['OMP_NUM_THREADS']='1'
#    os.environ['MKL_NUM_THREADS'] = '1'
#    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    loc = '../cmb-cross-correlation'
    import numpy as np
    import fitsio as fio
    import time
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, Kmap, Nmap = load_CMB()

    for i in range(4):  # i-th redshift bin
        zmin = 0.1 + 0.2 *i
        zmax = 0.3 + 0.2 *i
        print( '===> you are in %s-th redshift bin'%(i+1))
         # read galaxy and target selection
        filename = '../cmb-cross-correlation/galaxy_%sto%s.fits'%(zmin, zmax)
       
#        filename = '/data/s6/zysun/cmb-cross-correlation/galaxy_%sto%s.fits'%(zmin, zmax)
        galaxy = load_saved_galaxy(filename)
        g_l, g_b = fk5_2_galactic( galaxy['RA'], galaxy['DEC'] )
        g_ra, g_dec, g_c, g_mean, colorpix, g_mask = pix_gal(g_l, g_b)
        colorpix = colorpix/g_mean

        mask = np.load('%s/mask/comb_mask_%sto%s_%s.npy'%(loc, zmin, zmax, nside))
        g_ra, g_dec, g_c = in_mask(nside, mask, g_ra, g_dec, g_c)
        K_ra, K_dec, K_val = in_mask(nside, mask, K_ra, K_dec, K_val)

        filename = '../cmb-cross-correlation/2B_correlated/2B_correlated_%sto%s.fits'%(zmin, zmax)
        try:
            os.remove(filename)
        except OSError:
            pass
        f = fio.FITS(filename,'rw')
        f.write( [g_ra, g_dec, g_c], names=['g_ra', 'g_dec', 'g_c'] )
        #f.write( [R_ra, R_dec, R_c], names=['R_ra', 'R_dec', 'R_c'] )
        f.write( [K_ra, K_dec, K_val], names=['K_ra', 'K_dec', 'K_val'] )
        f.close()
    print('===> time to preprocess = ', time.time()-t0)
    return g_ra, g_dec, g_c, K_ra, K_dec, K_val
#preprocess()


def assign_jk(jk_mask, ra, dec):
    import time 
    t0 = time.time()

    import os
#    os.environ['OMP_NUM_THREADS']='1'
#    os.environ['MKL_NUM_THREADS'] = '1'
#    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    import numpy as np
    import healpy as hp
    
    jk = np.zeros_like(ra)
    # assign each jackknife region
    regmax = int(max(jk_mask))
    nside = hp.get_nside(jk_mask)

    for i in range(1, regmax+1):
        hparr = hp.ma(jk_mask)
        m = np.arange(hp.nside2npix(nside))
        m = m[hparr==i]
        pix = hp.ang2pix(nside, np.pi/2.-np.deg2rad(dec), np.deg2rad(ra), nest=False)
        slt = np.in1d(pix, m, assume_unique=False)
        jk[slt] = i
        RAg = ra[jk]
    print( '===> time to assige jackknife region = ', time.time()-t0)
    return jk


def run_jk():
    import time
    t0 = time.time()
    loc = '../cmb-cross-correlation'
    
    JK_g=[]; JK_K=[]
    for i in range(4):
        zmin = 0.1 + 0.2 *i
        zmax = 0.3 + 0.2 *i
        print( '===> you are in %s-th redshift bin'%(i+1))
        filename = '../cmb-cross-correlation/2B_correlated/2B_correlated_%sto%s.fits'%(zmin, zmax)
        f = fio.FITS(filename)[1].read()
        g_ra = f['g_ra']
        g_dec = f['g_dec']
        g_c = f['g_c']

        f = fio.FITS(filename)[2].read()
        K_ra = f['K_ra']
        K_dec = f['K_dec']
        K_val = f['K_val']

        jk_mask = np.load('%s/jk/jk_mask_%sto%s_%s.npy'%(loc, zmin, zmax, nside))
        g_jk = assign_jk(jk_mask, g_ra, g_dec)
        K_jk = assign_jk(jk_mask, K_ra, K_dec)
        JK_g.append(g_jk)
        JK_K.append(K_jk)
    np.save('../cmb-cross-correlation/2B_correlated/g_jk_'+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time())), JK_g)
    np.save('../cmb-cross-correlation/2B_correlated/K_jk_'+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time())), JK_K)
    print( '===> time to run g_jk, K_jk = ', time.time()-t0)
    return JK_g, JK_K 

def method3():
    import numpy as np
    import healpy as hp
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt 
    import time
    t0=time.time()
    loc = '../cmb-cross-correlation'
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, Kmap, Nmap = load_CMB()
    regN = 200
    Ckg_obs = np.ones((4, regN, lmax+1)) # !!!!
    err_std = np.ones((4, lmax+1))
    err_cov = np.ones((4, lmax+1))
    
    for i in range(4):
	zmin = 0.1 + 0.2 *i
        zmax = 0.3 + 0.2 *i
        print( '===> you are in %s-th redshift bin'%(i+1))
        jk_mask = np.load('%s/jk/jk_mask_%sto%s_%s_regN%s.npy'%(loc, zmin, zmax, nside, regN))
        mask = np.load('%s/mask/comb_mask_%sto%s_%s.npy'%(loc, zmin, zmax, nside))
        overdensity_map = np.load('%s/overdensity_map/overdensity_map_%sto%s_%s.npy'%(loc, zmin, zmax, nside))
        regmax = int(max(jk_mask))
	
        for j in range(1, regmax+1):
       	    hparr = hp.ma(jk_mask)
            m = np.arange(hp.nside2npix(nside))
            m = m[hparr==j]
            jack = np.ones(hp.nside2npix(nside))
            jack[m] = 0
            total_mask = jack * mask
            g_ma = Mask(overdensity_map, total_mask)
            k_ma = Mask(Kmap, total_mask)
            g_lm = hp.map2alm(g_ma)
            k_lm = hp.map2alm(k_ma)
            Ckg_obs[i, j-1] = hp.alm2cl(g_lm, k_lm, lmax=lmax)
    	err_std[i] = np.sqrt(regmax-1) * np.std(Ckg_obs[i], axis=0)
        cov = np.cov(Ckg_obs[i], rowvar=False)
	cov_diag = np.diag(cov)
        err_cov[i] = np.sqrt(regmax-1) * cov_diag 
    np.save('%s/jk/Ckg_obs_jk%s.npy'%(loc, regmax), Ckg_obs) 
    np.save('%s/jk/jk_err_std%s.npy'%(loc, regmax), err_std)
    np.save('%s/jk/jk_err_cov%s.npy'%(loc, regmax), err_cov)  	
    print '===> time to get err via jk method = ', time.time()-t0
        

#method3()



def method3_cov():
    import numpy as np
    import healpy as hp
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt 
    import time
    t0=time.time()
    loc = '../cmb-cross-correlation'
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, Kmap, Nmap = load_CMB()
    Ckg_obs = np.ones((4, 50, lmax+1))
    err = np.ones((4, lmax+1))
    regmax=50
    Ckg_obs = np.load('%s/jk/Ckg_obs_jk50.npy'%(loc))
    for i in range(4):
        zmin = 0.1 + 0.2 *i
        zmax = 0.3 + 0.2 *i
        print( '===> you are in %s-th redshift bin'%(i+1))
        cov = np.cov(Ckg_obs[i], rowvar=False)
        print cov.shape
        cov_diag = np.diag(cov)
        err[i] = np.sqrt(regmax-1) * cov_diag 
    np.save('%s/jk/jk_err_jk_cov_%s_'%(loc, regmax)+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time())), err)
    print '===> time to get err via jk method = ', time.time()-t0

#method3_cov()
