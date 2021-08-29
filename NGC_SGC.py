import numpy as np
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import fitsio as fio

nside=512
lmax=1024
stage='NGC+SGC'
Nsam=240
Q = 12
z_range = np.linspace(0.1,0.9,5)
tSZ=False
def read_DESIDR8(stage=stage):
    import time 
    t0 = time.time()
    print('===> you are reading %s......'%stage)
    if stage == 'NGC':
        filename = '../DESIDR8/DESI_NGC_group'
        group = np.loadtxt(filename)
    elif stage == 'SGC':
        filename = '../DESIDR8/DESI_SGC_group'
        group = np.loadtxt(filename)
    elif stage == 'NGCtest':
         filename = 'DESIDR8/DESI_NGCtest_group'
         group = np.loadtxt(filename)
    elif stage == 'NGC+SGC':
        filename_NGC = '../DESIDR8/DESI_NGC_group'
        filename_SGC = '../DESIDR8/DESI_SGC_group'
        group_NGC = np.loadtxt(filename_NGC)
        group_SGC = np.loadtxt(filename_SGC)
        print('===> NGC group.shape = ',group_NGC.shape)
        print('===> SGC group.shape = ',group_SGC.shape)
        group = np.vstack((group_NGC, group_SGC))
        print('===> %s group.shape = '%stage,group.shape)

    print('===> time to loadtxt = ',time.time()-t0)
#     print '===> you are loading DESI group data......'
    id = group[:,0]
    Nmember = group[:,1]
    ra = group[:,2]
    dec = group[:,3]
    photo_z = group[:,4]
    M = group[:,5]
    L = group[:,6]
    print('===> %s group photo_z.shape ='%stage,photo_z.shape)

#     print '===> original max ra, min ra = ',np.max(ra),np.min(ra)
#     print '===> original max dec, min dec = ',np.max(dec),np.min(dec)
    return id,Nmember,ra,dec,photo_z,M,L

def select_DESIDR8(member,zmin,zmax):
    import time
    t0 = time.time()
    id,Nmember,ra,dec,photo_z,M,L = read_DESIDR8()
    print('================= %s<z<%s member>=%s==============='%(zmin,zmax,member))
    print ('===> original number of group = ',len(photo_z))
    slt = (Nmember>=member) & (zmin<photo_z) & (photo_z<=zmax)
    photo_z = photo_z[slt]
    ra = ra[slt]
    dec = dec[slt]
    Nmember = Nmember[slt]
    M = M[slt]
    L = L[slt]
    id = id[slt]
    print ('===> after select member, Ngroup = ',len(photo_z))
    print ('===> mean mass of halo = ', np.mean(M))
    print ('===> select max M, min M=',np.max(M),np.min(M))
    print ('===> select max L, min L=',np.max(L),np.min(L))
    print('===> time to selection = ',time.time()-t0)
    return id,Nmember,ra,dec,photo_z,M,L

#print ('you are selectiong......')
#id,Nmember,ra,dec,photo_z,M,L = select_DESIDR8(member=5,zmin=0.1,zmax=1.0)
#print ('you have selected!')
#path = '../desi_data/SELECTION/SELECT5_tsz/'
#filename = '%s/arr.fits'%(path)
#f = fio.FITS(filename,'rw')
#f.write( [id,Nmember,ra,dec,photo_z,M,L], names=['id','Nmember','ra','dec','photo_z','M','L'] )
#f.close()
#print ('you have saved fits!')

def select_DESIDR8_M(member,zmin,zmax,M_cut,lab):
    import time
    t0 = time.time()
    id,Nmember,ra,dec,photo_z,M,L = read_DESIDR8()
    print('================= %s<z<%s member>=%s==============='%(zmin,zmax,member))
    print ('===> original number of group = ',len(photo_z))
    if lab=='less':
        slt = (Nmember>=member) & (zmin<photo_z) & (photo_z<=zmax) & (M<M_cut)
    if lab=='moreeq':
        slt = (Nmember>=member) & (zmin<photo_z) & (photo_z<=zmax) & (M>=M_cut)
    photo_z = photo_z[slt]
    ra = ra[slt]
    dec = dec[slt]
    Nmember = Nmember[slt]
    M = M[slt]
    L = L[slt]
    id = id[slt]
    print ('===> after select member, Ngroup = ',len(photo_z))
    print ('===> select max z, min z=',np.max(photo_z),np.min(photo_z))
    print ('===> select max M, min M=',np.max(M),np.min(M))
    print ('===> select max L, min L=',np.max(L),np.min(L))
    print('===> time to selection = ',time.time()-t0)
    return id,Nmember,ra,dec,photo_z,M,L



def draw_member_scatter():
    import time
    t0 = time.time()
    photo_z,ra,dec,Nmember,M = read_DESIDR8()
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    density = ax.scatter(photo_z, Nmember, s=Nmember, alpha=0.5)
    fontsize=20
    plt.xlabel('z',fontsize=fontsize)
    plt.ylabel('richness',fontsize=fontsize)    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    print ('===> you are drawing the colorbar map...')
#    cbar = fig.colorbar(density)
#    ticklabs = cbar.ax.get_yticklabels()
#    cbar.ax.set_yticklabels(ticklabs, fontsize=18)
#    cbar.set_label(label='Number of member per group', size=15, weight='bold')
    fig.tight_layout()
    plt.savefig('../figure/member_scatter.pdf')
#draw_member_scatter()

def sort_rich(zmin,zmax):
    Ngal = 1e5
    photo_z,ra,dec,Nmember,M = read_DESIDR8()
    slt= (zmin<photo_z) & (photo_z<zmax)
    richness = Nmember[slt]
    rich_sort_index = np.argsort(-richness)
    rich_sort = richness[rich_sort_index]

    searching = int(Ngal)
    while rich_sort[searching] == rich_sort[searching+1]:
        print (searching, round(rich_sort[searching],2))
        searching=searching+1
    else:
        cut_left = searching+1
        print ('slice of cut=',rich_sort[0],rich_sort[searching])
        slt1 = rich_sort_index[:(searching+1)]
        z = photo_z[slt1]
        rich = richness[slt1]
        print ('===> the number of groups when slt Ngroup%s='%(Ngal),slt1.shape)
        searching += int(Ngal)

    while rich_sort[searching] == rich_sort[searching+1]:
        print (searching, round(rich_sort[searching],2))
        searching=searching+1
    else:
        cut_left2 = searching+1
        print ('slice of cut=',rich_sort[cut_left],rich_sort[searching])
        slt2 = rich_sort_index[cut_left:(searching+1)]
        z = photo_z[slt2]
        rich = richness[slt2]
        print ('===> the number of groups when slt Ngroup%s='%(Ngal),slt2.shape)
        searching += int(Ngal)

    while rich_sort[searching] == rich_sort[searching+1]:
        print (searching, round(rich_sort[searching],2))
        searching=searching+1
    else:
        cut_left3 = searching+1
        print ('slice of cut=',rich_sort[cut_left2],rich_sort[searching])
        slt3 = rich_sort_index[cut_left2:(searching+1)]
        z = photo_z[slt3]
        rich = richness[slt3]
        print ('===> the number of groups when slt Ngroup%s='%(Ngal),slt3.shape)
        searching += int(Ngal)
#sort_rich(zmin=0.3, zmax=0.5)



    
def Ell_new(Q):
    lmax=1024
    cut=10
    ell = np.arange(0.,lmax+1)
    smallell = ell[ell<=cut]; largeell=ell[ell>cut]
    largex=np.logspace(np.log10(largeell[0]), np.log10(largeell[-1]), Q+1)
    largex_int = np.ones(len(largex))
    for i in range(len(largex_int)):
        largex_int[i] = round(largex[i],0)
    new_ell=[]
    for i in range(len(largex)-1):
        new_ell.append( round((largex[i]+largex[i+1]) / 2, 0) )
    ell_new = np.hstack(new_ell)
    Nbin = len(ell_new)
#     print 'bin edge = ', largex_int
#     print 'ell_new = ', ell_new
#     print '===> Number of bins =',len(ell_new)
    return largex_int, ell_new, lmax
# Q=10
# largex_int, ell_new, lmax = Ell_new(Q=Q)

def C_new(cl,Q):
    largex_int, ell_new, lmax = Ell_new(Q)
    ell = np.arange(0.,lmax+1)
    bin_averages=[]
    for i in range(len(largex_int)-1):
        bin_averages.append(np.mean(cl[int(largex_int[i]):int(largex_int[i+1])]))
    C_new = bin_averages * ell_new*(ell_new+1)/(2.*np.pi) 
    return ell_new,C_new
def C_new_noco(cl,Q):
    largex_int, ell_new, lmax = Ell_new(Q)
    ell = np.arange(0.,lmax+1)
    bin_averages=[]
    for i in range(len(largex_int)-1):
        bin_averages.append(np.mean(cl[int(largex_int[i]):int(largex_int[i+1])]))
    return ell_new,bin_averages


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


def pix_gal( ra, dec, nside=nside):
    import time
    t0 = time.time()
    from pandas import value_counts
    import healpy as hp
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    pxl = hp.ang2pix(nside, np.pi/2.-np.deg2rad(dec), np.deg2rad(ra) )
    pxlc = value_counts(pxl)
    meanfield = np.mean(pxlc.values)
    pxlc = pxlc - meanfield
    print('===> g_mean = ',meanfield)
    [pxl_dec, pxl_ra] = np.rad2deg(hp.pix2ang(nside, pxlc.index))
    pxl_dec = 90.-pxl_dec
    g_mask = np.zeros(hp.nside2npix(nside))
    g_mask[pxlc.index] = 1
#     if autoplot: # plot the galaxy number over density per pixel
    colorpix = np.ones(hp.nside2npix(nside)) * (-meanfield)
    colorpix[pxlc.index] = pxlc.values
    #print '===> time to pixelize the galaxies = ', time.time()-t0
    return pxl_ra, pxl_dec, pxlc.values, meanfield, colorpix, g_mask



def pix_gal_number_count( ra, dec, nside=nside):
    import time
    t0 = time.time()
    from pandas import value_counts
    import healpy as hp
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt 
    pxl = hp.ang2pix(nside, np.pi/2.-np.deg2rad(dec), np.deg2rad(ra) )
    pxlc = value_counts(pxl)
    meanfield = np.mean(pxlc.values)
    print('===> g_mean = ',meanfield)
    print ('===> pxlc = ', pxlc)
    [pxl_dec, pxl_ra] = np.rad2deg(hp.pix2ang(nside, pxlc.index))
    pxl_dec = 90.-pxl_dec
    colorpix = np.zeros(hp.nside2npix(nside)) 
    colorpix[pxlc.index] = pxlc.values
    print ('===> time to pixelize the galaxies = ', time.time()-t0)
    return colorpix


def pixelize_group_number_count(ra,dec):
    print ('===> you are pixelizing the group data......')
#    photo_z,ra,dec,Nmember,M = select_DESIDR8(member,zmin,zmax)
    g_l, g_b = fk5_2_galactic(ra, dec)
    colorpix = pix_gal_number_count(g_l, g_b)
   
    return colorpix


def load_CMB2048(tSZ):
    nside=2048
    if tSZ:
        K_mask = hp.read_map('../planck/COM_Lensing_Szdeproj_4096_R3.00/mask.fits.gz', verbose=False)
        K_lm, mmax = hp.read_alm('../planck/COM_Lensing_Szdeproj_4096_R3.00/TT/dat_klm.fits', return_mmax=True)
        print ('you are loading CMB2048 map... ---> mmax = ',mmax)
    Kmap = hp.alm2map(K_lm, nside, verbose=False)
    K_dec, K_ra = np.rad2deg( hp.pix2ang(nside, range(hp.nside2npix(nside))) )
    K_dec = 90.-K_dec
    K_ra = K_ra[K_mask>0]
    K_dec = K_dec[K_mask>0]
    K_val = Kmap[K_mask>0]
    return K_ra, K_dec, K_val, K_mask, K_lm, Kmap

def load_CMB_lcut(lcut, nside):
    mask = hp.read_map('../planck/COM_Lensing_Szdeproj_4096_R3.00/mask.fits.gz', verbose=False)
    K_lm, mmax=hp.read_alm('../planck/COM_Lensing_Szdeproj_4096_R3.00/TT/dat_klm.fits',return_mmax=True)
    index=[]
    lmax=hp.Alm.getlmax(len(K_lm))
    for l in range(lcut, lmax+1):
        for m in range(l+1):
            idx = hp.Alm.getidx(lmax=lmax, l=l, m=m)
            index.append(idx)
    K_mask = hp.ud_grade(mask, nside)
    K_mask[K_mask<1] = 0
    K_lm[index]=0.+0j
    Kmap_lcut=hp.alm2map(K_lm, nside=nside)
    K_dec, K_ra = np.rad2deg( hp.pix2ang(nside, range(hp.nside2npix(nside))))
    K_dec = 90.-K_dec
    K_ra = K_ra[K_mask>0]
    K_dec = K_dec[K_mask>0]
    K_val = Kmap_lcut[K_mask>0]
    return K_ra, K_dec, K_val, K_mask, K_lm, Kmap_lcut


def load_CMB(tSZ, nside_dg=nside):
    from time import time
    t0 = time()
    import healpy as hp
    import numpy as np
    if tSZ:
        mask = hp.read_map('../planck/COM_Lensing_Szdeproj_4096_R3.00/mask.fits.gz' ,verbose=False)

        K_lm, mmax = hp.read_alm('../planck/COM_Lensing_Szdeproj_4096_R3.00/TT/dat_klm.fits',return_mmax=True)
        tmp = np.loadtxt('../planck/COM_Lensing_Szdeproj_4096_R3.00/TT/nlkk.dat')
    else:
        mask = hp.read_map('../planck/mask.fits', verbose=False)
        K_lm, mmax = hp.read_alm('../planck/dat_klm.fits', return_mmax=True)
        tmp = np.loadtxt('../planck/nlkk.dat')

    K_mask = hp.ud_grade(mask, nside_dg)  # downgrade CMB mask to fit galaxy mask
    K_mask[K_mask<1] = 0 
    ell = np.array(tmp[:,0])
    N = np.array(tmp[:,1])
    C = np.array(tmp[:,2]-tmp[:,1])
    N_lm = hp.almxfl(K_lm, N/(C+N))
    K_lm = hp.almxfl(K_lm, C/(C+N))
    Kmap = hp.alm2map(K_lm, nside=nside_dg, verbose=False)
    Nmap = hp.alm2map(N_lm, nside=nside_dg, verbose=False)

    K_dec, K_ra = np.rad2deg( hp.pix2ang( nside, range(hp.nside2npix(nside)) ) )
    K_dec = 90.-K_dec
# here was changed, mask -> K_mask 
    K_ra = K_ra[K_mask>0]
    K_dec = K_dec[K_mask>0]
    K_val = Kmap[K_mask>0]
    #print '===> time to load CMB kappa map = ', time()-t0
    return K_ra, K_dec, K_val, K_mask, K_lm, N_lm, Kmap, Nmap



def Kmap_lcut(lcut):
    mask = hp.read_map('../planck/COM_lensing_Szdeproj_4096_R3.00/mask.fits.gz', verbose=False)
    K_lm, mmax = hp.read_alm('../planck/COM_Lensing_Szdeproj_4096_R3.00/TT/dat_klm.fits', return_mmax=True)



def in_mask( nside, mask, ra, dec, field ):
    t0 = time.time()

    # fit into mask
    hparr = hp.ma(mask)
    m = np.arange(hp.nside2npix(nside))
    m = m[hparr>0]
    pix = hp.ang2pix( nside, np.pi/2.-np.deg2rad(dec), np.deg2rad(ra), nest=False )
    slt = np.in1d(pix, m, assume_unique=False)

    ra = ra[slt]
    dec = dec[slt]
    field = field[slt]

    print ('===> time to fit into mask = ', time.time()-t0)
    return ra, dec, field




def pixelize_group(ra,dec):
    print ('===> you are pixelizing the group data......')
#    photo_z,ra,dec,Nmember,M = select_DESIDR8(member,zmin,zmax)
    g_l, g_b = fk5_2_galactic(ra, dec)
    g_ra, g_dec, g_c, g_mean, colorpix, g_mask = pix_gal(g_l, g_b)
    colorpix = colorpix/g_mean
    return colorpix, g_mask


def put_mask(Amap,mask):
    map_ma = hp.ma(Amap)
    map_ma.mask = np.logical_not(mask)
    alm = hp.map2alm(map_ma)
    return alm 
def Mask(Amap, m):
    Amap_ma = hp.ma(Amap)
    Amap_ma.mask = np.logical_not(m)
    return Amap_ma
def get_fsky(mask):
    f_sky = np.sum(mask) / np.array(mask).shape
    return f_sky
def Nlgg(g_mask,mask):
    f_sky = get_fsky(mask)
    n_bar = np.sum(g_mask) / (4*np.pi * f_sky)
    Nl_gg = 1./n_bar
    return Nl_gg
def fsky_after_pixelization(member,zmin,zmax):
    colorpix, g_mask = pixelize_group(member,zmin,zmax)
    print ('===> you are loading CMB ......')
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB()
    mask = g_mask * K_mask
    fsky_after_pixelization = get_fsky(mask)
    print ('===> fsky=',fsky_after_pixelization)
    return fsky_after_pixelization



def get_volume(member,zmin,zmax):
    import numpy as np
    import matplotlib.pyplot as plt
    from hmf import cosmo

#     total_area = 4*np.pi /(np.pi/180)**2
#     solid_angle = 200 / total_area # no unit
#     print ('===> solid angle = ',solid_angle)
    
    solid_angle = fsky_after_pixelization(member,zmin,zmax)
    print ('===> solid angle = ',solid_angle)

    my_cosmo = cosmo.Cosmology(cosmo_model=cosmo.Planck15)
    h0 = my_cosmo.cosmo.H0/100
#     print (h0)
    d_cmin = my_cosmo.cosmo.comoving_distance(zmin)      # 
    d_cmax = my_cosmo.cosmo.comoving_distance(zmax)      # 
#     print (d_cmin,d_cmax)
    V = 1./3 *4*np.pi * solid_angle * (d_cmax**3 - d_cmin**3)   
    print ('===> comoving volume in %s<z<%s I calculate = '%(zmin,zmax),V)
#     V_cmin = my_cosmo.cosmo.comoving_volume(zmin)       # 
#     V_cmax = my_cosmo.cosmo.comoving_volume(zmax) 
#     V_c = (V_cmax - V_cmin) * solid_angle
#     print ('===> comoving volume in %s<z<%s pakage calculate ='%(zmin,zmax),V_c)
    return V

def mass_function(member,zmin,zmax):
    N_bin = 30
    fontsize=20
    photo_z,ra,dec,Nmember,M = select_DESIDR8(member,zmin,zmax)
    N_group=len(photo_z)
    counts, bins = np.histogram(M,N_bin)  #  bins = bin_edges_new
    print('===> count=',counts)
    hist, bin_edges, patches = plt.hist(bins[:-1], bins, weights=counts/(bins[2]-bins[1]),color='orange',edgecolor='white',linewidth=2,alpha=0.9)
    median=[]
    for i in range(len(bins)-1):
        med =( bins[i]+bins[i+1])/2
        median.append(med)
    median=np.array(median)
    V = get_volume(member,zmin,zmax)
    dndlnm = hist / V
    error = dndlnm / np.sqrt(counts)
    error = np.array(error)
    print ('===> dndlnm / sqrtN = ',dndlnm / np.sqrt(counts))
    m = 10**median     
    return M,m,dndlnm,N_group,error

def plot_mass_function():
    import time
    t0 = time.time()
    fontsize=20
    alpha=1
    fit = 'Tinker08'
    z_range = np.linspace(0.2,1,5)
    fig,axs = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(15,10),sharey=False)
    for ax,i in zip(axs.flat,range(4)):
        zmin=round(z_range[i],1)
        zmax=round(z_range[i+1],1)
        zmean=str(round(np.mean([zmin,zmax]),1))
#         # theory Tinker08
#         x,y = theory_hmf(zmean)
#         ax.plot(x,y,'k--',lw=3,label='Theory: Tinker08, z=%s'%zmean)
        loc = '../cmb-cross-correlation/SMT_integral'
        ymean = np.load('%s/%s_dndlnm_integral_z%s.npy'%(loc,fit,zmean),)
#        ymid = np.load('%s/%s_dndlnm_mid_z%s.npy'%(loc,fit,zmean),)
        x = np.load('%s/%s_m_integral.npy'%(loc,fit),)
#         ax.loglog(x,ymid,'k:',lw=3, label='%s, z=%s'%(fit,zmean),alpha=alpha)
        ax.loglog(x,ymean,'k--',lw=3, label='%s, integral'%fit,alpha=alpha)
        
#         # theory SMT
#         x,y = hmf_SMT(zmean)
#         ax.plot(x,y,'k--',lw=3,label='Theory: SMT, z=%s'%zmean)
        
        M,m,dndlnm,N_group,error = mass_function(0,zmin,zmax)
        ax.errorbar(np.array(m),np.array(dndlnm),yerr=error,c='k',lw=3,capsize=5,label='%s group'%(stage))
        np.save('../desi_data/%s_dndm/%s_m_%sto%s.npy'%(stage,stage,zmin,zmax),np.array(m))
        np.save('../desi_data/%s_dndm/%s_dndlnm_%sto%s.npy'%(stage,stage,zmin,zmax),np.array(dndlnm))
        np.save('../desi_data/%s_dndm/%s_error_%sto%s.npy'%(stage,stage,zmin,zmax),error)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, linestyle='--',lw=2)
        ax.set_title('%s<z<%s'%(zmin,zmax),fontsize=fontsize)
        ax.set_xlim(10**min(M),10**max(M))
        ax.set_ylim(1e-9,1e-1)
        ax.set_xlabel(r'$M_{group} [M_{\odot}/h]$',fontsize=fontsize)
        ax.set_ylabel(r'$\left( \frac{dn}{d\ln M} \right) h^3 Mpc^{-3}$',fontsize=fontsize)
        ax.legend(fontsize=14,loc=1)
        ax.tick_params(labelsize=fontsize)
#     legend.get_frame().set_facecolor('none')
    plt.savefig('../figure/dndlnm_%s_Tinker08_integral.pdf'%(stage))
    print('===> time to get mass function = ',time.time()-t0)
#plot_mass_function()


def hmf(zmean,fit):
    import hmf 
    h = hmf.MassFunction(z = zmean,Mmax=16, hmf_model=fit)
    m = h.m
    dndm = h.dndlnm
    print('===> you get hmf %s, z = %s'%(fit,zmean))
    return m,dndm


def hmf_integral():
    fit = 'Tinker08'
    fontsize=20
    alpha=0.7
    Nz=21
    z_range = np.linspace(0.2,1,5)
    fig,axs = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(15,10),sharey=False)
    for ax,i in zip(axs.flat,range(4)):
        zmin=round(z_range[i],1)
        zmax=round(z_range[i+1],1)
        zmean=str(round(np.mean([zmin,zmax]),1))
        zrange = np.linspace(zmin,zmax,Nz)
        Y = []
        for j in range(len(zrange)):
            z = round(zrange[j],2)
            x,y = hmf(z, fit=fit)
            Y.append(y)
        ymean = np.mean(Y,axis=0)
        xmid, ymid = hmf_SMT(zmean)
        
        ax.loglog(x,ymid,'b',lw=3, label='%s, z=%s'%(fit,zmean),alpha=alpha)
        ax.loglog(x,ymean,'r',lw=3, label='%s, integral'%fit,alpha=alpha)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, linestyle='--',lw=2)
        ax.set_title('%s<z<%s'%(zmin,zmax),fontsize=fontsize)
        ax.set_xlim(1e12,1e15)
        ax.set_ylim(2e-9,1e-2)
        ax.set_xlabel(r'$M_{group} [M_{\odot}/h]$',fontsize=fontsize)
        ax.set_ylabel(r'$\left( \frac{dn}{d\ln M} \right) h^3 Mpc^{-3}$',fontsize=fontsize)
        ax.legend(fontsize=fontsize,loc=1)
        ax.tick_params(labelsize=fontsize)
        legend = plt.legend(fontsize=18,loc=1)
        legend.get_frame().set_facecolor('none')
        loc = 'cmb-cross-correlation/NGC_test_group/SMT_integral'
        np.save('%s/%s_dndlnm_integral_z%s.npy'%(loc,fit,zmean),ymean)
        np.save('%s/%s_dndlnm_mid_z%s.npy'%(loc,fit,zmean),ymid)
    np.save('%s/%s_m_integral.npy'%(loc,fit),x)
    plt.savefig('figure/%s_hmf.pdf'%(fit))
#hmf_integral()

def sub_mean_field():
#    loc = '../planck/COM_Lensing-SimMap_4096_R3.00/MV'
    loc = '../planck/COM_Lensing-SimMap_Szdeproj_4096_R3.00/TT' #Szdeproj
    from time import time
    t0 = time()
    import healpy as hp
    import numpy as np
    N_meanfield = 60
    K_lm_mf = np.ones((N_meanfield, 8394753),dtype=complex)
    for i in range(N_meanfield):
        K_lm_mf[i] = hp.read_alm('%s/sim_klm_%03d.fits'%(loc, i+240))
        print ('===> reading alm in sub mean field =', i)
    mean_field = np.mean(K_lm_mf, axis=0)
    np.save('%s/K_lm_mean_field_N%s.npy'%(loc, N_meanfield), mean_field)
    print ('===> number of calculating mean field = ',N_meanfield)
    print ('===> Klm mean field = ',mean_field)
#    K_lm = hp.read_alm('%s/sim_klm_%03d.fits'%(loc, j+N_meanfield)) - mean_field
    return K_lm_mf
#sub_mean_field()

def load_CMB_sim(Nsim,tSZ,nside_dg=nside, nside=nside):
    t0 = time.time()
    N_meanfield = 60
    if tSZ:
        loc = '../planck/COM_Lensing-SimMap_Szdeproj_4096_R3.00/TT' #Szdeproj
#        tmp = np.loadtxt('%s/nlkk.dat'%(loc))
        tmp = np.loadtxt('../planck/COM_Lensing_Szdeproj_4096_R3.00/TT/nlkk.dat')
        K_lm = hp.read_alm('%s/sim_klm_%03d.fits'%(loc, Nsim))
#        mean_field=hp.read_alm('../planck/COM_Lensing_Szdeproj_4096_R3.00/TT/mf_klm.fits')
        print ('load CMB sim : sim_klm_%03d.fits'%(Nsim))
    else:
        loc = '../planck/COM_Lensing-SimMap_4096_R3.00/MV'
        tmp = np.loadtxt('../planck/nlkk.dat')
        K_lm = hp.read_alm('%s/sim_klm_%03d.fits'%(loc, Nsim+N_meanfield))
        print ('load CMB sim : sim_klm_%03d.fits'%(Nsim+N_meanfield))
    mean_field = np.load('%s/K_lm_mean_field_N%s.npy'%(loc,N_meanfield))
    K_lm = K_lm - mean_field
    ell = np.array(tmp[:,0])
    N = np.array(tmp[:,1])
    C = np.array(tmp[:,2]-tmp[:,1])
    K_lm = hp.almxfl(K_lm, C/(C+N))
    Kmap = hp.alm2map(K_lm, nside=nside, verbose=False)
    return Kmap

def get_clkg(colorpix,g_mask,K,K_mask,get_cls=True,get_err_gg=False,get_err_Kg=True):
#    colorpix, g_mask = pixelize_group(zmin,zmax)
#    print ('=============== get cl %s<z<%s =============='%(zmin,zmax))
#    loc_pic = '../desi_data/%s_maps'%(stage)
#    colorpix = np.load('%s/colorpix_%s_group_%sto%s.npy'%(loc_pic,stage,zmin,zmax))
#    g_mask = np.load('%s/g_mask_%s_group_%sto%s.npy'%(loc_pic,stage,zmin,zmax))

#    print ('===> you are loading CMB ......')
#    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB()
    mask = g_mask * K_mask
    print ('===> fsky=',get_fsky(mask))
#    Nl_gg = Nlgg(g_mask,mask)
#    print ('===> Nlgg = ',Nl_gg)
    if get_cls:
        K_lm = put_mask(K,mask)
        g_lm = put_mask(colorpix,mask)
        C_Kg = hp.alm2cl(K_lm, g_lm)
#        C_gg = hp.alm2cl(g_lm)
    if get_err_gg:
        print ('======= error gg ==========')
        clsgg_mtd2 = np.ones((Nsam,len(C_gg)))
        for n in range(Nsam):
            print ('Nsam=',n)
            gsam = hp.synfast(C_gg, nside=nside, verbose=False)
            gsam_ma = Mask(gsam, mask)
            clsgg_mtd2[n] = hp.anafast(gsam_ma)
        print ('===> clsgg_mtd2 = ',clsgg_mtd2)
        print ('======= End of error gg ==========')
    
    if get_err_Kg:
        clskg_mtd2 = np.ones((Nsam,len(C_Kg)))
        print ('======= error Kg ==========')
        for n in range(Nsam):
            print ('Nsam=',n)
#            gsam = hp.synfast(C_gg, nside=nside, verbose=False)
#            gsam_ma = Mask(gsam, mask)
            gsam_ma = Mask(colorpix, mask)
            ksam = load_CMB_sim(n)      # load sim from 60-299 
            ksam_ma = Mask(ksam, mask)
            clskg_mtd2[n] = hp.anafast(gsam_ma, ksam_ma)
        print ('===> clskg_mtd2 = ',clskg_mtd2)
        print ('======= End of error kg ==========')
#    return C_Kg, C_gg, Nl_gg, clsgg_mtd2, clskg_mtd2
    return C_Kg, clskg_mtd2

def plot_cl():
    fontsize=20
    largex_int, ell_new, lmax = Ell_new(Q=Q)
    ell=np.arange(lmax+1)
    print('===> ell_new = ',ell_new)   
    z_range = np.linspace(0.1,0.9,5)
    fig,axs = plt.subplots(nrows=2, ncols=2,constrained_layout=True,figsize=(15,10),sharey=False)
    for ax,i in zip(axs.flat,range(4)):
        zmin=round(z_range[i],1)
        zmax=round(z_range[i+1],1)
        zmean=str(round(np.mean([zmin,zmax]),1))
        Nm = np.array([0,2,4,6])
        color = np.array(['k','r','g','b'])
        for m in range(len(Nm)):
            C_Kg, clskg_mtd2 = get_clkg(Nm[m],zmin,zmax,get_err_gg=False,get_err_Kg=True)
#            Cgg_NGC_new = C_new(C_gg,Q)
            Ckg_new = C_new(C_Kg,Q)
            Ckg_noco = C_new_noco(C_Kg, Q)
#            err_gg = np.std(clsgg_mtd2,axis=0)
#            err_gg_bin = C_new(err_gg,Q)[1]
#            print ('===> err_gg = ',err_gg_bin)
            err_kg = np.std(clskg_mtd2,axis=0)
            err_kg_bin = C_new(err_kg,Q)[1]
            print ('===> err_kg = ',err_kg_bin)
            np.save('../desi_data/%s_clkg/clkg_new_%s_%sto%s_Nm%s_Nsam%s.npy'%(stage,stage,zmin,zmax,Nm[m],Nsam), Ckg_new)
            np.save('../desi_data/%s_clkg/clkg_noco_%s_%sto%s_Nm%s_Nsam%s.npy'%(stage,stage,zmin,zmax,Nm[m],Nsam), Ckg_noco)
            np.save('../desi_data/%s_clkg/err_kg_bin_%s_%sto%s_Nm%s_Nsam%s.npy'%(stage,stage,zmin,zmax,Nm[m],Nsam), err_kg_bin)
#            ax.errorbar(Cgg_NGC_new[0],Cgg_NGC_new[1],c='k',marker='o',yerr=err_gg_bin,markersize=10,label='$\mathcal{C}_\ell^{gg}$',lw=3,capsize=5)
            ax.errorbar(Ckg_new[0],Ckg_new[1],c=color[m],marker='s',yerr=err_kg_bin,markersize=10,label='$N_{member}>%s$'%Nm[m],lw=3,capsize=5)
    #     plt.plot(ell, C_gg*ell*(ell+1)/2./np.pi,label='Cgg_NGC')
    #     plt.plot(ell, C_Kg*ell*(ell+1)/2./np.pi,label='Ckg_NGC')
    #     plt.loglog(ell,Nl_gg*ell*(ell+1)/2./np.pi,'grey',label='shot noise')
        ax.grid(True, linestyle='--',lw=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e1,1e3)
        ax.set_title('%s %s<z<%s'%(stage,zmin,zmax),fontsize=fontsize)
        ax.set_xlabel(r'$\ell$',fontsize=fontsize)
        ax.set_ylabel(r'$\frac{\ell(\ell+1)} {2\pi}\mathcal{C}_\ell$', fontsize = fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.legend(fontsize=14,loc=1,ncol=2)
    plt.tight_layout()
    plt.savefig('../figure/Clkg_%s_Nsam%s_'%(stage,Nsam)+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.pdf')

#plot_cl()

'''
===================================== SNR ================================= 
'''

def get_covariance(slt,key):
    import time
    t0 = time.time()
    Q = 12
    Nsam = 240
    fontsize=20
    alpha=0.7
    color = np.array(['k','orange','g','b'])
    z_range = np.linspace(0.1,0.9,5)
    member_cut = np.array([0,2,4,6])
    mass_cut = np.array([11,12.5,13.0,13.5])
    cov = np.ones((4, len(member_cut), Q, Q))
    cov_noco = np.ones((4, len(member_cut), Q, Q))
    r = np.ones((4, len(member_cut), Q, Q))
    r_noco = np.ones((4, len(member_cut), Q, Q))
    cls_new = np.ones((Nsam, Q))
    cls_noco = np.ones((Nsam,Q))
    for i in range(4):
        zmin=round(z_range[i],1)
        zmax=round(z_range[i+1],1)
        for j in range(len(member_cut)):
            member = member_cut[j]
            clskg_mtd2 = np.load('../desi_data/select_%s_cl/clskg_mtd2_%sto%s_%s%s_Nsam%s.npy'%(slt, zmin,zmax,key,member,Nsam))
            print ('clskg_mtd2.shape = ',clskg_mtd2.shape)
            for n in range(Nsam):
                cls_new[n] = C_new(clskg_mtd2[n],Q)[1]
                cls_noco[n] = C_new_noco(clskg_mtd2[n],Q)[1]
                print ('cls_new = ',cls_new)
            cov[i,j] = np.cov(cls_new,rowvar=False)
            cov_noco[i,j] = np.cov(cls_noco, rowvar=False)
            r[i,j] = np.corrcoef(cls_new,rowvar=False)
            r_noco[i,j] = np.corrcoef(cls_new,rowvar=False)
    print ('===> cov = ',cov)
    np.save('../desi_data/select_%s_cl/cov_Nsam%s.npy'%(slt,Nsam),cov)
    np.save('../desi_data/select_%s_cl/r_Nsam%s.npy'%(slt,Nsam),r)
    print ('===> time to get covariance = ',time.time()-t0)
    return cov_noco

def SNR(slt = 'member',key='Nm'):
    import time
    t0 = time.time()
    from numpy import linalg as LA
    Nsam = 240
    member_cut = np.array([0,2,4,6])
    cov = get_covariance(slt,key)
#    cov = np.load('../desi_data/select_%s_cl/cov_Nsam%s.npy'%(slt,Nsam))
    right = np.ones((4,len(member_cut)))
    for i in range(4):
        zmin=round(z_range[i],1)
        zmax=round(z_range[i+1],1)
        for j in range(len(member_cut)):
            member = member_cut[j]
            Ckg_new = np.load('../desi_data/select_%s_cl/clkg_new_%sto%s_%s%s_Nsam%s.npy'%(slt,zmin,zmax,key,member,Nsam))
            kg = Ckg_new[1]
            print ('kg %s<z<%s, member>%s= '%(zmin,zmax,member),kg)
            print ('kg.shape = ',kg.shape)
            kg_T = kg.reshape(-1,1)
            c = cov[i,j]
            cov_inv = LA.inv(c)
            print ('===> inv cov = ',cov_inv)
            left = np.dot(kg, cov_inv)
            print('===>left=',left)
            right[i,j] = np.dot(left, kg_T)
            print ('===>right=',right)
    print ('===> SNR for member cut in %s<z<%s = '%(zmin,zmax),right)
    SNR_eachz = np.ones(4)
    for i in range(4):
        each_z = right[i]
        SNR_eachz[i] = np.sqrt(np.sum(each_z**2))
        print ('===> SNR_z= %s = '%i, SNR_eachz)
    SNR = np.sqrt(np.sum(SNR_eachz**2))
    print ('===> SNR = ', SNR)
    print ('===> time to get SNR = ',time.time()-t0)
#SNR()

'''
============================= theory kg ==============================
'''
def wiener_filter():
    import numpy as np
    import healpy as hp
    tmp = np.loadtxt('../planck/nlkk.dat')

    ell = np.array(tmp[:,0])
    N = np.array(tmp[:,1])
    C = np.array(tmp[:,2]-tmp[:,1])
    wn_flt = C / (C+N)
    return wn_flt
#wn_flt = wiener_filter()

def Cl_th_pyccl(filename,i, bias):
    import numpy as np
    import pyccl as ccl
    import fitsio as fio
    import matplotlib.pyplot as plt
#    filename = '../desi_data/nz/group_nz_sigmaz0.052020-11-20-1254.fits'
    f = fio.FITS(filename)[-1].read()
    z = f['z']
    n_all = f['n_all']
    plt.figure(figsize=(10,7))
    Q = 12
    nz = f['n_%s'%(i+1)]   # i-th redshift bin in redshift distribution
    cosmo = ccl.Cosmology(Omega_c = 0.120/0.674**2, Omega_b = 0.0224/0.674**2, h = 0.674, sigma8 = 0.811, n_s = 0.965)
    b = bias * np.ones(len(z))
    cmbl = ccl.CMBLensingTracer(cosmo, 1090.) 
    clu = ccl.NumberCountsTracer(cosmo, has_rsd = False, dndz = (z, nz), bias=(z,b)) 
    ell = np.arange(3*512)
    cls_cmbl_cross_clu = ccl.angular_cl(cosmo, cmbl, clu, ell)
#    cls_cmbl = ccl.angular_cl(cosmo, cmbl, cmbl, ell)
#    cls_clu = ccl.angular_cl(cosmo, clu, clu, ell)
    zmin = 0.1 + i * 0.2
    zmax = 0.3 + i * 0.2  
    print (zmin,zmax)
    return cls_cmbl_cross_clu


def total_masking(i,slt = 'mass', key='M'):
    import time
    t0 = time.time()
    Nsam = 240
    mass_cut = np.array([11,12.5,13.0,13.5])
    member_cut = np.array([0,2,4,6])
    loc_pic = '../desi_data/select_%s_colorpix'%(slt)
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB()
    zmin=round(z_range[i],1)
    zmax=round(z_range[i+1],1)
#    for j in range(len(member_cut)):
    member = member_cut[0]
    mass = mass_cut[0]
#    g_mask = np.load('%s/g_mask_group_%sto%s_%s%s.npy'%(loc_pic,zmin,zmax,key,member))
    g_mask = np.load('%s/g_mask_group_%sto%s_%s%s.npy'%(loc_pic,zmin,zmax,key,mass))
    mask = g_mask*K_mask
#    np.save('%s/mask_group_%sto%s_%s%s.npy'%(loc_pic,zmin,zmax,key,member), mask)
    np.save('%s/mask_group_%sto%s_%s%s.npy'%(loc_pic,zmin,zmax,key,mass), mask)
    fsky = np.sum(mask) / len(mask)
    print ('===> %s<z<%s, member>%s fsky = '%(zmin,zmax,member),fsky)
    return mask













