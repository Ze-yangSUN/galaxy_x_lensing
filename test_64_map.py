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

t0 = time.time()
fontsize=20
alpha=0.7
z_range = np.linspace(0.1,0.9,5)
member_cut = np.array([0,2,4,6])
mass_cut = np.array([11,12.5,13.0,13.5])
color = np.array(['k','orange','g','b'])
print ('member cut = ',member_cut)


def load_saved_galaxy(filename):

    from time import time 
    t0 = time()

    import fitsio as fio
    galaxy = fio.FITS(filename)[-1].read()

    print ('===> time to load galaxies = ', time()-t0)
    return galaxy


new_nside=512
def pixelize_gal( ra, dec, nside=new_nside):
    import time
    t0 = time.time()
    from pandas import value_counts
    import healpy as hp
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt 
    pxl = hp.ang2pix(nside, np.pi/2.-np.deg2rad(dec), np.deg2rad(ra) )
    pxlc = value_counts(pxl)
    print ('min pxlc=',min(pxlc.values))
    meanfield = np.mean(pxlc.values)
    print('===> g_mean = ',meanfield)
    [pxl_dec, pxl_ra] = np.rad2deg(hp.pix2ang(nside, pxlc.index))
    pxl_dec = 90.-pxl_dec
#    g_mask = np.zeros(hp.nside2npix(nside))
#    g_mask[pxlc.index] = 1 
#     if autoplot: # plot the galaxy number over density per pixel
    colorpix = np.zeros(hp.nside2npix(nside))
    colorpix[pxlc.index] = pxlc.values
    #print '===> time to pixelize the galaxies = ', time.time()-t0
    return pxl_ra, pxl_dec, pxlc.values, meanfield, colorpix



def test_64():
    colorpix_out = np.zeros((4, len(member_cut), hp.nside2npix(new_nside)))
    G_mean = np.zeros((4, len(member_cut)))
    for i in range(4):
        zmin=round(z_range[i],1)
        zmax=round(z_range[i+1],1)
        for j in range(len(member_cut)):
            print('member=',j)
            member = member_cut[j]
            ra,dec,M = load_group_select_member(zmin,zmax,member)
            print('===> len(ra)=',len(ra))
            g_l, g_b = fk5_2_galactic(ra, dec)
            g_ra, g_dec, g_c, g_mean, colorpix = pixelize_gal(g_l, g_b)
            print ('===> gmean = ',g_mean)
            colorpix_out[i,j] = colorpix
            G_mean[i,j] = g_mean
            loc_pic = '../desi_data/test_64'
    np.save('%s/colorpix_group_allz_512.npy'%(loc_pic), colorpix_out)     
    np.save('%s/g_mean_group_allz_512.npy'%(loc_pic), G_mean)
#test_64()


def fi_mask():
    fimap = np.load('../desi_data/survey_mask/fimap_ud512_thrh0.5.npy')
    b_mask = cm.make_mask_total(512,b_mask = True, b_deg_min = -31, b_deg_max = 25)+0. #mask the middle 
    fi = b_mask * fimap # 0.5~1
    g_mask=np.zeros_like(fi)
    g_mask[fi>0]=1
    return fi,g_mask



def total_mask():
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB()
    g_fi, g_mask = fi_mask()
    mask = g_mask * K_mask
    return mask



def counts_galaxy(ra,dec):
    ra,dec = fk5_2_galactic(ra, dec)
    pxl = hp.ang2pix(nside, np.pi/2.-np.deg2rad(dec), np.deg2rad(ra) )
    print ('===>pxl.shape=',pxl.shape)
    pixel = pd.DataFrame({'pixel':pxl})
    f = pd.DataFrame({'pixel': pxl,
                      'counts':pxl})
    F = f.groupby('pixel',as_index=False)['counts'].agg('count')

    pixel = F.pop('pixel')
    counts = F.pop('counts')

    colorpix = np.zeros(hp.nside2npix(nside))
    colorpix[pixel] = counts
    return colorpix
def run_counts_galaxy():
    COLORPIX = np.zeros((4, hp.nside2npix(nside)))
    for i in range(4):
        zmin=round(z_range[i],1)
        zmax=round(z_range[i+1],1)
        print('===================== z=%s===================='%(i))
        filename = '../cmb-cross-correlation/galaxy_%sto%s.fits'%(zmin, zmax)
        galaxy = load_saved_galaxy(filename)
        ra,dec=galaxy['RA'], galaxy['DEC']
        cmap = counts_galaxy(ra,dec)
        COLORPIX[i]=cmap
    loc = '../desi_data/galaxy'
    np.save('%s/galaxy_number_counts.npy'%loc,COLORPIX)
    return COLORPIX
#run_counts_galaxy()
def modified_galaxy():
    from NPTFit import create_mask as cm
    loc = '../desi_data/galaxy'
    test64 = np.load('%s/galaxy_number_counts.npy'%(loc))
    
    fi,g_mask = fi_mask() # 0.5~1
    mask = total_mask()
    ans = np.where(fi!=0)
    Modified = np.zeros((4, hp.nside2npix(nside=512)))
    for i in range(4):
        print ('z=',i)
        zmin=round(z_range[i],1)
        zmax=round(z_range[i+1],1)
        density = test64[i]*g_mask
        mod_mean = np.sum(density) / np.sum(fi)
        density[ans] = density[ans] / fi[ans]
        print ('mod_mean=',mod_mean)
        Modified[i] = (density / mod_mean -1) * mask 
    np.save('%s/overdensity_modified_allz_512_totalmask.npy'%(loc), Modified)
    return Modified
#modified_galaxy()
def clkg_gal():
    kg = np.ones((4, 3*512))
    gg = np.ones((4, 3*512))
    kg_bin = np.ones((4, 12))
    gg_bin = np.ones((4, 12))
    loc = '../desi_data/galaxy'
    Modified = np.load('%s/overdensity_modified_allz_512_totalmask.npy'%(loc))
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB()
    fi, g_mask = fi_mask()
    mask = g_mask*K_mask
    for i in range(4):
        print ('z=',i)
        kg[i] = hp.anafast(K*mask, Modified[i]*mask)
        kg_bin[i] = C_new(kg[i],Q=12)[1]
        gg[i] = hp.anafast(Modified[i]*mask)
        gg_bin[i] = C_new(gg[i],Q=12)[1]
    print ('kg_bin=',kg_bin)
    print ('gg_bin=',gg_bin)
    loc = '../desi_data/galaxy'
    np.save('%s/clkg_totalmask.npy'%(loc),kg)
    np.save('%s/clgg_totalmask.npy'%(loc),gg)
    np.save('%s/clkg_bin_totalmask.npy'%(loc),kg_bin)
    np.save('%s/clgg_bin_totalmask.npy'%(loc),gg_bin)
#clkg_gal()
def gal_kgerr():
    loc = '../desi_data/galaxy'
    Modified = np.load('%s/overdensity_modified_allz_512_totalmask.npy'%(loc))
    Nsim=240
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB()
    clskg_mtd2 = np.ones((4, Nsim, lmax+1))
    fi, g_mask = fi_mask()
    mask = total_mask()
    print ('======= error Kg ==========')
    for i in range(4):
        zmin=round(z_range[i],1)
        zmax=round(z_range[i+1],1)
        print('z=%s'%(i))
        gsam_ma = Modified[i]*mask
        for n in range(Nsim):
            print ('Nsam=',n)
            ksam = load_CMB_sim(n)      # load sim from 60-299 
            ksam_ma = ksam *mask     # K_mask is obs mask, not sim mask
            clskg_mtd2[i,n] = hp.anafast(gsam_ma, ksam_ma, lmax=lmax)
    print ('===> clskg_mtd2 = ',clskg_mtd2)
    print ('======= End of error kg ==========')
    np.save('%s/clskg_totalmask.npy'%(loc),clskg_mtd2)
    return clskg_mtd2
#gal_kgerr()





def w_map(ra,dec,M):
    print ('MAX M, MIN M = ',10**max(M),10**min(M))
    w = 10**M / 10**13
    print('===> w = ',w)
    print('===> w.shape = ',w.shape) 
    print('=== max w, min w, mean w, med w = ',max(w),min(w),np.mean(w),np.median(w))
    ra,dec = fk5_2_galactic(ra, dec)
    pxl = hp.ang2pix(nside, np.pi/2.-np.deg2rad(dec), np.deg2rad(ra) ) 
    print ('===>pxl.shape=',pxl.shape)
    pixel = pd.DataFrame({'pixel':pxl})
    f = pd.DataFrame({'pixel': pxl,
                      'counts':pxl,
                      'weight': w})
    F = f.groupby('pixel',as_index=False)['counts'].agg('count')
    A = f.groupby(['pixel'],as_index=False)['weight'].agg({'weight':np.sum})
    mean_weight = A.pop('weight')
    F['mean_weight'] = mean_weight

    pixel = F.pop('pixel')
    counts = F.pop('counts')
    mean_weight = F.pop('mean_weight')

    meanfield = np.mean(counts)
    frac_density = counts / meanfield

    colorpix = np.zeros(hp.nside2npix(nside))
    wmap = np.zeros(hp.nside2npix(nside))
    fracmap = np.zeros(hp.nside2npix(nside))
    colorpix[pixel] = counts
    wmap[pixel]= mean_weight
    fracmap[pixel]=frac_density

    return colorpix, wmap



def mass_weighted_map():
    COLORPIX = np.zeros((4, len(member_cut), hp.nside2npix(nside)))
    WMAP = np.zeros((4, len(member_cut), hp.nside2npix(nside)))
    for i in range(4):
        zmin=round(z_range[i],1)
        zmax=round(z_range[i+1],1)
        for j in range(len(member_cut)):
            print('===================== z=%s, member=%s ===================='%(i,j))
            member = member_cut[j]
            ra,dec,photo_z,M,Nmember = load_group_select_member(zmin,zmax,member)

#            mass = mass_cut[j]
#            ra,dec,photo_z,M,Nmember = load_group_select_mass(zmin,zmax,mass)

            cmap, wmap = w_map(ra,dec,M)
            COLORPIX[i,j]=cmap
            WMAP[i,j]=wmap
    loc = '../desi_data/membercut_map'
    np.save('%s/number_count_modified_allz_512_masscut.npy'%loc,COLORPIX)
    np.save('%s/w_map_allz_512_masscut.npy'%loc, WMAP)
#mass_weighted_map()


def mass_weighted_map_2to3(cut):
    COLORPIX = np.zeros((len(member_cut), hp.nside2npix(nside)))
    WMAP = np.zeros((len(member_cut), hp.nside2npix(nside)))
    for j in range(len(member_cut)):
        print('===================== member=%s ===================='%(j))
#        member = member_cut[j]
#        ra,dec,photo_z,M,Nmember = load_group_select_member(0.2,0.3,member)

        mass = mass_cut[j]
        ra,dec,photo_z,M,Nmember = load_group_select_mass(0.2,0.3,mass)

        cmap, wmap = w_map(ra,dec,M)
        COLORPIX[j]=cmap
        WMAP[j]=wmap
    loc = '../desi_data/%scut_map'%cut
    np.save('%s/number_count_modified_0.2to0.3_512.npy'%loc,COLORPIX)
    np.save('%s/w_map_0.2to0.3_512.npy'%loc, WMAP)
#mass_weighted_map_2to3(cut='mass')
def modified_2to3(cut):
    from NPTFit import create_mask as cm
    loc = '../desi_data/%scut_map'%cut
    test64 = np.load('%s/number_count_modified_0.2to0.3_512.npy'%(loc))
    wmap = np.load('%s/w_map_0.2to0.3_512.npy'%(loc))
        
    fi,g_mask = fi_mask() # 0.5~1
    mask = total_mask()
    ans = np.where(fi!=0)
    Modified = np.zeros((len(member_cut), hp.nside2npix(nside=512)))
    for j in range(len(member_cut)):
        print('j=',j)
        density = test64[j]*g_mask
#        density = wmap[j]*g_mask  # number count map*mass-weighted map
        mod_mean = np.sum(density) / np.sum(fi)
        density[ans] = density[ans] / fi[ans]
        print ('mod_mean=',mod_mean)
        Modified[j] = (density / mod_mean -1)
    np.save('%s/overdensity_modified_0.2to0.3_512_gmask.npy'%(loc), Modified)
#    np.save('%s/overdensity_modified_mass_weight_0.2to0.3_512_mask.npy'%(loc), Modified)
    return Modified
#modified_2to3(cut='mass')
def clkg_2to3(cut,lab):
    kg = np.ones((len(member_cut), 3*512))
    gg = np.ones((len(member_cut), 3*512))
    kg_bin = np.ones((len(member_cut), 12))
    gg_bin = np.ones((len(member_cut), 12))
    loc = '../desi_data/%scut_map'%cut
    Modified = np.load('%s/overdensity_modified_0.2to0.3_512_gmask.npy'%(loc))
#    Modified = np.load('%s/overdensity_modified_mass_weight_0.2to0.3_512_mask.npy'%(loc))
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB()
    fi, g_mask = fi_mask()
    mask = g_mask*K_mask
    for j in range(len(member_cut)):
        print('member=',j)
        kg[j] = hp.anafast(K*mask, Modified[j]*mask)
        kg_bin[j] = C_new(kg[j],Q=12)[1]
        gg[j] = hp.anafast(Modified[j]*mask)
        gg_bin[j] = C_new(gg[j],Q=12)[1]
    print ('kg_bin=',kg_bin)
    print ('gg_bin=',gg_bin)
    loc = '../desi_data/%scut'%cut
    np.save('%s/clkg_%scut_%s_0.2to0.3.npy'%(loc,cut,lab),kg)
    np.save('%s/clgg_%scut_%s_0.2to0.3.npy'%(loc,cut,lab),gg)
    np.save('%s/clkg_bin_%scut_%s_0.2to0.3.npy'%(loc,cut,lab),kg_bin)
    np.save('%s/clgg_bin_%scut_%s_0.2to0.3.npy'%(loc,cut,lab),gg_bin)
#clkg_2to3(cut='mass',lab='')
def kgerr_2to3(cut,lab):
    loc = '../desi_data/%scut_map'%cut
    Modified = np.load('%s/overdensity_modified_0.2to0.3_512_gmask.npy'%(loc))
#    Modified = np.load('%s/overdensity_modified_mass_weight_0.2to0.3_512_mask.npy'%(loc))
    Nsim=240
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB()
    clskg_mtd2 = np.ones((len(member_cut), Nsim, lmax+1))
    fi, g_mask = fi_mask()
    mask = total_mask()
    print ('======= error Kg ==========')
    for j in range(len(mass_cut)):
        print('%s=%s,lab=%s'%(cut,j,lab))
        gsam_ma = Modified[j]*mask
        for n in range(Nsim):
            print ('Nsam=',n)
            ksam = load_CMB_sim(n)      # load sim from 60-299 
            ksam_ma = ksam * mask     # K_mask is obs mask, not sim mask
            clskg_mtd2[j,n] = hp.anafast(gsam_ma, ksam_ma, lmax=lmax)
    print ('===> clskg_mtd2 = ',clskg_mtd2)
    print ('======= End of error kg ==========')
    loc = '../desi_data/%scut'%cut
    np.save('%s/clskg_%s%scut_0.2to0.3.npy'%(loc,lab,cut),clskg_mtd2)
    return clskg_mtd2
kgerr_2to3(cut='mass',lab='')




def modified_test(cut):
    from NPTFit import create_mask as cm
    loc = '../desi_data/%scut_map'%cut
    test64 = np.load('%s/number_count_modified_allz_512_%scut.npy'%(loc,cut))
    wmap = np.load('%s/w_map_allz_512_%scut.npy'%(loc,cut))
    
    fi,g_mask = fi_mask() # 0.5~1
    mask = total_mask()
    ans = np.where(fi!=0)
    Modified = np.zeros((4, len(member_cut), hp.nside2npix(nside=512)))
    for i in range(4):
        print ('z=',i)
        zmin=round(z_range[i],1)
        zmax=round(z_range[i+1],1)
        for j in range(len(member_cut)):
            print('j=',j)
            density = test64[i,j]*g_mask
#            density = wmap[i,j]*g_mask  # number count map*mass-weighted map
            mod_mean = np.sum(density) / np.sum(fi)
            density[ans] = density[ans] / fi[ans]
            print ('mod_mean=',mod_mean)
            Modified[i,j] = (density / mod_mean -1) 
    np.save('%s/overdensity_modified_allz_512_%scut_gmask.npy'%(loc,cut), Modified)
#    np.save('%s/overdensity_modified_mass_weight_allz_512_%scut_gmask.npy'%(loc,cut), Modified)
    return Modified
#Modified = modified_test(cut='mass')




def clkg_test(cut,lab):
    kg = np.ones((4, len(member_cut), 3*512))
    gg = np.ones((4, len(member_cut), 3*512))
    kg_bin = np.ones((4, len(member_cut), 12))
    gg_bin = np.ones((4, len(member_cut), 12))
    loc = '../desi_data/%scut_map'%cut
    Modified = np.load('%s/overdensity_modified_mass_weight_allz_512_%scut_gmask.npy'%(loc,cut))
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB()
    fi, g_mask = fi_mask()
    mask = g_mask*K_mask
    for i in range(4):
        print ('z=',i)
        for j in range(len(member_cut)):
            print('member=',j)
            kg[i,j] = hp.anafast(K*K_mask, Modified[i,j])
            kg_bin[i,j] = C_new(kg[i,j],Q=12)[1]
            gg[i,j] = hp.anafast(Modified[i,j]*mask)
            gg_bin[i,j] = C_new(gg[i,j],Q=12)[1]
    print ('kg_bin=',kg_bin)
    print ('gg_bin=',gg_bin)
    loc = '../desi_data/%scut_eachmask'%cut
    np.save('%s/clkg_%scut_%s.npy'%(loc,cut,lab),kg)
    np.save('%s/clgg_%scut_%s.npy'%(loc,cut,lab),gg)
    np.save('%s/clkg_bin_%scut_%s.npy'%(loc,cut,lab),kg_bin)
    np.save('%s/clgg_bin_%scut_%s.npy'%(loc,cut,lab),gg_bin)

#clkg_test(cut='mass',lab='w')



def kgerr(cut,lab='mass_weight_'):
#    Modified = np.load('../desi_data/%scut_map/overdensity_modified_%sallz_512_%scut_gmask.npy'%(cut,lab,cut))  # respective mask
    Modified = np.load('../desi_data/%scut_map/overdensity_modified_%sallz_512_%scut.npy'%(cut,lab,cut))  # total mask
    Nsim=240
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB()
    clskg_mtd2 = np.ones((4, len(member_cut), Nsim, lmax+1))
    fi, g_mask = fi_mask()
    mask = total_mask()
    print ('======= error Kg ==========')
    for i in range(4):
        zmin=round(z_range[i],1)
        zmax=round(z_range[i+1],1)
        for j in range(len(mass_cut)):
            print('z=%s,%s=%s,lab=%s'%(i,cut,j,lab))
            gsam_ma = Modified[i,j]*mask  
#            gsam_ma = Modified[i,j]   # overdensity map already has its mask
            # if change g map
            #cgg = hp.anafast(colorpix)
            for n in range(Nsim):
                print ('Nsam=',n)
                ksam = load_CMB_sim(n)      # load sim from 60-299 
                ksam_ma = ksam * mask     # K_mask is obs mask, not sim mask
                # if change g map
#                gsam = hp.synfast(cgg, nside=nside, verbose=False)                
#                gsam_ma = gsam * g_mask
                clskg_mtd2[i,j,n] = hp.anafast(gsam_ma, ksam_ma, lmax=lmax)
    print ('===> clskg_mtd2 = ',clskg_mtd2)
    print ('======= End of error kg ==========')
    loc = '../desi_data/%scut'%cut
    np.save('%s/clskg_%s%scut.npy'%(loc,lab,cut),clskg_mtd2)
    return clskg_mtd2
#kgerr(cut='member')





def ggerr(C_gg):
    Nsim=200
    fi,g_mask = fi_mask() 
    clsgg_mtd2 = np.ones((Nsim, 3*nside))
    for n in range(Nsim):
        print ('Nsim=',n)
        gsam = hp.synfast(C_gg, nside=nside, verbose=False)
        gsam_ma = gsam * g_mask
        clsgg_mtd2[n] = hp.anafast(gsam_ma)
    print ('===> clsgg_mtd2 = ',clsgg_mtd2)
    print ('======= End of error gg ==========')
    return clsgg_mtd2


def cal_ggerr(gg):
#    gg=np.load('../desi_data/test_64/clgg.npy')
#    gg=np.load('../desi_data/test_64/clgg_mass_weight.npy')
# masscut
#    gg = np.load('../desi_data/masscut/clgg_masscut.npy')
#    gg = np.load('../desi_data/masscut/clgg_mass_weight_masscut.npy')
    print ('gg[0,0]',gg[0][0])
    Nsim=200
    Clsgg_mtd2 = np.ones((4, len(member_cut), Nsim, 3*nside))
    for i in range(4):
        zmin=round(z_range[i],1)
        zmax=round(z_range[i+1],1)
        for j in range(len(mass_cut)):
            print('z=%s,mass=%s'%(i,j))
            mass = mass_cut[j]
            C_gg = gg[i][j]  
            clsgg_mtd2 = ggerr(C_gg)
            Clsgg_mtd2[i,j] = clsgg_mtd2 
    np.save('../desi_data/masscut/clsgg_mtd2_512_allz_Nsim%s_masscut.npy'%(Nsim),Clsgg_mtd2)
#    np.save('../desi_data/masscut/clsgg_mtd2_512_allz_Nsim%s_mass_weight_masscut.npy'%(Nsim),Clsgg_mtd2)

#cal_ggerr(gg = np.load('../desi_data/masscut/clgg_masscut.npy'))
#cal_ggerr(gg = np.load('../desi_data/masscut/clgg_mass_weight_masscut.npy'))



def plot_M_Nmember(zmin,zmax,member=0,plot_yerr=False,plot_xerr=True):
    t0=time.time()
    fontsize=20
    ra,dec,photo_z,M,Nmember = load_group_select_member(zmin,zmax,member)
    plt.figure(figsize=(8,6))
    plt.grid(True,lw=1)
    plt.scatter(Nmember[:],M[:],c='steelblue', marker='.',s=5, alpha=0.1)
    if plot_yerr:
        MEMBER = np.array([1,2,3,5,7,10,14,20,30,40,50,70,100,140,200,300])
        MEAN = np.ones(len(MEMBER))
        STD = np.ones(len(MEMBER))
        for i in range(len(MEMBER)):
            MASS = M[Nmember==MEMBER[i]]
            MEAN[i] = np.mean(MASS)
            STD[i] = np.std(MASS)
            plt.text(MEMBER[i], MEAN[i], '%s'%round(MEAN[i],1), color='firebrick',ha='left',va='top')
        print ('MEAN IF MEMBER=%s'%(MEMBER[i]), MEAN)
        print ('STD IF MEMBER=%s'%(MEMBER[i]), STD)
        plt.errorbar(MEMBER, MEAN, yerr = STD, c='firebrick',marker='.',lw=1,capsize=5, label='mean')

    if plot_xerr:
        MASS = np.array([11.5,12.,12.5,13.,13.5,14.0,14.5,15.])
        MEAN = np.ones(len(MASS))
        STD = np.ones(len(MASS))
        for i in range(len(MASS)):
            MEMBER = Nmember[M==MASS[i]]
            MEAN[i] = np.mean(MEMBER)
            STD[i] = np.std(MEMBER)
            plt.text(MEAN[i], MASS[i],'%s'%round(MEAN[i],1), color='firebrick',ha='left',va='top')
        print ('MEAN IF MASS=%s'%(MASS[i]), MEAN)
        print ('STD IF MASS=%s'%(MASS[i]), STD)
        plt.errorbar(MEAN,MASS, xerr = STD, c='firebrick',marker='.',lw=1,capsize=5)
    
    plt.xlabel('$N_{member}$',fontsize=fontsize)
    plt.ylabel('log [$M_{h}/(h^{-1}M_{\odot})$]',fontsize=fontsize)
    plt.xscale('log')
    plt.xlim(None,700)
    plt.ylim(11,16)
    plt.legend()
    print ('===>min Nm=',min(Nmember),',max Nm=',max(Nmember))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title('%s<z<%s'%(zmin,zmax),fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('../figure/M_Nmember/M_Nmember_%sto%s_xerr.png'%(zmin,zmax),dpi=300)
    print ('===> time to plot scatter=', time.time()-t0)
#plot_M_Nmember(0.1,0.3)
#plot_M_Nmember(0.3,0.5)
#plot_M_Nmember(0.5,0.7)
#plot_M_Nmember(0.7,0.9)



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
    cls_clu = ccl.angular_cl(cosmo, clu, clu, ell)
    return cls_cmbl_cross_clu, cls_clu



def kg_th_galaxy():
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB()
    mask = total_mask()
    t0 = time.time()
    Nnoise=50
    wn_flt = wiener_filter()
    kg_th = np.ones((4, lmax+1))
    gg_th = np.ones((4, lmax+1))
    norm = (12*512**2/4./np.pi)
    cls_cmbl_cross_clu = np.ones((4,3*512))
    cls_clu = np.ones((4,3*512))
    for i in range(4):
        zmin = round(0.1 + i * 0.2,1)
        zmax = round(0.3 + i * 0.2,1)
        print ('=================== %s<z<%s =================='%(zmin,zmax))
        filename='../desi_data/nz/galaxy_nz_2020-11-20-1552.fits'
        cls_cmbl_cross_clu[i], cls_clu[i] = Cl_th_pyccl(filename, i=i, bias=1.)
        kg = np.ones((Nnoise, lmax+1))
        gg = np.ones((Nnoise, lmax+1))
        for n in range(Nnoise):
            print ('noise=',n)
            noise = np.load('../cmb-cross-correlation/noise/noise_No.%s.npy'%(n+1))
            Nlm = hp.map2alm(noise)

            Nlm1 = hp.almxfl(Nlm, np.sqrt(cls_cmbl_cross_clu[i]))
            Nlm2 = hp.almxfl(Nlm1, np.sqrt(wn_flt))
            maps = hp.alm2map(Nlm2,nside=nside,verbose=False)
            maps_ma = maps * mask
            kg[n] = hp.anafast(maps_ma, lmax=lmax)*norm

        kg_th[i] = np.mean(kg, axis=0)
        gg_th[i] = np.mean(gg, axis=0)
    print ('kg_th = ',kg_th)
    print ('gg_th = ',gg_th)
    print ('===> time to get kg_th,gg_th = ',time.time()-t0)
    np.save('../desi_data/kg_th/theory/gal_kg_th.npy',kg_th)
#kg_th_galaxy()




def kg_th(filename,sigmaz,Nm):
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB()
    mask = total_mask()
    t0 = time.time()
    Nnoise=50
    wn_flt = wiener_filter()
    kg_th = np.ones((4, lmax+1))
    gg_th = np.ones((4, lmax+1))
    norm = (12*512**2/4./np.pi)
    b1 = np.load('../desi_data/b/b1.npy')
    b2 = np.load('../desi_data/b/b2.npy')
    cls_cmbl_cross_clu = np.ones((4,3*512))
    cls_clu = np.ones((4,3*512))
    for i in range(4):
        zmin = round(0.1 + i * 0.2,1)
        zmax = round(0.3 + i * 0.2,1)
        print ('=================== %s<z<%s, sigmaz=%s, Nm=%s =================='%(zmin,zmax,sigmaz,Nm))
        cls_cmbl_cross_clu[i], cls_clu[i] = Cl_th_pyccl(filename, i=i, bias=1.)
        np.save('../desi_data/kg_th/theory/cls_cmbl_cross_clu_sigmaz%s_M%s.npy'%(sigmaz,Nm),cls_cmbl_cross_clu)
        np.save('../desi_data/kg_th/theory/cls_clu_sigmaz%s_M%s.npy'%(sigmaz,Nm),cls_clu)
        kg = np.ones((Nnoise, lmax+1))
        gg = np.ones((Nnoise, lmax+1))
        for n in range(Nnoise):
            print ('noise=',n)
            noise = np.load('../cmb-cross-correlation/noise/noise_No.%s.npy'%(n+1))
            Nlm = hp.map2alm(noise)
            
            Nlm1 = hp.almxfl(Nlm, np.sqrt(cls_cmbl_cross_clu[i]))
            Nlm2 = hp.almxfl(Nlm1, np.sqrt(wn_flt))
            maps = hp.alm2map(Nlm2,nside=nside,verbose=False)
            maps_ma = maps * mask
            kg[n] = hp.anafast(maps_ma, lmax=lmax)*norm

            Nlmg = hp.almxfl(Nlm, np.sqrt(cls_clu[i]))            
            mapsg = hp.alm2map(Nlmg,nside=nside,verbose=False)
            mapsg_ma = mapsg * mask
            gg[n] = hp.anafast(mapsg_ma, lmax=lmax)*norm

        kg_th[i] = np.mean(kg, axis=0)
        gg_th[i] = np.mean(gg, axis=0)
    print ('kg_th = ',kg_th)
    print ('gg_th = ',gg_th)
    print ('===> time to get kg_th,gg_th = ',time.time()-t0)
    np.save('../desi_data/kg_th/theory/kg_th_sigmaz%s_M%s.npy'%(sigmaz,Nm),kg_th)
    np.save('../desi_data/gg_th/theory/gg_th_sigmaz%s_M%s.npy'%(sigmaz,Nm),gg_th)


#kg_th()
#kg_th(filename='../desi_data/nz/group_nz_sigmaz0.02_2020-12-15-1919.fits',sigmaz=0.02)
#kg_th(filename='../desi_data/nz/group_nz_sigmaz0.08_2020-12-15-1919.fits',sigmaz=0.08)
#kg_th(filename='../desi_data/nz/group_nz_sigmaz0.008_2020-12-15-1920.fits',sigmaz=0.008)
#kg_th(filename='../desi_data/nz/group_nz_sigmaz0.01_2020-12-15-2014.fits',sigmaz=0.01)
#kg_th(filename='../desi_data/nz/group_nz_sigmaz0.052020-11-20-1254.fits',sigmaz=0.05)


#for a in range(4):
#    member = member_cut[a]
#    kg_th(filename='../desi_data/nz/group_nz_sigmaz0.01_Nm%s.fits'%(member),sigmaz=0.01,Nm=member)
#    kg_th(filename='../desi_data/nz/group_nz_sigmaz0.05_Nm%s.fits'%(member),sigmaz=0.05,Nm=member)

#    mass = mass_cut[a]
#    kg_th(filename='../desi_data/nz/group_nz_sigmaz0.05_M%s.fits'%(mass),sigmaz=0.05,Nm=mass)





def W_k(z):
    from astropy.cosmology import Planck15 as cosmo
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


def b1_b2(loc):
    import time
    from colossus.lss import bias
    from colossus.cosmology import cosmology
    cosmology.setCosmology('planck18')
    cosmo = cosmology.getCurrent()
    halobias=np.ones(4)
    fontsize=20
    plt.figure(figsize=(10,7))
    b1 = np.ones((4,4))
    b2 = np.ones((4,4))
    b3 = np.ones((4,4))
    b_Wk1 = np.ones((4,4))
    b_Wk2 = np.ones((4,4))
    for i in range(4):
        zmin=round(z_range[i],1)
        zmax=round(z_range[i+1],1)
        z_mean = (zmin+zmax)/2
        for j in range(4):
#            member=member_cut[j]
#            print ('========== %s<z<%s member>%s ========='%(zmin,zmax,member))
#            ra,dec,photo_z,M,Nmember = load_group_select_member(zmin,zmax,member)

            mass=mass_cut[j]
            print ('========== %s<z<%s mass>%s ========='%(zmin,zmax,mass))
            ra,dec,photo_z,M,Nmember = load_group_select_mass(zmin,zmax,mass)

            M = 10**M
            b = bias.haloBias(M, model = 'sheth01', z = photo_z)
            b = b**2  # gg
            W_L, comoving_d = W_k(z = photo_z)
            b1[i,j] = sum(b) / len(M)
            b2[i,j] = sum(b*M) / sum(M)
            b3[i,j] = np.mean(b*M) / np.mean(b) / np.mean(M) 
            print ('===> you are calculate b_Wk......')            
            b_Wk1[i,j] = sum(b * W_L * comoving_d) / sum(W_L * comoving_d)
            b_Wk2[i,j] = sum(b * W_L * comoving_d * M) / sum(W_L * comoving_d * M)
            
    print ('===> b1=',b1)
    print ('===> b2=',b2)
    print ('===> b3=',b3)
    print ('===> b_Wk1 = ',b_Wk1)
    print ('===> b_Wk2 = ',b_Wk2)
    np.save('%s/b1.npy'%loc,b1)
    np.save('%s/b2.npy'%loc,b2)
    np.save('%s/b_Wk1.npy'%loc,b_Wk1)
    np.save('%s/b_Wk2.npy'%loc,b_Wk2)
#b1_b2(loc = '../desi_data/b/b_gg/masscut')







def corr( g_ra,g_dec,g_c, K_ra,K_dec,K_val, type, autoplot=False ):
    t0 = time.time()

    dataK = tc.Catalog( k=K_val, ra=K_ra, dec=K_dec, ra_units='deg', dec_units='deg' )
    datag = tc.Catalog( k=g_c, ra=g_ra, dec=g_dec, ra_units='deg', dec_units='deg' )

    Kg = tc.KKCorrelation( nbins=15, min_sep=10., max_sep=300., bin_slop=0.01, verbose=0, sep_units='arcmin' )
    Kg.process(dataK,datag,metric='Arc')

    if type == 'KK':
        theta = Kg.meanr
        xi = Kg.xi
        err = np.sqrt(Kg.varxi)
        Npairs = Kg.npairs
    elif type == 'Kg':

        norm_g = tc.NKCorrelation( nbins=15, min_sep=10., max_sep=300., bin_slop=0.01, verbose=0, sep_units='arcmin' )
        norm_g.process(dataK,datag,metric='Arc')

        theta = Kg.meanr

        xi = Kg.xi * 1.
        slt = norm_g.xi == 0
        xi[slt] = 0.
        slt = norm_g.xi != 0
        xi[slt] = Kg.xi[slt] / norm_g.xi[slt]

        err = xi * np.sqrt(Kg.varxi/Kg.xi**2 + norm_g.varxi/norm_g.xi**2)
        Npairs = Kg.npairs * norm_g.xi

    # plot correlation
    if autoplot:
#         plt.figure()
#         plt.errorbar(theta,xi*np.sqrt(theta),err*np.sqrt(theta),fmt='o',label=r'$<\kappa g>$')
#         plt.plot(theta,np.zeros_like(theta),':')
#         plt.semilogx()
#         plt.xlabel(r'$\theta$')
#         plt.ylabel(r'$\xi(\theta)\times\sqrt{\theta}$')
#         plt.legend()
        theta3, KK_xi, KK_err, Npairs = corr(K_ra,K_dec,K_val,K_ra,K_dec,K_val, type='KK')
        plt.figure()
        plt.errorbar( theta3, KK_xi*theta3, KK_err*theta3, fmt='o',label=r'$<\kappa\kappa>$')
        plt.plot(theta3,np.zeros_like(theta3),':')
        plt.semilogx()
        plt.xlabel(r'$\theta$ [arcmin]')
        plt.ylabel(r'$\xi(\theta)\times\theta$')
        plt.legend()
        filename = '../figure/KK_'+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.png'
        plt.savefig(filename)
    print ('===> time to calculate correlation = ', time.time()-t0)
    return theta,xi,err,Npairs




def run_corr():
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB()
    corr(K_ra, K_dec, K_val,K_ra, K_dec, K_val,)




#run_corr()

def namaster():
    import pymaster as nmt
    mask=total_mask()
    f_0 = nmt.NmtField(mask, Modified[i,j])
    f_1 = nmt.NmtField(mask, K)

    b = nmt.NmtBin.from_nside_linear(nside, 1)

    # compute MASTER estimator
    cl_01 = nmt.compute_full_master(f_0, f_1, b)

    ell_arr = b.get_effective_ells()
    plt.plot(ell_arr, cl_01, 'r-', label='Namaster')
    plt.loglog()
    plt.xlabel()
    plt.ylabel()
    plt.legend()
























