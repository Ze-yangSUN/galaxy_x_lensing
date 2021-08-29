from NGC_SGC import *
from test_group import *
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
import treecorr
import treecorr as tc
from pixell import enmap, enplot, reproject, utils, curvedsky 
from astropy.coordinates import SkyCoord
from astropy import units as u


t0 = time.time()
fontsize=20
alpha=0.7
nside=512
z_arr = np.array([0.1, 0.3, 0.5, 0.7])
color = np.array(['r','orange','b'])
Q=12
N_lens_sim=100

thetaN=15
theta_min=1
theta_max=300
Npatch=100
z_sour = np.array([0.4,0.6,0.8,1.0])
def shear_cat(z_s):
    t0=time.time()
    filename='/data/s6/zysun/shear/DR8_ZRP_all.fits'
    f = fio.FITS(filename)[-1].read()
    f = f[['ra','dec','zp_mean','zp_err','e1','e2','w','1+m']]
    zmin=z_sour[z_s]
    zmax=z_sour[z_s+1]
    slt = (f['zp_mean']<=zmax) & (f['zp_mean']>zmin)
    t = f[slt]
    print ('---> minz,maxz=',min(t['zp_mean']), max(t['zp_mean']))
    print ('===> time to slt shear catalog = ', time.time()-t0)
    return t


def jk_corr( dataE, dataD, dataR, filename ):
    print('===> enter jk_corr',len(dataE),len(dataD),len(dataR))
    TCnC = 1 # TreeCorr number of cores

    # need to use -g2 or -e2 due to different shape definition with RA
    E = tc.Catalog(g1=dataE['e1'], g2=-dataE['e2'], w=dataE['w'], ra=dataE['ra'], dec=dataE['dec'], ra_units='deg', dec_units='deg',npatch=Npatch)
    m = tc.Catalog(k=dataE['1+m'], w=dataE['w'], ra=dataE['ra'], dec=dataE['dec'], ra_units='deg', dec_units='deg',npatch=Npatch)
    D = tc.Catalog(ra=dataD['ra'], dec=dataD['dec'], ra_units='deg', dec_units='deg',npatch=Npatch)
    R = tc.Catalog(ra=dataR['RA'], dec=dataR['DEC'], ra_units='deg', dec_units='deg',npatch=Npatch)
    # print('EmDR')

    ED=treecorr.NGCorrelation(nbins=thetaN, min_sep=theta_min, max_sep=theta_max, sep_units='arcmin', bin_slop=0.01, verbose=0)
    ED.process(D,E,metric='Euclidean',num_threads=TCnC)
    # print(ED.npairs)

    if np.count_nonzero(ED.npairs)>0:
        mD=treecorr.NKCorrelation(nbins=thetaN, min_sep=theta_min, max_sep=theta_max, sep_units='arcmin', bin_slop=0.01, verbose=0)
        mD.process(D,m,metric='Euclidean',num_threads=TCnC)

        ER=treecorr.NGCorrelation(nbins=thetaN, min_sep=theta_min, max_sep=theta_max, sep_units='arcmin', bin_slop=0.01, verbose=0)
        mR=treecorr.NKCorrelation(nbins=thetaN, min_sep=theta_min, max_sep=theta_max, sep_units='arcmin', bin_slop=0.01, verbose=0)
        ER.process(R,E,metric='Euclidean',num_threads=TCnC)
        mR.process(R,m,metric='Euclidean',num_threads=TCnC)

        try:
            os.remove(filename)
        except OSError:
            pass
        f=fio.FITS(filename,'rw')
        var_names = ['r','ED','XD','EDw',
                    'mD','mDw',
                    'ER','XR','ERw',
                    'mR','mRw']
        var_data = [ED.meanr,ED.xi,ED.xi_im,ED.weight,
                    mD.xi,mD.weight,
                    ER.xi,ER.xi_im,ER.weight,
                    mR.xi,mR.weight]
        f.write(var_data, names=var_names)
        f.close()




def main_corr(z_lens,z_s):
    t0=time.time()
    loc='SELECT5_tsz_ratio'
    dataD = load_AFTER_MASK_z(loc=loc, z=z_lens)
    dataE = shear_cat(z_s)
    dataR = fio.FITS('../desi_data/results/randcat/rand_zlens%s_50.fits'%z_lens)[-1].read()
    filename='../desi_data/results/corr/var_data_zlens%s_zs%s.fits'%(z_lens,z_s+1)
    jk_corr(dataE, dataD, dataR, filename)
    print ('===> you are in filename=%s'%filename)
    print ('===> time to main corr = ',time.time()-t0)



def corr_CMB_old(K_val, K_ra, K_dec, g_ra, g_dec, filename):
    print ('====  you are in corr_CMB ====')
    t0=time.time()
    TCnC=1
    dataK = tc.Catalog(k=K_val, ra=K_ra, dec=K_dec, ra_units='deg', dec_units='deg', npatch=Npatch)
    p_center = dataK.patch_centers
    print ('===> patch center=',p_center)
    datag = tc.Catalog(ra=g_ra, dec=g_dec, ra_units='deg', dec_units='deg', patch_centers=p_center)
    Kg = tc.NKCorrelation(nbins=thetaN, min_sep=theta_min, max_sep=theta_max, sep_units='arcmin', var_method='jackknife', bin_slop=0.01, verbose=0)
    Kg.process(datag, dataK, metric='Euclidean', num_threads=TCnC)

    f=fio.FITS(filename, 'rw')
    var_names = ['r', 'logKg', 'Kg', 'Kgvar', 'Kgw', 'Kgpair', 'Kgcov', 'Kgrawxi']
    var_data = [Kg.meanr, Kg.meanlogr, Kg.xi, Kg.varxi, Kg.weight, Kg.npairs, Kg.cov, Kg.raw_xi]
    f.write(var_data, names=var_names)
    f.close()
    print ('===> time to corr CMB with lens group = ', time.time()-t0)



def corr_CMB(dataK, p_center, g_ra, g_dec, filename):
    print ('====  you are in corr_CMB ====')
    t0=time.time()
    TCnC=2
    datag = tc.Catalog(ra=g_ra, dec=g_dec, ra_units='deg', dec_units='deg', patch_centers=p_center)
    Kg = tc.NKCorrelation(nbins=thetaN, min_sep=theta_min, max_sep=theta_max, sep_units='arcmin', var_method='jackknife', bin_slop=0.01, verbose=0)
    Kg.process(datag, dataK, metric='Euclidean', num_threads=TCnC)

    f=fio.FITS(filename, 'rw')
    var_names = ['r', 'logKg', 'Kg', 'Kgvar', 'Kgw', 'Kgpair', 'Kgcov', 'Kgrawxi']
    var_data = [Kg.meanr, Kg.meanlogr, Kg.xi, Kg.varxi, Kg.weight, Kg.npairs, Kg.cov, Kg.raw_xi]
    f.write(var_data, names=var_names)
    f.close()
    print ('===> time to corr CMB with lens group = ', time.time()-t0)



g_fi, g_mask, mask = total_mask(threshold)
#    K_ra, K_dec, K_val, K_mask, K_lm, Kmap = load_CMB2048(tSZ=tSZ)   # raw data
     
K_ra, K_dec, K_val, K_mask, K_lm, N_lm, Kmap, Nmap = load_CMB(tSZ=tSZ)   # wn filter
print ('---> len(K_val)=',len(K_val))

#    K_ra, K_dec, K_val, K_mask, K_lm, Kmap = load_CMB_lcut(lcut=l_cut, nside=512)  # l_cut=1536

K_ra, K_dec, K_val = in_mask(nside, mask, K_ra, K_dec, K_val)
print ('---> len(K_val)=',len(K_val))

dataK = tc.Catalog(k=K_val, ra=K_ra, dec=K_dec, ra_units='deg', dec_units='deg', npatch=Npatch)
p_center = dataK.patch_centers
print ('===> patch center=',p_center)
#dataK.write_patch_centers(cen_file)

def run_corr_CMB(z_lens, stage):
    if stage=='DK_CMB':
        loc='SELECT5_tsz_ratio'
        dataD = load_AFTER_MASK_z(loc=loc, z=z_lens)
        filename='../desi_data/results/corr/corr_CMB_zlens%s_wn.fits'%(z_lens)
#        corr_CMB(K_val, K_ra, K_dec, dataD['ra'], dataD['dec'], filename)
        corr_CMB(dataK, p_center, dataD['ra'], dataD['dec'], filename)

    elif stage=='RK_CMB':
        print ('==========RK_CMB,z%s==============='%z_lens)
        filen = '../desi_data/results/randcat/rand_zlens%s_50.fits'%z_lens
        dataR = fio.FITS(filen)[-1].read()
        filename='../desi_data/results/corr/corr_R_CMB_zlens%s_wn.fits'%(z_lens)
#        corr_CMB(K_val, K_ra, K_dec, dataR['RA'], dataR['DEC'], filename)
        corr_CMB(dataK, p_center, dataR['RA'], dataR['DEC'], filename)


def get_zdis(z, zmin=0., zmax=1):
    from scipy import stats
    z_tr = np.linspace(zmin,zmax,int(zmax-zmin)*100+1)
    nz = np.zeros((4,len(z_tr)))
    count = np.zeros(4)
    dz=0.05
    PDF = stats.norm.pdf(z_tr, loc=z.reshape(-1, 1), scale=dz)
    PDF_sum = np.diff(stats.norm.cdf(z_tr[[0, -1]], loc=z.reshape(-1, 1), scale=dz), axis=-1)    # if dz is a number 
    colors=['g','orange','c']
    z_arr = np.array([0.1, 0.3, 0.5, 0.7])
    for i in range(3):
        z_min=z_arr[i]
        z_max=z_arr[i+1]
        print ('z_min=',z_min, 'z_max=',z_max)

        slt = (z_min<z) & (z<=z_max)
        nz[i+1][:] = np.sum(PDF[slt],0)
        nz[0][:] += nz[i+1][:]
    print ('===> nz = ',nz)
    print (len(z))
    nz /= len(z)
    print ('===> nz / len(z) = ', nz) 
    filename='../desi_data/results/corr/nzlens_dis.fits'
    f=fio.FITS(filename,'rw')
    f.write([z_tr, nz[0], nz[1], nz[2], nz[3]], names=['z','n_all','n_1','n_2','n_3'])
    f.close()



def zlens_dis():
    loc='SELECT5_tsz_ratio'
    z1=load_AFTER_MASK_z(loc,1)
    z2=load_AFTER_MASK_z(loc,2)
    z3=load_AFTER_MASK_z(loc,3)
    print (len(z1),len(z2),len(z3))
    z=np.hstack((z1,z2,z3))
    print ('len(z)=',len(z),z.shape)
    get_zdis(z=z)

#zlens_dis()


for i in range(3):
    z_lens=i+1
    run_corr_CMB(z_lens, stage='DK_CMB')
    run_corr_CMB(z_lens, stage='RK_CMB')

#for i in range(2,3):
#    z_lens=3
#    z_s=i
#    main_corr(z_lens,z_s)



