from test_group import *
from group_nz import *
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
import multiprocessing

t0 = time.time()
fontsize=20
alpha=0.7
nside=512
#z_arr = np.array([0.1, 0.33, 0.67, 1.0])
z_arr = np.array([0.1,0.3,0.5,0.7])
#mass_cut = np.array([12.7,13.1,13.6])
#mass_cut = np.array([13.1, 13.4, 13.9])
color = np.array(['r','orange','b'])
Q=12

#g_fi, g_mask, mask = total_mask(threshold=0.5)
#print (np.sum(mask))

print (tSZ, threshold)
#K_ra, K_dec, K_val, K_mask, K_lm, N_lm, K, Nmap = load_CMB(tSZ=tSZ)
#np.save('../desi_data/kappa_data/kappa_mask_tszTrue.npy',K_mask)
#g_fi, g_mask, mask = total_mask(threshold)
#hp.mollview(mask, title='threshold=%s, tSZ=%s'%(threshold,tSZ))
#np.save('../desi_data/total_mask/total_mask_thrh%s_tsz%s.npy'%(threshold,tSZ),mask)
#plt.savefig('../desi_data/total_mask/total_mask_thrh%s_tsz%s.png'%(threshold,tSZ))

#id,Nmember,ra,dec,photo_z,M,L = select_DESIDR8(member=3,zmin=0.1,zmax=1.0)
#print ('min(ra), max(ra), min(dec), max(dec), min(photo_z),max(photo_z),max(Nmember),max(id)',min(ra), max(ra), min(dec), max(dec), min(photo_z),max(photo_z),max(Nmember),max(id))
#ra,dec = fk5_2_galactic(ra,dec)
#print ('min(ra), max(ra), min(dec), max(dec), min(photo_z),max(photo_z),max(Nmember),max(id)',min(ra), max(ra), min(dec), max(dec), min(photo_z),max(photo_z),max(Nmember),max(id))
#print (min(dec[dec>0]), max(dec[dec<0]))
#d1=dec[dec>0]
#d2=dec[dec<0]
#print (np.sort(d1), np.sort(d1)[:100])
#print (np.sort(d2), np.sort(d2)[-100:])

#g_fi,g_mask,mask=total_mask(threshold=0.5)
#np.save('../desi_data/total_mask/mask_b25.npy',mask)

def test_load_planck_tsz(tSZ):
    K_ra, K_dec, K_val, K_mask, K_lm, N_lm, Kmap, Nmap = load_CMB(tSZ)
    hp.mollview(Kmap, title='$\kappa map, tSZ=%s$'%tSZ)
    plt.savefig('../figure/Kmap_tSZ%s.png'%tSZ,dpi=300)
    fsky=np.sum(K_mask) / len(K_mask)
    hp.mollview(K_mask, title='$\kappa mask, tSZ=%s, f_{sky}=%s$'%(tSZ,fsky))
    plt.savefig('../figure/Kmask_tSZ%s.png'%tSZ,dpi=300)
#test_load_planck_tsz(tSZ=True)
#test_load_planck_tsz(tSZ=False)

def run_after_masking_z(member,i,loc):
    print ('pid=',os.getpid(),'parent pid=', os.getppid())
    t0=time.time()
    pool = multiprocessing.Pool()
    zmin=z_arr[i]
    zmax=z_arr[i+1]
    id,Nmember,ra,dec,photo_z,M,L = select_DESIDR8(member=member,zmin=zmin,zmax=zmax)

    Ng_origin = len(ra)
    result =  pool.apply_async(after_masking,(id,Nmember,ra,dec,photo_z,M,L,threshold,))
    pool.close()
    pool.join()
    arr = result.get()

    pxl,id,Nmember,ra,dec,photo_z,M,L = arr[0],arr[1],arr[2],arr[3],arr[4],arr[5],arr[6],arr[7]
    Ng_masked = len(ra)
    print ('===> Ng_origin, Ng_masked, Ng_origin-Ng_masked = ',Ng_origin, Ng_masked, int(Ng_origin)-int(Ng_masked))

    print ('===> mean halo mass = ', np.mean(M))
    print ('===> median halo mass = ', np.median(M))

    path = '../desi_data/SELECTION/%s/'%loc
    if not os.path.exists(path):
        os.mkdir(path)
    filename = '%s/arr_z%s.fits'%(path,(i+1))
    f = fio.FITS(filename,'rw')
    f.write( [id,Nmember,ra,dec,photo_z,M,L], names=['id','Nmember','ra','dec','photo_z','M','L'] )
    f.close()
    print ('===> time to run after masking=',time.time()-t0)




def run_after_masking(member,lab,i,loc):
    print ('pid=',os.getpid(),'parent pid=', os.getppid())
    t0=time.time()
    pool = multiprocessing.Pool()
    zmin=z_arr[i]
    zmax=z_arr[i+1]
    mass=mass_cut[i]
    id,Nmember,ra,dec,photo_z,M,L = select_DESIDR8_M(member=member,zmin=zmin,zmax=zmax,M_cut=mass,lab=lab)

    Ng_origin = len(ra)
    result =  pool.apply_async(after_masking,(id,Nmember,ra,dec,photo_z,M,L,threshold,))
    pool.close()
    pool.join()
    arr = result.get()

    pxl,id,Nmember,ra,dec,photo_z,M,L = arr[0],arr[1],arr[2],arr[3],arr[4],arr[5],arr[6],arr[7]
    Ng_masked = len(ra)
    print ('===> Ng_origin, Ng_masked, Ng_origin-Ng_masked = ',Ng_origin, Ng_masked, int(Ng_origin)-int(Ng_masked))

    print ('===> mean halo mass = ', np.mean(M))
    print ('===> median halo mass = ', np.median(M))

    path = '../desi_data/SELECTION/%s/'%loc
    if not os.path.exists(path):
        os.mkdir(path)
    filename = '%s/arr_%s_z%s.fits'%(path,lab,(i+1))
    f = fio.FITS(filename,'rw')
    f.write( [id,Nmember,ra,dec,photo_z,M,L], names=['id','Nmember','ra','dec','photo_z','M','L'] )
    f.close()
    print ('===> time to run after masking=',time.time()-t0)



def main(loadloc,lab,loc):
    KG = np.zeros((3,Q))
    Errkg = np.zeros((3,Q))
    COV = np.zeros((3,Q,Q))
    R = np.zeros((3,Q,Q))
    COLORPIX = np.zeros((3, hp.nside2npix(512)))
    MODIFIED = np.zeros((3, hp.nside2npix(512)))
    S=np.zeros(3)
    Err_s=np.zeros(3)
    t0=time.time()
    
    
    for i in range(3):
        zmin=z_arr[i]
        zmax=z_arr[i+1]
        mass = mass_cut[i]
  
        id,Nmember,ra,dec,photo_z,M,L = load_AFTER_MASK(loc='%s'%loadloc,lab='%s'%lab,z=i+1)
        print ('---> min Nmember=', np.min(Nmember))
        print ('---> max photo_z=', np.max(photo_z))       

        colorpix, meanfield = counts_group(ra,dec)
        Modified = modified_map(colorpix,threshold)
        kg,kg_bin,gg,gg_bin = clkg_test(Modified,threshold) 
        clskg = kgerr(Modified,threshold)
        err_kg,cov,r = get_cov(clskg)
        COLORPIX[i] = colorpix
        MODIFIED[i] = Modified
        KG[i] = kg_bin
        Errkg[i] = err_kg
        COV[i] = cov
        R[i] = r
  

    print ('MODIFIED=',MODIFIED)
    
    path = '../desi_data/%s/'%loc
    if not os.path.exists(path):
        os.mkdir(path)
    np.save('%s/colorpix_%s.npy'%(path,lab),COLORPIX)
    np.save('%s/Modified_%s.npy'%(path,lab),MODIFIED)
    np.save('%s/KG_%s.npy'%(path,lab),KG)
    np.save('%s/Errkg_%s.npy'%(path,lab),Errkg)
    np.save('%s/COV_%s.npy'%(path,lab),COV)
    np.save('%s/R_%s.npy'%(path,lab),R)
    print ('===> time to run total =',time.time()-t0)



def run_group_nz(dz,member,saveloc):
    for lab in ['less','moreeq']:
        for i in range(3):
            mass = mass_cut[i]
            get_group_nz(M_thrh=mass, slt='%s'%lab, Nm=member, dz=dz, saveloc=saveloc,total=False)




def run_kg_th(loadloc,loc,lab,sigmaz):
    for i in range(3):
        mass = mass_cut[i]
        if lab == 'less':
            print ('=================redshift=%s,mass<%s==================='%(i,mass))
            filename = '../desi_data/z_dis/%s/group_nz_sigmaz%s_less%s.fits'%(loadloc,sigmaz,mass)
        if lab == 'moreeq':
            print ('=================redshift=%s,mass>=%s==================='%(i,mass))
            filename = '../desi_data/z_dis/%s/group_nz_sigmaz%s_moreeq%s.fits'%(loadloc,sigmaz,mass)
        kg_mean = kg_th(filename, i)
        path = '../desi_data/%s/'%loc
        if not os.path.exists(path):
            os.mkdir(path)
        np.save('%s/kgth_%s%s_sigmaz%s.npy'%(path,lab,mass,sigmaz),kg_mean)
        kmu_mean = kmu_th(filename,i)
        np.save('%s/kmuth_%s%s_sigmaz%s.npy'%(path,lab,mass,sigmaz),kmu_mean)


def bias(loadloc,saveloc,lab,model,richness):
    B = np.ones(3)
    B_sigma = np.ones(3)
    for i in range(3):
        zmin=z_arr[i]
        zmax=z_arr[i+1]
        mass = mass_cut[i]
        id,Nmember,ra,dec,photo_z,M,L = load_AFTER_MASK(loc='%s'%loadloc,lab='%s'%lab,z=i+1)

        M = M_true(M, richness)

        b_Wk1, sigma_b = b1_b2(photo_z, M, model=model)
        B[i] = b_Wk1
        B_sigma[i] = sigma_b
    path = '../desi_data/%s/'%saveloc
    if not os.path.exists(path):
        os.mkdir(path)
    np.save('%s/B_%s_%s.npy'%(path,lab,model), B)
    np.save('%s/B_sigma_%s_%s.npy'%(path,lab,model), B_sigma)


def run_bias(loadloc,saveloc,richness):
    for lab in ['less','moreeq']:
        for mod in ['jing98','sheth01','tinker10','comparat17']:
            bias(loadloc=loadloc,saveloc=saveloc,lab=lab, model=mod, richness=richness)#jing98, sheth01, tinker10, cole89, seljak04, pillepich10

mmm=5
sigmaz=0.05
mass_cut = np.array([13.1, 13.4, 13.9])

#sltfile='SELECT%s_tsz'%mmm
#clfile='results/KG%s_tsz'%mmm
#zfile='z_distribution%s_tsz'%mmm

#id,Nmember,ra,dec,photo_z,M,L = select_DESIDR8(1,0.1,0.33)
#for lab in ['less','moreeq']:
#    for i in range(3):
#        print (lab,i+1)
#        id,Nmember,ra,dec,photo_z,M,L = load_AFTER_MASK(sltfile,lab,i+1)
#        print ('===> len(photo_z) = ', len(photo_z) )
#        print ('===> np.median(M) = ', np.median(M))
#        print ('===> np.mean(M) = ', np.mean(M) )


#sltfile='SELECT5_tsz'
#clfile='results/KG5_tsz'
#zfile='z_distribution5_tsz'

#sltfile='SELECT5_tsz_SGC'
#clfile='results/KG5_tsz_SGC'
#zfile='z_distribution5_tsz_SGC'

#sltfile='SELECT5_tsz_thrh7'
#clfile='results/KG5_tsz_thrh7'
#zfile='z_distribution5_tsz_thrh7'

#sltfile='SELECT5_tsz_ratio'
#clfile='results/KG5_tsz_ratio'
#zfile='z_distribution5_tsz_ratio'

#for i in range(3):
#    run_after_masking_z(member=mmm,i=i,loc=sltfile)

sltfile='SELECT5_tsz_newz'
clfile='results/KG5_tsz_newz'
zfile='z_distribution5_tsz_newz'

#if __name__=='__main__':
#    for lab in ['less','moreeq']:
#        for i in range(3):
#            run_after_masking(member=mmm,lab='%s'%lab,i=i,loc=sltfile)
#for lab in ['less','moreeq']:
#    main(loadloc = sltfile,lab='%s'%lab,loc=clfile)
#run_bias(loadloc=sltfile,saveloc=clfile, richness=mmm)

#run_group_nz(dz=sigmaz,member=mmm,saveloc=zfile)
for lab in ['less','moreeq']:
    run_kg_th(loadloc=zfile,loc=clfile,lab='%s'%lab,sigmaz=sigmaz)








