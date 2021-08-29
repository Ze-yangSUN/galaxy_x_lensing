# -------------- pixelize the galaxy position
nside = 512;
autoplot = True;
zmin = 0.1;
zmax = 0.3;
def pix_gal(ra, dec):

    import time
    t0 = time.time()

    import os
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    from pandas import value_counts
    import healpy as hp
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print '===> size of pixel in unit of arcmin = ',np.sqrt( 41253.*60.*60./hp.nside2npix(nside) )

    pxl = hp.ang2pix(nside, np.pi/2.-np.deg2rad(dec), np.deg2rad(ra) )
    pxlc = value_counts(pxl)
    meanfield = np.mean(pxlc.values)
    pxlc = pxlc - meanfield

    [pxl_dec, pxl_ra] = np.rad2deg(hp.pix2ang(nside, pxlc.index))
    pxl_dec = 90.-pxl_dec

    g_mask = np.zeros(hp.nside2npix(nside))
    g_mask[pxlc.index] = 1

    if autoplot: # plot the galaxy number over density per pixel
        colorpix = np.ones(hp.nside2npix(nside)) * (-meanfield)
        colorpix[pxlc.index] = pxlc.values
        plt.figure()
        hp.mollview(colorpix, title='galaxy count')
        filename = '../cmb-cross-correlation/figure/healpix_map_nside%s_%sto%s_'%(nside, zmin, zmax)+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.png'
        plt.savefig(filename)

    print '===> time to pixelize the galaxies = ', time.time()-t0
    return pxl_ra, pxl_dec, pxlc, meanfield, colorpix, g_mask
import numpy as np
import fitsio as fio 
import time
#filename = '../cmb-cross-correlation/galaxy_0.1to0.9_2020-04-16-211212.fits'
#filename = '../cmb-cross-correlation/galaxy_0.1to0.3_2020-04-17-232350.fits'
filename = '../cmb-cross-correlation/galaxy_0.1to0.3_2020-04-18-050044.fits'
galaxy = fio.FITS(filename)[-1].read()
ra = galaxy['RA']
dec = galaxy['DEC']

pxl_ra, pxl_dec, pxlc, meanfield, colorpix, g_mask = pix_gal(ra, dec)

np.save('pxl_ra_nside%s_%sto%s_'%(nside, zmin, zmax)+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.npy', pxl_ra)
np.save('pxl_dec_nside%s_%sto%s_'%(nside, zmin, zmax)+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.npy', pxl_dec)
np.save('pxlc_nside%s_%sto%s_'%(nside, zmin, zmax)+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.npy', pxlc)
np.save('meanfield_nside%s_%sto%s_'%(nside, zmin, zmax)+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.npy', meanfield)
np.save('colorpix_nside%s_%sto%s_'%(nside, zmin, zmax)+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.npy', colorpix)
np.save('g_mask_nside%s_%sto%s_'%(nside, zmin, zmax)+time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))+'.npy', g_mask)
