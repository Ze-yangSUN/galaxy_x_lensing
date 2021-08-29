def sort_mag(zmin,zmax,Ngal):
    import time
    t0 = time.time()
    import numpy as np
    import fitsio as fio  
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt 

    filename = '../zouhu/LS_DR8_total_csp.fits'
    f = fio.FITS(filename)[-1].read()
    photo_z = f['photo_z']
    MAG_R = f['MAG_R']
    RA = f['RA']
    DEC = f['DEC']
    photo_zerr = f['photo_zerr']

	#print '===> you are reading file %s<z<%s......'%(zmin,zmax)
    slt = (zmin<photo_z) & (photo_z<zmax)
    photo_z = photo_z[slt]
    MAG_R = MAG_R[slt]
    RA = RA[slt]
    DEC = DEC[slt]
    photo_zerr = photo_zerr[slt]
    print '===> the number of galaxies in %s<z<%s='%(zmin,zmax),MAG_R.shape
    # mag_r_sort = sorted(MAG_R, reverse=True)
    mag_r_sort_index = np.argsort(-MAG_R)  # sort the magnitude reversely
    mag_r_sort = MAG_R[mag_r_sort_index]  # return the magnitude value reversely

    # Ngal = 2e6  # the number of galaies in one magnitude cut bin
    searching = int(Ngal)
    while mag_r_sort[searching] == mag_r_sort[searching+1]:
        print searching, round(mag_r_sort[searching],2)
        searching += 1
    else:
    	cut_left=searching+1
        print 'slice of cut:',mag_r_sort[:10],mag_r_sort[searching+1-10:searching+1]
        slt1 = mag_r_sort_index[:(searching+1)]
        z = photo_z[slt1]
        mag_r = MAG_R[slt1]
        ra = RA[slt1]
        dec = DEC[slt1]
        zerr = photo_zerr[slt1]
        print '===> the number of galaxies when slt %s<z<%s='%(zmin,zmax),slt1.shape
    	filename = '../cmb-cross-correlation/galaxy_magr_cut/galaxy_%sto%s_cut%s.fits'%(zmin,zmax,round(mag_r_sort[searching+1],2))
    	f1 = fio.FITS(filename,'rw')
    	f1.write([z, mag_r, ra, dec, zerr], names=['photo_z', 'MAG_R', 'RA', 'DEC', 'photo_zerr'])
    	f1.close()
    	searching += int(Ngal)
	while mag_r_sort[searching] == mag_r_sort[searching+1]:
        print searching, round(mag_r_sort[searching],2)
        searching += 1
    else:
    	cut_left2 = searching+1
        print 'slice of cut:',mag_r_sort[cut_left:cut_left+10],mag_r_sort[searching+1-10:searching+1]
        slt1 = mag_r_sort_index[cut_left:(searching+1)]
        z = photo_z[slt1]
        mag_r = MAG_R[slt1]
        ra = RA[slt1]
        dec = DEC[slt1]
        zerr = photo_zerr[slt1]
        print '===> the number of galaxies when slt %s<z<%s='%(zmin,zmax),slt1.shape
    	filename = '../cmb-cross-correlation/galaxy_magr_cut/galaxy_%sto%s_cut%s.fits'%(zmin,zmax,round(mag_r_sort[searching+1],2))
    	f1 = fio.FITS(filename,'rw')
    	f1.write([z, mag_r, ra, dec, zerr], names=['photo_z', 'MAG_R', 'RA', 'DEC', 'photo_zerr'])
    	f1.close()
    	searching += int(Ngal)
	while mag_r_sort[searching] == mag_r_sort[searching+1]:
        print searching, round(mag_r_sort[searching],2)
        searching += 1
    else:
    	cut_left3 = searching+1
        print 'slice of cut:',mag_r_sort[cut_left2:cut_left2+10],mag_r_sort[searching+1-10:searching+1]
        slt1 = mag_r_sort_index[cut_left2:(searching+1)]
        z = photo_z[slt1]
        mag_r = MAG_R[slt1]
        ra = RA[slt1]
        dec = DEC[slt1]
        zerr = photo_zerr[slt1]
        print '===> the number of galaxies when slt %s<z<%s='%(zmin,zmax),slt1.shape
    	filename = '../cmb-cross-correlation/galaxy_magr_cut/galaxy_%sto%s_cut%s.fits'%(zmin,zmax,round(mag_r_sort[searching+1],2))
    	f1 = fio.FITS(filename,'rw')
    	f1.write([z, mag_r, ra, dec, zerr], names=['photo_z', 'MAG_R', 'RA', 'DEC', 'photo_zerr'])
    	f1.close()
    	searching += int(Ngal)
	while mag_r_sort[searching] == mag_r_sort[searching+1]:
        print searching, round(mag_r_sort[searching],2)
        searching += 1
    else:
    	cut_left4 = searching+1
        print 'slice of cut:',mag_r_sort[cut_left3:cut_left3+10],mag_r_sort[searching+1-10:searching+1]
        slt1 = mag_r_sort_index[cut_left3:(searching+1)]
        z = photo_z[slt1]
        mag_r = MAG_R[slt1]
        ra = RA[slt1]
        dec = DEC[slt1]
        zerr = photo_zerr[slt1]
        print '===> the number of galaxies when slt %s<z<%s='%(zmin,zmax),slt1.shape
    	filename = '../cmb-cross-correlation/galaxy_magr_cut/galaxy_%sto%s_cut%s.fits'%(zmin,zmax,round(mag_r_sort[searching+1],2))
    	f1 = fio.FITS(filename,'rw')
    	f1.write([z, mag_r, ra, dec, zerr], names=['photo_z', 'MAG_R', 'RA', 'DEC', 'photo_zerr'])
    	f1.close()
    	searching += int(Ngal)
	while mag_r_sort[searching] == mag_r_sort[searching+1]:
        print searching, round(mag_r_sort[searching],2)
        searching += 1
    else:
    	cut_left5 = searching+1
        print 'slice of cut:',mag_r_sort[cut_left4:cut_left4+10],mag_r_sort[searching+1-10:searching+1]
        slt1 = mag_r_sort_index[cut_left4:(searching+1)]
        z = photo_z[slt1]
        mag_r = MAG_R[slt1]
        ra = RA[slt1]
        dec = DEC[slt1]
        zerr = photo_zerr[slt1]
        print '===> the number of galaxies when slt %s<z<%s='%(zmin,zmax),slt1.shape
    	filename = '../cmb-cross-correlation/galaxy_magr_cut/galaxy_%sto%s_cut%s.fits'%(zmin,zmax,round(mag_r_sort[searching+1],2))
    	f1 = fio.FITS(filename,'rw')
    	f1.write([z, mag_r, ra, dec, zerr], names=['photo_z', 'MAG_R', 'RA', 'DEC', 'photo_zerr'])
    	f1.close()
    	searching += int(Ngal)

    print '===> time to get sort mag =', time.time()-t0

# slt_z(0.1,0.3,2e6)
sort_mag(0.3,0.5,4e6)
sort_mag(0.5,0.7,5e6)
sort_mag(0.7,0.9,3e6)



