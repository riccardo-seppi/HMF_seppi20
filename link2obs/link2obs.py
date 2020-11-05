from astropy.table import Table, Column, Row
#from astropy_healpix import healpy
import sys
import os, glob
import time
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
import astropy.constants as cc
import astropy.io.fits as fits
import scipy
from scipy.special import erf
from scipy.stats import norm
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d
from scipy.stats import scoreatpercentile
import h5py
import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import mass_function as mf
from colossus.lss import peaks
from sklearn import mixture
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import curve_fit
import ultranest
from ultranest.plot import cornerplot
import corner
import pandas as pd
import hydro_mc
import random


print('Looks for correlation between Xray center-BCG displacement with Xoff')
print('------------------------------------------------')
print('------------------------------------------------')
#set cosmology
cosmo = cosmology.setCosmology('multidark-planck')  
dc = peaks.collapseOverdensity(z = 0)
h = cosmo.Hz(z=0)/100
cosmo_astropy = FlatLambdaCDM(H0=67.77, Om0=0.307)  

direct = '.'
path_2_BCG = os.path.join(direct, 'SpidersXclusterBCGs-v2.0.fits')
path_2_clusters = os.path.join(direct,'catCluster-SPIDERS_RASS_CLUS-v3.0.fits')
#path_2_bcg_eFEDS = os.path.join(direct, 'BCG_eFEDS.fits')
#path_2_clusters_eFEDS = os.path.join(direct,'eFEDS_properties_18_3_2020.fits')
path_2_clusters_eFEDS = os.path.join(direct,'wcen','decals_dr8_run_32_efeds_extendedSourceCatalog_mllist_ph_22_11_2019_v940_final_catalog.fit')
#path_2_model = os.path.join('..','quarantine','HMF','g_sigma','coeff','HMD','z_0.000','model.fit')
path_2_model = os.path.join('..','quarantine','gsigma','extended','3d','z_0.000','hsigma_params.fit')
path_2_zevo = os.path.join('..','quarantine','gsigma','extended','3d','zevo','hsigma_params.fit')
path_2_clusters_eRASS = os.path.join(direct,'eRASS','decals_dr8_run_redmapper_v0.6.6_lgt20_catalog2.fit')
path_2_clusters_shear_sel = os.path.join(direct,'distance_shearsel_xray.txt')
path2lightcone_MD10 = os.path.join(direct,'MD10_eRO_CLU_b10_CM_0_pixS_20.0.fits')
path2lightcone_MD40 = os.path.join(direct,'MD40_eRO_CLU_b8_CM_0_pixS_20.0.fits')

#read catalogs SDSS
print('reading catalogs...')

t_clus = Table.read(path_2_clusters)
dt_clus = t_clus.to_pandas()
id_clus = np.array(dt_clus.CLUS_ID)
ra_bcg = np.array(dt_clus.RA_OPT)
dec_bcg = np.array(dt_clus.DEC_OPT)
z_sp = np.array(dt_clus.SCREEN_CLUZSPEC)
ra_clus = np.array(dt_clus.RA)
dec_clus = np.array(dt_clus.DEC)
rich = np.array(dt_clus.LAMBDA_CHISQ_OPT)
z_ph = np.array(dt_clus.Z_LAMBDA)
r200 = np.array(dt_clus.R200C_DEG)
richness = np.array(dt_clus.LAMBDA_CHISQ_OPT)
Ncomp = np.array(dt_clus.NCOMPONENT)
index1comp = np.where(Ncomp==1)
print(r200)
print(np.average(z_sp))


ind1 = (ra_clus > 100) & (ra_clus<315)
print(min(ra_clus[ind1]),max(ra_clus[ind1]))
print(min(dec_clus[ind1]),max(dec_clus[ind1]))

ind1 = (ra_clus < 100) #or (ra_clus>315)
print(min(ra_clus[ind1]),max(ra_clus[ind1]))
print(min(dec_clus[ind1]),max(dec_clus[ind1]))


print(min(z_sp),max(z_sp))
mass_rich = 3e14*(richness/38.56*((1+z_sp)/1.18)**1.13)**(1/0.99)*(200/178)**3
print('%.3g'%min(mass_rich),'%.3g'%max(mass_rich))
 
print('computing offset...')
dist_col = cosmo.comovingDistance(z_min=0.,z_max=z_sp)*u.pc*1e6/0.6777 #/h
#print('colossus = ', dist_col)
dist = cosmo_astropy.comoving_distance(z_sp).to(u.pc)
#print('astropy = ', dist)

bcg = SkyCoord(ra_bcg[index1comp]*u.degree,dec_bcg[index1comp]*u.degree,frame='fk5',distance=dist[index1comp])
clus = SkyCoord(ra_clus[index1comp]*u.degree, dec_clus[index1comp]*u.degree, frame='fk5',distance=dist[index1comp])
print(clus)

sep = bcg.separation_3d(clus)
#get separation in kpc (same as Xoff in distinct catalogs)
sep = sep.value*1e-3
#print(sep)

r_200 = cosmo_astropy.kpc_comoving_per_arcmin(z_sp).value*(r200*60)
x = (cosmo.Om(z_sp)-1)
delta_vir = (18*(np.pi)**2+82*x-39*x**2)/cosmo.Om(z_sp)
#print(delta_vir)
rvir = (200/delta_vir)**3*r_200
#print(rvir)

#get xoff from the data and get its PDF
#xoff_data = sep/rvir
xoff_data = sep
#binning = np.linspace(-3.,0.,20)
binning = np.linspace(min(xoff_data),max(xoff_data),20)
pdf, b = np.histogram(np.log10(xoff_data),bins=binning,density=True)
bins = (b[:-1]+b[1:])/2
xoff_data = np.sort(xoff_data)
cdf = np.arange(1,len(xoff_data)+1)/len(xoff_data)
print(cdf)

fig = plt.figure(figsize=(10,10))
psf = np.array([10,30,60,120,180])
shift = np.zeros(len(psf))
for k,kernel in enumerate(psf):
    #error = cosmo_astropy.kpc_comoving_per_arcmin(z_sp[index1comp]).value*(kernel/60)
    #print('error = ',error,' kpc')
    #compute a new centroid shifted by xoff and noise
    print(bcg,clus) 
    offset = bcg.separation(clus)+Angle(np.random.normal(0,Angle(kernel/3600, unit=u.deg).value,len(clus)), unit=u.deg)  #noise is a gaussian with sigma=kernel
    angle = np.random.randint(0,360,size=len(clus))
    print('angle = ',angle)
    clus_new = clus.directional_offset_by(angle, offset)
    clus_new = SkyCoord(clus_new,distance=dist[index1comp])
    print('clus_new = ',clus_new)
    sep_new = bcg.separation_3d(clus_new).value*1e-3
    print('sep_new =', sep_new)
    plt.scatter(sep, sep-sep_new, label='%d'%(kernel),s=12)
    print('convolution effect = ',np.average(sep-sep_new))
    shift[k] =  np.sqrt(np.average((sep-sep_new)**2))

plt.ylabel('Xoff - Xoff degraded',fontsize=20)
plt.xlabel('Xoff',fontsize=20)
plt.legend(fontsize=15)
plt.tick_params(labelsize=15)
plt.title('SPIDERS',fontsize=20)
plt.grid(True)
plt.tight_layout()
outf = os.path.join(direct,'xoff_degraded.png')
plt.savefig(outf,overwrite=True)

t=Table()
t.add_column(Column(name='kernel',data=psf,unit=''))
t.add_column(Column(name='shift', data=shift,unit=''))
outt = os.path.join(direct,'shift.fit')
t.write(outt,overwrite=True)

#xoff_spiders_shift_ = np.abs(xoff_data*((xoff_data-shift[1])/xoff_data))
#print('xoff data = ',(xoff_data))
#print('xoff_data shift', xoff_spiders_shift_)
#print('shift = ',np.average(np.abs(shift)))
#xoff_spiders_shift = np.sort(xoff_spiders_shift_)
#cdf_spiders_shift = np.arange(1,len(xoff_spiders_shift)+1)/len(xoff_spiders_shift)
#sys.exit()
#read catalogs eFEDS
print('reading catalogs...')

t_clus_eFEDS = Table.read(path_2_clusters_eFEDS)
print(t_clus_eFEDS['id_src'])
id_clus_eFEDS = t_clus_eFEDS['id_src']
ra_bcg_eFEDS = t_clus_eFEDS['ra']
dec_bcg_eFEDS = t_clus_eFEDS['dec']
ra_clus_eFEDS = t_clus_eFEDS['ra_orig']
dec_clus_eFEDS = t_clus_eFEDS['dec_orig']
z_lambda_eFEDS = t_clus_eFEDS['z_lambda']

print('computing offset...')
dist_col_eFEDS = cosmo.comovingDistance(z_min=0.,z_max=np.array(z_lambda_eFEDS))*u.pc*1e6/0.6777 #/h
print('colossus = ', dist_col_eFEDS)
dist_eFEDS = cosmo_astropy.comoving_distance(np.array(z_lambda_eFEDS)).to(u.pc)
print('astropy = ', dist)
bcg_eFEDS = SkyCoord(np.array(ra_bcg_eFEDS)*u.degree,np.array(dec_bcg_eFEDS)*u.degree,frame='fk5',distance=dist_eFEDS)
clus_eFEDS = SkyCoord(np.array(ra_clus_eFEDS)*u.degree, np.array(dec_clus_eFEDS)*u.degree, frame='fk5',distance=dist_eFEDS)

sep_eFEDS = bcg_eFEDS.separation_3d(clus_eFEDS)
#get separation in kpc (same as Xoff in distinct catalogs)
sep_eFEDS = sep_eFEDS.value*1e-3
print(len(sep_eFEDS))

#x_eFEDS = (cosmo.Om(z_mcmf_eFEDS)-1)
#delta_vir_eFEDS = (18*(np.pi)**2+82*x_eFEDS-39*x_eFEDS**2)/cosmo.Om(z_mcmf_eFEDS)
#print(delta_vir)
#vir_eFEDS = (200/delta_vir_eFEDS)**3*r200_eFEDS
#print(rvir)

#get xoff from the data and get its PDF
#xoff_data = sep/rvir
xoff_data_eFEDS = sep_eFEDS
#binning_eFEDS = np.linspace(-3.,0.,20)
binning_eFEDS = np.linspace(min(xoff_data_eFEDS),max(xoff_data_eFEDS),20)
pdf_eFEDS, b_eFEDS = np.histogram(np.log10(xoff_data_eFEDS),bins=binning,density=True)
bins_eFEDS = (b_eFEDS[:-1]+b_eFEDS[1:])/2
indsort_eFEDS = np.argsort(xoff_data_eFEDS)
xoff_data_eFEDS_sort = xoff_data_eFEDS[indsort_eFEDS]
cdf_eFEDS = np.arange(1,len(xoff_data_eFEDS_sort)+1)/len(xoff_data_eFEDS_sort)
print(cdf_eFEDS)
ind_new_eFEDS = []
for i in range(len(cdf_eFEDS)):
    ind_new_eFEDS.append(int(np.argwhere(indsort_eFEDS==i)))
cdf_eFEDS_back = cdf_eFEDS[ind_new_eFEDS]
t_clus_eFEDS.add_column(Column(name='Sep',data=xoff_data_eFEDS,unit=''))
t_clus_eFEDS.add_column(Column(name='cdf',data=cdf_eFEDS_back,unit=''))
outt_eFEDS = os.path.join(direct,'wcen','decals_dr8_run_32_efeds_extendedSourceCatalog_mllist_ph_22_11_2019_v940_final_catalog_cdf.fit')
t_clus_eFEDS.write(outt_eFEDS,overwrite=True)

#read catalogs eRASS
print('reading catalog eRASS...')

t_clus_eRASS_uncut = Table.read(path_2_clusters_eRASS)
richness_eRASS = t_clus_eRASS_uncut['lambda']
ext_like_eRASS = t_clus_eRASS_uncut['ext_like']
det_like_eRASS = t_clus_eRASS_uncut['det_like_0']
index = ((richness_eRASS > 30) & (ext_like_eRASS > 0) & (det_like_eRASS > 5))
print(index)
t_clus_eRASS = t_clus_eRASS_uncut[index]
id_clus_eRASS = t_clus_eRASS['id_src']#[index]
print(id_clus_eRASS)
ra_bcg_eRASS = t_clus_eRASS['ra']#[index]
dec_bcg_eRASS = t_clus_eRASS['dec']#[index]
ra_clus_eRASS = t_clus_eRASS['ra_orig']#[index]
dec_clus_eRASS = t_clus_eRASS['dec_orig']#[index]
z_lambda_eRASS = t_clus_eRASS['z_lambda']#[index]


print('computing offset...')
dist_col_eRASS = cosmo.comovingDistance(z_min=0.,z_max=np.array(z_lambda_eRASS))*u.pc*1e6/0.6777 #/h
print('colossus = ', dist_col_eRASS)
dist_eRASS = cosmo_astropy.comoving_distance(np.array(z_lambda_eRASS)).to(u.pc)
print('astropy = ', dist)
bcg_eRASS = SkyCoord(np.array(ra_bcg_eRASS)*u.degree,np.array(dec_bcg_eRASS)*u.degree,frame='fk5',distance=dist_eRASS)
clus_eRASS = SkyCoord(np.array(ra_clus_eRASS)*u.degree, np.array(dec_clus_eRASS)*u.degree, frame='fk5',distance=dist_eRASS)

sep_eRASS = bcg_eRASS.separation_3d(clus_eRASS)
#get separation in kpc (same as Xoff in distinct catalogs)
sep_eRASS = sep_eRASS.value*1e-3

#get xoff from the data and get its PDF
#xoff_data = sep/rvir
xoff_data_eRASS = sep_eRASS
binning_eRASS = np.linspace(min(xoff_data_eRASS),max(xoff_data_eRASS),20)
pdf_eRASS, b_eRASS = np.histogram(np.log10(xoff_data_eRASS),bins=binning,density=True)
bins_eRASS = (b_eRASS[:-1]+b_eRASS[1:])/2
indsort_eRASS = np.argsort(xoff_data_eRASS)
xoff_data_eRASS_sort = xoff_data_eRASS[indsort_eRASS]
cdf_eRASS = np.arange(1,len(xoff_data_eRASS_sort)+1)/len(xoff_data_eRASS_sort)
ind_new_eRASS = []
for i in range(len(cdf_eRASS)):
    ind_new_eRASS.append(int(np.argwhere(indsort_eRASS==i)))
cdf_eRASS_back = cdf_eRASS[ind_new_eRASS]
t_clus_eRASS.add_column(Column(name='Sep',data=xoff_data_eRASS,unit=''))
t_clus_eRASS.add_column(Column(name='cdf',data=cdf_eRASS_back,unit=''))
outt_eRASS = os.path.join(direct,'eRASS','decals_dr8_run_redmapper_v0.6.6_lgt30_catalog_eRASS_clusters_cdf.fit')
t_clus_eRASS.write(outt_eRASS,overwrite=True)
print(cdf_eRASS)


#work on shear selected efeds clusters

dfr = pd.read_csv(path_2_clusters_shear_sel, sep='\t', header=None, dtype='a')
print(dfr)
dist_shearsel = pd.to_numeric(dfr[9][1:].values)
dist_shear_sort = np.sort(dist_shearsel)
cdf_shear_sel = np.arange(1,len(dist_shear_sort)+1)/len(dist_shear_sort)

#ota2020

displ_ota = np.array([52,18,239,22,20,40,76,23,228,17,40,171,109,133,41,260,5,111,74,113,188,102,17,26,93,187,30,129,129,279,64,189,131,15,196,166,82])
displ_ota_sort = np.sort(displ_ota)
cdf_ota = np.arange(1,len(displ_ota_sort)+1)/len(displ_ota_sort)

#mann2012
displ_mann = np.array([357.3,279.3,50.7,23.7,130.3,98.1,69.7,72.5,32.7,463.1,138.8,90.8,316.5,147.5,61.8,23.5,180.1,107.3,88.9,96.1,319.7,129.1,44.8,31.4,155.8,79, 21.3,11.8,53.9,103.3,38.9,47.3,15.1,24.1,35.9,67.3,119.9,70.1,25.5,48.1,89.9,8.3,30.8,18,9.1,5.7,70.5,23.8,10.2,33.5,59.9,19.4,10.5,114,33.8,16.8,32.5,37.7,21.5,34.7, 15.5,7.1,2.5,14.1,7.2,4.1,14.8,5.7,20.5,19.5,25.6,9.9,5.6,22.0,10.9,14.4,21.4,9.9,5.4,14.6,20.8,19.2,20.1,7.6,7,27.3,2.5,32.6,10.3,5.9,4.9,5.3,10,10.8,12.2,22.2,12.9, 3.9,7.9,7.7,7.8,13.7,7.3,8.0,26.7,21.7,19.7])

displ_mann_sort = np.sort(displ_mann)
cdf_mann = np.arange(1,len(displ_mann_sort)+1)/len(displ_mann_sort)

#rossetti2016
displ_rossetti = np.array([143.8, 2.3, 48.4, 3.9, 7.2, 71.9, 2.8, 0.3, 20.1, 14, 2, 204.7, 8.6, 32.4, 3.9, 1015.8, 9.1, 185.7, 6.2, 54, 3.2, 157.1, 38.3, 53.1, 24.8, 0.7, 242.2, 341.3, 13.8, 7.2, 33.1, 4.8, 31.6, 160.5, 123.7, 716.9, 33.9, 96.2, 1.7, 250.2, 16.7, 45.6, 6.4, 3.7, 9.2, 2.7, 42.4, 58.9, 11.6, 7.1, 51.4, 7.9, 6.3, 8.4, 77.5, 10.5, 401, 2.6, 234.7, 6.3, 7.3, 12.2, 10.3, 11.4, 34.3, 192.6, 10, 218, 2.3, 726.4, 163.5, 225.3, 5.2, 65.4, 23.7, 15.7, 1004, 20.4, 1.3, 390.3, 29.3, 16.3, 89.6, 200.1, 29.2, 112.6, 349.6, 22.7, 18.8, 565.5, 13.8, 14.9, 2.3, 3.5, 581.5, 28.7, 24.8, 16.8, 7.5, 996.3, 87.9, 58.8, 168.9, 175.4, 25.8, 12.2, 69.3, 3.3, 814.2, 2.2, 5.7, 143.7, 3.2, 6.4, 1.7, 5.4, 89.5, 59.7, 1.6, 11.6, 7.6, 3.7, 12.4, 65.8, 3.3, 212, 7.1, 88.9, 15.1, 444.6, 25.3, 11.8]) 
displ_rossetti_sort = np.sort(displ_rossetti)
cdf_rossetti = np.arange(1,len(displ_rossetti_sort)+1)/len(displ_rossetti_sort)

#lightcone
t10 = Table.read(path2lightcone_MD10)
displ_lightcone_10 = t10['HALO_Xoff']/h*2/np.pi
#displ_lc_sort_10 = np.sort(displ_lightcone_10)
#cdf_lightcone_10 = np.arange(1,len(displ_lc_sort_10)+1)/len(displ_lc_sort_10)

t40 = Table.read(path2lightcone_MD40)
displ_lightcone_40 = t40['HALO_Xoff']/h*2/np.pi
#displ_lc_sort_40 = np.sort(displ_lightcone_40)
#cdf_lightcone_40 = np.arange(1,len(displ_lc_sort_40)+1)/len(displ_lc_sort_40)

index10_spiders = (t10['HALO_Mvir']/h>7e13) & (t10['redshift_S']<0.67) & (t10['redshift_S']>0.01) & (t10['HALO_pid']==-1) & (t10['CLUSTER_FX_soft']>1e-13) & ((t10['RA'].all()>110.2 and t10['RA'].all()<261.6 and t10['DEC'].all()>16 and t10['DEC'].all()<60.5) or (t10['RA'].all()>0 and t10['RA'].all()<43.2 and t10['DEC'].all()>-5.5 and t10['DEC'].all()<35.3))
#index40_spiders = (t40['HALO_Mvir']/h>7e13) & (t40['redshift_S']<0.67) & (t40['redshift_S']>0.01) & (t40['HALO_pid']==-1) & (t40['CLUSTER_FX_soft']>1e-13) & ((t40['RA'].all()>110.2 and t40['RA'].all()<261.6 and t40['DEC'].all()>16 and t40['DEC'].all()<60.5) or (t40['RA'].all()>0 and t40['RA'].all()<43.2 and t40['DEC'].all()>-5.5 and t40['DEC'].all()<35.3))
#displ_lightcone_concat_spiders_ = np.append(displ_lightcone_10[index10_spiders],displ_lightcone_40[index40_spiders])
displ_lightcone_concat_spiders_ = displ_lightcone_10[index10_spiders] #+ shift[2]
displ_lightcone_concat_spiders_low_ = displ_lightcone_10[index10_spiders] #+ shift[2] - shift[0]
displ_lightcone_concat_spiders_up_ = displ_lightcone_10[index10_spiders] + shift[4] #+ shift[0]
#displ_lc_concat_sort_spiders = np.sort(displ_lightcone_concat_spiders)
displ_lc_concat_sort_spiders_low = np.sort(displ_lightcone_concat_spiders_low_)
displ_lc_concat_sort_spiders_up = np.sort(displ_lightcone_concat_spiders_up_)
displ_lc_concat_sort_spiders = np.sort(displ_lightcone_concat_spiders_)
cdf_lightcone_concat_spiders_low = np.arange(1,len(displ_lc_concat_sort_spiders_low)+1)/len(displ_lc_concat_sort_spiders_low)
cdf_lightcone_concat_spiders_up = np.arange(1,len(displ_lc_concat_sort_spiders_up)+1)/len(displ_lc_concat_sort_spiders_up)
cdf_lightcone_concat_spiders = np.arange(1,len(displ_lc_concat_sort_spiders)+1)/len(displ_lc_concat_sort_spiders)


M_vir_ota = hydro_mc.mass_from_mm_relation('500c', 'vir', M=7e13, a=1/(1+0.37),omega_m = 0.307, omega_b = 0.048, sigma8=0.8228, h0=h)
print('%.3g'%(M_vir_ota))
M_vir_mann = hydro_mc.mass_from_mm_relation('500c', 'vir', M=7e13, a=1/(1+0.38),omega_m = 0.307, omega_b = 0.048, sigma8=0.8228, h0=h)
#index10_ota = (t10['HALO_Mvir']/h>M_vir_ota) & (t10['redshift_S']<1.1) & (t10['redshift_S']>0.1) & (t10['HALO_pid']==-1)
index10_ota = (t10['HALO_Mvir']/h>M_vir_ota) & (t10['redshift_S']<1.1) & (t10['redshift_S']>0.1) & (t10['HALO_pid']==-1) & (t10['CLUSTER_FX_soft']>2e-14) & ((t10['RA'].all()>0 and t10['RA'].all()<14.4 and t10['DEC'].all()>-7.2 and t10['DEC'].all()<7.2))
#displ_lightcone_concat_ota_ = np.append(displ_lightcone_10[index10_ota],displ_lightcone_40[index40_ota])
displ_lightcone_concat_ota_ = displ_lightcone_10[index10_ota]
displ_lc_concat_sort_ota = np.sort(displ_lightcone_concat_ota_)
cdf_lightcone_concat_ota = np.arange(1,len(displ_lc_concat_sort_ota)+1)/len(displ_lc_concat_sort_ota)

#index10_mann = (t10['HALO_Mvir']/h>M_vir_mann) & (t10['redshift_S']<0.7) & (t10['redshift_S']>0.15) & (t10['HALO_pid']==-1)
index10_mann = (t10['redshift_S']<0.7) & (t10['redshift_S']>0.15) & (t10['HALO_pid']==-1) & (t10['CLUSTER_FX_soft']>1e-12) & ((t10['DEC'].all()>-40 and t10['DEC'].all()<80)) & (t10['CLUSTER_LX_soft']>44.7)
#displ_lightcone_concat_mann_ = np.append(displ_lightcone_10[index10_mann],displ_lightcone_40[index40_mann])
displ_lightcone_concat_mann_ = displ_lightcone_10[index10_mann]
displ_lc_concat_sort_mann = np.sort(displ_lightcone_concat_mann_)
cdf_lightcone_concat_mann = np.arange(1,len(displ_lc_concat_sort_mann)+1)/len(displ_lc_concat_sort_mann)


#make prediction from the model
model = Table.read(path_2_model)
pars_model = model['pars']
zevo = Table.read(path_2_zevo)
zevo_pars = zevo['pars']

parameters = np.append(pars_model,zevo_pars)

#colossus wants masses in Msun/h, so if I want to use physical 5e13 Msun, I will give him 5e13*h= 3.39e13 Msun/h
M1 = 5e13*h
R1 = peaks.lagrangianR(M1)
sigma1 = cosmo.sigma(R1,z=0)
log1_sigma1 = np.log10(1/sigma1)

M2 = 2e14*h
R2 = peaks.lagrangianR(M2)
sigma2 = cosmo.sigma(R2,z=0)
log1_sigma2 = np.log10(1/sigma2)

M3 = 1e15*h
R3 = peaks.lagrangianR(M3)
sigma3 = cosmo.sigma(R3,z=0)
log1_sigma3 = np.log10(1/sigma3)

print('%.3g Msun'%(M1/h),' is ',log1_sigma1)
print('%.3g Msun'%(M2/h),' is ',log1_sigma2)
print('%.3g Msun'%(M3/h),' is ',log1_sigma3)

s_edges1 = np.arange(log1_sigma1,0.5,1e-2)
s_edges2 = np.arange(log1_sigma2,0.5,1e-2)
s_edges3 = np.arange(log1_sigma3,0.5,1e-2)
xoff_edges = np.linspace(-0.7,4.0,75)
spin_edges = np.linspace(-4.5,-0.12,51)
s_bins1 = (s_edges1[1:]+s_edges1[:-1])/2
s_bins2 = (s_edges2[1:]+s_edges2[:-1])/2
s_bins3 = (s_edges3[1:]+s_edges3[:-1])/2
xoff_bins = (xoff_edges[:-1] + xoff_edges[1:])/2
spin_bins = (spin_edges[1:]+spin_edges[:-1])/2

xoff_grid1, spin_grid1, s_grid1 = np.meshgrid(xoff_bins,spin_bins,s_bins1)
xoff_grid2, spin_grid2, s_grid2 = np.meshgrid(xoff_bins,spin_bins,s_bins2)
xoff_grid3, spin_grid3, s_grid3 = np.meshgrid(xoff_bins,spin_bins,s_bins3)

def h_func(data,A,a,q,mu,alpha,beta,e0,gamma,delta,e1,k0,k1,k2,k3,k4,k5,k6,k7,k8,k9):
    x_,y_,z_ = data      #x_ is log10(1/sigma) y_ is log10(Xoff)
    x = 1/10**x_ #sigma
    y = 10**y_   #Xoff
    z = 10**z_ #spin
    #opz = (1+np.average(z_sp))
    opz = (1+0.357)    
    return A*(opz)**k0+np.log10(np.sqrt(2/np.pi)) + (q*(opz)**k2)*np.log10(np.sqrt(a*(opz)**k1)*dc/x) - a*(opz)**k1/2/np.log(10)*dc**2/x**2 + (alpha*(opz)**k4)*np.log10(y/10**(mu*(opz)**k3)/x**e0) - 1/np.log(10)*(y/10**(mu*(opz)**k3)/(x**(e0*(opz)**k6)))**(0.05*alpha*(opz)**k4) + (gamma*(opz)**k7)*np.log10(z/(0.7*10**(mu*(opz)**k3))) - 1/np.log(10)*(y/10**(mu*(opz)**k3)/x**(e1*(opz)**k9))**(beta*(opz)**k5)*(z/(0.7*10**(mu*(opz)**k3)))**(delta*(opz)**k8)
  #  return A*(opz)**k0+np.log10(np.sqrt(2/np.pi)) + (q*(opz)**k2)*np.log10(np.sqrt(a*(opz)**k1)*dc/x) - a*(opz)**k1/2/np.log(10)*dc**2/x**2 + (alpha*(opz)**k4)*np.log10(y/10**(1.83*mu*(opz)**k3)) - 1/np.log(10)*(y/10**(1.83*mu*(opz)**k3))**(0.05*alpha*(opz)**k4) + (gamma*(opz)**k6)*np.log10(z/(10**(mu*(opz)**k3))) - 1/np.log(10)*(y/10**(1.83*mu*(opz)**k3)/x**e*(opz)**k8)**(beta*(opz)**k5)*(z/(10**(mu*(opz)**k3)))**(delta*(opz)**k7)

x_data1 = [s_grid1,xoff_grid1,spin_grid1]
print(parameters)
h_seppi20_1 = 10**h_func(x_data1,*parameters)

g_sigma_xoff1 = np.zeros((len(s_bins1),len(xoff_bins)))

for i in range(len(s_bins1)):
    for j in range(len(xoff_bins)):
        g_sigma_xoff1[i,j] = integrate.simps(h_seppi20_1[:,j,i],spin_bins)


f_xoff1 = np.zeros(len(xoff_bins))
for i in range(len(xoff_bins)):
    f_xoff1[i] = integrate.simps(g_sigma_xoff1[:,i],s_bins1)

cdf_model1 = np.zeros(len(xoff_bins))
cdf_model1[0] = f_xoff1[0]
for i in range(1,len(cdf_model1)):
    cdf_model1[i] = np.sum(f_xoff1[:i])
cdf_model1 = cdf_model1/np.max(cdf_model1)

x_data2 = [s_grid2,xoff_grid2,spin_grid2]
h_seppi20_2 = 10**h_func(x_data2,*parameters)

g_sigma_xoff2 = np.zeros((len(s_bins2),len(xoff_bins)))

for i in range(len(s_bins2)):
    for j in range(len(xoff_bins)):
        g_sigma_xoff2[i,j] = integrate.simps(h_seppi20_2[:,j,i],spin_bins)


f_xoff2 = np.zeros(len(xoff_bins))
for i in range(len(xoff_bins)):
    f_xoff2[i] = integrate.simps(g_sigma_xoff2[:,i],s_bins2)

cdf_model2 = np.zeros(len(xoff_bins))
cdf_model2[0] = f_xoff2[0]
for i in range(1,len(cdf_model2)):
    cdf_model2[i] = np.sum(f_xoff2[:i])
cdf_model2 = cdf_model2/np.max(cdf_model2)

x_data3 = [s_grid3,xoff_grid3,spin_grid3]
h_seppi20_3 = 10**h_func(x_data3,*parameters)

g_sigma_xoff3 = np.zeros((len(s_bins3),len(xoff_bins)))

for i in range(len(s_bins3)):
    for j in range(len(xoff_bins)):
        g_sigma_xoff3[i,j] = integrate.simps(h_seppi20_3[:,j,i],spin_bins)


f_xoff3 = np.zeros(len(xoff_bins))
for i in range(len(xoff_bins)):
    f_xoff3[i] = integrate.simps(g_sigma_xoff3[:,i],s_bins3)

cdf_model3 = np.zeros(len(xoff_bins))
cdf_model3[0] = f_xoff3[0]
for i in range(1,len(cdf_model3)):
    cdf_model3[i] = np.sum(f_xoff3[:i])
cdf_model3 = cdf_model3/np.max(cdf_model3)


#SHIFT cdf model of xoff to cdf of separation data, so you understand by how much you need to correct

xoff_bins_2D = xoff_bins + np.log10(2/np.pi) - np.log10(h)
#do not use the tails of the cdf to compute the shift, use only cdf between 0.05 and 0.95
ind = (cdf_model2 >= 0.025)&(cdf_model2 <= 0.975)
xoff_bins_2D_cut = xoff_bins_2D[ind]

#function for the rescaling
def fit(binning,factor):
    xoff_shifted = factor*10**(binning)
    return xoff_shifted

def fit_mann(binning,factor):
    xoff_shifted = factor*binning
    return xoff_shifted

#interpolate the data to compare similar intervals of the two cdf, between 0.05 and 0.95
yerr = 0.05
f = interp1d(cdf,xoff_data,fill_value="extrapolate")
xoff_interp = f(cdf_model2[ind])
print(xoff_interp)
print(10**xoff_bins_2D_cut)
#fit for the shifting factor
popt, pcov = curve_fit(fit, xoff_bins_2D_cut, xoff_interp)
pvar = np.diag(pcov)
t = Table()
t.add_column(Column(name='shift', data=popt, unit=''))
t.add_column(Column(name='err', data=pvar, unit=''))
out_table = os.path.join(direct,'shift_factor.fit')
os.makedirs(os.path.dirname(out_table), exist_ok=True)
t.write(out_table, overwrite=True)

xoff_bins_shift = np.log10(fit(xoff_bins_2D,popt))
print(popt)
#xoff_bins_shift = np.log10(5*10**(xoff_bins_2D))
print(10**xoff_bins_shift)

#SHIFT MANN MODEL
#interpolate the data to compare similar intervals of the two cdf, between 0.05 and 0.95
f_mann = interp1d(cdf_mann,displ_mann_sort,fill_value="extrapolate")
xoff_interp_mann = f_mann(cdf_model2[ind])
print(xoff_interp_mann)
#fit for the shifting factor
print(xoff_bins_2D_cut)
popt_mann, pcov_mann = curve_fit(fit_mann, 10**xoff_bins_2D_cut, xoff_interp_mann, sigma = 0.05*10**xoff_bins_2D_cut, absolute_sigma=True)
pvar_mann = np.diag(pcov_mann)
t = Table()
t.add_column(Column(name='shift', data=popt_mann, unit=''))
t.add_column(Column(name='err', data=pvar_mann, unit=''))
out_table = os.path.join(direct,'shift_factor_mann.fit')
os.makedirs(os.path.dirname(out_table), exist_ok=True)
t.write(out_table, overwrite=True)
xoff_bins_shift_mann = np.log10(fit_mann(10**xoff_bins_2D,popt_mann))
#xoff_bins_shift_mann2 = np.log10(fit_mann(10**xoff_bins_2D,0.4))


#SHIFT MANN LIGHTCONE
#interpolate the data to compare similar intervals of the two cdf, between 0.05 and 0.95
f_mann_lc = interp1d(cdf_mann,displ_mann_sort,fill_value="extrapolate")
ind_lc = (cdf_lightcone_concat_mann >= 0.025)&(cdf_lightcone_concat_mann <= 0.975)
xoff_interp_mann_lc = f_mann_lc(cdf_lightcone_concat_mann[ind_lc])
#fit for the shifting factor
popt_mann_lc, pcov_mann_lc = curve_fit(fit_mann, displ_lc_concat_sort_mann[ind_lc], xoff_interp_mann_lc, sigma = 0.05*displ_lc_concat_sort_mann[ind_lc], absolute_sigma=True)
pvar_mann_lc = np.diag(pcov_mann_lc)
t = Table()
t.add_column(Column(name='shift', data=popt_mann_lc, unit=''))
t.add_column(Column(name='err', data=pvar_mann_lc, unit=''))
out_table = os.path.join(direct,'shift_factor_mann_lc.fit')
os.makedirs(os.path.dirname(out_table), exist_ok=True)
t.write(out_table, overwrite=True)
xoff_bins_shift_mann_lc = np.log10(fit_mann(displ_lc_concat_sort_mann,popt_mann_lc))
#xoff_bins_shift_mann2 = np.log10(fit_mann(10**xoff_bins_2D,0.4))


#plot
plt.figure(figsize=(10,10))
plt.fill_between(np.log10(xoff_data),cdf-yerr,cdf+yerr,label='SPIDERS',lw=4, color=[1,0,0])
#plt.fill_between(np.log10(xoff_spiders_shift),cdf_spiders_shift-yerr,cdf_spiders_shift+yerr,label='SPIDERS shift',lw=4,color='C3',alpha = 0.5)
plt.fill_between(np.log10(displ_ota_sort),cdf_ota-yerr,cdf_ota+yerr,label='Ota20',lw=4, color=[0,1,0])
plt.fill_between(np.log10(displ_mann_sort),cdf_mann-yerr,cdf_mann+yerr,label='Mann12',lw=4, color=[0,0,1])
#plt.plot(np.log10(xoff_interp),cdf_model2[ind],label='cdf',lw=4,c='C7')
plt.fill_between(np.log10(xoff_data_eFEDS_sort),cdf_eFEDS-yerr,cdf_eFEDS+yerr,label='eFEDS',lw=4, color=[0.5,0.5,0])
plt.fill_between(np.log10(xoff_data_eRASS_sort),cdf_eRASS-yerr,cdf_eRASS+yerr,label='eRASS',lw=4, color=[0,0.5,0.5])
plt.fill_between(np.log10(dist_shear_sort),cdf_shear_sel-yerr,cdf_shear_sel+yerr,label='shear sel',lw=4, color=[0.5,0,0.5])
#plt.plot(np.log10(displ_lc_sort_10),cdf_lightcone_10,label='MD10 lightcone',lw=4,linestyle='dashdot')
#plt.plot(np.log10(displ_lc_sort_40),cdf_lightcone_40,label='MD40 lightcone',lw=4,linestyle='dashdot')
plt.plot(np.log10(displ_lc_concat_sort_spiders),cdf_lightcone_concat_spiders,label='MD spiders',lw=4,linestyle='dashdot', color=[1,0,0])
plt.plot(np.log10(displ_lc_concat_sort_ota),cdf_lightcone_concat_ota,label='MD ota20',lw=4,linestyle='dashdot', color=[0,1,0])
plt.plot(np.log10(displ_lc_concat_sort_mann),cdf_lightcone_concat_mann,label='MD mann12',lw=4,linestyle='dashdot', color=[0,0,1])

plt.plot(xoff_bins_2D,cdf_model1,label=r'$M > %.3g M_\odot$'%(M1/h),lw=4, color=[0.25,0.5,0.25])
plt.plot(xoff_bins_2D,cdf_model2,label=r'$M > %.3g M_\odot$'%(M2/h),lw=4, color=[0.5,0.25,0.25])
plt.plot(xoff_bins_2D,cdf_model3,label=r'$M > %.3g M_\odot$'%(M3/h),lw=4, color=[0.25,0.25,0.5])
plt.plot(xoff_bins_shift,cdf_model2,label='rescale',lw=4,linestyle='dashed', color=[0.5,0.25,0.25])


plt.xlim(0,3.5)
plt.legend(fontsize=15)
plt.grid(True)
plt.tick_params(labelsize=20)
plt.xlabel(r'$\log_{10}S\ [kpc]$',fontsize=20)
plt.ylabel(r'$CDF$',fontsize=20)
plt.tight_layout()
outmodelpdf = os.path.join(direct,'link2obs_all.png')
plt.savefig(outmodelpdf,overwrite=True)

plt.figure(figsize=(10,10))
#plt.fill_between(np.log10(xoff_data),cdf-yerr,cdf+yerr,label='SPIDERS',lw=4,color='C0',alpha = 0.7)
ind = (cdf >= 0.05)&(cdf <= 0.95)
#plt.plot(np.log10(xoff_data[ind]),cdf[ind],label='SPIDERS cut',lw=4,color='b',linestyle='dotted')
#plt.fill_between(np.log10(xoff_spiders_shift),cdf_spiders_shift-yerr,cdf_spiders_shift+yerr,label='SPIDERS shift',lw=4,color='C3',alpha = 0.5)
#plt.fill_between(np.log10(displ_ota_sort),cdf_ota-yerr,cdf_ota+yerr,label='ota20',lw=4,color='C1',alpha=0.7)
#plt.plot(np.log10(displ_lc_sort_10),cdf_lightcone_10,label='MD10 lightcone',lw=4,c='C3',linestyle='dashdot')
#plt.plot(np.log10(displ_lc_sort_40),cdf_lightcone_40,label='MD40 lightcone',lw=4,c='C4',linestyle='dashdot')
#plt.plot(np.log10(displ_lc_concat_sort_spiders),cdf_lightcone_concat_spiders,label='MD spiders',lw=4,linestyle='dashdot')
#plt.fill_betweenx(cdf_lightcone_concat_spiders,np.log10(displ_lc_concat_sort_spiders_low),np.log10(displ_lc_concat_sort_spiders_up),label='MD SPIDERS shift',lw=4,color='C3',alpha = 0.5)
#plt.plot(np.log10(displ_lc_concat_sort_ota),cdf_lightcone_concat_ota,label='MD ota20',lw=4,linestyle='dashdot')
#plt.plot(xoff_bins_2D,cdf_model1,label=r'$M > %.3g M_\odot$'%(M1/h),lw=4,c='C1')
plt.plot(xoff_bins_2D,cdf_model2,label=r'$M > %.3g M_\odot$'%(M2/h),lw=4,c='C0')
#plt.plot(xoff_bins_2D,cdf_model3,label=r'$M > %.3g M_\odot$'%(M3/h),lw=4,c='C0')
#plt.plot(xoff_bins_shift,cdf_model2,label='rescale',lw=4,c='C9',linestyle='dashed')
plt.fill_between(np.log10(displ_mann_sort),cdf_mann-yerr,cdf_mann+yerr,label='Mann12',lw=4,color='C1',alpha=0.6,linewidth=0.0)
plt.fill_between(np.log10(displ_rossetti_sort),cdf_rossetti-yerr,cdf_rossetti+yerr,label='Rossetti16',lw=4,color='C2',alpha=0.6,linewidth=0.0)
plt.plot(np.log10(displ_lc_concat_sort_mann),cdf_lightcone_concat_mann,label='MD mann12',lw=4,color='C1')
plt.plot(xoff_bins_shift_mann,cdf_model2,label='model rescale',lw=4,c='C0',linestyle='dashed')
plt.plot(xoff_bins_shift_mann_lc,cdf_lightcone_concat_mann,label='MD rescale',lw=4,c='C1',linestyle='dashed')
#plt.plot(xoff_bins_shift_mann2,cdf_model2,label='Mann12 rescale2',lw=4,c='r',linestyle='dashed')

plt.xlim(-0.5,3.5)
plt.legend(fontsize=15)
plt.grid(True)
plt.tick_params(labelsize=20)
plt.xlabel(r'$\log_{10}S\ [kpc]$',fontsize=20)
plt.ylabel(r'$CDF$',fontsize=20)
plt.tight_layout()
outmodelpdf = os.path.join(direct,'link2obs_spiders.png')
plt.savefig(outmodelpdf,overwrite=True)
print('spiders sample = ',len(cdf))
print('spiders mock = ',len(cdf_lightcone_concat_spiders))
print('ota sample = ',len(cdf_ota))
print('ota mock = ',len(cdf_lightcone_concat_ota))
print('mann sample = ',len(cdf_mann))
print('mann mock = ',len(cdf_lightcone_concat_mann))
plt.show()





