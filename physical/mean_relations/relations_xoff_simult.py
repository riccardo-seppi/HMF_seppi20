#!/usr/bin/env python
# coding: utf-8

# In[94]:


"""
Build relations for MD
"""

#MAKES THE 2D HISTOGRAM AND CONVERTS THE COUNTS TO g(sigma,xoff)

from astropy.table import Table, Column
#from astropy_healpix import healpy
import sys
import os, glob
import time
from astropy.cosmology import FlatLambdaCDM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import NullFormatter
import astropy.units as u
import astropy.constants as cc
import astropy.io.fits as fits
import scipy
from scipy.stats import chi2
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

print('Analyze sigma(M) - Xoff relation')
print('------------------------------------------------')
print('------------------------------------------------')
t0 = time.time()

#env = 'MD40' 
# initializes pathes to files
#test_dir = os.path.join(os.environ[env], 'Mass_Xoff_Concentration')
test_dir='/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration'
this_dir='.'

#plt.figure(figsize=(10,10))
#path_2_snapshot_data = np.array(glob.glob(os.path.join(test_dir, 'distinct_*.fits')))
path_2_snapshot_data = np.array([os.path.join(test_dir, 'distinct_1.0.fits.gz'),os.path.join(test_dir,'distinct_0.6565.fits.gz'),os.path.join(test_dir,'distinct_0.4922.fits.gz'),os.path.join(test_dir,'distinct_0.4123.fits.gz')])

dir_2_5 = '/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration'

path_2_snapshot_data2_5 = np.array([os.path.join(dir_2_5,'distinct_1.0.fits.gz'),os.path.join(dir_2_5,'distinct_0.6583.fits.gz'),os.path.join(dir_2_5,'distinct_0.5.fits.gz'),os.path.join(dir_2_5,'distinct_0.409.fits.gz')])
dir_1_0 = '/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration'

path_2_snapshot_data1_0 = np.array([os.path.join(dir_1_0,'distinct_1.0.fits.gz'),os.path.join(dir_1_0,'distinct_0.6565.fits.gz'),os.path.join(dir_1_0,'distinct_0.4922.fits.gz'),os.path.join(dir_1_0,'distinct_0.4123.fits.gz')])

dir_0_4 = '/data17s/darksim/simulation_3/MD/MD_0.4Gpc/Mass_Xoff_Concentration'
path_2_snapshot_data0_4 = os.path.join(dir_0_4,'distinct_1.0.fits')
fig1,ax1 = plt.subplots(1,1,figsize=(4.5,5.5))
zpl = np.array([1/1.0-1, 1/0.6565-1, 1/0.4922-1, 1/0.4123-1])
colors = ['b','r','c','m']

cosmo = cosmology.setCosmology('multidark-planck')  

#define arrays used to cut data: low resolution of HMD or low statistic of MDPL 
cuts_HMD_low = np.array([2.2,2.8,3.5,3.7])
cuts_BigMD_low = np.array([1.5,1.8,2.25,3.0])
cuts_BigMD_up = np.array([3.0,3.2,4.7,4.8])
cuts_MDPL_low = np.array([0.9,1.25,1.5,1.95])
cuts_MDPL_up = np.array([2.0,2.3,2.4,3.0])


#def xoff_sigma2(ar,a0,b0,c0,d0,e0,f0,g0,h0):
def xoff_sigma2(ar,a0,b0,c0):
#def xoff_sigma2(ar,a0,b0):
    x,z=ar
    Ez = cosmo.Ez(z)
    sigma = 1/x*dc
    return b0/((Ez)**0.06)*(1+2.39*(sigma/(a0/(Ez)**0.8))**(c0*sigma))



xt = np.arange(0,1,0.1)
zt = np.repeat(0,len(xt))
arr = np.vstack((xt,zt))
dc=1.6
#sys.exit()

t0=Table()
t1=Table()
t2=Table()
t3=Table()
peak_array_full = []
xoff_full = []
xoff_err_full = []
z_full = []
xoff_tot_full =[]

for i, p2s in enumerate(path_2_snapshot_data):
    print('HMD')
    print(i,' of ', len(path_2_snapshot_data)) 
    aexp = float(os.path.basename(p2s[:-8]).split('_')[1])
    z_snap = 1/aexp -1
    print('z=%.3g'%(z_snap))  
    E_z = cosmo.Ez(z=z_snap)
    Vol1 = (4e3/(1+z_snap))**3
    dc = peaks.collapseOverdensity(z = z_snap)
    rho_m = cosmo.rho_m(z=z_snap)*1e9

    hd1 = fits.open(p2s)

    mass1=hd1[1].data['Mvir']
    logmass1 = np.log10(mass1)
    R_1 = peaks.lagrangianR(mass1)
    sigf_1 = cosmo.sigma(R_1,z=z_snap)
    log1_sigf_1 = np.log10(1/sigf_1)
    Rvir1 = hd1[1].data['Rvir']
    Rs1 = hd1[1].data['Rs']
    xoff_data1 = np.log10(hd1[1].data['Xoff'])#/hd1[1].data['Rvir'])
    spin1 = hd1[1].data['Spin']
    spinpar1 = hd1[1].data['Spin_Bullock']

    conc1 = Rvir1/Rs1
    peak_bins = np.arange(0.9,5.,0.02)
    peak_array = (peak_bins[:-1]+peak_bins[1:])/2.

    def get_average(x,sel):
        return np.average(x[sel]),np.std(x[sel]),np.sum(sel)

    def get_median(x,sel):
        return np.median(x[sel])
    
    xoff_av1 = np.zeros(len(peak_array))
    xoff_std1 = np.zeros(len(peak_array))
    xoff_N1 = np.zeros(len(peak_array))

    z1 = np.repeat(z_snap, len(peak_array))
         
#BigMD
    print('BigMD')
    hd2 = fits.open(path_2_snapshot_data2_5[i])

    mass2=hd2[1].data['Mvir']
    logmass2 = np.log10(mass2)
    R_2 = peaks.lagrangianR(mass2)
    sigf_2 = cosmo.sigma(R_2,z=z_snap)
    log1_sigf_2 = np.log10(1/sigf_2)
    Rvir2 = hd2[1].data['Rvir']
    Rs2 = hd2[1].data['Rs']
    xoff_data2 = np.log10(hd2[1].data['Xoff'])#/hd1[1].data['Rvir'])
    spin2 = hd2[1].data['Spin']
    spinpar2 = hd2[1].data['Spin_Bullock']

    conc2 = Rvir2/Rs2
        
    xoff_av2 = np.zeros(len(peak_array))
    xoff_std2 = np.zeros(len(peak_array))
    xoff_N2 = np.zeros(len(peak_array))
    z2 = np.repeat(z_snap, len(peak_array))

#MDPL
    print('MDPL')
    hd3 = fits.open(path_2_snapshot_data1_0[i])

    mass3=hd3[1].data['Mvir']
    logmass3 = np.log10(mass3)
    R_3 = peaks.lagrangianR(mass3)
    sigf_3 = cosmo.sigma(R_3,z=z_snap)
    log1_sigf_3 = np.log10(1/sigf_3)
    Rvir3 = hd3[1].data['Rvir']
    Rs3 = hd3[1].data['Rs']
    xoff_data3 = np.log10(hd3[1].data['Xoff'])#/hd1[1].data['Rvir'])
    spin3 = hd3[1].data['Spin']
    spinpar3 = hd3[1].data['Spin_Bullock']

    conc3 = Rvir3/Rs3

    xoff_tot_ = np.hstack((xoff_data1,xoff_data2,xoff_data3))
    sigf_tot_ = np.hstack((sigf_1,sigf_2,sigf_3))

    xoff_tot_full.extend(xoff_tot_)    
    xoff_av3 = np.zeros(len(peak_array))
    xoff_std3 = np.zeros(len(peak_array))
    xoff_N3 = np.zeros(len(peak_array))

    z3 = np.repeat(z_snap, len(peak_array))

    
    print('computing values...')
    for jj, (x_min,x_max) in enumerate(zip(peak_bins[:-1],peak_bins[1:])):
        xoff_av1[jj],xoff_std1[jj],xoff_N1[jj] = get_average(xoff_data1,(dc/sigf_1>=x_min) & (dc/sigf_1<x_max))
        xoff_av2[jj],xoff_std2[jj],xoff_N2[jj] = get_average(xoff_data2,(dc/sigf_2>=x_min) & (dc/sigf_2<x_max)) 
        xoff_av3[jj],xoff_std3[jj],xoff_N3[jj] = get_average(xoff_data3,(dc/sigf_3>=x_min) & (dc/sigf_3<x_max)) 
    print('values computed!')


#computing averages on each cube
    ind_one = ((peak_array > cuts_HMD_low[i]) & (~np.isnan(xoff_av1)) & (xoff_N1 > 100))
    xoff_err_1 = xoff_std1[ind_one]/np.sqrt(xoff_N1[ind_one])
    peak_array_1 = np.array(peak_array[ind_one])
    z1_ = np.array(z1[ind_one])
    xoff_av_1 = np.array(xoff_av1[ind_one])
    #xoff_err_ = 10*np.array(xoff_err[~np.isnan(xoff_av)])
    #xoff_err_1 = 0.1*xoff_av_1

    ind_two = ((peak_array > cuts_BigMD_low[i]) & (peak_array < cuts_BigMD_up[i]) & (~np.isnan(xoff_av2))& (xoff_N2 > 100))
    xoff_err_2 = xoff_std2[ind_two]/np.sqrt(xoff_N2[ind_two])
    peak_array_2 = np.array(peak_array[ind_two])
    z2_ = np.array(z2[ind_two])
    xoff_av_2 = np.array(xoff_av2[ind_two])
    #xoff_err_2 = 0.1*xoff_av_2

    ind_three = ((peak_array > cuts_MDPL_low[i]) & (peak_array < cuts_MDPL_up[i]) & (~np.isnan(xoff_av3)) & (xoff_N3 > 100))
    xoff_err_3 = xoff_std3[ind_three]/np.sqrt(xoff_N3[ind_three])
    peak_array_3 = np.array(peak_array[ind_three])
    z3_ = np.array(z3[ind_three])
    xoff_av_3 = np.array(xoff_av3[ind_three])
    #xoff_err_3 = 0.1*xoff_av_3
    ax1.scatter(peak_array_1,xoff_av_1, label = r'$z= %.3g\ HMD$'%(z_snap), ls='None',c='%.c'%(colors[i]),marker='o',facecolors='none',s=13)
    ax1.scatter(peak_array_2,xoff_av_2, label = r'$z= %.3g\ BigMD$'%(z_snap), ls='None', edgecolors='%.c'%(colors[i]), marker='^',facecolors='none',s=13)
    ax1.scatter(peak_array_3,xoff_av_3, label = r'$z= %.3g\ MDPL$'%(z_snap), ls ='None', edgecolors='%.c'%(colors[i]), marker='s',facecolors='none',s=13)

    peak_array_ = np.hstack((peak_array_1,peak_array_2,peak_array_3))
    xoff_av_ = np.hstack((xoff_av_1,xoff_av_2,xoff_av_3))
    xoff_err_ = np.hstack((xoff_err_1,xoff_err_2,xoff_err_3))
    z_ = np.hstack((z1_,z2_,z3_))

    ind1 = np.argsort(peak_array_)

    peak_array_full.extend(peak_array_)
    xoff_full.extend(xoff_av_) 
    xoff_err_full.extend(xoff_err_)  
    z_full.extend(z_)


peak_array_full = np.array(peak_array_full)
xoff_full = np.array(xoff_full)
xoff_err_full = np.array(xoff_err_full)
z_full = np.array(z_full)


xdata = np.vstack((peak_array_full,z_full))
print("peak array shape full = ",peak_array_full.shape)
print('zfull shape = ',z_full.shape)
print('xdata shape = ',xdata.shape)
xdata_rav = np.vstack((peak_array_full.ravel(),z_full.ravel()))
print('xdata rav shape = ',xdata_rav.shape)


popt1,pcov1 = curve_fit(xoff_sigma2,xdata,xoff_full,sigma=xoff_err_full,maxfev=10000000,p0=[(0.17,0.75,-0.4)])   
t1.add_column(Column(name='params',data=popt1,unit=''))
t1.add_column(Column(name='errs',data=np.diag(pcov1),unit=''))

z0 = np.repeat(zpl[0],len(peak_array))
z1 = np.repeat(zpl[1],len(peak_array))
z2 = np.repeat(zpl[2],len(peak_array))
z3 = np.repeat(zpl[3],len(peak_array))

red_arr = [z0,z1,z2,z3]

for k,red in enumerate(red_arr):
    ax1.plot(peak_array,xoff_sigma2([peak_array,red],*popt1),c='%.c'%(colors[k]))

outt1 = os.path.join(this_dir,'tables','xoff_sigma_params.fit')
t1.write(outt1,overwrite=True)

h=cosmo.Hz(z=0)/100
def Mass_sigma(x):
    r=cosmo.sigma(1.68/x,z=0,inverse=True)
    M=peaks.lagrangianM(r)/h
    return np.log10(M)

def Xoff_log(x):
    y = 10**x
    return y/h

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(left=0.8)
ax1.set_xticks([0.8,1,2,3,4])
ax1.set_yticks([1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4])
ax1.set_ylim(1.0,2.4)

h=cosmo.Hz(z=0)/100
dc0 = peaks.collapseOverdensity(z = 0)
def Mass_peak(x):
    r=cosmo.sigma(dc0/x,z=0,inverse=True)
    M=peaks.lagrangianM(r)#/h
    return np.log10(M)

def peak_mass(x):
    M=10**x
    r=peaks.lagrangianR(M)#*h)
    sigma=cosmo.sigma(r,z=0)
    nu=dc/sigma
    return nu

ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.yaxis.set_major_formatter(ScalarFormatter())
ax1.ticklabel_format(axis='both', style='plain')

ax1_sec = ax1.twiny()
ax1_sec.set_xscale('log')
ax1_sec.set_xlim(ax1.get_xlim())

mass_values = np.array([13.0,14.0,14.5,15.0,15.5])
new_tick_locations = peak_mass(mass_values)
print(mass_values)
print(new_tick_locations)
ax1_sec.xaxis.set_major_formatter(NullFormatter())
ax1_sec.xaxis.set_minor_formatter(NullFormatter())
ax1_sec.tick_params(axis='x', which='minor', top=False)
ax1.tick_params(axis='x', which='minor', bottom=False)
ax1_sec.set_xticks(new_tick_locations)
ax1_sec.set_xticklabels(mass_values)

ax1.legend(fontsize=8,bbox_to_anchor=(-0.3, 1.16, 1.3, .3), loc='lower left', ncol=3, mode="expand", borderaxespad=0.)
ax1.set_xlabel(r'$\nu$ = $\delta_{\rm c}$/$\sigma$', fontsize=12)
ax1.set_ylabel(r'$\log_{10}$X$_{\rm off,P}$', fontsize=12)
ax1_sec.set_xlabel(r'$\log_{10}$M [M$_{\odot}$/h]', fontsize=12)
ax1_sec.tick_params(labelsize=12)
ax1.tick_params(labelsize=12)
#ax1.grid(True)
fig1.tight_layout()
outfi1 = os.path.join(this_dir,'figures','relation_xoff_sigma.png')
os.makedirs(os.path.dirname(outfi1), exist_ok=True)
fig1.savefig(outfi1, overwrite=True)

print('done!')
sys.exit()







