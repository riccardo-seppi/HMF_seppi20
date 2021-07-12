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

print('Analyze sigma(M) - spin relation')
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
fig2,ax2 = plt.subplots(1,1,figsize=(4.5,5.5))
zpl = np.array([1/1.0-1, 1/0.6565-1, 1/0.4922-1, 1/0.4123-1])
colors = ['b','r','c','m']

cosmo = cosmology.setCosmology('multidark-planck')  

#define arrays used to cut data: low resolution of HMD or low statistic of MDPL 
cuts_HMD_low = np.array([2.2,2.8,3.5,3.7])
cuts_BigMD_low = np.array([1.5,1.8,2.25,3.0])
cuts_BigMD_up = np.array([3.0,3.2,4.7,4.8])
cuts_MDPL_low = np.array([0.9,1.25,1.5,1.95])
cuts_MDPL_up = np.array([2.0,2.3,2.4,3.0])

def spin_sigma(ar,a0,b0):
    x,z=ar
  #  return b0*((x*a0)**c0)*(np.exp(-x*d0))#/(1+z)**e0
    return a0 + b0*x
#    return b0*(x/a0)**(c0)
    
def spinpar_sigma(ar,a0,b0,c0,d0,e0):
    x,z=ar
  #  return b0*((x*a0)**c0)*(np.exp(-x*d0))#/(1+z)**e0
    return 1/(1+z)**0.1 *(a0 + b0*x +c0*x**2 + d0*x**3)

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
spin_full = []
spin_err_full = []
z_full = []

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

    spin_av1 = np.zeros(len(peak_array))
    spin_std1 = np.zeros(len(peak_array))
    spin_N1 = np.zeros(len(peak_array))
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
    spin_av2 = np.zeros(len(peak_array))
    spin_std2 = np.zeros(len(peak_array))
    spin_N2 = np.zeros(len(peak_array))

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

    spin_tot_ = np.hstack((spin1,spin2,spin3))
    sigf_tot_ = np.hstack((sigf_1,sigf_2,sigf_3))

    spin_av3 = np.zeros(len(peak_array))
    spin_std3 = np.zeros(len(peak_array))
    spin_N3 = np.zeros(len(peak_array))

    z3 = np.repeat(z_snap, len(peak_array))

    
    print('computing values...')
    for jj, (x_min,x_max) in enumerate(zip(peak_bins[:-1],peak_bins[1:])):
        spin_av1[jj],spin_std1[jj],spin_N1[jj] = get_average(spin1,(dc/sigf_1>=x_min) & (dc/sigf_1<x_max))
        spin_av2[jj],spin_std2[jj],spin_N2[jj] = get_average(spin2,(dc/sigf_2>=x_min) & (dc/sigf_2<x_max)) 
        spin_av3[jj],spin_std3[jj],spin_N3[jj] = get_average(spin3,(dc/sigf_3>=x_min) & (dc/sigf_3<x_max)) 
    print('values computed!')

    #plt.figure(figsize=(10,10))
    #plt.scatter(logxoff_data1,log1_sigf_1, label = r'$z= %.3g$'%(z_snap))
    #plt.scatter(peak_array,spin_av1, label = r'$z= %.3g$'%(z_snap))
    #plt.fill_between(peak_array,spin_av1-spin_std1/np.sqrt(spin_N1),spin_av1+spin_std1/np.sqrt(spin_N1),alpha=0.5)
    #plt.tick_params(labelsize=15)
    #plt.legend(fontsize=15)
    #plt.xlabel(r'$\delta_c/\sigma$', fontsize=20)
    #plt.ylabel(r'$Spin$', fontsize=20)
    #plt.grid(True)
    #outf = os.path.join(this_dir,'figures','spin_sigma_%.3g.png'%(z_snap))
    #os.makedirs(os.path.dirname(outf), exist_ok=True)
    #plt.savefig(outf, overwrite=True)

#computing averages on each cube
    ind_one = ((peak_array > cuts_HMD_low[i]) & (~np.isnan(spin_av1)) & (spin_N1 > 100))
    spin_err_1 = spin_std1[ind_one]/np.sqrt(spin_N1[ind_one])
    peak_array_1 = np.array(peak_array[ind_one])
    z1_ = np.array(z1[ind_one])
    spin_av_1 = np.array(spin_av1[ind_one])
    #spin_err_1 = 0.1*spin_av_1
    spin_N1_ = np.array(spin_N1[ind_one])

    ind_two = ((peak_array > cuts_BigMD_low[i]) & (peak_array < cuts_BigMD_up[i]) & (~np.isnan(spin_av2))& (spin_N2 > 100))
    spin_err_2 = spin_std2[ind_two]/np.sqrt(spin_N2[ind_two])
    peak_array_2 = np.array(peak_array[ind_two])
    z2_ = np.array(z2[ind_two])
    spin_av_2 = np.array(spin_av2[ind_two])
    spin_err_2 = 0.1*spin_av_2
    spin_N2_ = np.array(spin_N2[ind_two])

    ind_three = ((peak_array > cuts_MDPL_low[i]) & (peak_array < cuts_MDPL_up[i]) & (~np.isnan(spin_av3)) & (spin_N3 > 100))
    spin_err_3 = spin_std3[ind_three]/np.sqrt(spin_N3[ind_three])
    peak_array_3 = np.array(peak_array[ind_three])
    z3_ = np.array(z3[ind_three])
    spin_av_3 = np.array(spin_av3[ind_three])
    #spin_err_3 = 0.1*spin_av_3
    spin_N3_ = np.array(spin_N3[ind_three])

    ax2.scatter(peak_array_1,spin_av_1, label = r'$z= %.3g\ HMD$'%(z_snap), ls='None',c='%.c'%(colors[i]),marker='o',facecolors='none',s=13)
    ax2.scatter(peak_array_2,spin_av_2, label = r'$z= %.3g\ BigMD$'%(z_snap), ls='None', edgecolors='%.c'%(colors[i]), marker='^',facecolors='none',s=13)
    ax2.scatter(peak_array_3,spin_av_3, label = r'$z= %.3g\ MDPL$'%(z_snap), ls ='None', edgecolors='%.c'%(colors[i]), marker='s',facecolors='none',s=13)
    peak_array_ = np.hstack((peak_array_1,peak_array_2,peak_array_3))
    spin_av_ = np.hstack((spin_av_1,spin_av_2,spin_av_3))
    spin_err_ = np.hstack((spin_err_1,spin_err_2,spin_err_3))
    z_ = np.hstack((z1_,z2_,z3_))

    ind1 = np.argsort(peak_array_)


    peak_array_full.extend(peak_array_)
    spin_full.extend(spin_av_)
    spin_err_full.extend(spin_err_)
    z_full.extend(z_)


peak_array_full = np.array(peak_array_full)
spin_full = np.array(spin_full)
spin_err_full = np.array(spin_err_full)
z_full = np.array(z_full)


xdata = np.vstack((peak_array_full,z_full))
xdata_rav = np.vstack((peak_array_full.ravel(),z_full.ravel()))

    
popt2,pcov2 = curve_fit(spin_sigma,xdata,spin_full,sigma=spin_err_full,maxfev=1000000)   
t2.add_column(Column(name='params',data=popt2,unit=''))
t2.add_column(Column(name='errs',data=np.diag(pcov2),unit=''))


z0 = np.repeat(zpl[0],len(peak_array))
z1 = np.repeat(zpl[1],len(peak_array))
z2 = np.repeat(zpl[2],len(peak_array))
z3 = np.repeat(zpl[3],len(peak_array))

red_arr = [z0,z1,z2,z3]

#for k,red in enumerate(red_arr):

#    ax2.plot(peak_array,spin_sigma([peak_array,red],*popt2),c='%.c'%(colors[k]))
#    ax1[1].plot(peak_array,spinpar_sigma([peak_array,red],*popt3),c='%.c'%(colors[k]))
ax2.plot(peak_array,spin_sigma([peak_array,z0],*popt2),c='k',label='model')

outt2 = os.path.join(this_dir,'tables','spin_sigma_params.fit')
t2.write(outt2,overwrite=True)

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

#ax2.set_xscale('log')
ax2.set_xlim(left=0.8,right=4.2)
ax2.set_xticks([1,2,3,4])
ax2.xaxis.set_major_formatter(ScalarFormatter())
ax2.yaxis.set_major_formatter(ScalarFormatter())
ax2.ticklabel_format(axis='both', style='plain')

ax2_sec = ax2.twiny()
#ax2_sec.set_xscale('log')
ax2_sec.set_xlim(ax2.get_xlim())

mass_values = np.array([13.0,14.0, 14.5, 15.0, 15.5])
new_tick_locations = peak_mass(mass_values)
print(mass_values)
print(new_tick_locations)
ax2_sec.xaxis.set_major_formatter(NullFormatter())
ax2_sec.xaxis.set_minor_formatter(NullFormatter())
ax2_sec.tick_params(axis='x', which='minor', top=False)
ax2_sec.set_xticks(new_tick_locations)
ax2_sec.set_xticklabels(mass_values)
#xmin,xmax=ax2.get_xlim()
#print(xmin,xmax)
#print(Mass_peak(xmin),Mass_peak(xmax))
#ax2_sec.set_xlim(Mass_peak(xmin),Mass_peak(xmax))
#ax2_sec.set_xscale('log')
#ax2_sec.set_xticks(new_tick_locations)

ax2.set_ylabel(r'$\lambda$', fontsize=12)
ax2.set_xlabel(r'$\nu$ = $\delta_{\rm c}$/$\sigma$', fontsize=12)
ax2_sec.set_xlabel(r'$\log_{10}$M [M$_{\odot}$/h]', fontsize=12)
ax2_sec.tick_params(labelsize=12)
#ax2.grid(True)
#ax2.set_xlim(right=4.2)
ax2.set_ylim(0.02)
ax2.legend(fontsize=8,bbox_to_anchor=(-0.3, 1.16, 1.3, .33), loc='lower left', ncol=3, mode="expand", borderaxespad=0.)
ax2.tick_params(labelsize=12,top=False, labeltop=False)
fig2.tight_layout()
outfig2 = os.path.join(this_dir,'figures','relations_spin_sigma.png')
os.makedirs(os.path.dirname(outfig2), exist_ok=True)
fig2.savefig(outfig2, overwrite=True)
print('done!')
sys.exit()







