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

print('Plots sigma(M) - Xoff relation')
print('------------------------------------------------')
print('------------------------------------------------')
t0 = time.time()

test_dir='/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration'
this_dir='.'

zpl = np.array([1/1.0-1, 1/0.6565-1, 1/0.4922-1, 1/0.4123-1])
colors = ['b','r','c','m']

cosmo = cosmology.setCosmology('multidark-planck')  

dc = peaks.collapseOverdensity(z = 0)

def xoff_sigma_pl(ar,a0,b0):
    x,z=ar
    Ez = cosmo.Ez(z)
    sigma = 1/x*dc
    return a0/Ez**0.136*sigma**(b0*Ez**-1.11)


tab_data = Table.read('tables/xoff_data_table.fit')
peak_array_full = tab_data['peak_array_full']
z_full = tab_data['z_full']
xoff_full = tab_data['xoff_full']
xoff_err_full = tab_data['xoff_err_full']
xdata = np.vstack((peak_array_full,z_full))
#popt1,pcov1 = curve_fit(xoff_sigma_pl,xdata,xoff_full,sigma=xoff_err_full,maxfev=10000000,p0=[(-1.3,0.16)])
popt1 = [-1.30418, 0.15084]   
#t1.add_column(Column(name='params',data=popt1,unit=''))
#t1.add_column(Column(name='errs',data=np.diag(pcov1),unit=''))

#make fill between plot 
z_arr = np.unique(z_full)
plt.figure(0,(10,10))

for k,red in enumerate(z_arr):
    sel = (z_full==red)
    sortid = np.argsort(peak_array_full[sel])
    peak = peak_array_full[sel][sortid]
    y = xoff_full[sel][sortid]
    dy = xoff_err_full[sel][sortid]  
    plt.fill_between(peak,y-dy,y+dy,color='%.c'%(colors[k]),label='z = '+str(red),alpha=0.5)
    plt.plot(peak,xoff_sigma_pl([peak,red],*popt1),c='%.c'%(colors[k]))
plt.grid(True)
plt.xscale('log')
plt.xticks([0.6,0.8,1,2,3,4])
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.gca().ticklabel_format(axis='both', style='plain')
plt.legend(fontsize=17)
plt.xlabel(r'$\nu = \delta_c/\sigma$', fontsize=30)
plt.ylabel(r'$\log_{10}X_{off}$', fontsize=30)
plt.tick_params(labelsize=25)
plt.tight_layout()
outf=os.path.join(this_dir,'figures','relation_xoff_sigma_Rvir_err.png')
plt.savefig(outf,overwrite=True)

plt.figure(0,(10,10))

for k,red in enumerate(z_arr):
    sel = (z_full==red)
    sortid = np.argsort(peak_array_full[sel])
    peak = peak_array_full[sel][sortid]
    y = xoff_full[sel][sortid]
    dy = xoff_err_full[sel][sortid]   
    plt.errorbar(peak,y,yerr=dy,xerr=None,color='%.c'%(colors[k]),label='z = '+str(red))
    plt.plot(peak,xoff_sigma_pl([peak,red],*popt1),c='%.c'%(colors[k]))
plt.grid(True)
plt.xscale('log')
plt.xticks([0.6,0.8,1,2,3,4])
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.gca().ticklabel_format(axis='both', style='plain')
plt.legend(fontsize=17)
plt.xlabel(r'$\nu = \delta_c/\sigma$', fontsize=30)
plt.ylabel(r'$\log_{10}X_{off}$', fontsize=30)
plt.tick_params(labelsize=25)
plt.tight_layout()
outf=os.path.join(this_dir,'figures','relation_xoff_sigma_Rvir_err_errorbar.png')
plt.savefig(outf,overwrite=True)
print(popt1)

sys.exit()







