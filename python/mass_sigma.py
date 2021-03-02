from astropy.table import Table, Column
#from astropy_healpix import healpy
import sys
import os, glob
import time
from astropy.cosmology import FlatLambdaCDM
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


cosmo = cosmology.setCosmology('multidark-planck')  

peak_bins = np.arange(0.7,5.,0.02)
peak_array = (peak_bins[:-1]+peak_bins[1:])/2.

h=cosmo.Hz(z=0)/100
dc0 = peaks.collapseOverdensity(z = 0)

sigma = dc0/peak_array

log10_sigma = np.log10(1/sigma)

def Mass_sigma(x):
    r=cosmo.sigma(x,z=0,inverse=True)
    M=peaks.lagrangianM(r)/h
    return np.log10(M)

def Mass_peak(x):
    r=cosmo.sigma(dc0/x,z=0,inverse=True)
    M=peaks.lagrangianM(r)/h
    return np.log10(M)

def Mass_log10sigma(x):
    r=cosmo.sigma(1/10**x,z=0,inverse=True)
    M=peaks.lagrangianM(r)/h
    return np.log10(M)

outfig = os.path.join('/home/rseppi/HMF_seppi20','figures','Mass_sigma_relations.png')
fig = plt.figure(figsize=(18,6))
gs = fig.add_gridspec(1,3)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

ax1.plot(sigma, Mass_sigma(sigma), lw=5)
ax2.plot(peak_array, Mass_peak(peak_array), lw=5)
ax3.plot(log10_sigma, Mass_log10sigma(log10_sigma), lw=5)

ax1.grid(True)
ax2.grid(True)
ax3.grid(True)

#ax1.set_xticks(np.array([0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5]))
#ax2.set_xticks(np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]))
#ax3.set_xticks(np.array([-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45]))

ax1.set_yticks(np.array([12.0,12.5,13.0,13.5,14.0,14.5,15.0,15.5,16.0]))
ax2.set_yticks(np.array([12.0,12.5,13.0,13.5,14.0,14.5,15.0,15.5,16.0]))
ax3.set_yticks(np.array([12.0,12.5,13.0,13.5,14.0,14.5,15.0,15.5,16.0]))

ax2.set_xscale('log')
ax2.xaxis.set_major_formatter(ScalarFormatter())
ax2.ticklabel_format(useOffset=False, style='plain')
ax2.set_xticks([0.8,1,2,3,4])

ax1.tick_params(labelsize=20)
ax2.tick_params(labelsize=20)
ax3.tick_params(labelsize=20)

ax1.set_xlabel(r'$\sigma$', fontsize=20)
ax2.set_xlabel(r'$\nu = dc/\sigma$', fontsize=20)
ax3.set_xlabel(r'$\log_{10}\sigma^{-1}$', fontsize=20)

ax1.set_ylabel(r'Mass [M$_{\odot}$]', fontsize=20)
ax2.set_ylabel(r'Mass [M$_{\odot}$]', fontsize=20)
ax3.set_ylabel(r'Mass [M$_{\odot}$]', fontsize=20)

plt.tight_layout()
fig.savefig(outfig, overwrite=True)

print('done!')


