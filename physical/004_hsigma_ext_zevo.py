#!/usr/bin/env python
# coding: utf-8

# In[94]:


"""
Integrates models of mass function (log1_sigma, logxoff) along xoff to get f_sigma
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
import emcee
from scipy.optimize import minimize

print('Models mass functions as function of mass and Xoff')
print('------------------------------------------------')
print('------------------------------------------------')
t0 = time.time()

#env = 'MD40' 
# initializes pathes to files
#test_dir = os.path.join(os.environ[env], 'Mass_Xoff_Concentration')
this_dir='.'
file_dir='/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration'

path_2_snapshot_data4_0 = np.array(['/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_1.0.fits','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.9567.fits','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.8951.fits','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.8192.fits','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.7016.fits','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.6565.fits','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.5876.fits','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.5622.fits','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.4922.fits','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.4123.fits'])

path_2_snapshot_data2_5 = np.array(['/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_1.0.fits','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.956.fits','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.8953.fits','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.8173.fits','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.7003.fits','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.6583.fits','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.5864.fits','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.5623.fits','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.5.fits','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.409.fits'])

path_2_snapshot_data1_0 = np.array(['/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_1.0.fits','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.9567.fits','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.8951.fits','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.8192.fits','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.7016.fits','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.6565.fits','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.5876.fits','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.5622.fits','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.4922.fits','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.4123.fits'])


cosmo = cosmology.setCosmology('multidark-planck')    
s_low_list_HMD = np.array([-0.08,-0.07,-0.06,-0.05,-0.015,0.01,0.04,0.065,0.11,0.18])
s_low_list_BigMD = np.array([-0.09,-0.08,-0.07,-0.06,-0.055,-0.04,-0.03,-0.015,-0.01,0.08])
s_low_list_MDPL = np.array([-0.09,-0.08,-0.07,-0.06,-0.055,-0.04,-0.03,-0.015,-0.01,0.0])
diff_sigma1 = 1e-2
diff_sigma2 = 1e-2
diff_sigma3 = 1e-2

hsigma_all = []
#hsigma_uncut = []
ll = range(10)
hsigma_uncut = np.array([np.array(list) for _ in ll])
s_grid_uncut = np.array([np.array(list) for _ in ll])
xoff_grid_uncut = np.array([np.array(list) for _ in ll])
spin_grid_uncut = np.array([np.array(list) for _ in ll])
redshift_uncut = np.array([np.array(list) for _ in ll])
counts_uncut = np.array([np.array(list) for _ in ll])
herr_uncut = np.array([np.array(list) for _ in ll])
herror_uncut = np.array([np.array(list) for _ in ll])
herr_all = []
s_grid_all = []
xoff_grid_all = []
spin_grid_all = []
redshift_all = []
len_s_bins = []
s_bins_all = []
f_dir1_list = np.array([np.array(list) for _ in ll])
f_dir2_list = np.array([np.array(list) for _ in ll])
f_dir3_list = np.array([np.array(list) for _ in ll])
f1_err_list = np.array([np.array(list) for _ in ll])
f2_err_list = np.array([np.array(list) for _ in ll])
f3_err_list = np.array([np.array(list) for _ in ll])
s1_all = np.array([np.array(list) for _ in ll])
s2_all = np.array([np.array(list) for _ in ll])
s3_all = np.array([np.array(list) for _ in ll])

for i, p2d4_0 in enumerate(path_2_snapshot_data4_0):

    aexp = float(os.path.basename(p2d4_0[:-5]).split('_')[1])   
    z_snap = 1/aexp -1
    print('processing ',z_snap,' ....')

    E_z = cosmo.Ez(z=z_snap)
    h = cosmo.Hz(z=0.0)/100
    Vol1 = (4e3/(1+z_snap))**3
    Vol2 = (2.5e3/(1+z_snap))**3
    Vol3 = (1e3/(1+z_snap))**3
    dc = peaks.collapseOverdensity(z = z_snap)
    print(dc)
    rho_m = cosmo.rho_m(z=z_snap)*1e9

    hd1 = fits.open(p2d4_0)

    #mass1_=hd1[1].data['Mmvir_all']
    mass1_=hd1[1].data['Mvir']
    R1 = peaks.lagrangianR(mass1_)
    sigma1 = cosmo.sigma(R1,z=z_snap)
    log1_sigma1 = np.log10(1/sigma1)
    logxoff_data1 = np.log10(hd1[1].data['Xoff'])#/hd1[1].data['Rvir'])
    log_spin1 = np.log10(hd1[1].data['Spin'])

    hd2 = fits.open(path_2_snapshot_data2_5[i])

    mass2_=hd2[1].data['Mvir']
    R2 = peaks.lagrangianR(mass2_)
    sigma2 = cosmo.sigma(R2,z=z_snap)
    log1_sigma2 = np.log10(1/sigma2)
    logxoff_data2 = np.log10(hd2[1].data['Xoff'])#/hd1[1].data['Rvir'])
    log_spin2 = np.log10(hd2[1].data['Spin'])

    hd3 = fits.open(path_2_snapshot_data1_0[i])

    mass3_=hd3[1].data['Mvir']
    R3 = peaks.lagrangianR(mass3_)
    sigma3 = cosmo.sigma(R3,z=z_snap)
    log1_sigma3 = np.log10(1/sigma3)
    logxoff_data3 = np.log10(hd3[1].data['Xoff'])#/hd1[1].data['Rvir'])
    log_spin3 = np.log10(hd3[1].data['Spin'])


    #fsigma1
    s_low_HMD = s_low_list_HMD[i]
    print('s_low_HMD = ',s_low_HMD)
    s_low_BigMD = s_low_list_BigMD[i]
    print('s_low_BigMD = ',s_low_BigMD)
    s_low_MDPL = s_low_list_MDPL[i]
    print('s_low_MDPL = ',s_low_MDPL)
    r_low_HMD = cosmo.sigma(1/10**s_low_HMD, z=z_snap, inverse = True)
    mass_low_HMD = peaks.lagrangianM(r_low_HMD)/h
    print('mass low HMD = %.3g M sun'%(mass_low_HMD))
    r_low_BigMD = cosmo.sigma(1/10**s_low_BigMD, z=z_snap, inverse = True)
    mass_low_BigMD = peaks.lagrangianM(r_low_BigMD)/h
    print('mass low BigMD = %.3g M sun'%(mass_low_BigMD))
    r_low_MDPL = cosmo.sigma(1/10**s_low_MDPL, z=z_snap, inverse = True)
    mass_low_MDPL = peaks.lagrangianM(r_low_MDPL)/h
    print('mass low MDPL = %.3g M sun'%(mass_low_MDPL))
    s_edges1 = np.arange(s_low_HMD,0.5,diff_sigma1)
    mass1 = np.log10(mass1_)
    counts_f1 = np.histogram(log1_sigma1,bins=s_edges1)[0]
    print('from 4Gpc = ', np.sum(counts_f1),'halos')
    s_bins1 = (s_edges1[:-1]+s_edges1[1:])/2
    Runo = cosmo.sigma(1/10**s_bins1,inverse=True,z=z_snap)
    Runo_ = cosmo.sigma(1/10**s_edges1,inverse=True,z=z_snap)
    M1 = peaks.lagrangianM(Runo)
    M1_ = peaks.lagrangianM(Runo_)
    diff1 = np.diff(np.log(M1_))
    dn_dlnM1 = counts_f1/Vol1/diff1
    fsigma_dir1 = dn_dlnM1*M1/rho_m/cosmo.sigma(Runo,z=z_snap,derivative=True)*(-3.0)
    ferr1_ = np.sqrt(1/counts_f1 + 0.02**2)*(fsigma_dir1)
    ferr1 = 1/np.log(10)*ferr1_/fsigma_dir1
    
    #fsigma2    
    s_edges2 = np.arange(s_low_BigMD-diff_sigma2/3,0.45,diff_sigma2)
    mass2 = np.log10(mass2_)
    counts_f2 = np.histogram(log1_sigma2,bins=s_edges2)[0]
    print('from 2.5Gpc = ', np.sum(counts_f2),'halos')
    s_bins2 = (s_edges2[:-1]+s_edges2[1:])/2
    Rtwo = cosmo.sigma(1/10**s_bins2,inverse=True,z=z_snap)
    Rtwo_ = cosmo.sigma(1/10**s_edges2,inverse=True,z=z_snap)
    M2 = peaks.lagrangianM(Rtwo)
    M2_ = peaks.lagrangianM(Rtwo_)
    diff2 = np.diff(np.log(M2_))
    dn_dlnM2 = counts_f2/Vol2/diff2
    fsigma_dir2 = dn_dlnM2*M2/rho_m/cosmo.sigma(Rtwo,z=z_snap,derivative=True)*(-3.0)
    ferr2_ = np.sqrt(1/counts_f2 + 0.03**2)*(fsigma_dir2)
    ferr2 = 1/np.log(10)*ferr2_/fsigma_dir2
    
    #fsigma3
    s_edges3 = np.arange(s_low_MDPL-diff_sigma3*2/3,0.4,diff_sigma3)
    mass3 = np.log10(mass3_)
    counts_f3 = np.histogram(log1_sigma3,bins=s_edges3)[0]
    print('from 1Gpc = ', np.sum(counts_f3),'halos')
    s_bins3 = (s_edges3[:-1]+s_edges3[1:])/2
    Rthree = cosmo.sigma(1/10**s_bins3,inverse=True,z=z_snap)
    Rthree_ = cosmo.sigma(1/10**s_edges3,inverse=True,z=z_snap)
    M3 = peaks.lagrangianM(Rthree)
    M3_ = peaks.lagrangianM(Rthree_)
    diff3 = np.diff(np.log(M3_))
    dn_dlnM3 = counts_f3/Vol3/diff3
    fsigma_dir3 = dn_dlnM3*M3/rho_m/cosmo.sigma(Rthree,z=z_snap,derivative=True)*(-3.0)
    ferr3_ = np.sqrt(1/counts_f3 + 0.04**2)*(fsigma_dir3)
    ferr3 = 1/np.log(10)*ferr3_/fsigma_dir3
    
    #hsigma1
    xoff_edges = np.linspace(-0.7,3.0,50)
    spin_edges = np.linspace(-4.5,-0.12,51)
    xoff_bins = (xoff_edges[:-1] + xoff_edges[1:])/2
    spin_bins = (spin_edges[1:]+spin_edges[:-1])/2
    print('s diff = ',np.diff(s_bins1))
    diff_xoff = np.diff(xoff_bins)[0]
    diff_spin = np.diff(spin_bins)[0]

    edges3d1 = (s_edges1,xoff_edges,spin_edges)
    counts_h1 = np.histogramdd((log1_sigma1,logxoff_data1,log_spin1),bins=edges3d1)[0]
    countsh1 = np.transpose(counts_h1)

    dn_dlnM1 = countsh1/Vol1/diff1
    hsigma1 = dn_dlnM1/diff_xoff/diff_spin*M1/rho_m/cosmo.sigma(Runo,z=z_snap,derivative=True)*(-3.0)
    herror1 = np.sqrt(1/countsh1 + 0.02**2)*(hsigma1)
    herr1 = 1/np.log(10)*herror1/hsigma1 #this is the log error

    #hsigma2
    edges3d2 = (s_edges2,xoff_edges,spin_edges)
    counts_h2 = np.histogramdd((log1_sigma2,logxoff_data2,log_spin2),bins=edges3d2)[0]
    countsh2 = np.transpose(counts_h2)
    
    dn_dlnM2 = countsh2/Vol2/diff2
    hsigma2 = dn_dlnM2/diff_xoff/diff_spin*M2/rho_m/cosmo.sigma(Rtwo,z=z_snap,derivative=True)*(-3.0)
    herror2 = np.sqrt(1/countsh2 + 0.03**2)*(hsigma2)
    herr2 = 1/np.log(10)*herror2/hsigma2 #this is the log error
    
    #hsigma3
    edges3d3 = (s_edges3,xoff_edges,spin_edges)
    counts_h3 = np.histogramdd((log1_sigma3,logxoff_data3,log_spin3),bins=edges3d3)[0]
    countsh3 = np.transpose(counts_h3)
    
    dn_dlnM3 = countsh3/Vol3/diff3
    hsigma3 = dn_dlnM3/diff_xoff/diff_spin*M3/rho_m/cosmo.sigma(Rthree,z=z_snap,derivative=True)*(-3.0)
    herror3 = np.sqrt(1/countsh3 + 0.04**2)*(hsigma3)
    herr3 = 1/np.log(10)*herror3/hsigma3 #this is the log error
    print('done')

    print(hsigma3.shape)
    print(hsigma2.shape)
    print(hsigma1.shape)
    countsh_ext = np.dstack((countsh3,countsh2,countsh1))
    hsigma_ext = np.dstack((hsigma3,hsigma2,hsigma1))
    print(hsigma_ext.shape)
    herror_ext = np.dstack((herror3,herror2,herror1))
    herr_ext = np.dstack((herr3,herr2,herr1))
    s_bins_ext = np.concatenate((s_bins3,s_bins2,s_bins1))
    sort_sigma = np.argsort(s_bins_ext)
    s_bins = s_bins_ext[sort_sigma]
    
    #print(countsh_ext[22:28,22:28,100])
    countsh = countsh_ext[:,:,sort_sigma]
    #print(countsh[22:28,22:28,100])
    hsigma = hsigma_ext[:,:,sort_sigma]
    herror = herror_ext[:,:,sort_sigma]
    herr = herr_ext[:,:,sort_sigma]
    
    print(s_bins)
    xoff_grid, spin_grid, s_grid = np.meshgrid(xoff_bins,spin_bins,s_bins)

    redshift = np.repeat(z_snap, len(hsigma.ravel())).reshape(hsigma.shape)

    c1 = 9
    hsigma_ = hsigma[np.where(countsh>c1)]
    herr_ = herr[np.where(countsh>c1)]
    s_grid_ = s_grid[np.where(countsh>c1)]
    xoff_grid_ = xoff_grid[np.where(countsh>c1)]
    spin_grid_ = spin_grid[np.where(countsh>c1)]
    redshift_ = redshift[np.where(countsh>c1)]

    hsigma_all.append(hsigma_)
#    hsigma_uncut.append(hsigma)
    hsigma_uncut[i] = hsigma
    s_grid_uncut[i] = s_grid
    xoff_grid_uncut[i] = xoff_grid
    spin_grid_uncut[i] = spin_grid
    redshift_uncut[i] = redshift
    counts_uncut[i] = countsh
    herr_uncut[i] = herr
    herror_uncut[i] = herror
    herr_all.append(herr_)
    s_grid_all.append(s_grid_)
    xoff_grid_all.append(xoff_grid_)
    spin_grid_all.append(spin_grid_)
    redshift_all.append(redshift_)

    len_s_bins.append(len(s_bins))
    s_bins_all.append(s_bins)

    f_dir1_list[i] = fsigma_dir1
    f_dir2_list[i] = fsigma_dir2
    f_dir3_list[i] = fsigma_dir3

    f1_err_list[i] = ferr1
    f2_err_list[i] = ferr2
    f3_err_list[i] = ferr3
    s1_all[i] = s_bins1
    s2_all[i] = s_bins2
    s3_all[i] = s_bins3    

    print('s_grid_ shape = ', s_grid_.shape)
    print('xoff_grid_ shape = ', xoff_grid_.shape)
    print('spin_grid_ shape = ', spin_grid_.shape)
    print('redshift_grid_ shape = ', redshift_.shape)
    print('len_s_bins = ', len_s_bins)
print('hsigma_uncut 0 shape = ', hsigma_uncut[0].shape)
print('hsigma_uncut 1 shape = ', hsigma_uncut[1].shape)
print('hsigma_uncut 2 shape = ', hsigma_uncut[2].shape)
print('hsigma_uncut 3 shape = ', hsigma_uncut[3].shape)
print('hsigma_uncut 4 shape = ', hsigma_uncut[4].shape)
print('hsigma_uncut 5 shape = ', hsigma_uncut[5].shape)
print('hsigma_uncut 6 shape = ', hsigma_uncut[6].shape)
print('hsigma_uncut 7 shape = ', hsigma_uncut[7].shape)
print('hsigma_uncut 8 shape = ', hsigma_uncut[8].shape)
print('hsigma_uncut 9 shape = ', hsigma_uncut[9].shape)

hsigma_all = np.array([item for sublist in hsigma_all for item in sublist])
#hsigma_uncut = np.array([item for sublist in hsigma_uncut for item in sublist])
#hsigma_uncut = np.array(hsigma_uncut)
herr_all = np.array([item for sublist in herr_all for item in sublist])
s_grid_all = np.array([item for sublist in s_grid_all for item in sublist])
xoff_grid_all = np.array([item for sublist in xoff_grid_all for item in sublist])
spin_grid_all = np.array([item for sublist in spin_grid_all for item in sublist])
redshift_all = np.array([item for sublist in redshift_all for item in sublist])
#len_s_bins = np.array([item for sublist in len_s_bins for item in sublist])
s_bins_all = np.array([item for sublist in s_bins_all for item in sublist])

print('len s_bins_all = ',len(s_bins_all))
print(len(hsigma_all.ravel()))
print(type(hsigma_all.ravel()))
    
    # Initial guesses to the fit parameters.

#def my_flat_priors(cube):
#    params = cube.copy()
#    params[0] = (cube[0])*1 - 22.4  #A
#    params[1] = (cube[1])*0.3 + 0.7        #a
#    params[2] = (cube[2])*0.3 + 2.1            #q
#    params[3] = (cube[3])*0.4 - 3.2      #mu
#    params[4] = (cube[4])*0.4 + 5.5        #alpha
#    params[5] = (cube[5])*0.1 - 0.4             #beta
#    params[6] = (cube[6])*0.4 - 1.9           #e0
#    params[7] = (cube[7])*0.4 + 2.8              #gamma
#    params[8] = (cube[8])*0.3 + 1        #delta
#    params[9] = (cube[9])*0.4 - 2.9        #e1

#    return params

#priors for zevo
def my_flat_priors(cube):
    params = cube.copy()
    params[0] = (cube[0])*0.15 - 0.08  #k0
    params[1] = (cube[1])*0.3 - 0.25        #k1
    params[2] = (cube[2])*0.2 - 0.1            #k2
    params[3] = (cube[3])*0.2 - 0.15      #k3
    params[4] = (cube[4])*0.2 - 0.05        #k4
    params[5] = (cube[5])*0.8 - 0.7             #k5
    params[6] = (cube[6])*0.2 - 0.15          #k6
    params[7] = (cube[7])*0.3 - 0.05              #k7
    params[8] = (cube[8])*0.3 - 0.25     #k8
    params[9] = (cube[9])*0.4 - 0.05        #k9

    return params



#THE FITTING FUNCTION IS A MASS FUNCTION MULTIPLIED BY A MODIFIED SCHECHTER IN XOFF and spinpar
#guess_prms = [( -3, 0.9, 1.7, 1, 1.,1.,0.,3.,0.7,0.005,0.3)]#, -10., 5., 0.)]
guess_prms = [( -22, 0.9, 2.0, -3, 5.,-0.35,-1.5, 3., 1.1, -2.7)]
guess_prms = [(0.04,-0.2,0.08,-0.04,-0.19,0.17,0.26,-0.41,-0.2,0.0)]
guess_prms = [(0.0,-0.2,0.04,-0.04,0.08,-0.41,-0.19,0.17,-0.2,0.26)]

#def h_func(data,A,a,q,mu,alpha,beta,e0,nu,gamma,delta,beta1,e1):
#def h_func(data,A,a,q,mu,alpha,beta,e0,gamma,delta,e1):
#def h_func(data,A,a,q,mu,alpha,beta,e0,gamma,delta,e1,k0,k1,k2,k3,k4,k5,k6,k7,k8,k9):
def h_func(data,k0,k1,k2,k3,k4,k5,k6,k7,k8,k9):
    x_,y_,z_,redshift = data      #x_ is log10(1/sigma) y_ is log10(Xoff)
    x = 1/10**x_ #sigma
    y = 10**y_   #Xoff
    z = 10**z_ #spin
    opz = (1+redshift)
#    return A+np.log10(np.sqrt(2/np.pi)) + (q*opz**0.072)*np.log10(np.sqrt(a*opz**-0.14)*dc/x) - a*opz**-0.14/2/np.log(10)*dc**2/x**2 + (alpha*opz**0.066)*np.log10(y/10**(mu*opz**-0.05)/x**(e0*opz**-0.19)) - 1/np.log(10)*(y/10**(mu*opz**-0.05)/(x**(e0*opz**-0.19)))**(0.05*alpha*opz**0.066) + (gamma*opz**0.17)*np.log10(z/(0.7*10**(mu*opz**-0.05))) - 1/np.log(10)*(y/10**(mu*opz**-0.05)/x**(e1*opz**0.234))**(beta*opz**-0.43)*(z/(0.7*10**(mu*opz**-0.05)))**(delta*opz**-0.21)

    return -22.00389533*(opz)**k0+np.log10(np.sqrt(2/np.pi)) + (2.25739005*opz**k2)*np.log10(np.sqrt(0.87846168*opz**k1)*dc/x) - 0.87846168*opz**k1/2/np.log(10)*dc**2/x**2 + (5.62366311*opz**k4)*np.log10(y/10**(-3.14881194*opz**k3)/x**(-1.60810632*opz**k6)) - 1/np.log(10)*(y/10**(-3.14881194*opz**k3)/(x**(-1.60810632*opz**k6)))**(0.05*5.62366311*opz**k4) + (3.09536219*opz**k7)*np.log10(z/(0.7*10**(-3.14881194*opz**k3))) - 1/np.log(10)*(y/10**(-3.14881194*opz**k3)/x**(-2.72033278*opz**k9))**(-0.36473562*opz**k5)*(z/(0.7*10**(-3.14881194*opz**k3)))**(1.16794809*opz**k8)

#    return (A*opz**k8)+np.log10(np.sqrt(2/np.pi)) + (q*opz**k9)*np.log10(np.sqrt(a*opz**k0)*dc/x) - a*opz**k0/2/np.log(10)*dc**2/x**2 + (alpha*opz**k1)*np.log10(y/10**(mu*opz**k2)/x**(e0*opz**k3)) - 1/np.log(10)*(y/10**(mu*opz**k2)/(x**(e0*opz**k3)))**(0.05*alpha*opz**k1) + (gamma*opz**k4)*np.log10(z/(0.7*10**(mu*opz**k2))) - 1/np.log(10)*(y/10**(mu*opz**k2)/x**(e1*opz**k5))**(beta*opz**k6)*(z/(0.7*10**(mu*opz**k2)))**(delta*opz**k7)

#    return A+np.log10(np.sqrt(2/np.pi)) + (q)*np.log10(np.sqrt(a*opz**k0)*dc/x) - a*opz**k0/2/np.log(10)*dc**2/x**2 + (alpha*opz**k1)*np.log10(y/10**(mu*opz**k2)/x**(e0*opz**k3)) - 1/np.log(10)*(y/10**(mu*opz**k2)/(x**(e0*opz**k3)))**(0.05*alpha*opz**k1) + (gamma*opz**k4)*np.log10(z/(0.7*10**(mu*opz**k2))) - 1/np.log(10)*(y/10**(mu*opz**k2)/x**(e1*opz**k5))**(beta*opz**k6)*(z/(0.7*10**(mu*opz**k2)))**(delta*opz**k7)
    


#names = [r'$A$',r'$a$',r'$q$',r'$\log_{10}\mu$',r'$\alpha$',r'$\beta$',r'$e_0$',r'$\log_{10}\nu$',r'$\gamma$',r'$\delta$',r'$\beta_1$',r'$e_1$']
#names = [r'$\log_{10}A$',r'$a$',r'$q$',r'$\log_{10}\mu$',r'$\alpha$',r'$\beta$',r'$e_0$',r'$\gamma$',r'$\delta$',r'$e_1$']
names = [r'$k_0$',r'$k_1$',r'$k_2$',r'$k_3$',r'$k_4$',r'$k_5$',r'$k_6$',r'$k_7$',r'$k_8$',r'$k_9$']

def likelihood(params):
    model = h_func([s_grid_all,xoff_grid_all,spin_grid_all,redshift_all],*params)
    loglike = -0.5*(((model - np.log10(hsigma_all))/herr_all)**2).sum()
    #loglike = -np.sum(model-(np.log10(gsigma_.ravel())*np.log(model)))
    return loglike

def likelihood_mcmc(params,x,y,err):
    x0,x1,x2,x3 = x
    model = h_func([x0,x1,x2,x3],*params)
#    loglike = -0.5*(((np.log10(y) - model)/err)**2).sum()
    loglike = -0.5*(((np.log10(hsigma_all-model))/herr_.ravel())**2 + np.log(2*np.pi*herr_all**2)).sum()
    #loglike = -np.sum(model-(np.log10(gsigma_.ravel())*np.log(model)))
    return loglike


#######UNCOMMENT TO USE ULTRANEST#############
sampler_rseppi = ultranest.ReactiveNestedSampler(names, likelihood, my_flat_priors)

print('running sampler...')
result = sampler_rseppi.run()
sampler_rseppi.print_results()

tafterfit = time.time()
tem = tafterfit - t0
print('ci ho messo ',tem,'s')
#from ultranest.plot import PredictionBand
#fit = PredictionBand([log1sig, logx])
#print(result)
popt = np.array(result['posterior']['mean'])
pvar = np.array(result['posterior']['stdev'])
print('Best fit parameters = ', popt)

parameters = np.array(result['weighted_samples']['v'])
weights = np.array(result['weighted_samples']['w'])
weights /= weights.sum()
cumsumweights = np.cumsum(weights)
mask = cumsumweights > 1e-4
fig=corner.corner(parameters[mask,:], weights=weights[mask], labels=names, show_titles=True, color='r',bins=50,smooth=True,smooth1d=True,quantiles=[0.025,0.16,0.84,0.975],label_kwargs={'fontsize':20,'labelpad':25},title_kwargs={"fontsize":17},levels=[0.68,0.95],fill_contours=True,title_fmt='.3f')
axes = np.array(fig.axes).reshape((len(names), len(names)))
print(axes.shape)
for i in range(len(names)):
    ax = axes[i, 0]
    ax.yaxis.set_tick_params(labelsize=14)
    ax1 = axes[len(names)-1, i]
    ax1.xaxis.set_tick_params(labelsize=14)
fig.tight_layout()
fig.savefig(os.path.join(this_dir,'results','zevo','corner_hsigma_zevo2.png'),overwrite=True)
#############END OF ULTRANEST#########

'''
#################UNCOMMENT TO USE MCMC###################  to fix for zevo
#nll = lambda *args: -likelihood_mcmc(*args)
#initial = np.array(guess_prms)
xdata = np.vstack((s_grid_.ravel(),xoff_grid_.ravel(),spin_grid_.ravel()))
popt_curvefit = curve_fit(h_func, xdata, np.log10(hsigma_.ravel()),guess_prms, sigma = herr_.ravel(),maxfev=100000)[0]
print('estimator = ',popt_curvefit)

def log_prior(pars):
    A,a,q,mu,alpha,beta,e0,nu,gamma,delta,beta1,e1 = pars    
    if -24 < A < -17 and 0.4 < a < 1.8 and 1.5 < q < 3.5 and -4.0 < mu < -2.0 and 3 < alpha < 7.0 and 0.15 < beta < 0.45 and -2 < e0 < -1 and -4.5 < nu < -1.5 and 2.0 < gamma < 4. and 0.5 < delta < 2.0 and -0.5 < beta1 < -0.1 and -3.5 < e1 < -1.5:
        return 0.0
    return -np.inf

def logprob(pars):
    lp = log_prior(pars)
    if not np.isfinite(lp):
        return -np.inf
    return lp + likelihood_mcmc(pars,[s_grid_.ravel(),xoff_grid_.ravel(),spin_grid_.ravel()],hsigma_.ravel(),herr_.ravel())

A,a,q,mu,alpha,beta,e0,nu,gamma,delta,beta1,e1 = popt_curvefit
pos = popt_curvefit + 1e-4 * np.random.randn(100, 12)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob)
sampler.run_mcmc(pos, 6000, progress=True)

fig, axes = plt.subplots(12, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(names[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
fig.savefig(os.path.join(this_dir,'extended','3d','z_%.3f'%(z_snap),'mcmc_chain.png'))

#tau = sampler.get_autocorr_time()
#print(tau)
thin_par = 50
flat_samples = sampler.get_chain(discard=300, thin=thin_par, flat=True)
popt = np.percentile(flat_samples[:, i], [50])
fig = corner.corner(flat_samples, labels=names, truths=popt,show_titles=True,color='r',bins=50,smooth=True,smooth1d=True,quantiles=[0.025,0.16,0.84,0.975],label_kwargs={'fontsize':20,'labelpad':20},title_kwargs={"fontsize":17},levels=[0.68,0.95],fill_contours=True,title_fmt='.3f')
axes = np.array(fig.axes).reshape((len(names), len(names)))
print(axes.shape)
for i in range(len(names)):
    ax = axes[i, 0]
    ax.yaxis.set_tick_params(labelsize=14)
    ax1 = axes[len(names)-1, i]
    ax1.xaxis.set_tick_params(labelsize=14)
fig.savefig(os.path.join(this_dir,'extended','3d','z_%.3f'%(z_snap),'corner_mcmc_hsigma.png'))

pvar = []
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    pvar.append(np.average(q))
pvar = np.array(pvar)    


###################END OF MCMC ###################
'''

# Flatten the initial guess parameter list.
print('s_grid_all shape = ',s_grid_all.shape)
print('xoff_grid_all shape = ',xoff_grid_all.shape)
print('spin_grid_all shape = ',spin_grid_all.shape)
print('redshift_all shape = ',redshift_all.shape)
xdata = np.vstack((s_grid_all.ravel(),xoff_grid_all.ravel(),spin_grid_all.ravel(),redshift_all.ravel()))

print('now printing s_grid_all_ravel and redshift_all_ravel')
print(s_grid_all.ravel(),redshift_all.ravel())

#popt, pcov = curve_fit(h_func, xdata, np.log10(hsigma_all.ravel()),guess_prms, sigma = herr_all.ravel(),maxfev=100000)
#pvar = np.diag(pcov)

print('Initial parameters:')
print(guess_prms)
print('Fitted parameters:')
print(popt)
#fit = h_func([s_grid,xoff_grid,spin_grid], *popt)
#print('fit is ', fit)

t = Table()
t.add_column(Column(name='pars', data=popt, unit=''))
t.add_column(Column(name='err', data=pvar, unit=''))
out_table = os.path.join(this_dir,'results','zevo','hsigma_params.fit')
os.makedirs(os.path.dirname(out_table), exist_ok=True)
t.write(out_table, overwrite=True)

#fit_ = fit[np.where(counts_tot>c3)]
fit_ = h_func([s_grid_all, xoff_grid_all,spin_grid_all,redshift_all], *popt)
print('min log10(hsigma_) =', min(np.log10(hsigma_all)))
print('min fit_ = ',min(fit_))

print('max log10(hsigma_) =', max(np.log10(hsigma_all)))
print('max fit_ = ',max(fit_))

chi_2 = np.sum((np.log10(hsigma_all)-fit_)**2/herr_all**2)
dof = len(hsigma_all) - len(names)
chi_2r = chi_2/dof
rv = chi2.stats(dof)
print('chi2 = ', chi_2)
print('dof = ', dof)
print('chi2r = ',chi_2r)
print('expected chi2 = ', rv)

#PDF OF THE RESIDUALS
res = ((np.log10(hsigma_all)-fit_)/herr_all)
pdf, b = np.histogram(res,bins=100,density=True)
bins = (b[:-1]+b[1:])/2

def ga(x, x0, sigma):
    a=1/sigma/np.sqrt(2*np.pi)
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
par,cov = curve_fit(ga,bins,pdf)
gauss = ga(bins,*par)
print(par)
plt.figure(figsize=(10,10))
plt.plot(bins,pdf,label='pdf')
plt.plot(bins,gauss,label="mu = %.3g, sigma = %.3g"%(par[0],par[1]))
#plt.plot(bins,ga(bins,0,1))
plt.legend(fontsize=20)
plt.grid(True)
plt.tick_params(labelsize=20)
plt.tight_layout()
outmodelfig_respdf = os.path.join(this_dir,'results','zevo','resid_pdf_hsigma.png')
os.makedirs(os.path.dirname(outmodelfig_respdf), exist_ok=True)
plt.savefig(outmodelfig_respdf,overwrite=True)

low=0
for j,p2d4_0 in enumerate(path_2_snapshot_data4_0):
    aexp = float(os.path.basename(p2d4_0[:-5]).split('_')[1])   
    z_snap = 1/aexp -1
    print('making plots of z = ',z_snap)
    g_sigma_xoff = np.zeros((len_s_bins[j],len(xoff_bins)))
    g_sigma_spin = np.zeros((len_s_bins[j],len(spin_bins)))
    g_xoff_spin = np.zeros((len(xoff_bins),len(spin_bins)))
    g_sigma_xoff_model = np.zeros((len_s_bins[j],len(xoff_bins)))
    g_sigma_spin_model = np.zeros((len_s_bins[j],len(spin_bins)))
    g_xoff_spin_model = np.zeros((len(xoff_bins),len(spin_bins)))

    counts_sigma_xoff = np.zeros((len_s_bins[j],len(xoff_bins)))
    counts_sigma_spin = np.zeros((len_s_bins[j],len(spin_bins)))
    counts_xoff_spin = np.zeros((len(xoff_bins),len(spin_bins)))

    err_g_sigma_xoff = np.zeros((len_s_bins[j],len(xoff_bins)))
    err_g_sigma_spin = np.zeros((len_s_bins[j],len(spin_bins)))
    err_g_xoff_spin = np.zeros((len(xoff_bins),len(spin_bins)))
    
    fit = h_func([s_grid_uncut[j], xoff_grid_uncut[j],spin_grid_uncut[j],redshift_uncut[j]], *popt)

    if(j!=0):
        low += len_s_bins[j-1]
        up = low+len_s_bins[j]

    for k in range(len_s_bins[j]):
        for m in range(len(xoff_bins)):
            if(j==0):
                g_sigma_xoff[k,m] = integrate.simps(hsigma_uncut[j][:len(spin_bins),m,k],spin_bins)
                g_sigma_xoff_model[k,m] = integrate.simps((10**fit[:len(spin_bins),m,k]),spin_bins)
                counts_sigma_xoff[k,m] = np.sum(counts_uncut[j][:len(spin_bins),m,k])
                err_g_sigma_xoff[k,m] = np.sqrt(np.sum((herror_uncut[j][:len(spin_bins),m,k][~np.isnan(herror_uncut[j][:len(spin_bins),m,k])])**2))*diff_spin
            else:
                g_sigma_xoff[k,m] = integrate.simps(hsigma_uncut[j][:,m,k],spin_bins)
                g_sigma_xoff_model[k,m] = integrate.simps((10**fit[:,m,k]),spin_bins)
                counts_sigma_xoff[k,m] = np.sum(counts_uncut[j][:,m,k])
                err_g_sigma_xoff[k,m] = np.sqrt(np.sum((herror_uncut[j][:,m,k][~np.isnan(herror_uncut[j][:,m,k])])**2))*diff_spin    

    for k in range(len_s_bins[j]):
        for m in range(len(spin_bins)):
            if(j==0):
                g_sigma_spin[k,m] = integrate.simps(hsigma_uncut[j][m,:len(xoff_bins),k],xoff_bins)
                g_sigma_spin_model[k,m] = integrate.simps((10**fit[m,:len(xoff_bins),k]),xoff_bins)
                counts_sigma_spin[k,m] = np.sum(counts_uncut[j][m,:len(xoff_bins),k])
                err_g_sigma_spin[k,m] = np.sqrt(np.sum((herror_uncut[j][m,:len(xoff_bins),k][~np.isnan(herror_uncut[j][m,:len(xoff_bins),k])])**2))*diff_xoff
            else:
                g_sigma_spin[k,m] = integrate.simps(hsigma_uncut[j][m,:,k],xoff_bins)
                g_sigma_spin_model[k,m] = integrate.simps((10**fit[m,:,k]),xoff_bins)
                counts_sigma_spin[k,m] = np.sum(counts_uncut[j][m,:,k])
                err_g_sigma_spin[k,m] = np.sqrt(np.sum((herror_uncut[j][m,:,k][~np.isnan(herror_uncut[j][m,:,k])])**2))*diff_xoff

    for k in range(len(xoff_bins)):
        for m in range(len(spin_bins)):
            if(j==0):
                g_xoff_spin[k,m] = integrate.simps(hsigma_uncut[j][m,k,:len_s_bins[j]],s_bins_all[:len_s_bins[j]])
                g_xoff_spin_model[k,m] = integrate.simps((10**fit[m,k,:len_s_bins[j]]),s_bins_all[:len_s_bins[j]])
                counts_xoff_spin[k,m] = np.sum(counts_uncut[j][m,k,:len_s_bins[j]])
                err_g_xoff_spin[k,m] = np.sqrt(np.sum((herror_uncut[j][m,k,:len_s_bins[j]][~np.isnan(herror_uncut[j][m,k,:len_s_bins[j]])])**2))*diff_sigma3
            else:
                g_xoff_spin[k,m] = integrate.simps(hsigma_uncut[j][m,k,:],s_bins_all[low:up])
                g_xoff_spin_model[k,m] = integrate.simps((10**fit[m,k,:]),s_bins_all[low:up])
                counts_xoff_spin[k,m] = np.sum(counts_uncut[j][m,k,:])
                err_g_xoff_spin[k,m] = np.sqrt(np.sum((herror_uncut[j][m,k,:][~np.isnan(herror_uncut[j][m,k,:])])**2))*diff_sigma3


    g_sigma_xoff_err = np.sqrt(1/counts_sigma_xoff + 0.04**2)*g_sigma_xoff
    g_sigma_spin_err = np.sqrt(1/counts_sigma_spin + 0.04**2)*g_sigma_spin
    g_xoff_spin_err = np.sqrt(1/counts_xoff_spin + 0.04**2)*g_xoff_spin

    f_integral = np.zeros(len_s_bins[j])
    f_integral_model = np.zeros(len_s_bins[j])
    counts_sigma = np.zeros(len_s_bins[j])
    err_f_sigma = np.zeros(len_s_bins[j])

    for k in range(len_s_bins[j]):
    #the model is gsigma, so we integrate gsigma
        f_integral[k] = integrate.simps(g_sigma_xoff[k,:],xoff_bins)
        f_integral_model[k] = integrate.simps(g_sigma_spin_model[k,:],spin_bins)
        counts_sigma[k] = np.sum(counts_sigma_xoff[k,:])
        err_f_sigma[k] = np.sqrt(np.sum((err_g_sigma_xoff[k,:])**2))*diff_xoff

    f_xoff = np.zeros(len(xoff_bins))
    f_spin = np.zeros(len(spin_bins))
    f_xoff_model = np.zeros(len(xoff_bins))
    f_spin_model = np.zeros(len(spin_bins))
    counts_xoff = np.zeros(len(xoff_bins))
    counts_spin= np.zeros(len(spin_bins))
    err_f_xoff = np.zeros(len(xoff_bins))
    err_f_spin = np.zeros(len(spin_bins))

    for k in range(len(xoff_bins)):
        if(j==0):
            f_xoff[k] = integrate.simps(g_sigma_xoff[:,k],s_bins_all[:len_s_bins[j]])
            f_xoff_model[k] = integrate.simps(g_sigma_xoff_model[:,k],s_bins_all[:len_s_bins[j]])
            counts_xoff[k] = np.sum(counts_sigma_xoff[:,k])
            err_f_xoff[k] = np.sqrt(np.sum((err_g_sigma_xoff[:,k])**2))*diff_sigma3
        else:
            f_xoff[k] = integrate.simps(g_sigma_xoff[:,k],s_bins_all[low:up])
            f_xoff_model[k] = integrate.simps(g_sigma_xoff_model[:,k],s_bins_all[low:up])
            counts_xoff[k] = np.sum(counts_sigma_xoff[:,k])
            err_f_xoff[k] = np.sqrt(np.sum((err_g_sigma_xoff[:,k])**2))*diff_sigma3


    for k in range(len(spin_bins)):
        if(j==0):
            f_spin[k] = integrate.simps(g_sigma_spin[:,k],s_bins_all[:len_s_bins[j]])
            f_spin_model[k] = integrate.simps(g_sigma_spin_model[:,k],s_bins_all[:len_s_bins[j]])
            counts_spin[k] = np.sum(counts_sigma_spin[:,k])
            err_f_spin[k] = np.sqrt(np.sum((err_g_sigma_spin[:,k])**2))*diff_sigma3
        else:
            f_spin[k] = integrate.simps(g_sigma_spin[:,k],s_bins_all[low:up])
            f_spin_model[k] = integrate.simps(g_sigma_spin_model[:,k],s_bins_all[low:up])
            counts_spin[k] = np.sum(counts_sigma_spin[:,k])
            err_f_spin[k] = np.sqrt(np.sum((err_g_sigma_spin[:,k])**2))*diff_sigma3


    #plot integral g(sigma,xoff)dspin vs xoff in mass slices
    plt.figure(figsize=(10,10))
    if(j==0):
        a=len(s_bins_all[:len_s_bins[j]])
    else:
        a=len(s_bins_all[low:up])        
    print(a)
    b=5
    l = 19
    u = a-35
    for i in range(b):
        k = l+int((i)/(b)*(u-l))
        if(j==0):
            R=cosmo.sigma(1/(10**s_bins_all[:len_s_bins[j]][k]),z=z_snap,inverse=True)
        else:
            R=cosmo.sigma(1/(10**s_bins_all[low:up][k]),z=z_snap,inverse=True)
        label = peaks.lagrangianM(R)/h
#        yerr=1/np.log(10)*g_sigma_xoff_err[j,:]/g_sigma_xoff[j,:] #get log error
        yerr=1/np.log(10)*err_g_sigma_xoff[k,:]/g_sigma_xoff[k,:] 
        plt.fill_between(xoff_bins,np.log10(g_sigma_xoff[k,:])-yerr,np.log10(g_sigma_xoff[k,:])+yerr,alpha=0.4,label=r'$M = %.3g$'%(label))
        plt.plot(xoff_bins,np.log10(g_sigma_xoff_model[k,:]))
    plt.legend(fontsize=15)
    plt.tick_params(labelsize=15)
    plt.ylim(-5,0)
    #plt.xlim(-3.5,0.)
    plt.ylabel(r'$\log_{10}g(\sigma,X_{off})$', fontsize=20)
    plt.xlabel(r'$\log_{10}X_{off}$', fontsize=20)
    plt.grid(True)
    outfig_xo = os.path.join(this_dir,'results','zevo','red_%.3f'%(z_snap),'g_sigma_xoff_mass_slices.png')
    os.makedirs(os.path.dirname(outfig_xo), exist_ok=True)
    plt.savefig(outfig_xo,overwrite=True)
    plt.close()

    #plot integral g(sigma,xoff)dspin vs sigma in xoff slices
    plt.figure(figsize=(10,10))
    a=len(xoff_bins)
    b=5
    l= int(len(xoff_bins)/2.0)
    u = a-8
    for i in range(b):
        k = l+int((i)/(b)*(u-l))
#       yerr=1/np.log(10)*g_sigma_xoff_err[:,j]/g_sigma_xoff[:,j] #get log error
        yerr=1/np.log(10)*err_g_sigma_xoff[:,k]/g_sigma_xoff[:,k]
        label = 10**xoff_bins[k]
        if(j==0):
            plt.fill_between(s_bins_all[:len_s_bins[j]],np.log10(g_sigma_xoff[:,k])-yerr,np.log10(g_sigma_xoff[:,k])+yerr,alpha=0.4,label=r'$Xoff = %.3g$'%(label))
            plt.plot(s_bins_all[:len_s_bins[j]],np.log10(g_sigma_xoff_model[:,k]))
        else:
            plt.fill_between(s_bins_all[low:up],np.log10(g_sigma_xoff[:,k])-yerr,np.log10(g_sigma_xoff[:,k])+yerr,alpha=0.4,label=r'$Xoff = %.3g$'%(label))
            plt.plot(s_bins_all[low:up],np.log10(g_sigma_xoff_model[:,k]))
    plt.legend(fontsize=15)
    plt.ylabel(r'$\log_{10}g(\sigma,X_{off})$', fontsize=20)
    plt.xlabel(r'$\log_{10}\sigma^{-1}$',fontsize=20)
    plt.tick_params(labelsize=15)
    plt.ylim(-5,0)
    plt.grid(True)
    outfig_mass = os.path.join(this_dir,'results','zevo','red_%.3f'%(z_snap),'g_sigma_xoff_xoff_slices.png')
    os.makedirs(os.path.dirname(outfig_mass), exist_ok=True)
    plt.savefig(outfig_mass,overwrite=True)
    plt.close()

    #plot integral g(sigma,spin)dxoff vs spin in mass slices
    plt.figure(figsize=(10,10))
    if(j==0):
        a=len(s_bins_all[:len_s_bins[j]])
    else:
        a=len(s_bins_all[low:up])        
    print(a)
    b=5
    l = 19
    u = a-35
    for i in range(b):
        k = l+int((i)/(b)*(u-l))
        if(j==0):
            R=cosmo.sigma(1/(10**s_bins_all[:len_s_bins[j]][k]),z=z_snap,inverse=True)
        else:
            R=cosmo.sigma(1/(10**s_bins_all[low:up][k]),z=z_snap,inverse=True)
        label = peaks.lagrangianM(R)/h
#        yerr=1/np.log(10)*g_sigma_spin_err[j,:]/g_sigma_spin[j,:] #get log error
        yerr=1/np.log(10)*err_g_sigma_spin[k,:]/g_sigma_spin[k,:] 
        plt.fill_between(spin_bins,np.log10(g_sigma_spin[k,:])-yerr,np.log10(g_sigma_spin[k,:])+yerr,alpha=0.4,label=r'$M = %.3g$'%(label))
        plt.plot(spin_bins,np.log10(g_sigma_spin_model[k,:]))
    plt.legend(fontsize=15)
    plt.tick_params(labelsize=15)
    plt.ylim(-5,0)
    #plt.xlim(-3.5,0.)
    plt.ylabel(r'$\log_{10}g(\sigma,\lambda_P)$', fontsize=20)
    plt.xlabel(r'$\log_{10}\lambda_P$', fontsize=20)
    plt.grid(True)
    outfig_xo = os.path.join(this_dir,'results','zevo','red_%.3f'%(z_snap),'g_sigma_spin_mass_slices.png')
    os.makedirs(os.path.dirname(outfig_xo), exist_ok=True)
    plt.savefig(outfig_xo,overwrite=True)
    plt.close()

    #plot integral g(sigma,spin)dxoff vs sigma in spin slices
    plt.figure(figsize=(10,10))
    a=len(spin_bins)
    b=5
    l= int(len(spin_bins)/2.6)
    u = a-12
    for i in range(b):
        k = l+int((i)/(b)*(u-l))
#        yerr=1/np.log(10)*g_sigma_spin_err[:,j]/g_sigma_spin[:,j] #get log error
        yerr=1/np.log(10)*err_g_sigma_spin[:,k]/g_sigma_spin[:,k]
        label = 10**spin_bins[k]
        if(j==0):
            plt.fill_between(s_bins_all[:len_s_bins[j]],np.log10(g_sigma_spin[:,k])-yerr,np.log10(g_sigma_spin[:,k])+yerr,alpha=0.4,label=r'$\lambda_P = %.3g$'%(label))
            plt.plot(s_bins_all[:len_s_bins[j]],np.log10(g_sigma_spin_model[:,k]))
        else:
            plt.fill_between(s_bins_all[low:up],np.log10(g_sigma_spin[:,k])-yerr,np.log10(g_sigma_spin[:,k])+yerr,alpha=0.4,label=r'$\lambda_P = %.3g$'%(label))
            plt.plot(s_bins_all[low:up],np.log10(g_sigma_spin_model[:,k]))   
    plt.legend(fontsize=15)
    plt.ylabel(r'$\log_{10}g(\sigma,\lambda_P)$', fontsize=20)
    plt.xlabel(r'$\log_{10}\sigma^{-1}$',fontsize=20)
    plt.tick_params(labelsize=15)
    plt.ylim(-5,0)
    plt.grid(True)
    outfig_mass = os.path.join(this_dir,'results','zevo','red_%.3f'%(z_snap),'g_sigma_spin_spin_slices.png')
    os.makedirs(os.path.dirname(outfig_mass), exist_ok=True)
    plt.savefig(outfig_mass,overwrite=True)
    plt.close()

    #plot integral g(xoff,spin)dsigma vs xoff in spin slices
    plt.figure(figsize=(10,10))
    a=len(spin_bins)
    b=5
    l= int(len(spin_bins)/2.6)
    u = a-12
    for i in range(b):
        k = l+int((i)/(b)*(u-l))
#        yerr=1/np.log(10)*g_xoff_spin_err[:,j]/g_xoff_spin[:,j] #get log error
        yerr=1/np.log(10)*err_g_xoff_spin[:,k]/g_xoff_spin[:,k]
        label = 10**spin_bins[k]
        plt.fill_between(xoff_bins,np.log10(g_xoff_spin[:,k])-yerr,np.log10(g_xoff_spin[:,k])+yerr,alpha=0.4,label=r'$\lambda_P = %.3g$'%(label))
        plt.plot(xoff_bins,np.log10(g_xoff_spin_model[:,k]))
    plt.legend(fontsize=15)
    plt.ylabel(r'$\log_{10}g(X_{off},\lambda_P)$', fontsize=20)
    plt.xlabel(r'$\log_{10}X_{off}$',fontsize=20)
    plt.tick_params(labelsize=15)
    plt.ylim(-5,0)
    plt.grid(True)
    outfig_mass = os.path.join(this_dir,'results','zevo','red_%.3f'%(z_snap),'g_xoff_spin_spin_slices.png')
    os.makedirs(os.path.dirname(outfig_mass), exist_ok=True)
    plt.savefig(outfig_mass,overwrite=True)
    plt.close()

    #plot integral g(xoff,spin)dsigma vs spin in xoff slices
    plt.figure(figsize=(10,10))
    a=len(xoff_bins)
    b=5
    l= int(len(xoff_bins)/2.0)
    u = a-8
    for i in range(b):
        k = l+int((i)/(b)*(u-l))
#        yerr=1/np.log(10)*g_xoff_spin_err[j,:]/g_xoff_spin[j,:] #get log error
        yerr=1/np.log(10)*err_g_xoff_spin[k,:]/g_xoff_spin[k,:]
        label = 10**xoff_bins[k]
        plt.fill_between(spin_bins,np.log10(g_xoff_spin[k,:])-yerr,np.log10(g_xoff_spin[k,:])+yerr,alpha=0.4,label=r'$X_{off} = %.3g$'%(label))
        plt.plot(spin_bins,np.log10(g_xoff_spin_model[k,:]))
    plt.legend(fontsize=15)
    plt.ylabel(r'$\log_{10}g(X_{off},\lambda_P)$', fontsize=20)
    plt.xlabel(r'$\log_{10}\lambda_P$',fontsize=20)
    plt.tick_params(labelsize=15)
    plt.ylim(-5,0)
    plt.grid(True)
    outfig_mass = os.path.join(this_dir,'results','zevo','red_%.3f'%(z_snap),'g_xoff_spin_xoff_slices.png')
    os.makedirs(os.path.dirname(outfig_mass), exist_ok=True)
    plt.savefig(outfig_mass,overwrite=True)

    #plot integral f(sigma,xoff,lambda)dsigma dlambda
    rat = (np.log10(f_xoff)-np.log10(f_xoff_model))#/f_xoff_model
    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(3,1)
    ax1 = fig.add_subplot(gs[0:2, :])
    ax2 = fig.add_subplot(gs[2, :])
    yerr=1/np.log(10)*err_f_xoff/f_xoff
    ax1.fill_between(xoff_bins,np.log10(f_xoff)-yerr,np.log10(f_xoff)+yerr, label=r'$f(X_{off})\ data$',alpha=0.8)
    ax2.fill_between(xoff_bins,rat-yerr,rat+yerr,alpha=0.8)
    ax2.hlines(0,min(xoff_bins),max(xoff_bins))
    ax1.plot(xoff_bins,np.log10(f_xoff_model), label=r'$f(X_{off})\ model$',linewidth=3, color = 'red')
    ax1.legend(fontsize=15)
    ax1.set_ylabel(r'$\log_{10}f(X_{off})$', fontsize=20)   
    ax2.set_xlabel(r'$\log_{10}X_{off}$',fontsize=20)
    ax2.set_ylabel(r'$\Delta\log_{10}f$', fontsize=20)
    ax1.tick_params(labelsize=15)
    ax2.tick_params(labelsize=15)
    ax1.set_ylim(-5,0)
    ax1.grid(True)
    ax2.set_ylim(-0.5,0.5)
    ax2.grid(True)
    plt.tight_layout()
    outfig_mass = os.path.join(this_dir,'results','zevo','red_%.3f'%(z_snap),'f_xoff_integral.png')
    os.makedirs(os.path.dirname(outfig_mass), exist_ok=True)
    plt.savefig(outfig_mass,overwrite=True)
    plt.close()

    #plot integral h(sigma,xoff,lambda)dsigma dxoff
    rat = (np.log10(f_spin)-np.log10(f_spin_model))#/f_spin_model
    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(3,1)
    ax1 = fig.add_subplot(gs[0:2, :])
    ax2 = fig.add_subplot(gs[2, :])
    #yerr=1/np.log(10)*g_spin_err/g_spin
    yerr=1/np.log(10)*err_f_spin/f_spin
    ax1.fill_between(spin_bins,np.log10(f_spin)-yerr,np.log10(f_spin)+yerr, label=r'$f(\lambda_P)\ data$',alpha=0.8)
    ax2.fill_between(spin_bins,rat-yerr,rat+yerr,alpha=0.8)
    ax2.hlines(0,min(spin_bins),max(spin_bins))
    ax1.plot(spin_bins,np.log10(f_spin_model), label=r'$f(\lambda_P)\ model$',linewidth=3, color = 'red')
    ax1.legend(fontsize=15)
    ax1.set_ylabel(r'$\log_{10}f(\lambda_P)$', fontsize=20)
    ax2.set_ylabel(r'$\Delta\log_{10}f$', fontsize=20)
    ax2.set_xlabel(r'$\log_{10}\lambda_P$',fontsize=20)
    ax1.tick_params(labelsize=15)
    ax2.tick_params(labelsize=15)
    ax1.set_ylim(-5,0)
    ax1.grid(True)
    ax2.set_ylim(-0.1,0.1)
    ax2.grid(True)
    plt.tight_layout()
    outfig_mass = os.path.join(this_dir,'results','zevo','red_%.3f'%(z_snap),'f_spin_integral.png')
    os.makedirs(os.path.dirname(outfig_mass), exist_ok=True)
    plt.savefig(outfig_mass,overwrite=True)
    plt.close()

    def Mass_sigma(x):
        r=cosmo.sigma(1/10**x,z=z_snap,inverse=True)
        M=peaks.lagrangianM(r)/h
        return np.log10(M)

    #   plot multiplicity function f = integral g(sigma,xoff,lambda)dxoff dlambda
    if(j==0):
        mf_comparat=mf.massFunction(1/(10**s_bins_all[:len_s_bins[j]]),q_in='sigma', z=z_snap, mdef = 'vir', model = 'comparat17', q_out = 'f')
    else:
        mf_comparat=mf.massFunction(1/(10**s_bins_all[low:up]),q_in='sigma', z=z_snap, mdef = 'vir', model = 'comparat17', q_out = 'f')         
    ratio1 = (f_integral_model-mf_comparat)/mf_comparat
    ratio2 = (f_integral-f_integral_model)/f_integral_model
    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(3,1)
    ax1 = fig.add_subplot(gs[0:2, :])
    ax2 = fig.add_subplot(gs[2, :])
    #yerr = ferr
    yerr = 1/np.log(10)*err_f_sigma/f_integral
    #ax1.plot(s_bins,np.log10(f_integral), label='integral data',linewidth=6,c='C9')
    if(j==0):
        ax1.fill_between(s_bins_all[:len_s_bins[j]],np.log10(f_integral)-yerr, np.log10(f_integral)+yerr,label='integral data',alpha=0.8,color='C9')
        ax1.plot(s_bins_all[:len_s_bins[j]],np.log10(f_integral_model), label='seppi20',linewidth=6,c='C0')
        ax1.plot(s_bins_all[:len_s_bins[j]],np.log10(mf_comparat),label='comparat17',linewidth=2,c='C4')
        #ax2.plot(s_bins_all[:len_s_bins[j]],ratio1,linewidth=4,c='C0',label='models')
        ax2.fill_between(s_bins_all[:len_s_bins[j]],ratio2-err_f_sigma/f_integral_model,ratio2+err_f_sigma/f_integral_model,alpha=0.8,color='C9',label='data')
        ax2.hlines(0,min(s_bins_all[:len_s_bins[j]]),max(s_bins_all[:len_s_bins[j]]))
    else:
        ax1.fill_between(s_bins_all[low:up],np.log10(f_integral)-yerr, np.log10(f_integral)+yerr,label='integral data',alpha=0.8,color='C9')
        ax1.plot(s_bins_all[low:up],np.log10(f_integral_model), label='seppi20',linewidth=6,c='C0')
        ax1.plot(s_bins_all[low:up],np.log10(mf_comparat),label='comparat17',linewidth=2,c='C4')
        #ax2.plot(s_bins_all[low:up],ratio1,linewidth=4,c='C0',label='models')
        ax2.fill_between(s_bins_all[low:up],ratio2-err_f_sigma/f_integral_model,ratio2+err_f_sigma/f_integral_model,alpha=0.8,color='C9',label='data')
        ax2.hlines(0,min(s_bins_all[low:up]),max(s_bins_all[low:up]))  
    ax1.fill_between(s1_all[j],np.log10(f_dir1_list[j])-f1_err_list[j], np.log10(f_dir1_list[j])+f1_err_list[j],label=r'$f(\sigma)\ HMD$',alpha=0.8,color='C1')
    ax1.fill_between(s2_all[j],np.log10(f_dir2_list[j])-f2_err_list[j], np.log10(f_dir2_list[j])+f2_err_list[j],label=r'$f(\sigma)\ BigMD$',alpha=0.8,color='C2')
    ax1.fill_between(s3_all[j],np.log10(f_dir3_list[j])-f3_err_list[j], np.log10(f_dir3_list[j])+f3_err_list[j],label=r'$f(\sigma)\ MDPL2$',alpha=0.8,color='C3')      
    ax1_sec = ax1.twiny()
    xmin,xmax=ax1.get_xlim()
    ax1_sec.set_xlim((Mass_sigma(xmin),Mass_sigma(xmax)))
    ax1_sec.plot([],[])
    ax1_sec.set_xlabel(r'$\log_{10}M\ [M_{\odot}]$', fontsize=25, labelpad=15)
    ax1_sec.tick_params(labelsize=20)
    ax1.set_ylim(bottom=-5)
    ax2.set_ylim(-0.1,0.1)
    ax1.grid(True)
    ax2.grid(True)
    ax2.set_xlabel(r'$\log_{10}\sigma^{-1}$', fontsize=20)
    ax2.set_ylabel(r'$\Delta f/f$', fontsize=20)
    ax1.set_ylabel(r'$\log_{10}f(\sigma)$', fontsize=20)
    ax1.legend(fontsize=15,loc=3)
    ax2.legend(fontsize=15,loc=3)
    #plt.yscale('log')
    #plt.xlim(left=log1_sigma_low,right=log1_sigma_up)
    ax1.set_ylim(-5)
    ax1.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)
    plt.tight_layout()
    outfig = os.path.join(this_dir,'results','zevo','red_%.3f'%(z_snap),'f_integral_gsigma.png')
    plt.savefig(outfig,overwrite=True)
    plt.close()

    print('Initial parameters:')
    print(guess_prms)
    print('Fitted parameters:')
    print(popt)

    print('done!')

sys.exit()







