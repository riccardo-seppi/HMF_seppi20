#!/usr/bin/env python
# coding: utf-8

# In[94]:


"""
Models mass functions as function of mass, Xoff and spin
"""

#MAKES THE 3D HISTOGRAM AND CONVERTS THE COUNTS TO h(sigma,xoff,lambda)

from astropy.table import Table, Column
#from astropy_healpix import healpy
import sys
import os, glob
import time
from astropy.cosmology import FlatLambdaCDM
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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

print('Models mass functions as function of mass, Xoff and spin')
print('------------------------------------------------')
print('------------------------------------------------')
t0 = time.time()

#env = 'MD40' 
# initializes pathes to files
#test_dir = os.path.join(os.environ[env], 'Mass_Xoff_Concentration')
this_dir='.'
file_dir='/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration'

path_2_snapshot_data = os.path.join(file_dir, 'distinct_1.0.fits.gz')
path_2_snapshot_data2_5 = '/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_1.0.fits.gz'
path_2_snapshot_data1_0 = '/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_1.0.fits.gz'

path_2_snapshot_data4_0_list = np.array(['/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_1.0.fits.gz','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.9567.fits','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.8951.fits','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.8192.fits','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.7016.fits','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.6565.fits','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.5876.fits','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.5622.fits','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.4922.fits','/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.4123.fits'])

path_2_snapshot_data2_5_list = np.array(['/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_1.0.fits.gz','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.956.fits','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.8953.fits','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.8173.fits','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.7003.fits','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.6583.fits','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.5864.fits','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.5623.fits','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.5.fits','/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration/distinct_0.409.fits'])

path_2_snapshot_data1_0_list = np.array(['/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_1.0.fits.gz','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.9567.fits','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.8951.fits','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.8192.fits','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.7016.fits','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.6565.fits','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.5876.fits','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.5622.fits','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.4922.fits','/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration/distinct_0.4123.fits'])

s_low_list_HMD = np.array([-0.08,-0.07,-0.06,-0.05,-0.015,0.01,0.04,0.065,0.11,0.18])  #-0.08
s_low_list = np.array([-0.09,-0.075,-0.06,-0.05,-0.04,-0.03,-0.02,-0.01,0.00,0.01])     #-0.09
index = 0
aexp = float(os.path.basename(path_2_snapshot_data4_0_list[index][:-8]).split('_')[1])
z_snap = 1/aexp -1

cosmo = cosmology.setCosmology('multidark-planck')    
E_z = cosmo.Ez(z=z_snap)
h = cosmo.Hz(z=0.0)/100
Vol1 = (4e3/(1+z_snap))**3
Vol2 = (2.5e3/(1+z_snap))**3
Vol3 = (1e3/(1+z_snap))**3
dc = peaks.collapseOverdensity(z = z_snap)
print(dc)
rho_m = cosmo.rho_m(z=z_snap)*1e9

hd1 = fits.open(path_2_snapshot_data4_0_list[index])

#mass1_=hd1[1].data['Mmvir_all']
mass1_=hd1[1].data['Mvir']
R1 = peaks.lagrangianR(mass1_)
sigma1 = cosmo.sigma(R1,z=z_snap)
log1_sigma1 = np.log10(1/sigma1)
logxoff_data1 = np.log10(hd1[1].data['Xoff']/hd1[1].data['Rvir'])
print(min(logxoff_data1),max(logxoff_data1))
log_spin1 = np.log10(hd1[1].data['Spin'])

hd2 = fits.open(path_2_snapshot_data2_5_list[index])

mass2_=hd2[1].data['Mvir']
R2 = peaks.lagrangianR(mass2_)
sigma2 = cosmo.sigma(R2,z=z_snap)
log1_sigma2 = np.log10(1/sigma2)
logxoff_data2 = np.log10(hd2[1].data['Xoff']/hd2[1].data['Rvir'])
print(min(logxoff_data2),max(logxoff_data2))
log_spin2 = np.log10(hd2[1].data['Spin'])

hd3 = fits.open(path_2_snapshot_data1_0_list[index])

mass3_=hd3[1].data['Mvir']
R3 = peaks.lagrangianR(mass3_)
sigma3 = cosmo.sigma(R3,z=z_snap)
log1_sigma3 = np.log10(1/sigma3)
logxoff_data3 = np.log10(hd3[1].data['Xoff']/hd3[1].data['Rvir'])
print(min(logxoff_data3),max(logxoff_data3))
log_spin3 = np.log10(hd3[1].data['Spin'])


#fsigma1
diff_sigma1 = 1e-2
diff_sigma2 = 1e-2
diff_sigma3 = 1e-2
s_low_HMD = s_low_list_HMD[index]
print('s_low_HMD = ',s_low_HMD)
s_low = s_low_list[index]
print('s_low = ',s_low)
r_low_HMD = cosmo.sigma(1/10**s_low_HMD, z=z_snap, inverse = True)
mass_low_HMD = peaks.lagrangianM(r_low_HMD)/h
print('mass low HMD = %.3g M sun'%(mass_low_HMD))
r_low = cosmo.sigma(1/10**s_low, z=z_snap, inverse = True)
mass_low = peaks.lagrangianM(r_low)/h
print('mass low = %.3g M sun'%(mass_low))
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
s_edges2 = np.arange(s_low-diff_sigma2/3,0.45,diff_sigma2)
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
s_edges3 = np.arange(s_low-diff_sigma3*2/3,0.4,diff_sigma3)
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
xoff_edges = np.linspace(-3.8,-0.2,50)
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


# Initial guesses to the fit parameters.

#def my_flat_priors(cube):
#    params = cube.copy()
#    params[0] = (cube[0])*1 - 22.3  #A
#    params[1] = (cube[1])*0.3 + 0.7        #a
#    params[2] = (cube[2])*0.6 + 1.9            #q
#    params[3] = (cube[3])*0.5 - 3.4      #mu
#    params[4] = (cube[4])*0.4 + 5.4        #alpha
#    params[5] = (cube[5])*0.25 + 0.1             #beta
#    params[6] = (cube[6])*0.6 - 1.8           #e0
#    params[7] = (cube[7])*0.5 - 3.45            #nu
#    params[8] = (cube[8])*0.5 + 2.8              #gamma
#    params[9] = (cube[9])*0.3 + 1        #delta
#    params[10] = (cube[10])*0.1 - 0.4              #beta1
#    params[11] = (cube[11])*0.5 - 2.9        #e1

#    return params

def my_flat_priors(cube):
    params = cube.copy()
    params[0] = (cube[0])*1 - 22.5  #A
    params[1] = (cube[1])*0.3 + 0.7        #a
    params[2] = (cube[2])*0.2 + 2.2            #q
    params[3] = (cube[3])*0.6 - 3.6      #mu
    params[4] = (cube[4])*0.4 + 5.4        #alpha
    params[5] = (cube[5])*0.2 - 0.5             #beta
    params[6] = (cube[6])*0.4 + 2.8              #gamma
    params[7] = (cube[7])*0.3 + 1        #delta
    params[8] = (cube[8])*0.4 - 1.3        #e

    return params



#THE FITTING FUNCTION IS A MASS FUNCTION MULTIPLIED BY A MODIFIED SCHECHTER IN XOFF and spinpar
#guess_prms = [( -2, 0.9, 2.0, -3,5.,-0.35,-1.5,3.,1.1,-2.7)] #paper
guess_prms = [( -2, 0.9, 2.0, -3.3,5.,-0.35,-1.5,3.,1.1,-2.7)]
guess_prms = [( -2, 0.9, 2.0, -3.3,5.,-0.35,3.,1.1,-1.1)] #no e0
def h_func(data,A,a,q,mu,alpha,beta,gamma,delta,e):
#def h_func(data,A,a,q,mu,alpha,beta,e0,gamma,delta,e1): #paper
    x_,y_,z_ = data      #x_ is log10(1/sigma) y_ is log10(Xoff)
    x = 1/10**x_ #sigma
    y = 10**y_   #Xoff
    z = 10**z_ #spin
#    return A+np.log10(np.sqrt(2/np.pi)) + q*np.log10(np.sqrt(a)*dc/x) - a/2/np.log(10)*dc**2/x**2 + (alpha)*np.log10(y/10**mu/x**e0) - 1/np.log(10)*(y/10**mu/(x**e0))**(0.05*alpha) + gamma*np.log10(z/(0.7*10**(mu))) - 1/np.log(10)*(y/10**(mu)/x**e1)**(beta)*(z/(0.7*10**(mu)))**(delta)   #paper

#    return A+np.log10(np.sqrt(2/np.pi)) + q*np.log10(np.sqrt(a)*dc/x) - a/2/np.log(10)*dc**2/x**2 + (alpha)*np.log10(y/10**(1.84*mu)/x**e0) - 1/np.log(10)*(y/10**(1.84*mu)/(x**e0))**(0.05*alpha) + gamma*np.log10(z/(10**(mu))) - 1/np.log(10)*(y/10**(1.84*mu)/x**e1)**(beta)*(z/(10**(mu)))**(delta)

    return A+np.log10(np.sqrt(2/np.pi)) + q*np.log10(np.sqrt(a)*dc/x) - a/2/np.log(10)*dc**2/x**2 + (alpha)*np.log10(y/10**(1.83*mu)) - 1/np.log(10)*(y/10**(1.83*mu))**(0.05*alpha) + gamma*np.log10(z/(10**(mu))) - 1/np.log(10)*(y/10**(1.83*mu)/x**e)**(beta)*(z/(10**(mu)))**(delta) #no e0


#names = [r'$\log_{10}A$',r'$a$',r'$q$',r'$\log_{10}\mu$',r'$\alpha$',r'$\beta$',r'$e_0$',r'$\log_{10}\nu$',r'$\gamma$',r'$\delta$',r'$\beta_1$',r'$e_1$']
#names = [r'$\log_{10}A$',r'$a$',r'$q$',r'$\log_{10}\mu$',r'$\alpha$',r'$\beta$',r'$e_0$',r'$\gamma$',r'$\delta$',r'$e_1$']
names = [r'$\log_{10}A$',r'$a$',r'$q$',r'$\log_{10}\mu$',r'$\alpha$',r'$\beta$',r'$\gamma$',r'$\delta$',r'$e$']
#fit only points where gsigma is different from 0
c1 = 30
hsigma_ = hsigma[np.where(countsh>c1)]
herr_ = herr[np.where(countsh>c1)]
s_grid_ = s_grid[np.where(countsh>c1)]
xoff_grid_ = xoff_grid[np.where(countsh>c1)]
spin_grid_ = spin_grid[np.where(countsh>c1)]

def likelihood(params):
    model = h_func([s_grid_.ravel(),xoff_grid_.ravel(),spin_grid_.ravel()],*params)
    loglike = -0.5*(((model - np.log10(hsigma_.ravel()))/herr_.ravel())**2).sum()
#    loglike = -0.5*(((model - np.log10(hsigma_.ravel()))/herr_.ravel())**2 + np.log(2*np.pi*herr_.ravel()**2)).sum()
    #loglike = -np.sum(model-(np.log10(gsigma_.ravel())*np.log(model)))
    return loglike

def likelihood_mcmc(params,x,y,err):
    x0,x1,x2 = x
    model = h_func([x0,x1,x2],*params)
#    loglike = -0.5*(((np.log10(y) - model)/err)**2).sum()
    loglike = -0.5*(((np.log10(hsigma_.ravel()-model))/herr_.ravel())**2 + np.log(2*np.pi*herr_.ravel()**2)).sum()
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
fig=corner.corner(parameters[mask,:], weights=weights[mask], labels=names, show_titles=True, color='r',bins=50,smooth=True,smooth1d=True,quantiles=[0.025,0.16,0.84,0.975],label_kwargs={'fontsize':15,'labelpad':20},title_kwargs={"fontsize":15},levels=[0.68,0.95],fill_contours=True,title_fmt='.3f')
axes = np.array(fig.axes).reshape((len(names), len(names)))
print(axes.shape)
for i in range(len(names)):
    ax = axes[i, 0]
    if(i>0):
        ax.yaxis.set_tick_params(labelsize=11,direction='in')
        if(i==7):
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax1 = axes[len(names)-1, i]
    ax1.xaxis.set_tick_params(labelsize=11,direction='in')
    if(i==0):
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.3f')) 
    elif(i==7):
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    else:
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))    
fig.tight_layout()
outfig = os.path.join(this_dir,'results','z_%.3f'%(z_snap),'corner_hsigma_Rvir.png')
os.makedirs(os.path.dirname(outfig), exist_ok=True)
fig.savefig(outfig,overwrite=True)
#############END OF ULTRANEST#######

'''
#################UNCOMMENT TO USE MCMC###################
#nll = lambda *args: -likelihood_mcmc(*args)
#initial = np.array(guess_prms)
#soln = minimize(nll,initial,args=([s_grid_.ravel(),xoff_grid_.ravel(),spin_grid_.ravel()],hsigma_.ravel(),herr_.ravel()))
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
fig.savefig(os.path.join(this_dir,'extended','3d','z_%.3f'%(z_snap),'corner_mcmc_hsigma_Rvir.png'))

pvar = []
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    pvar.append(np.average(q))
pvar = np.array(pvar)    


###################END OF MCMC ###################
'''

# Flatten the initial guess parameter list.
xdata = np.vstack((s_grid_.ravel(),xoff_grid_.ravel(),spin_grid_.ravel()))

#popt, pcov = curve_fit(h_func, xdata, np.log10(hsigma_.ravel()),guess_prms, sigma = herr_.ravel(),maxfev=100000)
#pvar = np.diag(pcov)

print('Initial parameters:')
print(guess_prms)
print('Fitted parameters:')
print(popt)
fit = h_func([s_grid,xoff_grid,spin_grid], *popt)
#print('fit is ', fit)
print(s_grid.shape)
print(xoff_grid.shape)
print(spin_grid.shape)
print(hsigma.shape)
print(fit.shape)

#tab = Table(fit,names=s_bins)
#out_table = os.path.join(this_dir,'z_%.3f'%(z_snap),'model.fit')
#os.makedirs(os.path.dirname(out_table), exist_ok=True)
#tab.write(out_table, overwrite=True)

t = Table()
t.add_column(Column(name='pars', data=popt, unit=''))
t.add_column(Column(name='err', data=pvar, unit=''))
out_table = os.path.join(this_dir,'results','z_%.3f'%(z_snap),'hsigma_params_Rvir.fit')
os.makedirs(os.path.dirname(out_table), exist_ok=True)
t.write(out_table, overwrite=True)

#fit_ = fit[np.where(counts_tot>c3)]
fit_ = h_func([s_grid_, xoff_grid_,spin_grid_], *popt)
print('min log10(hsigma_) =', min(np.log10(hsigma_)))
print('min fit_ = ',min(fit_))

print('max log10(hsigma_) =', max(np.log10(hsigma_)))
print('max fit_ = ',max(fit_))

chi_2 = np.sum((np.log10(hsigma_)-fit_)**2/herr_**2)
dof = len(hsigma_.ravel()) - len(names)
chi_2r = chi_2/dof
rv = chi2.stats(dof)
print('chi2 = ', chi_2)
print('dof = ', dof)
print('chi2r = ',chi_2r)
print('expected chi2 = ', rv)

#PDF OF THE RESIDUALS
res = ((np.log10(hsigma_)-fit_)/herr_)
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
outmodelfig_respdf = os.path.join(this_dir,'results','z_%.3f'%(z_snap),'resid_pdf_hsigma_Rvir.png')
os.makedirs(os.path.dirname(outmodelfig_respdf), exist_ok=True)
plt.savefig(outmodelfig_respdf,overwrite=True)



g_sigma_xoff = np.zeros((len(s_bins),len(xoff_bins)))
g_sigma_spin = np.zeros((len(s_bins),len(spin_bins)))
g_xoff_spin = np.zeros((len(xoff_bins),len(spin_bins)))
g_sigma_xoff_model = np.zeros((len(s_bins),len(xoff_bins)))
g_sigma_spin_model = np.zeros((len(s_bins),len(spin_bins)))
g_xoff_spin_model = np.zeros((len(xoff_bins),len(spin_bins)))

counts_sigma_xoff = np.zeros((len(s_bins),len(xoff_bins)))
counts_sigma_spin = np.zeros((len(s_bins),len(spin_bins)))
counts_xoff_spin = np.zeros((len(xoff_bins),len(spin_bins)))

err_g_sigma_xoff = np.zeros((len(s_bins),len(xoff_bins)))
err_g_sigma_spin = np.zeros((len(s_bins),len(spin_bins)))
err_g_xoff_spin = np.zeros((len(xoff_bins),len(spin_bins)))

for i in range(len(s_bins)):
    for j in range(len(xoff_bins)):
        g_sigma_xoff[i,j] = integrate.simps(hsigma[:,j,i],spin_bins)
        g_sigma_xoff_model[i,j] = integrate.simps((10**fit[:,j,i]),spin_bins)
        counts_sigma_xoff[i,j] = np.sum(countsh[:,j,i])
        err_g_sigma_xoff[i,j] = np.sqrt(np.sum((herror[:,j,i][~np.isnan(herror[:,j,i])])**2))*diff_spin

for i in range(len(s_bins)):
    for j in range(len(spin_bins)):
        g_sigma_spin[i,j] = integrate.simps(hsigma[j,:,i],xoff_bins)
        g_sigma_spin_model[i,j] = integrate.simps((10**fit[j,:,i]),xoff_bins)
        counts_sigma_spin[i,j] = np.sum(countsh[j,:,i])
        err_g_sigma_spin[i,j] = np.sqrt(np.sum((herror[j,:,i][~np.isnan(herror[j,:,i])])**2))*diff_xoff

for i in range(len(xoff_bins)):
    for j in range(len(spin_bins)):
        g_xoff_spin[i,j] = integrate.simps(hsigma[j,i,:],s_bins)
        g_xoff_spin_model[i,j] = integrate.simps((10**fit[j,i,:]),s_bins)
        counts_xoff_spin[i,j] = np.sum(countsh[j,i,:])
        err_g_xoff_spin[i,j] = np.sqrt(np.sum((herror[j,i,:][~np.isnan(herror[j,i,:])])**2))*diff_sigma3

g_sigma_xoff_err = np.sqrt(1/counts_sigma_xoff + 0.04**2)*g_sigma_xoff
g_sigma_spin_err = np.sqrt(1/counts_sigma_spin + 0.04**2)*g_sigma_spin
g_xoff_spin_err = np.sqrt(1/counts_xoff_spin + 0.04**2)*g_xoff_spin

f_integral = np.zeros(len(s_bins))
f_integral_model = np.zeros(len(s_bins))
counts_sigma = np.zeros(len(s_bins))
err_f_sigma = np.zeros(len(s_bins))

for i in range(len(s_bins)):
    #the model is gsigma, so we integrate gsigma
    f_integral[i] = integrate.simps(g_sigma_xoff[i,:],xoff_bins)
    f_integral_model[i] = integrate.simps(g_sigma_spin_model[i,:],spin_bins)
    counts_sigma[i] = np.sum(counts_sigma_xoff[i,:])
    err_f_sigma[i] = np.sqrt(np.sum((err_g_sigma_xoff[i,:])**2))*diff_xoff

f_xoff = np.zeros(len(xoff_bins))
f_spin = np.zeros(len(spin_bins))
f_xoff_model = np.zeros(len(xoff_bins))
f_spin_model = np.zeros(len(spin_bins))
counts_xoff = np.zeros(len(xoff_bins))
counts_spin= np.zeros(len(spin_bins))
err_f_xoff = np.zeros(len(xoff_bins))
err_f_spin = np.zeros(len(spin_bins))

for i in range(len(xoff_bins)):
    f_xoff[i] = integrate.simps(g_sigma_xoff[:,i],s_bins)
    f_xoff_model[i] = integrate.simps(g_sigma_xoff_model[:,i],s_bins)
    counts_xoff[i] = np.sum(counts_sigma_xoff[:,i])
    err_f_xoff[i] = np.sqrt(np.sum((err_g_sigma_xoff[:,i])**2))*diff_sigma3

for i in range(len(spin_bins)):
    f_spin[i] = integrate.simps(g_sigma_spin[:,i],s_bins)
    f_spin_model[i] = integrate.simps(g_sigma_spin_model[:,i],s_bins)
    counts_spin[i] = np.sum(counts_sigma_spin[:,i])
    err_f_spin[i] = np.sqrt(np.sum((err_g_sigma_spin[:,i])**2))*diff_sigma3

f_sigma_err = np.sqrt(1/counts_sigma + 0.04**2)*f_integral
f_xoff_err = np.sqrt(1/counts_xoff + 0.04**2)*f_xoff 
f_spin_err = np.sqrt(1/counts_spin + 0.04**2)*f_spin 

#plot integral g(sigma,xoff)dspin vs xoff in mass slices
plt.figure(figsize=(4.5,4.5))
a=len(s_bins)
print(a)
b=5
l = 19
u = a-35
for i in range(b):
    j = l+int((i)/(b)*(u-l))
    R=cosmo.sigma(1/(10**s_bins[j]),z=z_snap,inverse=True)
    label = peaks.lagrangianM(R)/h
#    yerr=1/np.log(10)*g_sigma_xoff_err[j,:]/g_sigma_xoff[j,:] #get log error
    yerr=1/np.log(10)*err_g_sigma_xoff[j,:]/g_sigma_xoff[j,:] 
    plt.fill_between(xoff_bins,np.log10(g_sigma_xoff[j,:])-yerr,np.log10(g_sigma_xoff[j,:])+yerr,alpha=0.4,label=r'$M = %.3g$'%(label))
    plt.plot(xoff_bins,np.log10(g_sigma_xoff_model[j,:]))
plt.legend(fontsize=10)
plt.tick_params(labelsize=12)
plt.ylim(-5,0)
#plt.xlim(-3.5,0.)
plt.ylabel(r'$\log_{10}g_{\lambda}(\sigma,X_{off})$', fontsize=12)
plt.xlabel(r'$\log_{10}X_{off}$', fontsize=12)
plt.grid(True)
plt.tight_layout()
outfig_xo = os.path.join(this_dir,'results','z_%.3f'%(z_snap),'g_sigma_xoff_mass_slices_Rvir.png')
os.makedirs(os.path.dirname(outfig_xo), exist_ok=True)
plt.savefig(outfig_xo,overwrite=True)

#plot integral g(sigma,xoff)dspin vs sigma in xoff slices
plt.figure(figsize=(4.5,4.5))
a=len(xoff_bins)
b=5
l= int(len(xoff_bins)/1.8)
u = a-8
for i in range(b):
    j = l+int((i)/(b)*(u-l))
#    yerr=1/np.log(10)*g_sigma_xoff_err[:,j]/g_sigma_xoff[:,j] #get log error
    yerr=1/np.log(10)*err_g_sigma_xoff[:,j]/g_sigma_xoff[:,j]
    label = 10**xoff_bins[j]
    plt.fill_between(s_bins,np.log10(g_sigma_xoff[:,j])-yerr,np.log10(g_sigma_xoff[:,j])+yerr,alpha=0.4,label=r'$Xoff = %.3g$'%(label))
    plt.plot(s_bins,np.log10(g_sigma_xoff_model[:,j]))
plt.legend(fontsize=10)
plt.ylabel(r'$\log_{10}g_{\lambda}(\sigma,X_{off})$', fontsize=12)
plt.xlabel(r'$\log_{10}\sigma^{-1}$',fontsize=12)
plt.tick_params(labelsize=12)
plt.ylim(-5,0)
plt.grid(True)
plt.tight_layout()
outfig_mass = os.path.join(this_dir,'results','z_%.3f'%(z_snap),'g_sigma_xoff_xoff_slices_Rvir.png')
os.makedirs(os.path.dirname(outfig_mass), exist_ok=True)
plt.savefig(outfig_mass,overwrite=True)

#plot integral g(sigma,spin)dxoff vs spin in mass slices
plt.figure(figsize=(4.5,4.5))
a=len(s_bins)
print(a)
b=5
l = 19
u = a-35
for i in range(b):
    j = l+int((i)/(b)*(u-l))
    R=cosmo.sigma(1/(10**s_bins[j]),z=z_snap,inverse=True)
    label = peaks.lagrangianM(R)/h
#    yerr=1/np.log(10)*g_sigma_spin_err[j,:]/g_sigma_spin[j,:] #get log error
    yerr=1/np.log(10)*err_g_sigma_spin[j,:]/g_sigma_spin[j,:] 
    plt.fill_between(spin_bins,np.log10(g_sigma_spin[j,:])-yerr,np.log10(g_sigma_spin[j,:])+yerr,alpha=0.4,label=r'$M = %.3g$'%(label))
    plt.plot(spin_bins,np.log10(g_sigma_spin_model[j,:]))
plt.legend(fontsize=10)
plt.tick_params(labelsize=12)
plt.ylim(-5,0)
#plt.xlim(-3.5,0.)
plt.ylabel(r'$\log_{10}g_{X_{off}}(\sigma,\lambda)$', fontsize=12)
plt.xlabel(r'$\log_{10}\lambda$', fontsize=12)
plt.grid(True)
plt.tight_layout()
outfig_xo = os.path.join(this_dir,'results',r'z_%.3f'%(z_snap),'g_sigma_spin_mass_slices_Rvir.png')
os.makedirs(os.path.dirname(outfig_xo), exist_ok=True)
plt.savefig(outfig_xo,overwrite=True)

#plot integral g(sigma,spin)dxoff vs sigma in spin slices
plt.figure(figsize=(4.5,4.5))
a=len(spin_bins)
b=5
l= int(len(spin_bins)/2.6)
u = a-12
for i in range(b):
    j = l+int((i)/(b)*(u-l))
#    yerr=1/np.log(10)*g_sigma_spin_err[:,j]/g_sigma_spin[:,j] #get log error
    yerr=1/np.log(10)*err_g_sigma_spin[:,j]/g_sigma_spin[:,j]
    label = 10**spin_bins[j]
    plt.fill_between(s_bins,np.log10(g_sigma_spin[:,j])-yerr,np.log10(g_sigma_spin[:,j])+yerr,alpha=0.4,label=r'$\lambda = %.3g$'%(label))
    plt.plot(s_bins,np.log10(g_sigma_spin_model[:,j]))
plt.legend(fontsize=10)
plt.ylabel(r'$\log_{10}g_{X_{off}}(\sigma,\lambda)$', fontsize=12)
plt.xlabel(r'$\log_{10}\sigma^{-1}$',fontsize=12)
plt.tick_params(labelsize=12)
plt.ylim(-5,0)
plt.grid(True)
plt.tight_layout()
outfig_mass = os.path.join(this_dir,'results','z_%.3f'%(z_snap),'g_sigma_spin_spin_slices_Rvir.png')
os.makedirs(os.path.dirname(outfig_mass), exist_ok=True)
plt.savefig(outfig_mass,overwrite=True)

#plot integral g(xoff,spin)dsigma vs xoff in spin slices
plt.figure(figsize=(4.5,4.5))
a=len(spin_bins)
b=5
l= int(len(spin_bins)/2.6)
u = a-12
for i in range(b):
    j = l+int((i)/(b)*(u-l))
#    yerr=1/np.log(10)*g_xoff_spin_err[:,j]/g_xoff_spin[:,j] #get log error
    yerr=1/np.log(10)*err_g_xoff_spin[:,j]/g_xoff_spin[:,j]
    label = 10**spin_bins[j]
    plt.fill_between(xoff_bins,np.log10(g_xoff_spin[:,j])-yerr,np.log10(g_xoff_spin[:,j])+yerr,alpha=0.4,label=r'$\lambda = %.3g$'%(label))
    plt.plot(xoff_bins,np.log10(g_xoff_spin_model[:,j]))
plt.legend(fontsize=10)
plt.tight_layout()
plt.ylabel(r'$\log_{10}g_{\sigma}(X_{off},\lambda)$', fontsize=12)
plt.xlabel(r'$\log_{10}X_{off}$',fontsize=12)
plt.tick_params(labelsize=12)
plt.ylim(-5,0)
plt.grid(True)
plt.tight_layout()
outfig_mass = os.path.join(this_dir,'results','z_%.3f'%(z_snap),'g_xoff_spin_spin_slices_Rvir.png')
os.makedirs(os.path.dirname(outfig_mass), exist_ok=True)
plt.savefig(outfig_mass,overwrite=True)

#plot integral g(xoff,spin)dsigma vs spin in xoff slices
plt.figure(figsize=(4.5,4.5))
a=len(xoff_bins)
b=5
l= int(len(xoff_bins)/1.8)
u = a-8
for i in range(b):
    j = l+int((i)/(b)*(u-l))
#    yerr=1/np.log(10)*g_xoff_spin_err[j,:]/g_xoff_spin[j,:] #get log error
    yerr=1/np.log(10)*err_g_xoff_spin[j,:]/g_xoff_spin[j,:]
    label = 10**xoff_bins[j]
    plt.fill_between(spin_bins,np.log10(g_xoff_spin[j,:])-yerr,np.log10(g_xoff_spin[j,:])+yerr,alpha=0.4,label=r'$X_{off} = %.3g$'%(label))
    plt.plot(spin_bins,np.log10(g_xoff_spin_model[j,:]))
plt.legend(fontsize=10)
plt.ylabel(r'$\log_{10}g_{\sigma}(X_{off},\lambda)$', fontsize=12)
plt.xlabel(r'$\log_{10}\lambda$',fontsize=12)
plt.tick_params(labelsize=12)
plt.ylim(-5,0)
plt.grid(True)
plt.tight_layout()
outfig_mass = os.path.join(this_dir,'results','z_%.3f'%(z_snap),'g_xoff_spin_xoff_slices_Rvir.png')
os.makedirs(os.path.dirname(outfig_mass), exist_ok=True)
plt.savefig(outfig_mass,overwrite=True)

#plot integral f(sigma,xoff,lambda)dsigma dlambda
rat = (np.log10(f_xoff)-np.log10(f_xoff_model))#/f_xoff_model
fig = plt.figure(figsize=(4.5,4.5))
gs = fig.add_gridspec(3,1,hspace=0.0)
ax1 = fig.add_subplot(gs[0:2, :])
ax2 = fig.add_subplot(gs[2, :], sharex = ax1)
yerr=1/np.log(10)*err_f_xoff/f_xoff
ax1.fill_between(xoff_bins,np.log10(f_xoff)-yerr,np.log10(f_xoff)+yerr, label=r'$f(X_{off})\ data$',alpha=0.8)
ax2.fill_between(xoff_bins,rat-yerr,rat+yerr,alpha=0.8)
ax2.hlines(0,min(xoff_bins),max(xoff_bins))
ax1.plot(xoff_bins,np.log10(f_xoff_model), label=r'$f(X_{off})\ model$',linewidth=3, color = 'red')
ax1.legend(fontsize=10)
ax1.set_ylabel(r'$\log_{10}f_{\sigma,\lambda}(X_{off})$', fontsize=12)
ax2.set_xlabel(r'$\log_{10}X_{off}$',fontsize=12)
ax2.set_ylabel(r'$\Delta\log_{10}f$', fontsize=12)
ax1.tick_params(labelsize=12)
ax2.tick_params(labelsize=12)
ax1.set_ylim(-5,0)
ax1.grid(True)
ax2.set_ylim(-0.5,0.5)
ax2.grid(True)
plt.setp(ax1.get_xticklabels(), visible=False)
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
plt.tight_layout()
outfig_mass = os.path.join(this_dir,'results','z_%.3f'%(z_snap),'f_xoff_integral_Rvir.png')
os.makedirs(os.path.dirname(outfig_mass), exist_ok=True)
plt.savefig(outfig_mass,overwrite=True)

#plot integral h(sigma,xoff,lambda)dsigma dxoff
rat = (np.log10(f_spin)-np.log10(f_spin_model))#/f_spin_model
fig = plt.figure(figsize=(4.5,4.5))
gs = fig.add_gridspec(3,1,hspace=0.0)
ax1 = fig.add_subplot(gs[0:2, :])
ax2 = fig.add_subplot(gs[2, :])
#yerr=1/np.log(10)*g_spin_err/g_spin
yerr=1/np.log(10)*err_f_spin/f_spin
ax1.fill_between(spin_bins,np.log10(f_spin)-yerr,np.log10(f_spin)+yerr, label=r'$f(\lambda)\ data$',alpha=0.8)
ax2.fill_between(spin_bins,rat-yerr,rat+yerr,alpha=0.8)
ax2.hlines(0,min(spin_bins),max(spin_bins))
ax1.plot(spin_bins,np.log10(f_spin_model), label=r'$f(\lambda)\ model$',linewidth=3, color = 'red')
ax1.legend(fontsize=10)
ax1.set_ylabel(r'$\log_{10}f_{\sigma,X_{off}}(\lambda)$', fontsize=12)
ax2.set_ylabel(r'$\Delta\log_{10}f$', fontsize=12)
ax2.set_xlabel(r'$\log_{10}\lambda$',fontsize=12)
ax1.tick_params(labelsize=12)
ax2.tick_params(labelsize=12)
ax1.set_ylim(-5,0)
ax1.grid(True)
ax2.set_ylim(-0.5,0.5)
ax2.grid(True)
plt.setp(ax1.get_xticklabels(), visible=False)
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
plt.tight_layout()
outfig_mass = os.path.join(this_dir,'results','z_%.3f'%(z_snap),'f_spin_integral_Rvir.png')
os.makedirs(os.path.dirname(outfig_mass), exist_ok=True)
plt.savefig(outfig_mass,overwrite=True)

def Mass_sigma(x):
    r=cosmo.sigma(1/10**x,z=z_snap,inverse=True)
    M=peaks.lagrangianM(r)#/h
    return np.log10(M)

#plot multiplicity function f = integral g(sigma,xoff,lambda)dxoff dlambda
mf_comparat=mf.massFunction(1/(10**s_bins),q_in='sigma', z=z_snap, mdef = 'vir', model = 'comparat17', q_out = 'f') 
ratio1 = (f_integral_model-mf_comparat)/mf_comparat
ratio2 = (f_integral-f_integral_model)/f_integral_model
fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(4,1,hspace=0.0)
ax1 = fig.add_subplot(gs[0:3, :])
ax2 = fig.add_subplot(gs[3, :])
#yerr = ferr
yerr = 1/np.log(10)*err_f_sigma/f_integral
#ax1.plot(s_bins,np.log10(f_integral), label='integral data',linewidth=6,c='C9')
ax1.fill_between(s_bins,np.log10(f_integral)-yerr, np.log10(f_integral)+yerr,label='integral data',alpha=0.8,color='C9')
ax1.plot(s_bins,np.log10(f_integral_model), label='seppi20',linewidth=5,c='C0')
ax1.plot(s_bins,np.log10(mf_comparat),label='comparat17',linewidth=3,c='k',linestyle='dashed')
ax1.fill_between(s_bins1,np.log10(fsigma_dir1)-ferr1, np.log10(fsigma_dir1)+ferr1,label=r'$f(\sigma)\ HMD$',alpha=0.8,color='C1')
ax1.fill_between(s_bins2,np.log10(fsigma_dir2)-ferr2, np.log10(fsigma_dir2)+ferr2,label=r'$f(\sigma)\ BigMD$',alpha=0.8,color='C2')
ax1.fill_between(s_bins3,np.log10(fsigma_dir3)-ferr3, np.log10(fsigma_dir3)+ferr3,label=r'$f(\sigma)\ MDPL2$',alpha=0.8,color='C3')
ax2.plot(s_bins,ratio1,linewidth=3,c='C0',label='models')
ax2.fill_between(s_bins,ratio2-err_f_sigma/f_integral_model,ratio2+err_f_sigma/f_integral_model,alpha=0.8,color='C9',label='data')
ax2.hlines(0,min(s_bins),max(s_bins))
#ax2.axvline(np.log10(1/cosmo.sigma(peaks.lagrangianR(5e13*h),z=z_snap)),color='C3',label=r'$5e13 M_\odot$')
#ax2.axvline(np.log10(1/cosmo.sigma(peaks.lagrangianR(1e14*h),z=z_snap)),color='C8',label=r'$1e14 M_\odot$')
#ax2.axvline(np.log10(1/cosmo.sigma(peaks.lagrangianR(1e15*h),z=z_snap)),color='C7',label=r'$1e15 M_\odot$')
#ax2.axvline(np.log10(1/cosmo.sigma(peaks.lagrangianR(5e15*h),z=z_snap)),color='C6',label=r'$5e15 M_\odot$')
ax1_sec = ax1.twiny()
xmin,xmax=ax1.get_xlim()
ax1_sec.set_xlim((Mass_sigma(xmin),Mass_sigma(xmax)))
ax1_sec.plot([],[])
ax1_sec.set_xlabel(r'$\log_{10}M\ [M_{\odot}/h]$', fontsize=24, labelpad=15)
ax1_sec.tick_params(labelsize=24)
ax1.set_ylim(bottom=-5)
ax2.set_ylim(-0.05,0.05)
ax1.grid(True)
ax2.grid(True)
ax2.set_xlabel(r'$\log_{10}\sigma^{-1}$', fontsize=24)
ax2.set_ylabel(r'$\Delta f/f$', fontsize=24)
ax1.set_ylabel(r'$\log_{10}f_{X_{off},\lambda}(\sigma)$', fontsize=24)
ax1.legend(fontsize=20,loc=3)
ax2.legend(fontsize=15,loc=3)
#plt.yscale('log')
#plt.xlim(left=log1_sigma_low,right=log1_sigma_up)
ax1.set_ylim(-5)
ax1.tick_params(labelsize=24)
ax2.tick_params(labelsize=24)
plt.setp(ax1.get_xticklabels(), visible=False)
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
plt.tight_layout()
outfig = os.path.join(this_dir,'results','z_%.3f'%(z_snap),'f_integral_gsigma_Rvir.png')
plt.savefig(outfig,overwrite=True)

print('Initial parameters:')
print(guess_prms)
print('Fitted parameters:')
print(popt)

print('done!')
sys.exit()







