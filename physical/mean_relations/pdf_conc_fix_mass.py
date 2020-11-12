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
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
import astropy.units as u
import astropy.constants as cc
import astropy.io.fits as fits
import scipy
from scipy.stats import chi2 as chi2scipy
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

print('Plots distributions of concentration and predictions of the model obtained with pdf_conc.py in consistent mass slices at different z')
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
path_2_snapshot_data = np.array([os.path.join(test_dir, 'distinct_1.0.fits'),os.path.join(test_dir,'distinct_0.6565.fits'),os.path.join(test_dir,'distinct_0.4922.fits'),os.path.join(test_dir,'distinct_0.4123.fits')])

dir_2_5 = '/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration'

path_2_snapshot_data2_5 = np.array([os.path.join(dir_2_5,'distinct_1.0.fits'),os.path.join(dir_2_5,'distinct_0.6583.fits'),os.path.join(dir_2_5,'distinct_0.5.fits'),os.path.join(dir_2_5,'distinct_0.409.fits')])
dir_1_0 = '/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration'

path_2_snapshot_data1_0 = np.array([os.path.join(dir_1_0,'distinct_1.0.fits'),os.path.join(dir_1_0,'distinct_0.6565.fits'),os.path.join(dir_1_0,'distinct_0.4922.fits'),os.path.join(dir_1_0,'distinct_0.409.fits')])

dir_0_4 = '/data17s/darksim/simulation_3/MD/MD_0.4Gpc/Mass_Xoff_Concentration'
path_2_snapshot_data0_4 = os.path.join(dir_0_4,'distinct_1.0.fits')

zpl = np.array([1/1.0-1, 1/0.6565-1, 1/0.4922-1, 1/0.4123-1])
colors = ['b','r','c','m']

#define arrays used to cut data: low resolution of HMD or low statistic of MDPL 

def modified_sch(x,A,alpha,beta,x0):
    f = (x/x0)**(alpha)*np.exp(-(x/x0)**beta)
    return np.log10(A*f)


def modified_sch_list(data,A,alpha,beta,x0,e0,e1,e2):
    x,sigma = data
    f = (x/x0/sigma**e0)**(alpha*sigma**e1)*np.exp(-(x/x0/sigma**e0)**(beta*sigma**e2))
    return np.log10(A*f)


print('HMD')
aexp = float(os.path.basename(path_2_snapshot_data[0][:-5]).split('_')[1])
z_snap = 1/aexp -1
print('z=%.3g'%(z_snap))
cosmo = cosmology.setCosmology('multidark-planck')    
E_z = cosmo.Ez(z=z_snap)
Vol1 = (4e3/(1+z_snap))**3
dc = peaks.collapseOverdensity(z = z_snap)
rho_m = cosmo.rho_m(z=z_snap)*1e9
h = cosmo.Hz(z=0)/100

hd1 = fits.open(path_2_snapshot_data[0])
mass1=hd1[1].data['Mvir']
logmass1 = np.log10(mass1)
R_1 = peaks.lagrangianR(mass1)
sigf_1 = cosmo.sigma(R_1,z=z_snap)
log1_sigf_1 = np.log10(1/sigf_1)
Rvir1 = hd1[1].data['Rvir']
Rs1 = hd1[1].data['Rs']
xoff = np.log10(hd1[1].data['Xoff'])#/hd1[1].data['Rvir'])
conc1 = Rvir1/Rs1

print('min = ',min(log1_sigf_1))
print('max = ',max(log1_sigf_1))
#sys.exit()

#log1sig_intervals = [-0.15, -0.11,  -0.08, -0.01,  0.07, 0.20, 0.4]
#log1sig_intervals = [-0.1, 0.08,  0.11, 0.14,  0.19, 0.22, 0.44]
#log1sig_intervals = [-0.05, 0.13,  0.16, 0.2,  0.25, 0.28, 0.48]
#log1sig_intervals = [0.1, 0.18,  0.2, 0.23, 0.27, 0.3, 0.52]
#sig_intervals = 1/(10**np.array(log1sig_intervals))
#sig_bins = (sig_intervals[1:] + sig_intervals[:-1])/2
#R_intervals = cosmo.sigma(sig_intervals, z=z_snap,inverse=True)
#M_intervals = peaks.lagrangianM(R_intervals)/h
#print(log1sig_intervals)

#create mass interval and compute corresponding sigma value
#mass_interval = 10**np.array([13.3,13.5,13.6,13.75,13.9,14.0,14.3])
mass_interval = np.array([2e13,3e13,4.5e13,6e13,8e13,1e14,2e14])
R_interval = peaks.lagrangianR(mass_interval)
sigma_interval = cosmo.sigma(R_interval,z=z_snap)
sig_bins = (sigma_interval[1:] + sigma_interval[:-1])/2

interval0 = ((mass1<= mass_interval[1]) & (mass1 > mass_interval[0]))
interval1 = ((mass1<= mass_interval[2]) & (mass1 > mass_interval[1]))
interval2 = ((mass1<= mass_interval[3]) & (mass1 > mass_interval[2]))
interval3 = ((mass1<= mass_interval[4]) & (mass1 > mass_interval[3]))
interval4 = ((mass1<= mass_interval[5]) & (mass1 > mass_interval[4]))
interval5 = ((mass1<= mass_interval[6]) & (mass1 > mass_interval[5]))

#compute histograms of conc
edges_conc = np.linspace(3,11,80)
#edges_conc = np.linspace(3,6,50)
bins_conc = (edges_conc[:-1]+edges_conc[1:])/2
diff_bins_conc=np.diff(edges_conc)

def get_pdf_conc(x):
    pdf = np.histogram(x,bins=edges_conc,density=True)[0]
    pdf = pdf*diff_bins_conc

    N = np.histogram(x,bins=edges_conc)[0]
    err_ = np.sqrt(1/N)*pdf#*3
    err = 1/np.log(10)*err_/pdf
    pdf = np.log10(pdf)
    
    return pdf, err

pdf_conc, yerr_conc = get_pdf_conc(conc1)
pdf_conc0, yerr_conc0 = get_pdf_conc(conc1[interval0])
pdf_conc1, yerr_conc1 = get_pdf_conc(conc1[interval1])
pdf_conc2, yerr_conc2 = get_pdf_conc(conc1[interval2])
pdf_conc3, yerr_conc3 = get_pdf_conc(conc1[interval3])
pdf_conc4, yerr_conc4 = get_pdf_conc(conc1[interval4])
pdf_conc5, yerr_conc5 = get_pdf_conc(conc1[interval5])

popt_conc, pcov_conc = curve_fit(modified_sch,bins_conc,pdf_conc,sigma=yerr_conc,p0=[(1,1.,1,1)],maxfev=1000000)#,maxfev=100000)#,p0=[(1e-3,4.,0.6,1e-3)])   
model_pdf_conc = modified_sch(bins_conc,*popt_conc)

#read the parameters
tab = os.path.join('tables','schechter_HMD_conc_z_%.3g_simult.fit'%(z_snap))
t = Table.read(tab)
popt_pdf_conc_list = t['pars']

#make the figures

fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0],sharex=ax1)
ax4 = fig.add_subplot(gs[1,1],sharex=ax2)


ax1.plot(bins_conc,model_pdf_conc,label='full sample')
ax1.scatter(bins_conc,pdf_conc, ls ='None', marker='o',s=10)
ax1.fill_between(bins_conc,pdf_conc0-yerr_conc0+0.3,pdf_conc0+yerr_conc0+0.3, label = r'$%.3g < M_\odot \leq %.3g$'%(mass_interval[0],mass_interval[1]), alpha=0.5)
ax1.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[0]],*popt_pdf_conc_list)+0.3)
ax1.fill_between(bins_conc,pdf_conc1-yerr_conc1+0.2,pdf_conc1+yerr_conc1+0.2, label = r'$%.3g < M_\odot \leq %.3g$'%(mass_interval[1],mass_interval[2]), alpha=0.5)
ax1.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[1]],*popt_pdf_conc_list)+0.2)
ax1.fill_between(bins_conc,pdf_conc2-yerr_conc2+0.1,pdf_conc2+yerr_conc2+0.1, label = r'$%.3g < M_\odot \leq %.3g$'%(mass_interval[2],mass_interval[3]), alpha=0.5)
ax1.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[2]],*popt_pdf_conc_list)+0.1)
ax1.fill_between(bins_conc,pdf_conc3-yerr_conc3,pdf_conc3+yerr_conc3, label = r'$%.3g < M_\odot \leq %.3g$'%(mass_interval[3],mass_interval[4]), alpha=0.5)
ax1.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[3]],*popt_pdf_conc_list))
ax1.fill_between(bins_conc,pdf_conc4-yerr_conc4-0.1,pdf_conc4+yerr_conc4-0.1, label = r'$%.3g < M_\odot \leq %.3g$'%(mass_interval[4],mass_interval[5]), alpha=0.5)
ax1.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[4]],*popt_pdf_conc_list)-0.1)
ax1.fill_between(bins_conc,pdf_conc5-yerr_conc5-0.2,pdf_conc5+yerr_conc5-0.2, label = r'$%.3g < M_\odot \leq %.3g$'%(mass_interval[5],mass_interval[6]), alpha=0.5)
ax1.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[5]],*popt_pdf_conc_list)-0.2)
ax1.set_ylim(bottom=-2.5,top=-1.2)
ax1.tick_params(labelsize=12)
ax1.grid(True)
ax1.legend(fontsize=10,bbox_to_anchor=(0, 1.1, 2.15, .3), loc='lower left', ncol=3, mode="expand", borderaxespad=0.)
#ax1.set_xlabel(r'$c = R_{vir}/R_s$',fontsize=12)
ax1.set_ylabel(r'$\log_{10}P(c)+C_0$',fontsize=12)
ax1.set_title('z = 0.00', fontsize=12)



############################### z=0.5 ###############################################

aexp = float(os.path.basename(path_2_snapshot_data[1][:-5]).split('_')[1])
z_snap = 1/aexp -1
print('z=%.3g'%(z_snap))
cosmo = cosmology.setCosmology('multidark-planck')    
E_z = cosmo.Ez(z=z_snap)
Vol1 = (4e3/(1+z_snap))**3
dc = peaks.collapseOverdensity(z = z_snap)
rho_m = cosmo.rho_m(z=z_snap)*1e9
h = cosmo.Hz(z=0)/100

hd1 = fits.open(path_2_snapshot_data[1])
mass1=hd1[1].data['Mvir']
Rvir1 = hd1[1].data['Rvir']
Rs1 = hd1[1].data['Rs']
conc1 = Rvir1/Rs1

#create mass interval and compute corresponding sigma value
sigma_interval = cosmo.sigma(R_interval,z=z_snap)
sig_bins = (sigma_interval[1:] + sigma_interval[:-1])/2

interval0 = ((mass1<= mass_interval[1]) & (mass1 > mass_interval[0]))
interval1 = ((mass1<= mass_interval[2]) & (mass1 > mass_interval[1]))
interval2 = ((mass1<= mass_interval[3]) & (mass1 > mass_interval[2]))
interval3 = ((mass1<= mass_interval[4]) & (mass1 > mass_interval[3]))
interval4 = ((mass1<= mass_interval[5]) & (mass1 > mass_interval[4]))
interval5 = ((mass1<= mass_interval[6]) & (mass1 > mass_interval[5]))

pdf_conc, yerr_conc = get_pdf_conc(conc1)
pdf_conc0, yerr_conc0 = get_pdf_conc(conc1[interval0])
pdf_conc1, yerr_conc1 = get_pdf_conc(conc1[interval1])
pdf_conc2, yerr_conc2 = get_pdf_conc(conc1[interval2])
pdf_conc3, yerr_conc3 = get_pdf_conc(conc1[interval3])
pdf_conc4, yerr_conc4 = get_pdf_conc(conc1[interval4])
pdf_conc5, yerr_conc5 = get_pdf_conc(conc1[interval5])

popt_conc, pcov_conc = curve_fit(modified_sch,bins_conc,pdf_conc,sigma=yerr_conc,p0=[(1,1.,1,1)],maxfev=1000000)#,maxfev=100000)#,p0=[(1e-3,4.,0.6,1e-3)])   
model_pdf_conc = modified_sch(bins_conc,*popt_conc)

#read the parameters
tab = os.path.join('tables','schechter_HMD_conc_z_%.3g_simult.fit'%(z_snap))
t = Table.read(tab)
popt_pdf_conc_list = t['pars']

ax2.plot(bins_conc,model_pdf_conc,label='full sample')
ax2.scatter(bins_conc,pdf_conc, ls ='None', marker='o',s=10)
ax2.fill_between(bins_conc,pdf_conc0-yerr_conc0+0.3,pdf_conc0+yerr_conc0+0.3, alpha=0.5)
ax2.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[0]],*popt_pdf_conc_list)+0.3)
ax2.fill_between(bins_conc,pdf_conc1-yerr_conc1+0.2,pdf_conc1+yerr_conc1+0.2, alpha=0.5)
ax2.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[1]],*popt_pdf_conc_list)+0.2)
ax2.fill_between(bins_conc,pdf_conc2-yerr_conc2+0.1,pdf_conc2+yerr_conc2+0.1, alpha=0.5)
ax2.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[2]],*popt_pdf_conc_list)+0.1)
ax2.fill_between(bins_conc,pdf_conc3-yerr_conc3,pdf_conc3+yerr_conc3, alpha=0.5)
ax2.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[3]],*popt_pdf_conc_list))
ax2.fill_between(bins_conc,pdf_conc4-yerr_conc4-0.1,pdf_conc4+yerr_conc4-0.1, alpha=0.5)
ax2.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[4]],*popt_pdf_conc_list)-0.1)
ax2.fill_between(bins_conc,pdf_conc5-yerr_conc5-0.2,pdf_conc5+yerr_conc5-0.2,alpha=0.5)
ax2.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[5]],*popt_pdf_conc_list)-0.2)
ax2.set_ylim(bottom=-2.5,top=-1.2)
ax2.tick_params(labelsize=12)
ax2.grid(True)
#ax2.set_xlabel(r'$c = R_{vir}/R_s$',fontsize=12)
#ax2.set_ylabel(r'$\log_{10}P(c)+C_0$',fontsize=12)
ax2.set_title('z = 0.52', fontsize=12)


############################### z=1.03 ###############################################

aexp = float(os.path.basename(path_2_snapshot_data[2][:-5]).split('_')[1])
z_snap = 1/aexp -1
print('z=%.3g'%(z_snap))
cosmo = cosmology.setCosmology('multidark-planck')    
E_z = cosmo.Ez(z=z_snap)
Vol1 = (4e3/(1+z_snap))**3
dc = peaks.collapseOverdensity(z = z_snap)
rho_m = cosmo.rho_m(z=z_snap)*1e9
h = cosmo.Hz(z=0)/100

hd1 = fits.open(path_2_snapshot_data[2])
mass1=hd1[1].data['Mvir']
Rvir1 = hd1[1].data['Rvir']
Rs1 = hd1[1].data['Rs']
conc1 = Rvir1/Rs1

#create mass interval and compute corresponding sigma value
sigma_interval = cosmo.sigma(R_interval,z=z_snap)
sig_bins = (sigma_interval[1:] + sigma_interval[:-1])/2

interval0 = ((mass1<= mass_interval[1]) & (mass1 > mass_interval[0]))
interval1 = ((mass1<= mass_interval[2]) & (mass1 > mass_interval[1]))
interval2 = ((mass1<= mass_interval[3]) & (mass1 > mass_interval[2]))
interval3 = ((mass1<= mass_interval[4]) & (mass1 > mass_interval[3]))
interval4 = ((mass1<= mass_interval[5]) & (mass1 > mass_interval[4]))
interval5 = ((mass1<= mass_interval[6]) & (mass1 > mass_interval[5]))

pdf_conc, yerr_conc = get_pdf_conc(conc1)
pdf_conc0, yerr_conc0 = get_pdf_conc(conc1[interval0])
pdf_conc1, yerr_conc1 = get_pdf_conc(conc1[interval1])
pdf_conc2, yerr_conc2 = get_pdf_conc(conc1[interval2])
pdf_conc3, yerr_conc3 = get_pdf_conc(conc1[interval3])
pdf_conc4, yerr_conc4 = get_pdf_conc(conc1[interval4])
pdf_conc5, yerr_conc5 = get_pdf_conc(conc1[interval5])

popt_conc, pcov_conc = curve_fit(modified_sch,bins_conc,pdf_conc,sigma=yerr_conc,p0=[(1,1.,1,1)],maxfev=1000000)#,maxfev=100000)#,p0=[(1e-3,4.,0.6,1e-3)])   
model_pdf_conc = modified_sch(bins_conc,*popt_conc)

#read the parameters
tab = os.path.join('tables','schechter_HMD_conc_z_%.3g_simult.fit'%(z_snap))
t = Table.read(tab)
popt_pdf_conc_list = t['pars']

ax3.plot(bins_conc,model_pdf_conc,label='full sample')
ax3.scatter(bins_conc,pdf_conc, ls ='None', marker='o',s=10)
ax3.fill_between(bins_conc,pdf_conc0-yerr_conc0+0.3,pdf_conc0+yerr_conc0+0.3, alpha=0.5)
ax3.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[0]],*popt_pdf_conc_list)+0.3)
ax3.fill_between(bins_conc,pdf_conc1-yerr_conc1+0.2,pdf_conc1+yerr_conc1+0.2, alpha=0.5)
ax3.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[1]],*popt_pdf_conc_list)+0.2)
ax3.fill_between(bins_conc,pdf_conc2-yerr_conc2+0.1,pdf_conc2+yerr_conc2+0.1, alpha=0.5)
ax3.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[2]],*popt_pdf_conc_list)+0.1)
ax3.fill_between(bins_conc,pdf_conc3-yerr_conc3,pdf_conc3+yerr_conc3, alpha=0.5)
ax3.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[3]],*popt_pdf_conc_list))
ax3.fill_between(bins_conc,pdf_conc4-yerr_conc4-0.1,pdf_conc4+yerr_conc4-0.1, alpha=0.5)
ax3.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[4]],*popt_pdf_conc_list)-0.1)
ax3.fill_between(bins_conc,pdf_conc5-yerr_conc5-0.2,pdf_conc5+yerr_conc5-0.2,alpha=0.5)
ax3.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[5]],*popt_pdf_conc_list)-0.2)
ax3.set_ylim(bottom=-2.5,top=-1.2)
ax3.tick_params(labelsize=12)
ax3.grid(True)
ax3.set_xlabel(r'$c = R_{vir}/R_s$',fontsize=12)
ax3.set_ylabel(r'$\log_{10}P(c)+C_0$',fontsize=12)
ax3.set_title('z = 1.03', fontsize=12)



############################### z=1.43 ###############################################

aexp = float(os.path.basename(path_2_snapshot_data[3][:-5]).split('_')[1])
z_snap = 1/aexp -1
print('z=%.3g'%(z_snap))
cosmo = cosmology.setCosmology('multidark-planck')    
E_z = cosmo.Ez(z=z_snap)
Vol1 = (4e3/(1+z_snap))**3
dc = peaks.collapseOverdensity(z = z_snap)
rho_m = cosmo.rho_m(z=z_snap)*1e9
h = cosmo.Hz(z=0)/100

hd1 = fits.open(path_2_snapshot_data[3])
mass1=hd1[1].data['Mvir']
Rvir1 = hd1[1].data['Rvir']
Rs1 = hd1[1].data['Rs']
conc1 = Rvir1/Rs1

#create mass interval and compute corresponding sigma value
sigma_interval = cosmo.sigma(R_interval,z=z_snap)
sig_bins = (sigma_interval[1:] + sigma_interval[:-1])/2

interval0 = ((mass1<= mass_interval[1]) & (mass1 > mass_interval[0]))
interval1 = ((mass1<= mass_interval[2]) & (mass1 > mass_interval[1]))
interval2 = ((mass1<= mass_interval[3]) & (mass1 > mass_interval[2]))
interval3 = ((mass1<= mass_interval[4]) & (mass1 > mass_interval[3]))
interval4 = ((mass1<= mass_interval[5]) & (mass1 > mass_interval[4]))
interval5 = ((mass1<= mass_interval[6]) & (mass1 > mass_interval[5]))

pdf_conc, yerr_conc = get_pdf_conc(conc1)
pdf_conc0, yerr_conc0 = get_pdf_conc(conc1[interval0])
pdf_conc1, yerr_conc1 = get_pdf_conc(conc1[interval1])
pdf_conc2, yerr_conc2 = get_pdf_conc(conc1[interval2])
pdf_conc3, yerr_conc3 = get_pdf_conc(conc1[interval3])
pdf_conc4, yerr_conc4 = get_pdf_conc(conc1[interval4])
pdf_conc5, yerr_conc5 = get_pdf_conc(conc1[interval5])

popt_conc, pcov_conc = curve_fit(modified_sch,bins_conc,pdf_conc,sigma=yerr_conc,p0=[(1,1.,1,1)],maxfev=1000000)#,maxfev=100000)#,p0=[(1e-3,4.,0.6,1e-3)])   
model_pdf_conc = modified_sch(bins_conc,*popt_conc)

#read the parameters
tab = os.path.join('tables','schechter_HMD_conc_z_%.3g_simult.fit'%(z_snap))
t = Table.read(tab)
popt_pdf_conc_list = t['pars']

ax4.plot(bins_conc,model_pdf_conc,label='full sample')
ax4.scatter(bins_conc,pdf_conc, ls ='None', marker='o',s=10)
ax4.fill_between(bins_conc,pdf_conc0-yerr_conc0+0.3,pdf_conc0+yerr_conc0+0.3, alpha=0.5)
ax4.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[0]],*popt_pdf_conc_list)+0.3)
ax4.fill_between(bins_conc,pdf_conc1-yerr_conc1+0.2,pdf_conc1+yerr_conc1+0.2, alpha=0.5)
ax4.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[1]],*popt_pdf_conc_list)+0.2)
ax4.fill_between(bins_conc,pdf_conc2-yerr_conc2+0.1,pdf_conc2+yerr_conc2+0.1, alpha=0.5)
ax4.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[2]],*popt_pdf_conc_list)+0.1)
ax4.fill_between(bins_conc,pdf_conc3-yerr_conc3,pdf_conc3+yerr_conc3, alpha=0.5)
ax4.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[3]],*popt_pdf_conc_list))
ax4.fill_between(bins_conc,pdf_conc4-yerr_conc4-0.1,pdf_conc4+yerr_conc4-0.1, alpha=0.5)
ax4.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[4]],*popt_pdf_conc_list)-0.1)
ax4.fill_between(bins_conc,pdf_conc5-yerr_conc5-0.2,pdf_conc5+yerr_conc5-0.2, alpha=0.5)
ax4.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[5]],*popt_pdf_conc_list)-0.2)
ax4.set_ylim(bottom=-2.5,top=-1.2)
ax4.tick_params(labelsize=12)
ax4.grid(True)
ax4.set_xlabel(r'$c = R_{vir}/R_s$',fontsize=12)
ax4.set_title('z = 1.43', fontsize=12)





plt.tight_layout()
outpl = os.path.join(this_dir,'figures','pdf_conc_HMD_mass_slices_z_all.png')
os.makedirs(os.path.dirname(outpl), exist_ok=True)
plt.savefig(outpl, overwrite=True)

print('done!')
sys.exit()

