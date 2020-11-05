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
path_2_snapshot_data = np.array([os.path.join(test_dir, 'distinct_1.0.fits'),os.path.join(test_dir,'distinct_0.6565.fits'),os.path.join(test_dir,'distinct_0.4922.fits'),os.path.join(test_dir,'distinct_0.4123.fits')])

dir_2_5 = '/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration'

path_2_snapshot_data2_5 = np.array([os.path.join(dir_2_5,'distinct_1.0.fits'),os.path.join(dir_2_5,'distinct_0.6583.fits'),os.path.join(dir_2_5,'distinct_0.5.fits'),os.path.join(dir_2_5,'distinct_0.409.fits')])
dir_1_0 = '/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration'

path_2_snapshot_data1_0 = np.array([os.path.join(dir_1_0,'distinct_1.0.fits'),os.path.join(dir_1_0,'distinct_0.6565.fits'),os.path.join(dir_1_0,'distinct_0.4922.fits'),os.path.join(dir_1_0,'distinct_0.409.fits')])

dir_0_4 = '/data17s/darksim/simulation_3/MD/MD_0.4Gpc/Mass_Xoff_Concentration'
path_2_snapshot_data0_4 = os.path.join(dir_0_4,'distinct_1.0.fits')
fig,ax = plt.subplots(1,2,figsize=(20,10))
fig1,ax1 = plt.subplots(1,2,figsize=(20,10))
zpl = np.array([1/1.0-1, 1/0.6565-1, 1/0.4922-1, 1/0.4123-1])
colors = ['b','r','c','m']

#define arrays used to cut data: low resolution of HMD or low statistic of MDPL 


def lognormal(x, x0, sigma):
    return np.log10(1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(np.log10(x/x0))**2/(2*sigma**2)))

def lognormallog(x, x0, sigma):
    x=10**x
    return np.log10(1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(np.log10(x/x0))**2/(2*sigma**2)))

def lognormalA(x, A, x0, sigma):
    return A*np.exp(-(np.log10(x/x0))**2/(2*sigma**2))

def modified_sch(x,A,alpha,beta,x0):
    f = (x/x0)**(alpha)*np.exp(-(x/x0)**beta)
    return np.log10(A*f)

def modified_sch_list(data,A,alpha,beta,x0,e0,e1,e2):
    x,sigma = data
    f = (x/x0/sigma**e0)**(alpha*sigma**e1)*np.exp(-(x/x0/sigma**e0)**(beta*sigma**e2))
    return np.log10(A*f)

def modified_sch_log0(x,A,alpha,beta,x0):
    f = (10**(x/x0))**(alpha)*np.exp(-(10**(x/x0))**beta)
    return A+np.log10(f)

def modified_sch_log0_list(data,A,alpha,beta,x0,e0,e1,e2):
    x,sigma = data
    f = (10**(x/x0/sigma**e0))**(alpha*sigma**e1)*np.exp(-(10**(x/x0/sigma**e0))**(beta*sigma**e2))
    return A+np.log10(f)


def double_schechter(data,A,alpha0,beta0,x0,alpha1,beta1,y0,e0,e1,e2,e3,e4,e5,x1,x2,x3,y1,y2,y3):
    x,y=data
    x_=10**x
    y_=10**y
    #f = (10**(x/x0))**(alpha0)*np.exp(-(10**(x/x0))**(beta0))*(10**(y/y0))**(alpha1)*np.exp(-(10**(y/y0))**(beta1))
    #f = (10**(x/x0))**(alpha0)*np.exp(-(10**(x/x0))**(beta0))*(10**(y/y0))**(alpha1)*np.exp(-(10**(y/y0))**(beta1))
    f = (10**(x/x0*(y_/y1)**e0))**(alpha0*(y_/y2)**e1)*np.exp(-(10**(x/x0/(y_/y1)**e0))**(beta0*(y_/y3)**e2))*(10**(y/y0*(x_/x1)**e3))**(alpha1*(x_/x2)**e4)*np.exp(-(10**(y/y0*(x_/x1)**e3))**(beta1*(x_/x3)**e5))
    #f = (10**(x/x0))**(alpha0)*np.exp(-(10**(x/x0))**(beta0))*(10**(y/y0))**(alpha1)*np.exp(-(10**(y/y0))**(beta1))*(10**(x*y/e0))**(e1)*np.exp(-(10**(x*y/e2))**(e3))
    return A+np.log10(f)


print('HMD')
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
logmass1 = np.log10(mass1)
R_1 = peaks.lagrangianR(mass1)
sigf_1 = cosmo.sigma(R_1,z=z_snap)
log1_sigf_1 = np.log10(1/sigf_1)
Rvir1 = hd1[1].data['Rvir']
Rs1 = hd1[1].data['Rs']
xoff = np.log10(hd1[1].data['Xoff'])#/hd1[1].data['Rvir'])
#spin1 = hd1[1].data['Spin']
spinpar1 = hd1[1].data['Spin']
conc1 = Rvir1/Rs1

print('min = ',min(log1_sigf_1))
print('max = ',max(log1_sigf_1))
#sys.exit()
print('computing spinparameter pdf...')

#log1sig_intervals = [-0.15, -0.11,  -0.08, -0.01,  0.07, 0.20, 0.4]
#log1sig_intervals = [-0.1, 0.08,  0.11, 0.14,  0.19, 0.22, 0.44]
#log1sig_intervals = [-0.05, 0.13,  0.16, 0.2,  0.25, 0.28, 0.48]
log1sig_intervals = [0.1, 0.18,  0.2, 0.23, 0.27, 0.3, 0.52]
sig_intervals = 1/(10**np.array(log1sig_intervals))
sig_bins = (sig_intervals[1:] + sig_intervals[:-1])/2
R_intervals = cosmo.sigma(sig_intervals, z=z_snap,inverse=True)
M_intervals = peaks.lagrangianM(R_intervals)/h
print(log1sig_intervals)
interval0 = ((log1_sigf_1<= log1sig_intervals[1]))
interval1 = ((log1_sigf_1<= log1sig_intervals[2]) & (log1_sigf_1 > log1sig_intervals[1]))
interval2 = ((log1_sigf_1<= log1sig_intervals[3]) & (log1_sigf_1 > log1sig_intervals[2]))
interval3 = ((log1_sigf_1<= log1sig_intervals[4]) & (log1_sigf_1 > log1sig_intervals[3]))
interval4 = ((log1_sigf_1<= log1sig_intervals[5]) & (log1_sigf_1 > log1sig_intervals[4]))
interval5 = ((log1_sigf_1 > log1sig_intervals[5]))

edges_conc = np.linspace(3,11,80)
#edges_conc = np.linspace(3,6,50)
bins_conc = (edges_conc[:-1]+edges_conc[1:])/2
diff_bins_conc=np.diff(edges_conc)

def get_pdf_conc(x):
    pdf = np.histogram(x,bins=edges_conc,density=True)[0]
    pdf = pdf*diff_bins_conc

    N = np.histogram(x,bins=edges_conc)[0]
    err_ = np.sqrt(1/N)*pdf*3
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


index_conc0 = (pdf_conc0 > -4)
index_conc1 = (pdf_conc1 > -4)
index_conc2 = (pdf_conc2 > -4)
index_conc3 = (pdf_conc3 > -4)
index_conc4 = (pdf_conc4 > -4)
index_conc5 = (pdf_conc5 > -4)



#simulatenous fit of conc
bins_conc_list = np.hstack((bins_conc[index_conc0],bins_conc[index_conc1],bins_conc[index_conc2],bins_conc[index_conc3],bins_conc[index_conc4],bins_conc[index_conc5]))
pdf_conc_list = np.hstack((pdf_conc0[index_conc0],pdf_conc1[index_conc1],pdf_conc2[index_conc2],pdf_conc3[index_conc3],pdf_conc4[index_conc4],pdf_conc5[index_conc5]))
yerr_conc_list = np.hstack((yerr_conc0[index_conc0],yerr_conc1[index_conc1],yerr_conc2[index_conc2],yerr_conc3[index_conc3],yerr_conc4[index_conc4],yerr_conc5[index_conc5]))
print('bins conc list = ',bins_conc_list)
print('pdf conc list = ', pdf_conc_list)
print('yerr conc list = ', yerr_conc_list)
sigma0c = np.repeat(sig_bins[0],len(pdf_conc0[index_conc0]))
sigma1c = np.repeat(sig_bins[1],len(pdf_conc1[index_conc1]))
sigma2c = np.repeat(sig_bins[2],len(pdf_conc2[index_conc2]))
sigma3c = np.repeat(sig_bins[3],len(pdf_conc3[index_conc3]))
sigma4c = np.repeat(sig_bins[4],len(pdf_conc4[index_conc4]))
sigma5c = np.repeat(sig_bins[5],len(pdf_conc5[index_conc5]))
sigma_listc = np.hstack((sigma0c,sigma1c,sigma2c,sigma3c,sigma4c,sigma5c))

xdatac = np.vstack((bins_conc_list,sigma_listc))
popt_pdf_conc_list,pcov_pdf_conc_list = curve_fit(modified_sch_list,xdatac,pdf_conc_list,sigma=yerr_conc_list,maxfev=1000000,p0=[(0.05, 1.7, 2.1, 6.6,0,0,0)])

chi2concl = np.sum((pdf_conc_list-modified_sch_list(xdatac,*popt_pdf_conc_list))**2/(yerr_conc_list)**2)
dofconcl = len(pdf_conc_list)-7
print('chi2 xoff l = ',chi2concl)
print('dof xoff l = ',dofconcl)
chi2rconcl=chi2concl/dofconcl
print('chi2r conc l = ',chi2rconcl)
print('parameters from simultaneous fitting = ',popt_pdf_conc_list)

tab_sch_conc_list = Table()
tab_sch_conc_list.add_column(Column(name='pars',data=popt_pdf_conc_list,unit=''))
tab_sch_conc_list.add_column(Column(name='err',data=np.diag(pcov_pdf_conc_list),unit=''))
outschc_list = os.path.join(this_dir,'tables','schechter_HMD_conc_z_%.3g_simult.fit'%(z_snap))
tab_sch_conc_list.write(outschc_list,overwrite=True)

#make the figures

plt.figure(figsize=(16,8))
plt.plot(bins_conc,model_pdf_conc,label='mod sch - full sample')
#plt.scatter(bins_conc,pdf_conc, label = r'HMD', ls ='None', marker='o')
plt.fill_between(bins_conc,pdf_conc - yerr_conc,pdf_conc + yerr_conc, alpha=0.5, label = r'HMD')
plt.tick_params(labelsize=20)
plt.grid(True)
plt.legend(fontsize=15)
plt.xlabel(r'$c = R_{vir}/R_s$',fontsize=25)
plt.ylabel(r'$P(c)$',fontsize=25)
plt.title('z = %.3g'%(z_snap), fontsize=25)
plt.tight_layout()
outpl = os.path.join(this_dir,'figures','pdf_conc_HMD_z_%.3g.png'%(z_snap))
os.makedirs(os.path.dirname(outpl), exist_ok=True)
plt.savefig(outpl, overwrite=True)

plt.figure(figsize=(16,8))
plt.plot(bins_conc,model_pdf_conc,label='mod sch - full sample')
plt.scatter(bins_conc,pdf_conc, label = r'HMD', ls ='None', marker='o')
plt.fill_between(bins_conc,pdf_conc0-yerr_conc0+0.3,pdf_conc0+yerr_conc0+0.3, label = r'$M_\odot \leq %.3g$'%(M_intervals[1]), alpha=0.5)
plt.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[0]],*popt_pdf_conc_list)+0.3)
plt.fill_between(bins_conc,pdf_conc1-yerr_conc1+0.2,pdf_conc1+yerr_conc1+0.2, label = r'$%.3g < M_\odot \leq %.3g$'%(M_intervals[1],M_intervals[2]), alpha=0.5)
plt.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[1]],*popt_pdf_conc_list)+0.2)
plt.fill_between(bins_conc,pdf_conc2-yerr_conc2+0.1,pdf_conc2+yerr_conc2+0.1, label = r'$%.3g < M_\odot \leq %.3g$'%(M_intervals[2],M_intervals[3]), alpha=0.5)
plt.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[2]],*popt_pdf_conc_list)+0.1)
plt.fill_between(bins_conc,pdf_conc3-yerr_conc3,pdf_conc3+yerr_conc3, label = r'$%.3g < M_\odot \leq %.3g$'%(M_intervals[3],M_intervals[4]), alpha=0.5)
plt.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[3]],*popt_pdf_conc_list))
plt.fill_between(bins_conc,pdf_conc4-yerr_conc4-0.1,pdf_conc4+yerr_conc4-0.1, label = r'$%.3g < M_\odot \leq %.3g$'%(M_intervals[4],M_intervals[5]), alpha=0.5)
plt.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[4]],*popt_pdf_conc_list)-0.1)
plt.fill_between(bins_conc,pdf_conc5-yerr_conc5-0.2,pdf_conc5+yerr_conc5-0.2, label = r'$M_\odot > %.3g$'%(M_intervals[5]),alpha=0.5)
plt.plot(bins_conc,modified_sch_list([bins_conc,sig_bins[5]],*popt_pdf_conc_list)-0.2)
plt.ylim(bottom=-2.6,top=-1.2)
plt.tick_params(labelsize=20)
plt.grid(True)
plt.legend(fontsize=15)
plt.xlabel(r'$c = R_{vir}/R_s$',fontsize=25)
plt.ylabel(r'$\log_{10}P(c)+C_0$',fontsize=25)
plt.title('z = 1.43', fontsize=25)
plt.tight_layout()
outpl = os.path.join(this_dir,'figures','pdf_conc_HMD_slices_z_%.3g.png'%(z_snap))
os.makedirs(os.path.dirname(outpl), exist_ok=True)
plt.savefig(outpl, overwrite=True)

print('done!')
sys.exit()

