"""
Shows relaxed and unrelaxed mass function
"""

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
from matplotlib import cm
import matplotlib.gridspec as gridspec


print('Analyze Mass Function of relaxed and unrelaxed halos')
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
path_2_snapshot_data = np.array([os.path.join(test_dir, 'distinct_1.0.fits.gz'),os.path.join(test_dir,'distinct_0.6565.fits.gz'),os.path.join(test_dir,'distinct_0.4922.fits.gz'),os.path.join(test_dir,'distinct_0.353.fits.gz')])

dir_2_5 = '/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration'

path_2_snapshot_data2_5 = np.array([os.path.join(dir_2_5,'distinct_1.0.fits'),os.path.join(dir_2_5,'distinct_0.6583.fits'),os.path.join(dir_2_5,'distinct_0.5.fits'),os.path.join(dir_2_5,'distinct_0.318.fits')])
dir_1_0 = '/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration'

path_2_snapshot_data1_0 = np.array([os.path.join(dir_1_0,'distinct_1.0.fits'),os.path.join(dir_1_0,'distinct_0.6565.fits'),os.path.join(dir_1_0,'distinct_0.4922.fits'),os.path.join(dir_1_0,'distinct_0.353.fits')])

dir_0_4 = '/data17s/darksim/simulation_3/MD/MD_0.4Gpc/Mass_Xoff_Concentration'
path_2_snapshot_data0_4 = os.path.join(dir_0_4,'distinct_1.0.fits')
zpl = np.array([1/1.0-1, 1/0.6565-1, 1/0.4922-1, 1/0.353-1])
colors = ['b','r','c','m']
print(colors[0])

aexp = float(os.path.basename(path_2_snapshot_data[0][:-8]).split('_')[1])
z_snap = 1/aexp -1
print('z=%.3g'%(z_snap))
cosmo = cosmology.setCosmology('multidark-planck')    
h = cosmo.Hz(0)/100
E_z = cosmo.Ez(z=z_snap)
Vol1 = (4e3/(1+z_snap))**3
dc = peaks.collapseOverdensity(z = z_snap)
rho_m = cosmo.rho_m(z=z_snap)*1e9
R=cosmo.sigma(1/10**(-0.1),z=0,inverse=True)
M=peaks.lagrangianM(R)
print(M/h)
R=cosmo.sigma(1/10**(0.19),z=0,inverse=True)
M=peaks.lagrangianM(R)
print(M/h)
#sys.exit()


print('1e15 Msun is log1_sigma = ',np.log10(1/cosmo.sigma(peaks.lagrangianR(1e15*h),z=z_snap)))
print('1e14 Msun is log1_sigma = ',np.log10(1/cosmo.sigma(peaks.lagrangianR(1e14*h),z=z_snap)))
print('1e13 Msun is log1_sigma = ',np.log10(1/cosmo.sigma(peaks.lagrangianR(1e13*h),z=z_snap)))

print('reading data...')
hd1 = fits.open(path_2_snapshot_data[0])
mass1=hd1[1].data['Mmvir_all']
logmass1 = np.log10(mass1)
R_1 = peaks.lagrangianR(mass1)
sigf_1 = cosmo.sigma(R_1,z=z_snap)
log1_sigf_1 = np.log10(1/sigf_1)
Rvir1 = hd1[1].data['Rvir']
Rs1 = hd1[1].data['Rs']
xoff_data1 = (hd1[1].data['Xoff'])#/hd1[1].data['Rvir'])
#spin1 = hd1[1].data['Spin']
spinpar1 = hd1[1].data['Spin']

xoff_split = 100
spinpar_split = 0.07
relax = ((xoff_data1<=xoff_split) & (spinpar1<=spinpar_split))
unrelax = ~relax

print('plotting xoff spin plane...')
xoff_rel = xoff_data1[relax]
xoff_unrel = xoff_data1[unrelax]
spinpar_rel = spinpar1[relax]
spinpar_unrel = spinpar1[unrelax]

edge_xoff = np.logspace(0,3,51)
edge_spin = np.logspace(-4,0,51)
bins = [edge_xoff,edge_spin]
counts = np.histogram2d(xoff_data1,spinpar1,bins=bins)[0]
counts = np.transpose(counts)
bins_xoff = (edge_xoff[1:]+edge_xoff[:-1])/2
bins_spin =(edge_spin[1:]+edge_spin[:-1])/2
x,y = np.meshgrid(bins_xoff,bins_spin)
'''
plt.figure(figsize=(4.5,4.5))
plt.scatter(xoff_rel,spinpar_rel, ls='None',label='relaxed',c='C0',marker='.',s=5)
plt.scatter(xoff_unrel,spinpar_unrel, ls='None',label='unrelaxed',c='C1',marker='.',s=5)
levels = [0.01*np.max(counts),0.05*np.max(counts),0.1*np.max(counts),0.3*np.max(counts),0.5*np.max(counts),np.max(counts)]
plt.contour(x,y,counts, levels = levels)
plt.axvline(xoff_split, ls='--',c='g')
plt.axhline(spinpar_split, ls='--',c='g')
plt.xlabel(r'X$_{\rm off,P}$ [kpc/h]',fontsize=12)
plt.ylabel(r'$\lambda$',fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.ylim(bottom=3e-4)
plt.tick_params(labelsize=12)
#plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
outf=os.path.join(this_dir,'figures','Xoff_lambdaP_physical.png')
plt.savefig(outf,overwrite=True)
#sys.exit()
'''

print('now working on mass function...')

log1_sigf_1_relax = log1_sigf_1[relax]
log1_sigf_1_unrelax = log1_sigf_1[unrelax]
s_bins1 = np.arange(-0.08,0.5,1e-2)
log1sig1_bin = (s_bins1[:-1]+s_bins1[1:])/2
Runo = cosmo.sigma(1/10**log1sig1_bin,inverse=True,z=z_snap)
M1 = peaks.lagrangianM(Runo)
Runo_ = cosmo.sigma(1/10**s_bins1,inverse=True,z=z_snap)
M1_ = peaks.lagrangianM(Runo_)
diff1 = np.diff(np.log(M1_))

counts_tot = np.histogram(log1_sigf_1,bins=s_bins1)[0]
counts_rel = np.histogram(log1_sigf_1_relax,bins=s_bins1)[0]
counts_unrel = np.histogram(log1_sigf_1_unrelax,bins=s_bins1)[0]

dn_dlnM1_tot = counts_tot/Vol1/diff1
dn_dlnM1_rel = counts_rel/Vol1/diff1
dn_dlnM1_unrel = counts_unrel/Vol1/diff1
fsigma_tot = dn_dlnM1_tot*M1/rho_m/cosmo.sigma(Runo,z=z_snap,derivative=True)*(-3.0)
fsigma_rel = dn_dlnM1_rel*M1/rho_m/cosmo.sigma(Runo,z=z_snap,derivative=True)*(-3.0)
fsigma_unrel = dn_dlnM1_unrel*M1/rho_m/cosmo.sigma(Runo,z=z_snap,derivative=True)*(-3.0)

ftoterr_ = np.sqrt(1/counts_tot + 0.02**2)*(fsigma_tot)
f_tot_err = 1/np.log(10)*ftoterr_/fsigma_tot

frelerr_ = np.sqrt(1/counts_rel + 0.02**2)*(fsigma_rel)
f_rel_err = 1/np.log(10)*frelerr_/fsigma_rel

funrelerr_ = np.sqrt(1/counts_unrel + 0.02**2)*(fsigma_unrel)
f_unrel_err = 1/np.log(10)*funrelerr_/fsigma_unrel

mf_comparat=mf.massFunction(1/(10**log1sig1_bin),q_in='sigma', z=z_snap, mdef = 'vir', model = 'comparat17', q_out = 'f') 

ratio_rel = (fsigma_rel-mf_comparat)/mf_comparat
ratio_unrel = (fsigma_unrel-mf_comparat)/mf_comparat
ratio_comp = (fsigma_tot-mf_comparat)/mf_comparat

def Mass_sigma(x):
    r=cosmo.sigma(1/10**x,z=z_snap,inverse=True)
    M=peaks.lagrangianM(r)#/h
    return np.log10(M)

print('plotting...')
fig = plt.figure(figsize=(4.5,4.5))
gs = fig.add_gridspec(3,1,hspace=0.0)
#gss = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs,hspace=0.0)
ax1 = fig.add_subplot(gs[0:2, :])
ax2 = fig.add_subplot(gs[2, :], sharex = ax1)
ax1.fill_between(log1sig1_bin,np.log10(fsigma_rel)-f_rel_err, np.log10(fsigma_rel)+f_rel_err,label='relaxed',alpha=1.0,color='C0')
ax1.fill_between(log1sig1_bin,np.log10(fsigma_unrel)-f_unrel_err, np.log10(fsigma_unrel)+f_unrel_err,label='unrelaxed',alpha=1.0,color='C1')
ax1.plot(log1sig1_bin,np.log10(mf_comparat),label='comparat17',linewidth=2,c='C3')
ax1.fill_between(log1sig1_bin,np.log10(fsigma_tot)-f_tot_err, np.log10(fsigma_tot)+f_tot_err,label='full sample',alpha=1.0,color='C2')

ax2.plot(log1sig1_bin,ratio_rel,linewidth=3)
ax2.plot(log1sig1_bin,ratio_unrel,linewidth=3)
ax2.plot(log1sig1_bin,ratio_comp,linewidth=3)

#ax2.axvline(np.log10(1/cosmo.sigma(peaks.lagrangianR(3e13*h),z=z_snap)),color='r',label=r'$3e13 M_\odot$')
#ax2.axvline(np.log10(1/cosmo.sigma(peaks.lagrangianR(1e14*h),z=z_snap)),color='g',label=r'$1e14 M_\odot$')
#ax2.axvline(np.log10(1/cosmo.sigma(peaks.lagrangianR(1e15*h),z=z_snap)),color='y',label=r'$1e15 M_\odot$')

ax2.set_ylim(-1.0,0.3)
ax2.set_xlabel(r'$\log_{10}\sigma^{-1}$', fontsize=12)
ax1.set_ylabel(r'$\log_{10}$f($\sigma$)', fontsize=12)
ax2.set_ylabel(r'$\Delta$f/f',fontsize=12)
ax1.set_ylim(bottom=-5)
ax1_sec = ax1.twiny()
xmin,xmax=ax1.get_xlim()
ax1_sec.set_xlim((Mass_sigma(xmin),Mass_sigma(xmax)))
ax1_sec.plot([],[])
ax1_sec.set_xlabel(r'$\log_{10}$M [M$_{\odot}$/h]',labelpad=15,fontsize=12)
ax1.set_xticks([-0.1,0.0,0.1,0.2,0.3,0.4,0.5])
ax1.tick_params(labelsize=12)
ax1_sec.tick_params(labelsize=12)
ax2.tick_params(labelsize=12)
#ax1.grid(True)
#ax2.grid(True)
plt.setp(ax1.get_xticklabels(), visible=False)
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
plt.subplots_adjust(hspace=.0)
plt.tight_layout()
ax1.legend(fontsize=10)
#ax2.legend(fontsize=15)
outfig = os.path.join(this_dir,'figures','MF_comparison_physical.png')
fig.savefig(outfig,overwrite=True)



print('plotting...')
fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(3,1,hspace=0.0)
#gss = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs,hspace=0.0)
ax1 = fig.add_subplot(gs[0:2, :])
ax2 = fig.add_subplot(gs[2, :], sharex = ax1)
ax1.fill_between(log1sig1_bin,np.log10(fsigma_rel)-f_rel_err, np.log10(fsigma_rel)+f_rel_err,label='relaxed',alpha=1.0,color='C0')
ax1.fill_between(log1sig1_bin,np.log10(fsigma_unrel)-f_unrel_err, np.log10(fsigma_unrel)+f_unrel_err,label='unrelaxed',alpha=1.0,color='C1')
ax1.plot(log1sig1_bin,np.log10(mf_comparat),label='comparat17',linewidth=4,c='C3')
ax1.fill_between(log1sig1_bin,np.log10(fsigma_tot)-f_tot_err, np.log10(fsigma_tot)+f_tot_err,label='full sample',alpha=1.0,color='C2')

ax2.plot(log1sig1_bin,ratio_rel,linewidth=5)
ax2.plot(log1sig1_bin,ratio_unrel,linewidth=5)
ax2.plot(log1sig1_bin,ratio_comp,linewidth=5)

#ax2.axvline(np.log10(1/cosmo.sigma(peaks.lagrangianR(3e13*h),z=z_snap)),color='r',label=r'$3e13 M_\odot$')
#ax2.axvline(np.log10(1/cosmo.sigma(peaks.lagrangianR(1e14*h),z=z_snap)),color='g',label=r'$1e14 M_\odot$')
#ax2.axvline(np.log10(1/cosmo.sigma(peaks.lagrangianR(1e15*h),z=z_snap)),color='y',label=r'$1e15 M_\odot$')

ax2.set_ylim(-1.0,0.3)
ax2.set_xlabel(r'$\log_{10}\sigma^{-1}$', fontsize=25)
ax1.set_ylabel(r'$\log_{10}$f($\sigma$)', fontsize=25)
ax2.set_ylabel(r'$\Delta$f/f',fontsize=25)
ax1.set_ylim(bottom=-5)
ax1_sec = ax1.twiny()
xmin,xmax=ax1.get_xlim()
ax1_sec.set_xlim((Mass_sigma(xmin),Mass_sigma(xmax)))
ax1_sec.plot([],[])
ax1_sec.set_xlabel(r'$\log_{10}$M [M$_{\odot}$/h]',labelpad=15,fontsize=25)
ax1.set_xticks([-0.1,0.0,0.1,0.2,0.3,0.4,0.5])
ax1.tick_params(labelsize=25)
ax1_sec.tick_params(labelsize=25)
ax2.tick_params(labelsize=25)
#ax1.grid(True)
#ax2.grid(True)
plt.setp(ax1.get_xticklabels(), visible=False)
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
plt.subplots_adjust(hspace=.0)
plt.tight_layout()
ax1.legend(fontsize=22)
#ax2.legend(fontsize=15)
outfig = os.path.join(this_dir,'figures','MF_comparison_physical_high_res.png')
fig.savefig(outfig,overwrite=True)


print('done!')





