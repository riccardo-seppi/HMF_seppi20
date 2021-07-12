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
path_2_snapshot_data = np.array([os.path.join(test_dir, 'distinct_1.0.fits.gz'),os.path.join(test_dir,'distinct_0.6565.fits.gz'),os.path.join(test_dir,'distinct_0.4922.fits.gz'),os.path.join(test_dir,'distinct_0.4123.fits.gz')])

dir_2_5 = '/data39s/simulation_2/MD/MD_2.5Gpc/Mass_Xoff_Concentration'

path_2_snapshot_data2_5 = np.array([os.path.join(dir_2_5,'distinct_1.0.fits.gz'),os.path.join(dir_2_5,'distinct_0.6583.fits.gz'),os.path.join(dir_2_5,'distinct_0.5.fits.gz'),os.path.join(dir_2_5,'distinct_0.409.fits.gz')])
dir_1_0 = '/data37s/simulation_1/MD/MD_1.0Gpc/Mass_Xoff_Concentration'

path_2_snapshot_data1_0 = np.array([os.path.join(dir_1_0,'distinct_1.0.fits.gz'),os.path.join(dir_1_0,'distinct_0.6565.fits.gz'),os.path.join(dir_1_0,'distinct_0.4922.fits.gz'),os.path.join(dir_1_0,'distinct_0.409.fits.gz')])

dir_0_4 = '/data17s/darksim/simulation_3/MD/MD_0.4Gpc/Mass_Xoff_Concentration'
path_2_snapshot_data0_4 = os.path.join(dir_0_4,'distinct_1.0.fits.gz')

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

def modified_sch_log0(x,A,alpha,beta,x0):
    f = (10**(x/x0))**(alpha)*np.exp(-(10**(x/x0))**beta)
    return A+np.log10(f)

def modified_sch_log0_list(data,A,alpha,beta,x0,e0):
    x,sigma = data
    f = (10**(x/x0)/sigma**e0)**(alpha)*np.exp(-(10**(x/x0)/sigma**e0)**(beta))
#    f = ((x/x0/sigma**e0))**(alpha)*np.exp(-((x/x0/sigma**e0))**(beta))
    return A+np.log10(f)


print('HMD')
choose_snap = 3
aexp = float(os.path.basename(path_2_snapshot_data[choose_snap][:-8]).split('_')[1])
z_snap = 1/aexp -1
print('z=%.3g'%(z_snap))
cosmo = cosmology.setCosmology('multidark-planck')    
E_z = cosmo.Ez(z=z_snap)
Vol1 = (4e3/(1+z_snap))**3
dc = peaks.collapseOverdensity(z = z_snap)
rho_m = cosmo.rho_m(z=z_snap)*1e9
h = cosmo.Hz(z=0)/100

hd1 = fits.open(path_2_snapshot_data[choose_snap])
mass1=hd1[1].data['Mvir']
logmass1 = np.log10(mass1)
R_1 = peaks.lagrangianR(mass1)
sigf_1 = cosmo.sigma(R_1,z=z_snap)
log1_sigf_1 = np.log10(1/sigf_1)
Rvir1 = hd1[1].data['Rvir']
Rs1 = hd1[1].data['Rs']
#xoff = np.log10(hd1[1].data['Xoff']/1000)#/hd1[1].data['Rvir'])
xoff = np.log10(hd1[1].data['Xoff'])#/hd1[1].data['Rvir'])
#spin1 = hd1[1].data['Spin']
spinpar1 = hd1[1].data['Spin']
conc1 = Rvir1/Rs1

print('min = ',min(log1_sigf_1))
print('max = ',max(log1_sigf_1))
#sys.exit()

if choose_snap == 0:
    log1sig_intervals = [-0.15, -0.11,-0.108,  -0.08,-0.078, -0.01,-0.008,  0.07,0.08, 0.20,0.24, 0.4]
elif choose_snap == 1:
    log1sig_intervals = [-0.1, 0.08,0.095,  0.11,0.13, 0.14,0.17,  0.19,0.21, 0.22,0.3, 0.44]
elif choose_snap == 2:
    log1sig_intervals = [-0.05, 0.13,0.15,  0.16,0.19, 0.2,0.23,  0.25,0.28, 0.29,0.35, 0.48]
elif choose_snap == 3:
    log1sig_intervals = [0.1, 0.18,0.195,  0.2,0.22, 0.23,0.265, 0.27,0.3, 0.31,0.4, 0.52]
else:
    print('snapshot not valid\naborting...')
    sys.exit()
sig_intervals = 1/(10**np.array(log1sig_intervals))
#sig_bins = (sig_intervals[1:] + sig_intervals[:-1])/2
sig_bins = [(sig_intervals[0]+sig_intervals[1])/2, (sig_intervals[1]+sig_intervals[2])/2, (sig_intervals[3]+sig_intervals[4])/2, (sig_intervals[5]+sig_intervals[6])/2, (sig_intervals[7]+sig_intervals[8])/2, (sig_intervals[9]+sig_intervals[10])/2] 
print(sig_bins)
R_intervals = cosmo.sigma(sig_intervals, z=z_snap,inverse=True)
M_intervals = peaks.lagrangianM(R_intervals)/h
print(log1sig_intervals)

interval0 = ((log1_sigf_1<= log1sig_intervals[1]))
interval1 = ((log1_sigf_1<= log1sig_intervals[2]) & (log1_sigf_1 > log1sig_intervals[1]))
interval2 = ((log1_sigf_1<= log1sig_intervals[4]) & (log1_sigf_1 > log1sig_intervals[3]))
interval3 = ((log1_sigf_1<= log1sig_intervals[6]) & (log1_sigf_1 > log1sig_intervals[5]))
interval4 = ((log1_sigf_1<= log1sig_intervals[8]) & (log1_sigf_1 > log1sig_intervals[7]))
interval5 = ((log1_sigf_1<= log1sig_intervals[10]) & (log1_sigf_1 > log1sig_intervals[9]))


low = min(xoff)
#low = -0.7
up = max(xoff)
#up=3
print(low,up)
bins_xoff = np.linspace(low,up,80)
#bins_xoff = np.logspace(np.log10(low),np.log10(up),80)
diff_bins_xoff = np.diff(bins_xoff)
bins_final_xoff = (bins_xoff[1:]+bins_xoff[:-1])/2.

def get_pdf_xoff(x):
    pdf = np.histogram(x,bins=bins_xoff,density=True)[0]
    pdf = pdf*diff_bins_xoff

    N = np.histogram(x,bins=bins_xoff)[0]
    err_ = np.sqrt(1/N)*pdf
    err = 1/np.log(10)*err_/pdf
    pdf = np.log10(pdf)
    
    return pdf, err, N

pdf_xoff0, yerr_xoff0, N0 = get_pdf_xoff(xoff[interval0])
pdf_xoff1, yerr_xoff1, N1 = get_pdf_xoff(xoff[interval1])
pdf_xoff2, yerr_xoff2, N2 = get_pdf_xoff(xoff[interval2])
pdf_xoff3, yerr_xoff3, N3 = get_pdf_xoff(xoff[interval3])
pdf_xoff4, yerr_xoff4, N4 = get_pdf_xoff(xoff[interval4])
pdf_xoff5, yerr_xoff5, N5 = get_pdf_xoff(xoff[interval5])

pdf_xoff, yerr_xoff, Ntot = get_pdf_xoff(xoff)

print('working on xoff pdf...')

indexx = ((pdf_xoff>= -4) & (Ntot>=100))
index_xoff0 = (pdf_xoff0 > -4)
index_xoff1 = (pdf_xoff1 > -4)
index_xoff2 = (pdf_xoff2 > -4)
index_xoff3 = (pdf_xoff3 > -4)
index_xoff4 = (pdf_xoff4 > -4)
index_xoff5 = (pdf_xoff5 > -4)


#fit pdf of xoff (also simultaneous)

popt_pdf_xoff,pcov_pdf_xoff = curve_fit(modified_sch_log0,bins_final_xoff[indexx],pdf_xoff[indexx],sigma=yerr_xoff[indexx],maxfev=1000000,p0=[(-6,4.5,0.45,0.5)])
#popt_pdf_xoff,pcov_pdf_xoff = curve_fit(modified_sch_log_double,bins_final_xoff[indexx],pdf_xoff[indexx],sigma=yerr_xoff[indexx],maxfev=1000000,p0=[(-6,4.5,0.45,0.5,4.5,0.45,0.5)])
model_pdf_xoff = modified_sch_log0(bins_final_xoff,*popt_pdf_xoff)
#model_pdf_xoff = modified_sch_log_double(bins_final_xoff,*popt_pdf_xoff)
chi2xoff = np.sum((pdf_xoff[indexx]-model_pdf_xoff[indexx])**2/(yerr_xoff[indexx])**2)
dofxoff = len(pdf_xoff[indexx])-5
print('chi2 xoff= ',chi2xoff)
print('dof xoff = ',dofxoff)
chi2rxoff=chi2xoff/dofxoff


bins_xoff_list = np.hstack((bins_final_xoff[index_xoff0],bins_final_xoff[index_xoff1],bins_final_xoff[index_xoff2],bins_final_xoff[index_xoff3],bins_final_xoff[index_xoff4],bins_final_xoff[index_xoff5]))
pdf_xoff_list = np.hstack((pdf_xoff0[index_xoff0],pdf_xoff1[index_xoff1],pdf_xoff2[index_xoff2],pdf_xoff3[index_xoff3],pdf_xoff4[index_xoff4],pdf_xoff5[index_xoff5]))
yerr_list = np.hstack((yerr_xoff0[index_xoff0],yerr_xoff1[index_xoff1],yerr_xoff2[index_xoff2],yerr_xoff3[index_xoff3],yerr_xoff4[index_xoff4],yerr_xoff5[index_xoff5]))
sigma0 = np.repeat(sig_bins[0],len(pdf_xoff0[index_xoff0]))
sigma1 = np.repeat(sig_bins[1],len(pdf_xoff1[index_xoff1]))
sigma2 = np.repeat(sig_bins[2],len(pdf_xoff2[index_xoff2]))
sigma3 = np.repeat(sig_bins[3],len(pdf_xoff3[index_xoff3]))
sigma4 = np.repeat(sig_bins[4],len(pdf_xoff4[index_xoff4]))
sigma5 = np.repeat(sig_bins[5],len(pdf_xoff5[index_xoff5]))
sigma_list = np.hstack((sigma0,sigma1,sigma2,sigma3,sigma4,sigma5))

xdata = np.vstack((bins_xoff_list,sigma_list))


def my_flat_priors(cube):
    params = cube.copy()
    params[0] = (cube[0])*1 - 6.0  #A
    params[1] = (cube[1])*3 + 9.0       #alpha
    params[2] = (cube[2])*1 + 0.8        #beta
    params[3] = (cube[3])*1 + 1.9   #x0
    params[4] = (cube[4])*1 - 1       #e0
    
    return params

names = [r'$A$',r'$\alpha$',r'$\beta$',r'$x_0$',r'$e_0$']

def likelihood(params):
    model = modified_sch_log0_list(xdata,*params)
    like = -0.5*(((model - pdf_xoff_list)/yerr_list)**2).sum()
    #like = -np.sum(model-(pdf_xoff[indexx])*np.log(model))
    return like

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
popt_pdf_xoff_list = np.array(result['posterior']['mean'])
pvar_pdf_xoff_list = np.array(result['posterior']['stdev'])
print('Best fit parameters = ', popt_pdf_xoff)

parameters = np.array(result['weighted_samples']['v'])
weights = np.array(result['weighted_samples']['w'])
weights /= weights.sum()
cumsumweights = np.cumsum(weights)
mask = cumsumweights > 1e-4
fig=corner.corner(parameters[mask,:], weights=weights[mask], labels=names, show_titles=True, color='r',bins=50,smooth=True,smooth1d=True,quantiles=[0.025,0.16,0.84,0.975],label_kwargs={'fontsize':20,'labelpad':20},title_kwargs={"fontsize":17},levels=[0.68,0.95],fill_contours=True,title_fmt='.3f')
axes = np.array(fig.axes).reshape((len(names), len(names)))
print(axes.shape)
for i in range(len(names)):
    ax = axes[i, 0]
    ax.yaxis.set_tick_params(labelsize=14)
    ax1 = axes[len(names)-1, i]
    ax1.xaxis.set_tick_params(labelsize=14)
#fig.savefig(os.path.join(this_dir,'pdf_xoff_HMD_corner_z_%.3g_slices.png'%(z_snap)))

#END OF ULTRANEST

#popt_pdf_xoff_list,pcov_pdf_xoff_list = curve_fit(modified_sch_log0_list,xdata,pdf_xoff_list,sigma=yerr_list,maxfev=1000000,p0=[(-6,4.5,0.45,0.5,0,0,0)])
#popt_pdf_xoff_list,pcov_pdf_xoff_list = curve_fit(modified_sch_log0_list,xdata,pdf_xoff_list,sigma=yerr_list,maxfev=1000000,p0=[(-5,7.5,0.9,2,-0.7)])
#pvar_pdf_xoff_list = np.diag(pcov_pdf_xoff_list)

chi2xoffl = np.sum((pdf_xoff_list-modified_sch_log0_list(xdata,*popt_pdf_xoff_list))**2/(yerr_list)**2)
dofxoffl = len(pdf_xoff_list)-7
print('chi2 xoff l = ',chi2xoffl)
print('dof xoff l = ',dofxoffl)
chi2rxoffl=chi2xoffl/dofxoffl
print('chi2r xoff l = ',chi2rxoffl)
print('parameters from simultaneous fitting = ',popt_pdf_xoff_list)

tab_sch_xoff_list = Table()
tab_sch_xoff_list.add_column(Column(name='pars',data=popt_pdf_xoff_list,unit=''))
tab_sch_xoff_list.add_column(Column(name='err',data=pvar_pdf_xoff_list,unit=''))
outschx_list = os.path.join(this_dir,'tables','schechter_HMD_xoff_z_%.3g_simult.fit'%(z_snap))
tab_sch_xoff_list.write(outschx_list,overwrite=True)


#plt.figure(figsize=(4.5,5.5))
#plt.scatter(bins_final_xoff,pdf_xoff, label = r'HMD', ls ='None', marker='o')
#plt.plot(bins_final_xoff,model_pdf_xoff,label='mod sch - full sample')
#plt.ylim(bottom=-5,top=-0.5)
#plt.tick_params(labelsize=12)
#plt.grid(True)
#plt.legend(fontsize=)
#plt.xlabel(r'$\log_{10}X_{off,P}$',fontsize=12)
#plt.ylabel(r'$P(X_{off,P})$',fontsize=12)
#plt.title('z = %.3g'%(z_snap), fontsize=12)
#plt.tight_layout()
#outpl = os.path.join(this_dir,'figures','pdf_xoff_HMD_z_%.3g.png'%(z_snap))
#os.makedirs(os.path.dirname(outpl), exist_ok=True)
#plt.savefig(outpl, overwrite=True)

plt.figure(figsize=(4.5,5.5))
plt.scatter(bins_final_xoff-0.6,pdf_xoff0+0.6,label=r'M$_\odot \leq$%.3g'%(M_intervals[1]),marker='o',ls='None',s=15)
plt.plot(bins_final_xoff-0.6,modified_sch_log0_list([bins_final_xoff,sig_bins[0]],*popt_pdf_xoff_list)+0.6)
plt.scatter(bins_final_xoff-0.4,pdf_xoff1+0.4,label=r'%.3g < M$_\odot \leq$%.3g'%(M_intervals[1],M_intervals[2]),marker='o',ls='None',s=15)
plt.plot(bins_final_xoff-0.4,modified_sch_log0_list([bins_final_xoff,sig_bins[1]],*popt_pdf_xoff_list)+0.4)
plt.scatter(bins_final_xoff-0.2,pdf_xoff2+0.2,label=r'%.3g < M$_\odot \leq$%.3g'%(M_intervals[3],M_intervals[4]),marker='o',ls='None',s=15)
plt.plot(bins_final_xoff-0.2,modified_sch_log0_list([bins_final_xoff,sig_bins[2]],*popt_pdf_xoff_list)+0.2)
plt.scatter(bins_final_xoff,pdf_xoff3,label=r'%.3g < M$_\odot \leq$%.3g'%(M_intervals[5],M_intervals[6]),marker='o',ls='None',s=15)
plt.plot(bins_final_xoff,modified_sch_log0_list([bins_final_xoff,sig_bins[3]],*popt_pdf_xoff_list))
plt.scatter(bins_final_xoff+0.2,pdf_xoff4-0.2,label=r'%.3g < M$_\odot \leq$ %.3g'%(M_intervals[7],M_intervals[8]),marker='o',ls='None',s=15)
plt.plot(bins_final_xoff+0.2,modified_sch_log0_list([bins_final_xoff,sig_bins[4]],*popt_pdf_xoff_list)-0.2)
plt.scatter(bins_final_xoff+0.4,pdf_xoff5-0.4,label=r'M$_\odot$ > %.3g'%(M_intervals[9]),marker='o',ls='None',s=15)
plt.plot(bins_final_xoff+0.4,modified_sch_log0_list([bins_final_xoff,sig_bins[5]],*popt_pdf_xoff_list)-0.4)
#plt.plot(bins_final_xoff,modified_sch_log0(bins_final_xoff,-5,4.,0.4,0.4),label='check')
#plt.xscale('log')
#plt.yscale('log')
plt.xlim(0,3.1)
plt.ylim(bottom=-5,top=-0.5)
plt.tick_params(labelsize=12)
#plt.grid(True)
plt.legend(fontsize=8,bbox_to_anchor=(-0.2, 1.1, 1.2, .3), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel(r'$\log_{10}$X$_{\rm off,P}$ + C$_1$',fontsize=12)
plt.ylabel(r'$\log_{10}$P(X$_{\rm off,P}$) + C$_0$',fontsize=12)
plt.title('z = %.2f'%z_snap, fontsize=12)
plt.tight_layout()
outpl = os.path.join(this_dir,'figures','pdf_xoff_HMD_slices_z_%.3g.png'%(z_snap))
os.makedirs(os.path.dirname(outpl), exist_ok=True)
plt.savefig(outpl, overwrite=True)
'''
plt.figure(figsize=(20,10))
plt.scatter(bins_final_xoff,pdf_xoff0,label=r'$M_\odot \leq %.3g$'%(M_intervals[1]),marker='o',ls='None')
plt.plot(bins_final_xoff,modified_sch_log0_list([bins_final_xoff,sig_bins[0]],*popt_pdf_xoff_list))
plt.scatter(bins_final_xoff,pdf_xoff1,label=r'$%.3g < M_\odot \leq %.3g$'%(M_intervals[1],M_intervals[2]),marker='o',ls='None')
plt.plot(bins_final_xoff,modified_sch_log0_list([bins_final_xoff,sig_bins[1]],*popt_pdf_xoff_list))
plt.scatter(bins_final_xoff,pdf_xoff2,label=r'$%.3g < M_\odot \leq %.3g$'%(M_intervals[3],M_intervals[4]),marker='o',ls='None')
plt.plot(bins_final_xoff,modified_sch_log0_list([bins_final_xoff,sig_bins[2]],*popt_pdf_xoff_list))
plt.scatter(bins_final_xoff,pdf_xoff3,label=r'$%.3g < M_\odot \leq %.3g$'%(M_intervals[5],M_intervals[6]),marker='o',ls='None')
plt.plot(bins_final_xoff,modified_sch_log0_list([bins_final_xoff,sig_bins[3]],*popt_pdf_xoff_list))
plt.scatter(bins_final_xoff,pdf_xoff4,label=r'$%.3g < M_\odot \leq %.3g$'%(M_intervals[7],M_intervals[8]),marker='o',ls='None')
plt.plot(bins_final_xoff,modified_sch_log0_list([bins_final_xoff,sig_bins[4]],*popt_pdf_xoff_list))
plt.scatter(bins_final_xoff,pdf_xoff5,label=r'$M_\odot > %.3g$'%(M_intervals[9]),marker='o',ls='None')
plt.plot(bins_final_xoff,modified_sch_log0_list([bins_final_xoff,sig_bins[5]],*popt_pdf_xoff_list))
#plt.plot(bins_final_xoff,modified_sch_log0(bins_final_xoff,-5,4.,0.4,0.4),label='check')
plt.xscale('log')
#plt.yscale('log')
plt.ylim(bottom=-5,top=-0.5)
plt.tick_params(labelsize=20)
plt.grid(True)
plt.legend(fontsize=15)
plt.xlabel(r'$\log_{10}X_{off}$',fontsize=25)
plt.ylabel(r'$P(X_{off})$',fontsize=25)
plt.title('z = %.3g'%(z_snap), fontsize=25)
plt.tight_layout()
outpl = os.path.join(this_dir,'figures','pdf_xoff_HMD_slices_z_%.3g.png'%(z_snap))
os.makedirs(os.path.dirname(outpl), exist_ok=True)
plt.savefig(outpl, overwrite=True)
'''
print('done!')

