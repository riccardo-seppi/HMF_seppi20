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
    return A+np.log10(f)

def modified_sch2(x,A,alpha,beta,x0):
    f = (x/10**x0)**(alpha)*np.exp(-(x/10**x0)**beta)
    return A+np.log10(f)

t0 = Table()
t1 = Table()
t2 = Table()
t3 = Table()

t = [t0,t1,t2,t3]

fig,ax = plt.subplots(1,1,figsize=(4.5,4.5))
print('HMD')
for i, p2s in enumerate(path_2_snapshot_data):
    aexp = float(os.path.basename(p2s[:-8]).split('_')[1])
    z_snap = 1/aexp -1
    print('z=%.3g'%(z_snap))
    cosmo = cosmology.setCosmology('multidark-planck')    
    E_z = cosmo.Ez(z=z_snap)
    Vol1 = (4e3/(1+z_snap))**3
    dc = peaks.collapseOverdensity(z = z_snap)
    rho_m = cosmo.rho_m(z=z_snap)*1e9
    h = cosmo.Hz(z=0)/100

    hd1 = fits.open(p2s)
    mass1=hd1[1].data['Mvir']
    logmass1 = np.log10(mass1)
    R_1 = peaks.lagrangianR(mass1)
    sigf_1 = cosmo.sigma(R_1,z=z_snap)
    log1_sigf_1 = np.log10(1/sigf_1)
    Rvir1 = hd1[1].data['Rvir']
    Rs1 = hd1[1].data['Rs']
    xoff = (hd1[1].data['Xoff']/hd1[1].data['Rvir'])
    spinpar1 = hd1[1].data['Spin']
    conc1 = Rvir1/Rs1

    print('min = ',min(log1_sigf_1))
    print('max = ',max(log1_sigf_1))
    #sys.exit()
    print('computing xoff pdf...')

    low = min(xoff)
    up = max(xoff)
    print(low,up)
    #bins_xoff = np.linspace(low,up,80)
    bins_xoff = np.logspace(np.log10(low),np.log10(up),80)
    bins_final = (bins_xoff[1:]+bins_xoff[:-1])/2.
    diff_bins = np.diff((bins_xoff))

    def get_pdf_xoff(x):
        pdf = np.histogram(x,bins=bins_xoff,density=True)[0]
        pdf = pdf*diff_bins
    
        N = np.histogram(x,bins=bins_xoff)[0]
        err_ = np.sqrt(1/N)*pdf
        err = 1/np.log(10)*err_/pdf
        pdf = np.log10(pdf)
    
        return pdf, err, N

    pdf_xoff, yerr, Ntot = get_pdf_xoff(xoff)

    index = ((pdf_xoff>=-4) & (Ntot>=100))

    def my_flat_priors(cube):
        params = cube.copy()
        params[0] = (cube[0])*2.5 - 3.5  #A
        params[1] = (cube[1])*1 + 3.5       #alpha
        params[2] = (cube[2])*0.5 + 0.5        #beta
#        params[3] = (cube[3])*0.2 + 1e-4    #x0
        params[3] = (cube[3])*2.5 - 3.   #x0
        
        return params

    names = [r'$A$',r'$\alpha$',r'$\beta$',r'$x_0$']

    def likelihood(params):
        model = modified_sch2(bins_final[index],*params)
        like = -0.5*(((model - pdf_xoff[index])/(0.1*pdf_xoff[index]))**2).sum()
        return like
    if(z_snap==0):    
    #######UNCOMMENT TO USE ULTRANEST#############
        sampler_rseppi = ultranest.ReactiveNestedSampler(names, likelihood, my_flat_priors)

        print('running sampler...')
        result = sampler_rseppi.run()
        sampler_rseppi.print_results()

        #from ultranest.plot import PredictionBand
        #fit = PredictionBand([log1sig, logx])
        #print(result)
        popt_pdf = np.array(result['posterior']['mean'])
        pvar_pdf = np.array(result['posterior']['stdev'])
        print('Best fit parameters = ', popt_pdf)

        parameters = np.array(result['weighted_samples']['v'])
        weights = np.array(result['weighted_samples']['w'])
        weights /= weights.sum()
        cumsumweights = np.cumsum(weights)
        mask = cumsumweights > 1e-4
        figcor=corner.corner(parameters[mask,:], weights=weights[mask], labels=names, show_titles=True, color='r',bins=50,smooth=True,smooth1d=True,quantiles=[0.025,0.16,0.84,0.975],label_kwargs={'fontsize':20,'labelpad':20},title_kwargs={"fontsize":17},levels=[0.68,0.95],fill_contours=True,title_fmt='.3f')
        axes = np.array(figcor.axes).reshape((len(names), len(names)))
        print(axes.shape)
        for j in range(len(names)):
            ax_ = axes[j, 0]
            ax_.yaxis.set_tick_params(labelsize=14)
            ax1_ = axes[len(names)-1, j]
            ax1_.xaxis.set_tick_params(labelsize=14)
        figcor.savefig(os.path.join(this_dir,'pdf_xoff_HMD_corner_rvir_z_%.3g.png'%(z_snap)))
        #END OF ULTRANEST
    else:    
        popt_pdf,pcov_pdf = curve_fit(modified_sch2,bins_final[index],pdf_xoff[index],sigma=0.1*pdf_xoff[index],p0=[(-2,3.8,0.6,-1.8)],maxfev=100000)#,p0=[(1e-3,4.,0.6,1e-3)])
        pvar_pdf=np.diag(pcov_pdf)   
    print('popt_pdf2 = ', popt_pdf)
    model_pdf = modified_sch2(bins_final,*popt_pdf)
    chi2 = np.sum((pdf_xoff[index]-model_pdf[index])**2/(yerr[index])**2)
    dof = len(pdf_xoff[index])-5
    print('chi2 = ',chi2)
    print('dof = ',dof)
    chi2r=chi2/dof
    print('chi2r = ',chi2r)
    chi2_expected = chi2scipy.stats(dof)[0]
    print('expexted chi2 = ',chi2_expected)
    t[i].add_column(Column(name='pars',data=popt_pdf,unit=''))
    t[i].add_column(Column(name='errors',data=pvar_pdf,unit=''))
    outsch = os.path.join(this_dir,'tables','schechter_HMD_xoff_z_%.3g_simult_code.fit'%(z_snap))
    t[i].write(outsch,overwrite=True)

    ax.scatter(bins_final,pdf_xoff, label = r'$z=%.3g$'%(z_snap), ls ='None', marker='o',s=15)
#    ax.fill_between(bins_final[index],pdf_spinpar[index] - yerr[index],pdf_spinpar[index] + yerr[index], alpha=0.4 )
    ax.plot(bins_final,model_pdf)
    
ax.set_xscale('log')
ax.set_ylim(bottom=-4.0,top=-1.0)
ax.set_xlim(0.9e-3,0.7)
#ax.grid(True)
ax.tick_params(labelsize=12)
ax.legend(fontsize=10)
ax.set_xlabel(r'X$_{\rm off}$',fontsize=12)
ax.set_ylabel(r'$\log_{10}$P(X$_{\rm off}$)',fontsize=12)
fig.tight_layout()
outpl = os.path.join(this_dir,'figures','pdf_xoff_HMD_all_z.png')
os.makedirs(os.path.dirname(outpl), exist_ok=True)
fig.savefig(outpl, overwrite=True)
    
print('done!')
    




