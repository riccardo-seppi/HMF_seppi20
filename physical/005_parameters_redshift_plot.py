''''

Checks for redshift evolution in best fit parameters of  h(sigma,xoff,lambda)

'''

print('-----------------------------------')
print('PLOTS REDSHIFT EVOLUTION OF BEST FIT h(sigma,xoff,lambda) PARAMETERS')
print('-----------------------------------')



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

cosmo = cosmology.setCosmology('multidark-planck')    

def zevolution(data,alpha):
    x0,z = data
    E_z = cosmo.Ez(z=z)
#    return x0*(E_z)**alpha + C
    return x0*(1+z)**alpha 


bestfit = os.path.join('/home','rseppi','HMF_seppi20','physical','results','z_0.000','hsigma_params.fit')
zevo = os.path.join('/home','rseppi','HMF_seppi20','physical','results','zevo','hsigma_params.fit')
z = np.array([0,0.045,0.117,0.221,0.425,0.523,0.702,0.779,1.032,1.425])

t=Table.read(bestfit)
bestfit_pars = t['pars']
bestfit_err = t['err']

t2=Table.read(zevo)
param_z = t2['pars']
err_z = t2['err']

names = [r'$\log_{10}A$',r'$a$',r'$q$',r'$\log_{10}\mu$',r'$\alpha$',r'$\beta$',r'$e_0$',r'$\gamma$',r'$\delta$',r'$e_1$']


fig = plt.figure(figsize=(15,10))
#fig,ax1,ax2,ax3 = plt.subplot((3,1),figsize=(10,10),sharex=True)
'''
gs = fig.add_gridspec(10,1)
ax10 = fig.add_subplot(gs[9,:])
ax1 = fig.add_subplot(gs[0,:],sharex=ax10)
ax2 = fig.add_subplot(gs[1,:],sharex=ax10)
ax3 = fig.add_subplot(gs[2,:],sharex=ax10)
ax4 = fig.add_subplot(gs[3,:],sharex=ax10)
ax5 = fig.add_subplot(gs[4,:],sharex=ax10)
ax6 = fig.add_subplot(gs[5,:],sharex=ax10)
ax7 = fig.add_subplot(gs[6,:],sharex=ax10)
ax8 = fig.add_subplot(gs[7,:],sharex=ax10)
ax9 = fig.add_subplot(gs[8,:],sharex=ax10)
'''
gs = fig.add_gridspec(5,2)
ax10 = fig.add_subplot(gs[4,1])
ax1 = fig.add_subplot(gs[0,0],sharex=ax10)
ax2 = fig.add_subplot(gs[1,0],sharex=ax10)
ax3 = fig.add_subplot(gs[2,0],sharex=ax10)
ax4 = fig.add_subplot(gs[3,0],sharex=ax10)
ax5 = fig.add_subplot(gs[4,0],sharex=ax10)
ax6 = fig.add_subplot(gs[0,1],sharex=ax10)
ax7 = fig.add_subplot(gs[1,1],sharex=ax10)
ax8 = fig.add_subplot(gs[2,1],sharex=ax10)
ax9 = fig.add_subplot(gs[3,1],sharex=ax10)
'''
gs = fig.add_gridspec(2,5)
ax10 = fig.add_subplot(gs[1,4])
ax1 = fig.add_subplot(gs[0,0],sharex=ax10)
ax2 = fig.add_subplot(gs[1,0],sharex=ax10)
ax3 = fig.add_subplot(gs[0,1],sharex=ax10)
ax4 = fig.add_subplot(gs[1,1],sharex=ax10)
ax5 = fig.add_subplot(gs[0,2],sharex=ax10)
ax6 = fig.add_subplot(gs[1,2],sharex=ax10)
ax7 = fig.add_subplot(gs[0,3],sharex=ax10)
ax8 = fig.add_subplot(gs[1,3],sharex=ax10)
ax9 = fig.add_subplot(gs[0,4],sharex=ax10)
'''

for i in range(len(bestfit_pars)):
    err_tot = (((1+z)**(param_z[i])*bestfit_err[i])**2+(bestfit_pars[i]*(1+z)**param_z[i]*np.log(1+z)*err_z[i])**2)**0.5
    xdata = np.vstack((np.repeat(np.array(bestfit_pars[i]),len(z)),z))
    y = zevolution(xdata,np.array(param_z[i]))
    
    if(i==0):
        ax10.fill_between(z,y-np.array(err_tot),y+np.array(err_tot),label=names[i],alpha=0.8,color='C%d'%(i))
    elif(i==4):
        ax1.fill_between(z,y-np.array(err_tot),y+np.array(err_tot),label=names[i],alpha=0.8,color='C%d'%(i))
    elif(i==7):
        ax2.fill_between(z,y-np.array(err_tot),y+np.array(err_tot),label=names[i],alpha=0.8,color='C%d'%(i))
    elif(i==2):
        ax3.fill_between(z,y-np.array(err_tot),y+np.array(err_tot),label=names[i],alpha=0.8,color='C%d'%(i))
    elif(i==8):
        ax4.fill_between(z,y-np.array(err_tot),y+np.array(err_tot),label=names[i],alpha=0.8,color='C%d'%(i))
    elif(i==1):
        ax5.fill_between(z,y-np.array(err_tot),y+np.array(err_tot),label=names[i],alpha=0.8,color='C%d'%(i))
    elif(i==5):
        ax6.fill_between(z,y-np.array(err_tot),y+np.array(err_tot),label=names[i],alpha=0.8,color='C%d'%(i))
    elif(i==6):
        ax7.fill_between(z,y-np.array(err_tot),y+np.array(err_tot),label=names[i],alpha=0.8,color='C%d'%(i))
    elif(i==9):
        ax8.fill_between(z,y-np.array(err_tot),y+np.array(err_tot),label=names[i],alpha=0.8,color='C%d'%(i))  
    else:
        ax9.fill_between(z,y-np.array(err_tot),y+np.array(err_tot),label=names[i],alpha=0.8,color='C%d'%(i))       

ax1.legend(fontsize=15, loc = 'lower right')
ax2.legend(fontsize=15, loc = 'lower right')
ax3.legend(fontsize=15, loc = 'lower right')
ax4.legend(fontsize=15)
ax5.legend(fontsize=15)
ax6.legend(fontsize=15, loc = 'lower right')
ax7.legend(fontsize=15, loc = 'lower right')
ax8.legend(fontsize=15)
ax9.legend(fontsize=15, loc = 'lower right')
ax10.legend(fontsize=15, loc = 'lower right')
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)
ax5.grid(True)
ax6.grid(True)
ax7.grid(True)
ax8.grid(True)
ax9.grid(True)
ax10.grid(True)
ax10.set_xlabel('z',fontsize=20)
#ax2.set_xlabel('z',fontsize=20)
#ax4.set_xlabel('z',fontsize=20)
#ax6.set_xlabel('z',fontsize=20)
#ax8.set_xlabel('z',fontsize=20)
ax5.set_xlabel('z',fontsize=20)
#ax5.set_ylabel('parameters',fontsize=20)
ax1.tick_params(labelsize=20)
ax2.tick_params(labelsize=20)
ax3.tick_params(labelsize=20)
ax4.tick_params(labelsize=20)
ax5.tick_params(labelsize=20)
ax6.tick_params(labelsize=20)
ax7.tick_params(labelsize=20)
ax8.tick_params(labelsize=20)
ax9.tick_params(labelsize=20)
ax10.tick_params(labelsize=20)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax4.get_xticklabels(), visible=False)
#plt.setp(ax5.get_xticklabels(), visible=False)
plt.setp(ax6.get_xticklabels(), visible=False)
plt.setp(ax7.get_xticklabels(), visible=False)
plt.setp(ax8.get_xticklabels(), visible=False)
plt.setp(ax9.get_xticklabels(), visible=False)
fig.tight_layout()
outf = os.path.join('/home','rseppi','HMF_seppi20','physical','results','zevo','parameters_redshift_plot.png')
os.makedirs(os.path.dirname(outf),exist_ok=True)
fig.savefig(outf,overwrite=True)




