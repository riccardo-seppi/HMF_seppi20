import sys
import os, glob
import matplotlib.pyplot as plt
import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.lss import bias
from colossus.lss import mass_function as mf

outfig = os.path.join('/home/rseppi/HMF_seppi20','figures','MF_cosmo.png')
cosmo = cosmology.setCosmology('multidark-planck') 

omega = lambda zz: cosmo.Om0*(1+zz)**3. / cosmo.Ez(zz)**2
DeltaVir_bn98 = lambda zz : (18.*np.pi**2. + 82.*(omega(zz)-1)- 39.*(omega(zz)-1)**2.) /omega(zz)

Mass = np.geomspace(1e12,8e15,50)
redshift = np.array([0,2.0])

fig0, ax0 = plt.subplots(1,1,figsize=(6,6))
colors = ['C0','C1']
for jj,(z,col) in enumerate(zip(redshift,colors)):
    mfunc = mf.massFunction(Mass,q_in='M', z=z, mdef = 'vir', model = 'despali16', q_out = 'dndlnM') 
    if jj==0:
        ax0.loglog(Mass, mfunc, label = 'Planck', lw=4, ls = 'solid', c=col)
    else:
        ax0.loglog(Mass, mfunc, lw=4, ls = 'solid', c=col)    
    
params1 = {'flat': True, 'H0': 67.74, 'Om0': 0.35, 'Ob0': 0.049, 'sigma8': 0.85, 'ns':0.96}
cosmo1 = cosmology.setCosmology('cosmo1', params1)

for jj,(z,col) in enumerate(zip(redshift,colors)):
    mfunc = mf.massFunction(Mass,q_in='M', z=z, mdef = 'vir', model = 'despali16', q_out = 'dndlnM') 
    if jj==0:
        ax0.loglog(Mass, mfunc, label = r'$\Omega_M$=0.35 $\sigma_8=0.85$', lw=4, ls = '--', c=col)
    else:
        ax0.loglog(Mass, mfunc, lw=4, ls = '--', c=col) 

params2 = {'flat': True, 'H0': 67.74, 'Om0': 0.25, 'Ob0': 0.049, 'sigma8': 0.75, 'ns':0.96}
cosmo2 = cosmology.setCosmology('cosmo2', params2)

for jj,(z,col) in enumerate(zip(redshift,colors)):
    mfunc = mf.massFunction(Mass,q_in='M', z=z, mdef = 'vir', model = 'despali16', q_out = 'dndlnM') 
    if jj==0:
        ax0.loglog(Mass, mfunc, label = r'$\Omega_M$=0.25 $\sigma_8=0.75$', lw=4, ls = 'dotted', c=col)
    else:
        ax0.loglog(Mass, mfunc, lw=4, ls = 'dotted', c=col) 

ax0.text(2e12, 1e-7, 'z=0', color='C0', fontsize=14)#, bbox=dict(facecolor='none', edgecolor='red'))
ax0.text(2e12, 5e-8, 'z=2', color='C1', fontsize=14)#, bbox=dict(facecolor='none', edgecolor='red'))

ax0.set_xlabel(r'Mass [M$_\odot$/h]', fontsize=16)
ax0.set_ylabel('dn/dlnM', fontsize=16)
ax0.tick_params(labelsize=16,direction='in')
ax0.set_ylim(1e-8,1e-2)
ax0.legend(fontsize=14)
fig0.tight_layout()
fig0.savefig(outfig,overwrite=True)


Mass = np.geomspace(1e13,8e15,50)
redshift = np.array([0,1.0])
outfig = os.path.join('/home/rseppi/HMF_seppi20','figures','MF_cosmo_Planck_WMAP.png')
fig0, ax0 = plt.subplots(1,1,figsize=(6,6))
colors = ['C0','C1']

#params = {'flat': True, 'H0': 67.74, 'Om0': 0.3089, 'Ob0': 0.0486, 'sigma8': 0.8159, 'ns':0.96}
#cosmo = cosmology.setCosmology('cosmo0', params)
cosmo = cosmology.setCosmology('planck18') 
for jj,(z,col) in enumerate(zip(redshift,colors)):
    mfunc = mf.massFunction(Mass,q_in='M', z=z, mdef = 'vir', model = 'tinker08', q_out = 'dndlnM') 
    if jj==0:
        ax0.loglog(Mass, mfunc, label = 'Planck', lw=4, ls = 'solid', c=col)
    else:
        ax0.loglog(Mass, mfunc, lw=4, ls = 'solid', c=col)    
    
#params1 = {'flat': True, 'H0': 70.4, 'Om0': 0.272, 'Ob0': 0.0456, 'sigma8': 0.809, 'ns':0.96}
#cosmo = cosmology.setCosmology('cosmo1', params1)
cosmo = cosmology.setCosmology('WMAP9') 

for jj,(z,col) in enumerate(zip(redshift,colors)):
    mfunc = mf.massFunction(Mass,q_in='M', z=z, mdef = 'vir', model = 'tinker08', q_out = 'dndlnM') 
    if jj==0:
        ax0.loglog(Mass, mfunc, label = 'WMAP', lw=4, ls = '--', c=col)
    else:
        ax0.loglog(Mass, mfunc, lw=4, ls = '--', c=col) 

ax0.text(2e13, 1e-7, 'z=0', color='C0', fontsize=14)#, bbox=dict(facecolor='none', edgecolor='red'))
ax0.text(2e13, 5e-8, 'z=1', color='C1', fontsize=14)#, bbox=dict(facecolor='none', edgecolor='red'))

ax0.set_xlabel(r'Mass [M$_\odot$/h]', fontsize=16)
ax0.set_ylabel('dn/dlnM', fontsize=16)
ax0.tick_params(labelsize=16,direction='in')
ax0.set_ylim(1e-8,1e-3)
ax0.legend(fontsize=14)
fig0.tight_layout()
fig0.savefig(outfig,overwrite=True)
print('done!')



