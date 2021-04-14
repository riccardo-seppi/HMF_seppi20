import sys
import os, glob
import matplotlib.pyplot as plt
import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.lss import bias

outdir = os.path.join('/home/rseppi/HMF_seppi20','figures','cosmo_trends')
cosmo = cosmology.setCosmology('multidark-planck') 

omega = lambda zz: cosmo.Om0*(1+zz)**3. / cosmo.Ez(zz)**2
DeltaVir_bn98 = lambda zz : (18.*np.pi**2. + 82.*(omega(zz)-1)- 39.*(omega(zz)-1)**2.) /omega(zz)

z0 = np.linspace(0,1.5,25)
z1 = np.linspace(0,10,25)


print('Delta vir...')
outfig = os.path.join(outdir,'Delta_vir.png')
fig = plt.figure(figsize=(6,6))
gs = fig.add_gridspec(2,1)
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(z0,DeltaVir_bn98(z0), label='Bryan-Norman+98')
ax2 = fig.add_subplot(gs[1,0])
ax2.plot(z1,DeltaVir_bn98(z1), label='Bryan-Norman+98')
ax2.hlines(178,0,10,color='C1', label=r'$\Delta_{vir}$=178')
ax1.grid(True)
ax2.grid(True)
ax1.set_xlabel('z', fontsize=14)
ax2.set_xlabel('z', fontsize=14)
ax1.set_ylabel(r'$\Delta_{vir}$', fontsize=14)
ax2.set_ylabel(r'$\Delta_{vir}$', fontsize=14)
ax1.tick_params(direction='in',labelsize=14)
ax2.tick_params(direction='in',labelsize=14)
ax1.legend(fontsize=14)
ax2.legend(fontsize=14)
ax1.set_title('Overdenisty on matter background', fontsize=15)
fig.tight_layout()
fig.savefig(outfig, overwrite=True)
plt.close()

print('OmegaM...')
outfig = os.path.join(outdir,'OmegaM.png')
fig = plt.figure(figsize=(6,6))
gs = fig.add_gridspec(2,1)
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(z0,cosmo.Om(z0))
ax2 = fig.add_subplot(gs[1,0])
ax2.plot(z1,cosmo.Om(z1))
ax1.grid(True)
ax2.grid(True)
ax1.set_xlabel('z', fontsize=14)
ax2.set_xlabel('z', fontsize=14)
ax1.set_ylabel(r'$\Omega_M$', fontsize=14)
ax2.set_ylabel(r'$\Omega_M$', fontsize=14)
ax1.tick_params(direction='in',labelsize=14)
ax2.tick_params(direction='in',labelsize=14)
fig.tight_layout()
fig.savefig(outfig, overwrite=True)
plt.close()

print('sigma8...')
outfig = os.path.join(outdir,'sigma8.png')
fig = plt.figure(figsize=(6,6))
gs = fig.add_gridspec(2,1)
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(z0,cosmo.sigma(8,z0))
ax2 = fig.add_subplot(gs[1,0])
ax2.plot(z1,cosmo.sigma(8,z1))
ax1.grid(True)
ax2.grid(True)
ax1.set_xlabel('z', fontsize=14)
ax2.set_xlabel('z', fontsize=14)
ax1.set_ylabel(r'$\sigma_8$', fontsize=14)
ax2.set_ylabel(r'$\sigma_8$', fontsize=14)
ax1.tick_params(direction='in',labelsize=14)
ax2.tick_params(direction='in',labelsize=14)
fig.tight_layout()
fig.savefig(outfig, overwrite=True)
plt.close()

print('Omega baryons...')
outfig = os.path.join(outdir,'OmegaB.png')
fig = plt.figure(figsize=(6,6))
gs = fig.add_gridspec(2,1)
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(z0,cosmo.Ob(z0))
ax2 = fig.add_subplot(gs[1,0])
ax2.plot(z1,cosmo.Ob(z1))
ax1.grid(True)
ax2.grid(True)
ax1.set_xlabel('z', fontsize=14)
ax2.set_xlabel('z', fontsize=14)
ax1.set_ylabel(r'$\Omega_b$', fontsize=14)
ax2.set_ylabel(r'$\Omega_b$', fontsize=14)
ax1.tick_params(direction='in',labelsize=14)
ax2.tick_params(direction='in',labelsize=14)
fig.tight_layout()
fig.savefig(outfig, overwrite=True)
plt.close()

print('Omega rad...')
outfig = os.path.join(outdir,'Omega_rad.png')
fig = plt.figure(figsize=(6,6))
gs = fig.add_gridspec(2,1)
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(z0,cosmo.Ogamma(z0))
ax2 = fig.add_subplot(gs[1,0])
ax2.plot(z1,cosmo.Ogamma(z1))
ax1.grid(True)
ax2.grid(True)
ax1.set_xlabel('z', fontsize=14)
ax2.set_xlabel('z', fontsize=14)
ax1.set_ylabel(r'$\Omega_{rad}$', fontsize=14)
ax2.set_ylabel(r'$\Omega_{rad}$', fontsize=14)
ax1.tick_params(direction='in',labelsize=14)
ax2.tick_params(direction='in',labelsize=14)
fig.tight_layout()
fig.savefig(outfig, overwrite=True)
plt.close()

print('Omega nu...')
outfig = os.path.join(outdir,'Omega_neutrinos.png')
fig = plt.figure(figsize=(6,6))
gs = fig.add_gridspec(2,1)
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(z0,cosmo.Onu(z0))
ax2 = fig.add_subplot(gs[1,0])
ax2.plot(z1,cosmo.Onu(z1))
ax1.grid(True)
ax2.grid(True)
ax1.set_xlabel('z', fontsize=14)
ax2.set_xlabel('z', fontsize=14)
ax1.set_ylabel(r'$\Omega_{\nu}$', fontsize=14)
ax2.set_ylabel(r'$\Omega_{\nu}$', fontsize=14)
ax1.tick_params(direction='in',labelsize=14)
ax2.tick_params(direction='in',labelsize=14)
fig.tight_layout()
fig.savefig(outfig, overwrite=True)
plt.close()

print('growth factor...')
outfig = os.path.join(outdir,'growth_factor.png')
fig = plt.figure(figsize=(6,6))
gs = fig.add_gridspec(2,1)
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(z0,cosmo.growthFactor(z0))
ax2 = fig.add_subplot(gs[1,0])
ax2.plot(z1,cosmo.growthFactor(z1))
ax1.grid(True)
ax2.grid(True)
ax1.set_xlabel('z', fontsize=14)
ax2.set_xlabel('z', fontsize=14)
ax1.set_ylabel(r'$D_{+}/D_0$', fontsize=14)
ax2.set_ylabel(r'$D_{+}/D_0$', fontsize=14)
ax1.tick_params(direction='in',labelsize=14)
ax2.tick_params(direction='in',labelsize=14)
fig.tight_layout()
fig.savefig(outfig, overwrite=True)
plt.close()

print('linear matter power spectrum...')
outfig = os.path.join(outdir,'matter_power_spectrum.png')
k = np.geomspace(1e-5,10,100)
zarr = np.linspace(0,10,6)
plt.figure(figsize=(6,6))
for z in zarr:
    plt.loglog(k,cosmo.matterPowerSpectrum(k,z), label='z={:.2f}'.format(z))
plt.grid(True)
plt.legend(fontsize=14)
plt.xlabel(r'k [h Mpc$^{-1}$]', fontsize=14)
plt.ylabel('P(k)', fontsize=14)
plt.tick_params(direction='in',labelsize=14)
plt.tight_layout()
plt.savefig(outfig, overwrite=True)
plt.close()

print('linear matter correlation function...')
outfig = os.path.join(outdir,'matter_corrfunc.png')
R = np.geomspace(1e-1,150,100)
zarr = np.linspace(0,10,6)
plt.figure(figsize=(6,6))
for z in zarr:
    plt.loglog(R,cosmo.correlationFunction(R,z), label='z={:.2f}'.format(z))
plt.grid(True)
plt.legend(fontsize=14)
plt.xlabel('r [Mpc/h]', fontsize=14)
plt.ylabel(r'$\xi$(r)', fontsize=14)
plt.tick_params(direction='in',labelsize=14)
plt.tight_layout()
plt.savefig(outfig, overwrite=True)
plt.close()

print('done!')


