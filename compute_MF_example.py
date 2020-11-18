import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from colossus.cosmology import cosmology
from colossus.lss import mass_function as mf
from colossus.lss import peaks

'''
This is an example code that shows how to compute multiplicity functions in Nbody simulations
'''

#Choose the data
mydir = '/home/rseppi/HMF_seppi20'
file_dir = '/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_1.0.fits'
#file_dir = '/data39s/simulation_2/MD/MD_4.0Gpc/Mass_Xoff_Concentration/distinct_0.4123.fits'

#Compute redshift from the name of the snapshot (the table name contains the scale factor, so a=1 is z=0, a=0.5 is z=1...)
aexp = float(os.path.basename(file_dir[:-5]).split('_')[1])
z_snap = 1/aexp -1

#set the cosmology
cosmo = cosmology.setCosmology('multidark-planck')    
h = cosmo.Hz(z=0.0)/100
rho_m = cosmo.rho_m(z=z_snap)*1e9 #get matter density in Msun h^2/Mpc**3

#compute the Volume considering comoving expansion
Vol=(4e3/(1+z_snap))**3 #(Mpc/h)^3

#open the data
print('Reading halos...')
hd1 = fits.open(file_dir)

mass = hd1[1].data['Mvir'] #Msun/h

#now we start computing the number density of halos
#compute number of halos in mass bins
print('Computing mass function...')
mass_edges = 10**np.linspace(np.log10(np.min(mass))+0.5,np.log10(np.max(mass)),50)
cts = np.histogram(mass, bins=mass_edges)[0] #number
 
mass_bins = (mass_edges[1:]+mass_edges[:-1])/2.
#compute lagrangian radius corresponding to each middle value of mass bin
R = peaks.lagrangianR(mass_bins) # Mpc/h
#compute rms variance inside these radii
sigma = cosmo.sigma(R,z=z_snap)
#compute the width of each mass bin in natural log units
dlnM = np.diff(np.log(mass_edges))

#compute number density of halos weighted by the mass bin width
dn_dlnM = cts/Vol/dlnM     #(Mpc/h)^-3

f = dn_dlnM*mass_bins/rho_m/cosmo.sigma(R,z=z_snap,derivative=True)*(-3.) # (Mpc/h)^-3 * Msun/h * (Msun h^2/Mpc^3)^-1 = number

#compute the error: poisson count 1/sqrt(N) and cosmic variance (~2% in HMD)
ferr = np.sqrt(1/cts + 0.02**2)*f

#add three models for comparison
mf_tinker = mf.massFunction(sigma,q_in='sigma', z=z_snap, mdef = 'vir', model = 'tinker08', q_out = 'f')
mf_despali = mf.massFunction(sigma,q_in='sigma', z=z_snap, mdef = 'vir', model = 'despali16', q_out = 'f')
mf_comparat = mf.massFunction(sigma,q_in='sigma', z=z_snap, mdef = 'vir', model = 'comparat17', q_out = 'f')

#make the figure
outf = os.path.join(mydir,'figures','HMD_z_%.3f_MF.png'%(z_snap))

fig,ax = plt.subplots(figsize=(8,8))
x = np.log10(1/sigma)
ax.fill_between(x,f-ferr,f+ferr,label='data',alpha=0.8)
ax.plot(x,mf_tinker,label='tinker08',lw=3)
ax.plot(x,mf_despali,label='despali16',lw=3)
ax.plot(x,mf_comparat,label='comparat17',lw=3)

#define transformations to add secondary x-axis from x=log10(1/sigma) to log10(Mass) and viceversa
def sigma_mass(x):
    sig=1/10**x
    r=cosmo.sigma(sig,z=z_snap,inverse=True)  #Mpc/h
    M=peaks.lagrangianM(r)  #Msun/h
    mas = np.log10(M)
    return mas

def mass_sigma(mas):
    r=peaks.lagrangianR(10**mas) #Mpc/h
    if bool(r.any()) == False:  #need this check because matplotlib can check empty values to create new axis and cosmo.sigma needs an np.array
        sigm = np.array([])
    else:
        sigm = cosmo.sigma(r,z=z_snap)
    x_ = np.log10(1/sigm)
    return x_

ax1_sec = ax.secondary_xaxis('top',functions=(sigma_mass,mass_sigma))
ax1_sec.set_xlabel(r'$\log_{10}M\ [M_{\odot}/h]$', fontsize=24, labelpad=10)
ax1_sec.tick_params(labelsize=24)
ax.set_xlabel(r'$\log_{10}\sigma^{-1}$',fontsize=24)
ax.set_ylabel(r'f($\sigma$)',fontsize=24)
ax.set_title('z = %.3g'%z_snap, fontsize=24)
ax.set_yscale('log')
ax.legend(fontsize=20)
ax.tick_params(labelsize=24)
ax.grid(True)
fig.tight_layout()
fig.savefig(outf, overwrite=True)

print('done!')
