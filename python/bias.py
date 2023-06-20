import sys
import os, glob
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import mass_function as mf
from colossus.lss import peaks
from colossus.lss import bias


cosmo = cosmology.setCosmology('multidark-planck')  

peak_bins = np.arange(0.7,5.,0.02)
peak_array = (peak_bins[:-1]+peak_bins[1:])/2.

h=cosmo.Hz(z=0)/100
dc0 = peaks.collapseOverdensity(z = 0)


M = np.logspace(12,15,100)
z = np.array([0.0,0.1,0.2,0.4,0.6,0.8,1.0])

Mdefs = ['200c','500c']
for Mdef in Mdefs:
	print(Mdef)
	outfig = os.path.join('/home/rseppi/HMF_seppi20','figures','bias_'+Mdef+'.png')
	plt.figure(figsize=(6,6))

	for z_ in z:
    		b = bias.haloBias(M, model = 'tinker10', z = z_, mdef = Mdef)
    		plt.plot(M,b, label='z={:.1f}'.format(z_))

	plt.xlabel(r'M'+Mdef+' [M$_\odot$/h]', fontsize=13)
	plt.ylabel('b', fontsize=13)
	plt.xscale('log')
	plt.ylim(0,6)
	plt.legend(fontsize=13)
	plt.tick_params(labelsize=13)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(outfig) 
	
	outfig = os.path.join('/home/rseppi/HMF_seppi20','figures','bias_'+Mdef+'_Msun.png')
	plt.figure(figsize=(6,6))

	for z_ in z:
    		b = bias.haloBias(M, model = 'tinker10', z = z_, mdef = Mdef)
    		plt.plot(M/h,b, label='z={:.1f}'.format(z_))

	plt.xlabel(r'M'+Mdef+' [M$_\odot$]', fontsize=13)
	plt.ylabel('b', fontsize=13)
	plt.xscale('log')
	plt.ylim(0,6)
	plt.legend(fontsize=13)
	plt.tick_params(labelsize=13)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(outfig)    



print('done!')


