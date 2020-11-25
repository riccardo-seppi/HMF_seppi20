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

path_2_snapshot_data1_0 = np.array([os.path.join(dir_1_0,'distinct_1.0.fits'),os.path.join(dir_1_0,'distinct_0.6565.fits'),os.path.join(dir_1_0,'distinct_0.4922.fits'),os.path.join(dir_1_0,'distinct_0.4123.fits')])

dir_0_4 = '/data17s/darksim/simulation_3/MD/MD_0.4Gpc/Mass_Xoff_Concentration'
path_2_snapshot_data0_4 = os.path.join(dir_0_4,'distinct_1.0.fits')

fig0,ax0 = plt.subplots(1,1,figsize=(4.5,5.5))

zpl = np.array([1/1.0-1, 1/0.6565-1, 1/0.4922-1, 1/0.4123-1])
colors = ['b','r','c','m']

cosmo = cosmology.setCosmology('multidark-planck')  

#define arrays used to cut data: low resolution of HMD or low statistic of MDPL 
cuts_HMD_low = np.array([2.2,2.8,3.5,3.7])
cuts_BigMD_low = np.array([1.5,1.8,2.25,3.0])
cuts_BigMD_up = np.array([3.0,3.2,4.7,4.8])
cuts_MDPL_low = np.array([0.9,1.25,1.5,1.95])
cuts_MDPL_up = np.array([2.0,2.3,2.4,3.0])

def conc_sigma(ar,a0,b0):
    x,z=ar
    sigma = 1/x*dc
    return b0*(1+z)**(-0.2)*(1+7.37*(sigma/a0/(1+z)**0.5)**0.75)*(1+0.14*(sigma/a0/(1+z)**0.5)**(-2))


xt = np.arange(0,1,0.1)
zt = np.repeat(0,len(xt))
arr = np.vstack((xt,zt))
dc=1.6
#sys.exit()

t0=Table()
t1=Table()
t2=Table()
t3=Table()
peak_array_full = []
conc_full = []
conc_err_full = []
z_full = []

for i, p2s in enumerate(path_2_snapshot_data):
    print('HMD')
    print(i,' of ', len(path_2_snapshot_data)) 
    aexp = float(os.path.basename(p2s[:-5]).split('_')[1])
    z_snap = 1/aexp -1
    print('z=%.3g'%(z_snap))  
    E_z = cosmo.Ez(z=z_snap)
    Vol1 = (4e3/(1+z_snap))**3
    dc = peaks.collapseOverdensity(z = z_snap)
    rho_m = cosmo.rho_m(z=z_snap)*1e9

    hd1 = fits.open(p2s)

    mass1=hd1[1].data['Mvir']
    logmass1 = np.log10(mass1)
    R_1 = peaks.lagrangianR(mass1)
    sigf_1 = cosmo.sigma(R_1,z=z_snap)
    log1_sigf_1 = np.log10(1/sigf_1)
    Rvir1 = hd1[1].data['Rvir']
    Rs1 = hd1[1].data['Rs']
    xoff_data1 = np.log10(hd1[1].data['Xoff'])#/hd1[1].data['Rvir'])
    spin1 = hd1[1].data['Spin']
    spinpar1 = hd1[1].data['Spin_Bullock']

    conc1 = Rvir1/Rs1
    peak_bins = np.arange(0.9,5.,0.02)
    peak_array = (peak_bins[:-1]+peak_bins[1:])/2.

    def get_average(x,sel):
        return np.average(x[sel]),np.std(x[sel]),np.sum(sel)

    def get_median(x,sel):
        return np.median(x[sel])

    conc_av1 = np.zeros(len(peak_array))
    conc_std1 = np.zeros(len(peak_array))
    conc_N1 = np.zeros(len(peak_array))
    
    z1 = np.repeat(z_snap, len(peak_array))
         
#BigMD
    print('BigMD')
    hd2 = fits.open(path_2_snapshot_data2_5[i])

    mass2=hd2[1].data['Mvir']
    logmass2 = np.log10(mass2)
    R_2 = peaks.lagrangianR(mass2)
    sigf_2 = cosmo.sigma(R_2,z=z_snap)
    log1_sigf_2 = np.log10(1/sigf_2)
    Rvir2 = hd2[1].data['Rvir']
    Rs2 = hd2[1].data['Rs']
    xoff_data2 = np.log10(hd2[1].data['Xoff'])#/hd1[1].data['Rvir'])
    spin2 = hd2[1].data['Spin']
    spinpar2 = hd2[1].data['Spin_Bullock']

    conc2 = Rvir2/Rs2

    conc_av2 = np.zeros(len(peak_array))
    conc_std2 = np.zeros(len(peak_array))
    conc_N2 = np.zeros(len(peak_array))
    z2 = np.repeat(z_snap, len(peak_array))

#MDPL
    print('MDPL')
    hd3 = fits.open(path_2_snapshot_data1_0[i])

    mass3=hd3[1].data['Mvir']
    logmass3 = np.log10(mass3)
    R_3 = peaks.lagrangianR(mass3)
    sigf_3 = cosmo.sigma(R_3,z=z_snap)
    log1_sigf_3 = np.log10(1/sigf_3)
    Rvir3 = hd3[1].data['Rvir']
    Rs3 = hd3[1].data['Rs']
    xoff_data3 = np.log10(hd3[1].data['Xoff'])#/hd1[1].data['Rvir'])
    spin3 = hd3[1].data['Spin']
    spinpar3 = hd3[1].data['Spin_Bullock']

    conc3 = Rvir3/Rs3
    sigf_tot_ = np.hstack((sigf_1,sigf_2,sigf_3))

    conc_av3 = np.zeros(len(peak_array))
    conc_std3 = np.zeros(len(peak_array))
    conc_N3 = np.zeros(len(peak_array))
    z3 = np.repeat(z_snap, len(peak_array))

    spinpar_median = np.zeros(len(peak_array))
    spin_median = np.zeros(len(peak_array))
    
    print('computing values...')
    for jj, (x_min,x_max) in enumerate(zip(peak_bins[:-1],peak_bins[1:])):
        conc_av1[jj],conc_std1[jj],conc_N1[jj] = get_average(conc1,(dc/sigf_1>=x_min) & (dc/sigf_1<x_max)) 
        conc_av2[jj],conc_std2[jj],conc_N2[jj] = get_average(conc2,(dc/sigf_2>=x_min) & (dc/sigf_2<x_max))      
        conc_av3[jj],conc_std3[jj],conc_N3[jj] = get_average(conc3,(dc/sigf_3>=x_min) & (dc/sigf_3<x_max)) 
    print('values computed!')

#computing averages on each cube

    conc_err1 = conc_std1/np.sqrt(conc_N1)

   # print('conc_N1 = ', conc_N1)
   # print('xoff_N1 = ', xoff_N1)

    ind_one = ((peak_array > cuts_HMD_low[i]) & (~np.isnan(conc_av1)) & (conc_N1 > 100))
    peak_array_1 = np.array(peak_array[ind_one])
    z1_ = np.array(z1[ind_one])
    conc_av_1 = np.array(conc_av1[ind_one])
    #conc_err_ = 10*np.array(conc_err[~np.isnan(conc_av)])
    conc_err_1 = 0.1*conc_av_1
    
    conc_err2 = conc_std2/np.sqrt(conc_N2)
    ind_two = ((peak_array > cuts_BigMD_low[i]) & (peak_array < cuts_BigMD_up[i]) & (~np.isnan(conc_av2))& (conc_N2 > 100))
    peak_array_2 = np.array(peak_array[ind_two])
    z2_ = np.array(z2[ind_two])
    conc_av_2 = np.array(conc_av2[ind_two])
    conc_err_2 = 0.1*conc_av_2
    conc_err3 = conc_std3/np.sqrt(conc_N3)
    ind_three = ((peak_array > cuts_MDPL_low[i]) & (peak_array < cuts_MDPL_up[i]) & (~np.isnan(conc_av3)) & (conc_N3 > 100))
    conc_av_3 = np.array(conc_av3[ind_three])
    conc_err_3 = 0.1*conc_av_3
    peak_array_3 = np.array(peak_array[ind_three])
    z3_ = np.array(z3[ind_three])

    ax0.scatter(peak_array_1,conc_av_1, label = r'$z= %.3g\ HMD$'%(z_snap), ls='None',c='%.c'%(colors[i]),marker='o',facecolors='none',s=13)

    ax0.scatter(peak_array_2,conc_av_2, label = r'$z= %.3g\ BigMD$'%(z_snap), ls='None', edgecolors='%.c'%(colors[i]), marker='^',facecolors='none',s=13)

    ax0.scatter(peak_array_3,conc_av_3, label = r'$z= %.3g\ MDPL$'%(z_snap), ls ='None', edgecolors='%.c'%(colors[i]), marker='s',facecolors='none',s=13)
    peak_array_ = np.hstack((peak_array_1,peak_array_2,peak_array_3))
    conc_av_ = np.hstack((conc_av_1,conc_av_2,conc_av_3))
    conc_err_ = np.hstack((conc_err_1,conc_err_2,conc_err_3))
    z_ = np.hstack((z1_,z2_,z3_))

    ind1 = np.argsort(peak_array_)


    peak_array_full.extend(peak_array_)
    conc_full.extend(conc_av_)
    conc_err_full.extend(conc_err_)
    z_full.extend(z_)


peak_array_full = np.array(peak_array_full)
conc_full = np.array(conc_full)
conc_err_full = np.array(conc_err_full)
z_full = np.array(z_full)


xdata = np.vstack((peak_array_full,z_full))
xdata_rav = np.vstack((peak_array_full.ravel(),z_full.ravel()))


popt,pcov = curve_fit(conc_sigma,xdata_rav,conc_full.ravel(),sigma=conc_err_full.ravel(),p0=[(0.5,0.7)])  
t0.add_column(Column(name='params',data=popt,unit=''))
t0.add_column(Column(name='errs',data=np.diag(pcov),unit=''))

z0 = np.repeat(zpl[0],len(peak_array))
z1 = np.repeat(zpl[1],len(peak_array))
z2 = np.repeat(zpl[2],len(peak_array))
z3 = np.repeat(zpl[3],len(peak_array))

red_arr = [z0,z1,z2,z3]

for k,red in enumerate(red_arr):
    ax0.plot(peak_array,conc_sigma([peak_array,red],*popt),c='%.c'%(colors[k]))


outt = os.path.join(this_dir,'tables','conc_sigma_params.fit')
t0.write(outt,overwrite=True)

x_klypin_z0=np.array([0.64468515421979,0.672062199796709,0.73713460864555,0.79004131186337,0.827405622700013,0.866537045842464,0.981685855246754,1.11728713807222, 1.21419488439505,1.3134245558784,1.3947436663504,1.5655468325982,1.71713087287551,1.90968320782083,2.08493152168224,2.30803750352711,2.56685179512581,2.76382575993555, 3.18950675357279,3.63007662126864,3.98155871641291,4.28709385014517,4.63745516350235])

#y_klypin_z0 = np.array([11.3114182167257,10.9837063649641,10.3260876714099,9.85160010757356,9.53809875032521,9.20746421379086,8.55499332852549,7.92542382731734,7.49485127540264, 7.17151317636077,6.88232847741283,6.45127382535001,6.22764289143584,5.85478019408151,5.58575086956991,5.37629353513157,5.18992638085984,5.02477057402113,4.86487041794332, 4.75178475824874,4.75178475824874,4.75178475824874,4.77980731235357,])
y_klypin_z0 = 0.567*(1+7.37*(1.68647/x_klypin_z0/0.75)**0.75)*(1+0.14*(1.68647/x_klypin_z0/0.75)**(-2))

x_klypin_z05 = np.array([0.6474747482294254,0.6681142183180427,0.6837207745120811,0.7173090145819012,0.7449005805141552,0.77204356667661,0.8058093310275328,0.8307578929497537, 0.8623303128594496,0.8929853011147799,0.9335316451558245,0.9691246652991021,1.0022697408738286,1.0463317011874522,1.087920335189708,1.1276911692813791,1.168471077059915, 1.2138016930886795,1.2618510878535811,1.2936224907667826,1.4467886313489733,1.5292286842690128,1.5766682882815193,1.705437242174511,1.7797093714129788,1.8493158741291793, 1.9243772734535025,1.9934163347419833,2.080435291831805,2.167185906533847,2.2635766597671743,2.375245095059301,2.4621110488831532,2.5712386879054834,2.6777414482604502, 2.766700861377479,2.8718475498425984,3.0942808904100816,3.222162305324112,3.3458068597228725,3.4791362204630425,3.62099307235141,3.77533369331493,4.0198616047034825, 4.663915552973153])

#y_klypin_z05 = np.array([8.918638717515265, 8.729516582975736, 8.595409431087191, 8.294823264983455, 8.084292010135396, 7.898716992209758, 7.776274467128623, 7.643771604092884, 7.4617006568618685, 7.312833028457753, 7.1051112039534186, 6.94554436869463, 6.8013358830494575, 6.6648062299329665, 6.511161451463365, 6.363854745456169,6.213608123006287, 6.085873723945776, 5.9736094398132495, 5.9131629106059505, 5.441865923252947, 5.359476448238832, 5.284899698822923, 5.0943237598617905, 5.003607216006128, 4.924883851517751, 4.865876173002063, 4.811500797138367, 4.747066727916322, 4.705673878369441, 4.652975872340827, 4.598275705777504, 4.570071627190047, 4.521272037885673, 4.490212633629277, 4.493258171836891, 4.471549562492173, 4.455718400237229, 4.432237134323113, 4.4343321602829615, 4.437294211450071,4.467941671550245,4.528942449762775, 4.592505669724429, 4.759824373420251])
y_klypin_z05 = 0.529*(1+7.37*(1.68647/x_klypin_z05/0.97)**0.75)*(1+0.14*(1.68647/x_klypin_z05/0.97)**(-2))

x_klypin_z1 = np.array([0.6417129487814519, 0.7371346086455502, 0.842841544754699, 0.9862327044933592, 1.1328838852957983, 1.6170153043197235, 1.4076935840339886, 1.8066704015823631, 2.0467477839935477, 2.38391588704717, 2.7006998923363783, 2.989698497269875,3.264057939953779, 3.563594872561355, 3.854828473566203, 4.169863043364483, 4.594793419988135])
#y_klypin_z1 = np.array([7.6282322432444465, 6.94329856251493, 6.39462427894103, 5.820455401461421, 5.4079989856383754, 4.879194056741039, 4.60057166039697, 4.262011536001448, 4.415057822817232, 4.138533501158129, 4.162939547486202, 4.224586210888821, 4.199818748953865, 4.299768409552162, 4.428057078314735, 4.560173391025831, 4.793880499835053])
y_klypin_z1 = 0.496*(1+7.37*(1.68647/x_klypin_z1/1.12)**0.75)*(1+0.14*(1.68647/x_klypin_z1/1.12)**(-2))

x_klypin_z15 = np.array([0.73035, 0.84284,0.94606,1.07177,1.15402,1.24833,1.40769,1.55833,1.80667,2.05623,2.4509,2.76383,3.03143,3.48220,3.68075,3.96320])
#y_klypin_z15 = np.array([6.15484, 5.65183, 5.25132, 4.95145, 4.77981, 4.60057, 4.38917, 4.1998, 4.04233, 3.97165, 3.95999, 3.98334, 4.06617, 4.26201, 4.33786, 4.50686, 3.99507])
y_klypin_z15 = 0.474*(1+7.37*(1.68647/x_klypin_z15/1.28)**0.75)*(1+0.14*(1.68647/x_klypin_z15/1.28)**(-2))

x_white_ = np.array([12.95428, 13.19431, 13.43430, 13.61911, 13.75937, 13.96767, 14.22190, 14.39802, 14.58162, 14.77839, 15.04843, 15.21553])
x_wh = 10**x_white_
x_whi = x_wh * (200/178)**3 
r_white = peaks.lagrangianR(x_whi)
s_white = cosmo.sigma(r_white,z=0)
dc = peaks.collapseOverdensity(z = z_snap)
x_white = dc/s_white

y_white_up = np.array([9.81966, 9.32432, 8.83185, 8.27883, 7.84794, 7.34993, 6.91603, 6.55599, 6.25745, 5.8362, 5.51008, 5.29120])

y_white_low =np.array([3.19233, 3.29416, 3.37779, 3.44187, 3.48475, 3.59550, 3.64572, 3.73355, 3.87689, 4.00045, 4.12771, 4.20153])
y_white_low = np.flip(y_white_low)

ax0.plot(x_klypin_z0,y_klypin_z0,label=r'Klypin16 z=0',ls='dashed')
ax0.plot(x_klypin_z05,y_klypin_z05,label=r'Klypin16 z=0.5',ls='dashed')
ax0.plot(x_klypin_z1,y_klypin_z1,label=r'Klypin16 z=1.0',ls='dashed')
ax0.plot(x_klypin_z15,y_klypin_z15,label=r'Klypin16 z=1.5',ls='dashed')
ax0.fill_between(x_white,y_white_low,y_white_up,label='Wang19', alpha=0.5)

h=cosmo.Hz(z=0)/100
dc0 = peaks.collapseOverdensity(z = 0)
def Mass_peak(x):
    r=cosmo.sigma(dc0/x,z=0,inverse=True)
    M=peaks.lagrangianM(r)/h
    return np.log10(M)


ax0.set_ylabel(r'$conc$', fontsize=12)
ax0.set_xlabel(r'$\nu = \delta_c/\sigma$', fontsize=12)
ax0.grid(True)
ax0.set_xlim(left=0.8)
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_xticks([0.8,1,2,3,4])
ax0.set_yticks([3,4,5,6,7,8,9,10,11])
ax0.set_ylim(3.0,11)

ax0.xaxis.set_major_formatter(ScalarFormatter())
ax0.yaxis.set_major_formatter(ScalarFormatter())
ax0.ticklabel_format(axis='both', style='plain')

ax0_sec = ax0.twiny()
new_tick_locations = np.array([12.5,13.0,13.5,14.0, 14.5, 15.0, 15.5])
xmin,xmax=ax0.get_xlim()
ax0_sec.set_xlim(Mass_peak(xmin),Mass_peak(xmax))
print(xmin,xmax)
print(Mass_peak(xmin),Mass_peak(xmax))
#ax0_sec.set_xscale('log')
ax0_sec.set_xticks(new_tick_locations)

ax0.legend(fontsize=8,bbox_to_anchor=(-0.2, 1.16, 1.2, .3), loc='lower left', ncol=3, mode="expand", borderaxespad=0.)
ax0.tick_params(labelsize=12)
ax0_sec.set_xlabel(r'$\log_{10}$M [M$_{\odot}$]', fontsize=12)
ax0_sec.tick_params(labelsize=12)
fig0.tight_layout()
outfi0 = os.path.join(this_dir,'figures','relation_c_sigma.png')
os.makedirs(os.path.dirname(outfi0), exist_ok=True)
fig0.savefig(outfi0, overwrite=True)

print('done!')
sys.exit()







