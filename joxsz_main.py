#!/usr/bin/env python3

# hydrostatic version of profile fit assuming parametric modified-beta
# density model

# note xspec will run for a while on first invocation to create grids,
# so don't worry if it hangs on Fitting

#from __future__ import division, print_function

import six.moves.cPickle as pickle
import numpy as np
import mbproj2 as mb
from scipy.interpolate import interp1d
from joxsz_funcs import (SZ_data, read_xy_err, mybeam, centdistmat, read_tf, filt_image, getEdges, loadBand, CmptPressure,
                         CmptUPPTemperature, CmptMyMass, mydefPars, myvikhFunction, myprior, get_sz_like, getLikelihood, 
                         prelimfit, traceplot, triangle, fitwithmod, best_fit_xsz, plot_best_sz, plot_rad_profs)
from types import MethodType
import h5py

#################
### SZ (see https://github.com/fcastagna/preprofit for more details)

## Global and local variables

mystep = 2. # constant step in arcsec (values higher than (1/3)*FWHM of the beam are not recommended)
m_e = 0.5109989*1e3 # electron rest mass (keV)
sigma_T = 6.6524587158*1e-25 # Thomson cross section (cm^2)
R_b = 5000 # Radial cluster extent (kpc), serves as upper bound for Compton y parameter integration

# Cluster and cosmology
redshift = 0.888
cosmology = mb.Cosmology(redshift)
cosmology.H0 = 67.32 # Hubble's constant (km/s/Mpc)
cosmology.WM = 0.3158 # matter density
cosmology.WV = 0.6842 # vacuum density

# Data
files_sz_dir = './data/SZ' # SZ data directory
beam_filename = '%s/Beam150GHz.fits' %files_sz_dir
tf_filename = '%s/TransferFunction150GHz_CLJ1227.fits' %files_sz_dir
flux_filename = '%s/flux_density.dat' %files_sz_dir 
convert_filename = '%s/Jy_per_beam_to_Compton.dat' %files_sz_dir

# Beam and transfer function. From raw data or Gaussian approximation?
beam_approx = False
tf_approx = False
fwhm_beam = None # fwhm of the normal distribution for the beam approximation
loc, scale, c = None, None, None # location, scale and normalization parameters of the normal cdf for the tf approximation

## Code
phys_const = [m_e, sigma_T]
kpc_as = cosmology.kpc_per_arcsec # number of kpc per arcsec
flux_data = read_xy_err(flux_filename, ncol=3) # radius (arcsec), flux density, statistical error
maxr_data = flux_data[0][-1] # highest radius in the data
beam_2d, fwhm_beam = mybeam(mystep, maxr_data, approx=beam_approx, filename=beam_filename, normalize=True, fwhm_beam=fwhm_beam)
mymaxr = (maxr_data+3*fwhm_beam)//mystep*mystep # max radius needed (arcsec)
radius = np.arange(0, mymaxr+mystep, mystep) # array of radii in arcsec
radius = np.append(-radius[:0:-1], radius) # from positive to entire axis
sep = radius.size//2 # index of radius 0
r_pp = np.arange(mystep*kpc_as, R_b+mystep*kpc_as, mystep*kpc_as) # radius in kpc used to compute the pressure profile
ub = min(sep, r_pp.size) # ub=sep unless r500 is too low and then r_pp.size < sep
d_mat = centdistmat(radius*kpc_as)
wn_as, tf = read_tf(tf_filename, approx=tf_approx, loc=loc, scale=scale, c=c) # wave number in arcsec^(-1), transmission
filtering = filt_image(wn_as, tf, d_mat.shape[0], mystep)
t_keV, compt_Jy_beam = np.loadtxt(convert_filename, skiprows=1, unpack=True)
convert = interp1d(t_keV, compt_Jy_beam*1e3, 'linear', fill_value='extrapolate')
sz_data = SZ_data(phys_const, mystep, kpc_as, convert, flux_data, beam_2d,
                  radius, sep, r_pp, ub, d_mat, filtering)

#################
### X-ray

## Global and local variables

# energy bands in eV
bandEs = [[700,1000], [1000,2000], [2000,3000], [3000,5000], [5000,7000]]

# Cluster parameters
NH_1022pcm2 = .01830000000000000000 # absorbing column density in 10^22 cm^(-2) 
Z_solar = 0.3

# Chandra data
files_x_dir = './data/X' # X-ray data directory
rmf = '%s/source.rmf' %files_x_dir
arf = '%s/source.arf' %files_x_dir
infgtempl = files_x_dir+'/fg_prof_%04i_%04i.dat' # foreground profile
inbgtempl = files_x_dir+'/bg_prof_%04i_%04i.dat' # background profile

# name for outputs
name = 'fit_modified_beta_nonhydro'
plotdir = './plots/'
savedir = './save/'

# Confidence interval
ci = 95

# MCMC parameters
nburn = 2000 # number to burn
nlength = 2000 # length of chain
nwalkers = 30 # number of walkers
nthreads = 8 # number of processes/threads

## Code

#remove cache
mb.xspechelper.deleteFile('countrate_cache.hdf5')

# annuli object contains edges of annuli
annuli = mb.Annuli(getEdges(infgtempl, bandEs), cosmology)

# load each band, chopping outer radius
bands = []
for bandE in bandEs:
    bands.append(loadBand(infgtempl, inbgtempl, bandE, rmf, arf))

# Data object represents annuli and bands (+ sz_data)
data = mb.Data(bands, annuli)
data.sz = sz_data


# flat metallicity profile
Z_cmpt = mb.CmptFlat('Z', annuli, defval = Z_solar, minval = 0)
# this is the modified beta model density described in Sanders+17 (used in McDonald+12)
ne_cmpt = mb.CmptVikhDensity('ne', annuli, mode='single')
# (single mode means only one beta model, as described in Vikhlinin+06)

# change parameter names
mb.CmptVikhDensity.vikhFunction = MethodType(myvikhFunction, ne_cmpt)
mb.CmptVikhDensity.defPars = MethodType(mydefPars, ne_cmpt)
mb.CmptVikhDensity.prior = MethodType(myprior, ne_cmpt)

press_cmpt = CmptPressure('p', annuli)
T_cmpt = CmptUPPTemperature('T', annuli, press_cmpt, ne_cmpt)

## NON-HYDRO
# non-hydrostatic model combining density, temperature and metallicity
model = mb.ModelNullPot(annuli, ne_cmpt, T_cmpt, Z_cmpt, NH_1022pcm2 = NH_1022pcm2)

# get default parameters
pars = model.defPars()
pars.update(press_cmpt.defPars())

# add parameter which allows variation of background with a Gaussian prior with sigma = 0.1
pars['backscale'] = mb.ParamGaussian(1., prior_mu = 1, prior_sigma = 0.1)

# stop radii going beyond edge of data
pars['log(r_c)'].maxval = annuli.edges_logkpc[-2]
pars['log(r_s)'].maxval = annuli.edges_logkpc[-2]

# some ranges of parameters to allow for the density model
pars['gamma'].val = 3.
pars['gamma'].frozen = True
pars['log(r_c)'].val = 2.
pars['epsilon'].maxval = 10

# Adam
pars['alpha'].val = 0.
pars['alpha'].frozen = True

#pars['a'].frozen = True
#pars['b'].frozen = True
pars['c'].frozen = True

# rimpiazzo di veusz
edges = annuli.edges_arcmin
xfig = 0.5 * (edges[1:] + edges[:-1])
errxfig = 0.5 * (edges[1:] - edges[:-1])
geomareas = np.pi * (edges[1:]**2 - edges[:-1]**2)

# do fitting of data with model
fit = mb.Fit(pars, model, data)
fit.press = press_cmpt
fit.mode = 'single'
fit.mass_cmpt = CmptMyMass('m', annuli, press_cmpt, ne_cmpt)
fit.savedir = savedir
mb.Fit.get_sz_like = MethodType(get_sz_like, fit)
mb.Fit.getLikelihood = MethodType(getLikelihood, fit)
# fit.refreshThawed() 
# refreshThawed is required if frozen is changed after Fit is constructed before doFitting (it's not required here)
fit.doFitting()

# save best fit
with open('%s%s_fit.pickle' % (savedir, name), 'wb') as f:
    pickle.dump(fit, f, -1)

# construct MCMC object and do burn in
mcmc = mb.MCMC(fit, walkers=nwalkers, processes=nthreads)
mcmc.burnIn(nburn)

# run mcmc proper
chainfilename = '%s%s_chain.hdf5' % (savedir, name)
mcmc.run(nlength)
mcmc.save(chainfilename)

# trace plot
# mcmc.fit.thawed dà i nomi dei parametri
# mcmc.sampler.chain.shape dà la dimensione
# mcmc.sampler.chain dà i valori
# tolgo la 3dim (data dai 200 samplers)
mysamples = mcmc.sampler.chain.reshape(-1, mcmc.sampler.chain.shape[2], order='F')

# corner plot, useless command if all params are plotted
# construct a set of physical median profiles from the chain and save
profs = mb.replayChainPhys(chainfilename, model, pars, thin=10, confint=ci)
mb.savePhysProfilesHDF5('%s%s_medians%s.hdf5' % (savedir, name, ci), profs)
mb.savePhysProfilesText('%s%s_medians%s.txt' % (savedir, name, ci), profs)
flatchain = mcmc.sampler.flatchain[::100]
mcmc_thawed = mcmc.fit.thawed

myprofs = fit.calcProfiles()
prelimfit(data, myprofs, geomareas, xfig, errxfig, plotdir)
traceplot(mysamples, mcmc_thawed, nsteps=nlength, nw=nwalkers, plotdir=plotdir)
triangle(mysamples, mcmc_thawed, plotdir)
profs = []
for pars in flatchain: #[-1000:]:
    fit.updateThawed(pars)
    profs.append(fit.calcProfiles())
    lxsz, mxsz, hxsz = np.percentile(profs, [50-ci/2., 50, 50+ci/2.], axis=0)
# lo, med and hi have shapes (numberofbands, numberannuli)

# fig model on data
fitwithmod(data, lxsz, mxsz, hxsz, geomareas, xfig, errxfig, plotdir)

# SZ data fit
med_xsz, lo_xsz, hi_xsz = best_fit_xsz(sz_data, flatchain, fit, ci, plotdir)
plot_best_sz(sz_data, med_xsz, lo_xsz, hi_xsz, ci, plotdir)

# Radial profiles plot
med_data = h5py.File('%s%s_medians%s.hdf5' % (savedir, name, ci), 'r')
plot_rad_profs(med_data, plotdir)
