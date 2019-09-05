"""
Authors: Castagna Fabio, Andreon Stefano, Pranjal RS.
"""

from preprofit_funcs import (Pressure, mybeam, centdistmat, read_xy_err, read_tf, filt_image, log_lik, mcmc_run, traceplot, 
                             triangle, plot_best)
import numpy as np
import mbproj2 as mb
import emcee
import six.moves.cPickle as pickle

plotdir = './' # plot directory
output_file = 'mychain.pickle'


### Global-global variables

# Physical constants
m_e = 0.5109989*1e3 # electron rest mass (keV)
sigma_T = 6.6524587158*1e-25 # Thomson cross section (cm^2)
kpc_cm = mb.physconstants.kpc_cm # cm in 1 kpc
phys_const = [m_e, sigma_T, kpc_cm]


### Global variables

# Pressure parameters
press = Pressure()
pars = press.defPars()
name_pars = list(pars.keys())

# Parameters that we want to fit (among P_0, r_p, a, b, c)
fit_pars = ['P_0', 'r_p', 'a', 'b']
# To see the default parameter space extent, use: print(pars)
# For each parameter, use the following to change the bounds of the prior distribution:
#pars['P_0'].minval = 0.1
#pars['P_0'].maxval = 10.

# Sampling step
mystep = 2. # constant step in arcsec (values higher than (1/3)*FWHM of the beam are not recommended)

# MCMC parameters
ndim = len(fit_pars)
nwalkers = 14
nthreads = 8
nburn = 3000
nsteps = 2000
ci = 95. # confidence interval level
comp_time = True # if you wanna output the computation time


### Local variables

redshift = 0.888
R_b = 5000 # Radial cluster extent (kpc), serves as upper bound for Compton y parameter integration
compt_mJy_beam = -11e3 # Compton parameter to mJy/beam conversion

# File names (FITS and ASCII formats are accepted)
files_dir = './data' # files directory
beam_filename = '%s/Beam150GHz.fits' %files_dir 
# The first two columns must be [radius (arcsec), beam]
tf_filename = '%s/TransferFunction150GHz_CLJ1227.fits' %files_dir
# The first two columns must be [wave number (1/arcsec), tf]
flux_filename = '%s/flux_density.dat' %files_dir 
# The first three columns must be [radius (arcsec), flux (mJy/beam), error]

# Without beam or tf data, switch beam_approx and tf_approx to True and set the other parameters
beam_approx = False
tf_approx = False
fwhm_beam = None # fwhm of the normal distribution for the beam approximation
loc, scale, c = None, None, None # location, scale and normalization parameters of the normal cdf for the tf approximation

# Cosmological parameters
cosmology = mb.Cosmology(redshift)
cosmology.H0 = 67.32 # Hubble's constant (km/s/Mpc)
cosmology.WM = 0.3158 # matter density
cosmology.WV = 0.6842 # vacuum density
kpc_as = cosmology.kpc_per_arcsec # number of kpc per arcsec

# -------------------------------------------------------------------------------------------------------------------------------
# Code
# -------------------------------------------------------------------------------------------------------------------------------

# Parameter definition
for i in name_pars:
    if i not in fit_pars:
        pars[i].frozen = True

# Flux density data
flux_data = read_xy_err(flux_filename, ncol=3) # radius (arcsec), flux density, statistical error
maxr_data = flux_data[0][-1] # highest radius in the data

# Beam computation and creation of the 2D image
beam_2d, fwhm_beam = mybeam(mystep, maxr_data, approx=beam_approx, filename=beam_filename, normalize=True, fwhm_beam=fwhm_beam)

# Radius definition
mymaxr = (maxr_data+3*fwhm_beam)//mystep*mystep # max radius needed (arcsec)
radius = np.arange(0, mymaxr+mystep, mystep) # array of radii in arcsec
radius = np.append(-radius[:0:-1], radius) # from positive to entire axis
sep = radius.size//2 # index of radius 0
r_pp = np.arange(mystep*kpc_as, R_b+mystep*kpc_as, mystep*kpc_as) # radius in kpc used to compute the pressure profile
ub = min(sep, r_pp.size) # ub=sep unless r500 is too low and then r_pp.size < sep

# Matrix of distances in kpc centered on 0 with step=mystep
d_mat = centdistmat(radius*kpc_as)

# Transfer function
wn_as, tf = read_tf(tf_filename, approx=tf_approx, loc=loc, scale=scale, c=c) # wave number in arcsec^(-1), transmission
filtering = filt_image(wn_as, tf, d_mat.shape[0], mystep)

# Bayesian fit
starting_guess = [pars[i].val for i in fit_pars]
starting_var = np.array(np.repeat(.1, ndim))
starting_guesses = np.random.random((nwalkers, ndim))*starting_var+starting_guess
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_lik, args=[
    press, pars, fit_pars, r_pp, phys_const, radius, d_mat, beam_2d, 
    mystep, filtering, sep, ub, flux_data, compt_mJy_beam], threads=nthreads)
mcmc_run(sampler, p0=starting_guesses, nburn=nburn, nsteps=nsteps, comp_time=comp_time)
mysamples = sampler.chain.reshape(-1, ndim, order='F')

## Save the chain
file = open(output_file, 'wb') # create file
res = list([sampler.chain, sampler.lnprobability])
pickle.dump(res, file) # write
file.close()

# Posterior distribution's parameters
param_med = np.empty(ndim)
param_std = np.empty(ndim)
for i in np.arange(ndim):
    param_med[i] = np.median(mysamples[:,i])
    param_std[i] = np.std(mysamples[:,i])
    print('{:>13}'.format('Median(%s):' %fit_pars[i])+'%9s' %format(param_med[i], '.3f')+ 
          ';{:>12}'.format('Sd(%s):' %fit_pars[i])+'%9s' %format(param_std[i], '.3f'))
print('Best fit: [%s] = [%s] \nChi2 = %s with %s df' % 
      (', '.join(fit_pars), ', '.join(['{:.2f}'.format(i) for i in param_med]), 
       '{:.4f}'.format(log_lik(param_med, press, pars, fit_pars, r_pp, phys_const, radius, d_mat,beam_2d, mystep, filtering, 
                               sep, ub, flux_data, compt_mJy_beam, output='chisq')), flux_data[1].size-ndim))


### Plots
## Traceplot
traceplot(mysamples, fit_pars, nsteps, nwalkers, plotdir=plotdir)

## Corner plot
triangle(mysamples, fit_pars, plotdir)

# Random samples of at most 1000 profiles
prof_size = min(1000, mysamples.shape[0])
out_prof = np.array([log_lik(mysamples[j], press, pars, fit_pars, r_pp, phys_const, radius, d_mat, beam_2d, mystep,
                             filtering, sep, ub, flux_data, compt_mJy_beam, output='flux') for j in 
                     np.random.choice(mysamples.shape[0], size=prof_size, replace=False)])
quant = np.percentile(out_prof, [50., 50-ci/2, 50+ci/2], axis=0)

# Best-fit
plot_best(param_med, fit_pars, quant[0], quant[1], quant[2], radius, sep, flux_data, int(ci), plotdir)
