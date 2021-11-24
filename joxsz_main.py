#!/usr/bin/env python3

# note xspec will run for a while on first invocation to create grids,
# so don't worry if it hangs on Fitting

import six.moves.cPickle as pickle
import numpy as np
import mbproj2 as mb
from scipy.interpolate import interp1d
from joxsz_funcs import (SZ_data, read_xy_err, mybeam, centdistmat, read_tf, filt_image, getEdges, loadBand, add_param_unit, Z_defPars, CmptPressure, CmptUPPTemperature, 
			 CmptMyMass, mydens_defPars, mydens_vikhFunction, mydens_prior, get_sz_like, mylikeFromProfs, getLikelihood, mcmc_run, add_backend_attrs, 
			 addCountCache)
from joxsz_plots import traceplot, triangle, best_fit_prof, fitwithmod, comp_rad_profs, plot_rad_profs, comp_mass_prof, mass_plot, frac_gas_prof, frac_gas_plot
from types import MethodType
import emcee as mc
from multiprocessing import Pool

#################
## Global and local variables

mystep = 2. # constant sampling step in arcsec for SZ analysis (values higher than (1/3)*FWHM of the SZ beam are not recommended)
m_e = 0.5109989*1e3 # electron rest mass (keV/c^2)
sigma_T = 6.6524587158*1e-25 # Thomson cross section (cm^2)
R_b = 5000. # Radial cluster extent (kpc), serves as upper bound for Compton y parameter integration

# Cluster cosmology
redshift = 0.888
cosmology = mb.Cosmology(redshift)
cosmology.H0 = 67.32 # Hubble's constant (km/s/Mpc)
cosmology.WM = 0.3158 # matter density
cosmology.WV = 0.6842 # vacuum density

# name for outputs
name = 'joxsz'
plotdir = './' # directory for the plots
savedir = './' # directory for saved files

# Uncertainty level
ci = 95

# MCMC parameters
nburn = 2000 # number of burn-in iterations
nlength = 5000 # number of chain iterations (after burn-in)
nwalkers = 30 # number of random walkers
nthin = 5 # thinning
seed = None # random seed

#################
### SZ

# Data
files_sz_dir = './data/SZ' # SZ data directory
beam_filename = '%s/Beam150GHz.fits' %files_sz_dir
tf_filename = '%s/TransferFunction150GHz_CLJ1227.fits' %files_sz_dir
flux_filename = '%s/press_data_cl1226_flagsource_Xraycent.dat' %files_sz_dir 
convert_filename = '%s/Compton_to_Jy_per_beam.dat' %files_sz_dir # conversion Compton -> Jy/beam

# Beam and transfer function. From raw data or Gaussian approximation?
beam_approx = False
tf_approx = False
fwhm_beam = None # fwhm of the normal distribution for the beam approximation
loc, scale, c = None, None, None # location, scale and normalization parameters of the normal cdf for the tf approximation

# Integrated Compton parameter option
calc_integ = False # apply or do not apply?
integ_mu = .94/1e3 # from Planck 
integ_sig = .36/1e3 # from Planck

#################
### X-ray

# energy bands in eV
bandEs = [[700, 1000], [1000, 1300], [1300, 1600], [1600, 2000], [2000, 2700],
	  [2700, 3400], [3400, 3800], [3800, 4300], [4300, 5000], [5000, 7000]]

# Cluster parameters
NH_1022pcm2 = 0.0183 # absorbing column density in 10^22 cm^(-2) 
Z_solar = 0.3 # assumed metallicity or, if free, any value in the valid range

# Chandra data
files_x_dir = './data/X' # X-ray data directory
rmf = '%s/source.rmf' %files_x_dir
arf = '%s/source.arf' %files_x_dir
infgtempl = files_x_dir+'/fg_profnew_%04i_%04i.dat' # foreground profile
inbgtempl = files_x_dir+'/bg_profnew_%04i_%04i.dat' # background profile

# whether to exclude unphysical masses from fit
exclude_unphy_mass = True

#################
### Code

def main():
    # setting up the elements for SZ data analysis
    phys_const = [m_e, sigma_T]
    kpc_as = cosmology.kpc_per_arcsec # number of kpc per arcsec
    flux_data = read_xy_err(flux_filename, ncol=3) # radius (arcsec), flux density, statistical error
    maxr_data = flux_data[0][-1] # highest radius in the data
    beam_2d, fwhm = mybeam(mystep, maxr_data, approx=beam_approx, filename=beam_filename, normalize=True, fwhm_beam=fwhm_beam)
    mymaxr = (maxr_data+3*fwhm)//mystep*mystep # max radius needed (arcsec)
    radius = np.arange(0., mymaxr+mystep, mystep) # array of radii in arcsec
    radius = np.append(-radius[:0:-1], radius) # from positive to entire axis
    sep = radius.size//2 # index of radius 0
    r_pp = np.arange(mystep*kpc_as, R_b+mystep*kpc_as, mystep*kpc_as) # radius in kpc used to compute the pressure profile
    d_mat = centdistmat(radius*kpc_as) # matrix of distances in kpc centered on 0 with step=mystep
    wn_as, tf = read_tf(tf_filename, approx=tf_approx, loc=loc, scale=scale, c=c) # wave number in arcsec^(-1), transmission
    filtering = filt_image(wn_as, tf, d_mat.shape[0], mystep) # transfer function matrix
    t_keV, compt_Jy_beam = np.loadtxt(convert_filename, skiprows=1, unpack=True) # Temp-dependent conversion Compton to Jy
    convert = interp1d(t_keV, 1e3*compt_Jy_beam, 'linear', fill_value='extrapolate')
    sz_data = SZ_data(phys_const, mystep, kpc_as, convert, flux_data, beam_2d, radius, sep, r_pp, d_mat, filtering, calc_integ, 
		      integ_mu, integ_sig) 
    # remove cache
    mb.xspechelper.deleteFile('countrate_cache.hdf5')

    # annuli object contains edges of annuli (one annulus for each X-ray data measurement)
    annuli = mb.Annuli(getEdges(infgtempl, bandEs), cosmology)

    # load each X-ray band, chopping outer radius
    bands = []
    for bandE in bandEs:
        bands.append(loadBand(infgtempl, inbgtempl, bandE, rmf, arf))

    # Data object represents annuli and bands
    data = mb.Data(bands, annuli)
    data.sz = sz_data # add SZ data

    # add units to parameters
    add_param_unit()

    # flat metallicity profile
    Z_cmpt = mb.CmptFlat('Z', annuli, defval=Z_solar, minval=0., maxval=1.)
    mb.CmptFlat.defPars = Z_defPars

    # density profile
    ne_cmpt = mb.CmptVikhDensity('ne', annuli, mode='single')
    # change parameter names for plotting reasons
    mb.CmptVikhDensity.vikhFunction = mydens_vikhFunction
    mb.CmptVikhDensity.defPars = mydens_defPars
    mb.CmptVikhDensity.prior = mydens_prior

    # pressure profile
    press_cmpt = CmptPressure('p', annuli)

    # temperature profile (from pressure and electron-density)
    T_cmpt = CmptUPPTemperature('T', annuli, press_cmpt, ne_cmpt)

    # Model combining density, temperature and metallicity
    model = mb.ModelNullPot(annuli, ne_cmpt, T_cmpt, Z_cmpt, NH_1022pcm2=NH_1022pcm2)

    # get default parameters
    pars = model.defPars()
    # update for pressure parameters
    pars.update(press_cmpt.defPars())

    # add parameter which allows variation of background with a Gaussian prior with sigma = 0.1
    pars['backscale'] = mb.ParamGaussian(1., prior_mu=1., prior_sigma=0.1)
    pars['calibration'] = mb.ParamGaussian(1., prior_mu=1., prior_sigma=0.07)

    # stop radii going beyond edge of data
    pars['log(r_c)'].maxval = annuli.edges_logkpc[-2]
    pars['log(r_s)'].maxval = annuli.edges_logkpc[-2]

    # some ranges of parameters to allow for the density model
    pars[r'\gamma'].val = 3.
    pars[r'\gamma'].frozen = True
    pars['log(r_c)'].val = 2.
    pars[r'\epsilon'].maxval = 10.
    pars[r'\alpha'].val = 0.
    pars[r'\alpha'].frozen = True

    # constraints on pressure parameters
    pars['c'].frozen = True

    # parameter regulating the ratio between X-ray temperature and SZ temperature
    pars['log(T_X/T_{SZ})'].frozen = False

    # do fitting of data with model
    fit = mb.Fit(pars, model, data)
    fit.thawed = [name for name, par in fit.pars.items() if not par.frozen] # change order for plotting reasons
    fit.exclude_unphy_mass = exclude_unphy_mass
    fit.savedir = savedir
    # add pressure and mass components
    fit.press = press_cmpt
    fit.mass_cmpt = CmptMyMass('m', annuli, press_cmpt, ne_cmpt)
    # update likelihood computation
    mb.Fit.get_sz_like = MethodType(get_sz_like, fit)
    mb.Fit.getLikelihood = MethodType(getLikelihood, fit)
    mb.Fit.mylikeFromProfs = MethodType(mylikeFromProfs, fit)
    #mb.countrate.CountRate.addCountCache = MethodType(addCountCache, fit.data.annuli.ctrate)
    #
    fit.doFitting()
    # save best fit
    with open('%s%s_fit.pickle' % (savedir, name), 'wb') as f:
        pickle.dump(fit, f, -1)
    #
    chainfilename = '%s%s_chain.hdf5' % (savedir, name)
    try:
        backend = mc.backends.HDFBackend(chainfilename)
        backend.reset(nwalkers, len(fit.thawedParVals()))
    except:
        pass
	
    with Pool() as pool:
        np.random.seed(seed)
        try:
             mcmc = mc.EnsembleSampler(nwalkers, len(fit.thawed), fit.getLikelihood, pool=pool, backend=backend)	
        except:
             mcmc = mc.EnsembleSampler(nwalkers, len(fit.thawed), fit.getLikelihood, pool=pool)
        mcmc.initspread = .1
        mcmc_run(mcmc, fit, nburn, nlength, nthin)
        add_backend_attrs(chainfilename, fit, nburn, nthin)
#    print('Autocorrelation: %.3f' %np.mean(mcmc.acor))
    cube_chain = mcmc.chain # (nwalkers x niter x nparams)
    flat_chain = cube_chain.reshape(-1, cube_chain.shape[2], order='F') # ((nwalkers x niter) x nparams)
    mcmc_thawed = fit.thawed # names of fitted parameters

    # Posterior distribution parameters
    param_med = np.median(flat_chain, axis=0)
    param_std = np.std(flat_chain, axis=0)
    print('{:>18}'.format('|')+'%11s' % 'Median |'+'%11s' % 'Sd |'+'%14s' % 'Unit\n'+'-'*53)
    for i in range(len(mcmc_thawed)):
        print('{:>18}'.format('%s |' %mcmc_thawed[i])+'%9s |' %format(param_med[i], '.3f')+
              '%9s |' %format(param_std[i], '.3f')+'%13s' % [pars[n].unit for n in mcmc_thawed][i])

    #################
    ### Plots

    # Bayesian diagnostics
    traceplot(cube_chain, mcmc_thawed, seed=None, plotdir=plotdir)
    triangle(flat_chain, mcmc_thawed, plotdir=plotdir)

    # Best fitting profiles on SZ and X-ray surface brightness
    perc_x, perc_sz = best_fit_prof(cube_chain, fit, num='all', seed=seed, ci=ci)
    fitwithmod(data, perc_x, perc_sz, ci=ci, plotdir=plotdir)

    # Main thermodynamic radial profiles (density, temperature(s), pressure, entropy, cooling time, gas mass)
    dens, temp, prss, entr, cool, gmss, xtmp = comp_rad_profs(cube_chain, fit, num='all', seed=seed, ci=ci)
    plot_rad_profs(r_pp, dens, temp, prss, entr, cool, gmss, xtmp, xmin=100., xmax=1000., plotdir=plotdir)

    # Cumulative total mass profile
    mass_prof, r_delta, m_delta = comp_mass_prof(cube_chain, fit, num='all', seed=seed, delta=500, start_opt=1000., ci=ci)
    mass_plot(r_pp, mass_prof, cosmology, delta=500, r_delta=r_delta, m_delta=m_delta, xmin=100., xmax=1500., plotdir=plotdir)

    # Gas fraction profile
    f_gas = frac_gas_prof(cube_chain, fit, num='all', seed=seed, ci=ci)
    frac_gas_plot(r_pp, f_gas, ci=ci, plotdir=plotdir)

if __name__ == '__main__':
    main()
