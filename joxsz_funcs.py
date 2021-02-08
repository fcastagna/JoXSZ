from astropy.io import fits
import numpy as np
from scipy import optimize
from scipy.stats import norm
import mbproj2 as mb
from astropy import units as u
from astropy import constants as const
from mbproj2.physconstants import keV_erg, kpc_cm, mu_g, G_cgs, solar_mass_g
from abel.direct import direct_transform
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.fftpack import fft2, ifft2
from scipy.integrate import simps
import time

def read_xy_err(filename, ncol, units):
    '''
    Read the data from FITS or ASCII file
    -------------------------------------
    ncol = number of columns to read
    '''
    if filename[filename.find('.', -5)+1:] == 'fits':
        data = fits.open(filename)[''].data[0]
    elif filename[filename.find('.', -5)+1:] in ('txt', 'dat'):
        data = np.loadtxt(filename, unpack=True)
    else:
        raise RuntimeError('Unrecognised file extension (not in fits, dat, txt)')
    return list(map(lambda x, y: x*y, data[:ncol], units))

def read_beam(filename):
    '''
    Read the beam data from the specified file up to the first negative or nan value
    --------------------------------------------------------------------------------
    '''
    radius, beam_prof = read_xy_err(filename, ncol=2, units=[u.arcsec, u.beam])
    if np.isnan(beam_prof).sum() > 0.:
        first_nan = np.where(np.isnan(beam_prof))[0][0]
        radius = radius[:first_nan]
        beam_prof = beam_prof[:first_nan]
    if beam_prof.min() < 0.:
        first_neg = np.where(beam_prof < 0.)[0][0]
        radius = radius[:first_neg]
        beam_prof = beam_prof[:first_neg]
    return radius, beam_prof

def mybeam(step, maxr_data, approx=False, filename=None, normalize=True, fwhm_beam=None):
    '''
    Set the 2D image of the beam, alternatively from file data or from a normal distribution with given FWHM
    --------------------------------------------------------------------------------------------------------
    step = binning step
    maxr_data = highest radius in the data
    approx = whether to approximate or not the beam to the normal distribution (boolean, default is False)
    filename = name of the file including the beam data
    normalize = whether to normalize or not the output 2D image (boolean, default is True)
    fwhm_beam = Full Width at Half Maximum
    -------------------------------------------------------------------
    RETURN: the 2D image of the beam and his Full Width at Half Maximum
    '''
    if not approx:
        r_irreg, b = read_beam(filename)
        f = interp1d(np.append(-r_irreg, r_irreg), np.append(b, b), 'cubic', bounds_error=False, fill_value=(0., 0.))
        inv_f = lambda x: f(x)-f(0.)/2
        fwhm_beam = 2*optimize.newton(inv_f, x0=5.)*r_irreg.unit
    maxr = (maxr_data+3*fwhm_beam)//step*step
    rad = np.arange(0., (maxr+step).value, step.value)*step.unit
    rad = np.append(-rad[:0:-1], rad)
    rad_cut = rad[np.where(abs(rad) <= 3*fwhm_beam)]
    beam_mat = centdistmat(rad_cut)
    if approx:
        sigma_beam = fwhm_beam/(2*np.sqrt(2*np.log(2)))
        beam_2d = norm.pdf(beam_mat, loc=0., scale=sigma_beam)
    else:
        beam_2d = f(beam_mat)
    if normalize:
        beam_2d /= beam_2d.sum()*step.value**2
    return beam_2d*u.beam, fwhm_beam

def centdistmat(r, offset=0.):
    '''
    Create a symmetric matrix of distances from the radius vector
    -------------------------------------------------------------
    r = vector of negative and positive distances with a given step (center value has to be 0)
    offset = value to be added to every distance in the matrix (default is 0)
    ---------------------------------------------
    RETURN: the matrix of distances centered on 0
    '''
    x, y = np.meshgrid(r, r)
    return np.sqrt(x**2+y**2)+offset

def read_tf(filename, approx=False, loc=0., scale=0.02, c=0.95):
    '''
    Read the transfer function data from the specified file
    -------------------------------------------------------
    approx = whether to approximate or not the tf to the normal cdf (boolean, default is False)
    loc, scale, c = location, scale and normalization parameters for the normal cdf approximation
    ---------------------------------------------------------------------------------------------
    RETURN: the vectors of wave numbers and transmission values
    '''
    wn, tf = read_xy_err(filename, ncol=2, units=[u.arcsec**-1, u.Unit('')]) # wave number, transmission
    if approx:
        tf = c*norm.cdf(wn, loc, scale)
    return wn, tf

def dist(naxis):
    '''
    Returns a matrix in which the value of each element is proportional to its frequency 
    (https://www.harrisgeospatial.com/docs/DIST.html)
    If you shift the 0 to the centre using fftshift, you obtain a symmetric matrix
    ------------------------------------------------------------------------------------
    naxis = number of elements per row and per column
    -------------------------------------------------
    RETURN: the (naxis x naxis) matrix
    '''
    axis = np.linspace(-naxis//2+1, naxis//2, naxis)
    result = np.sqrt(axis**2+axis[:,np.newaxis]**2)
    return np.roll(result, naxis//2+1, axis=(0, 1))

def filt_image(wn_as, tf, side, step):
    '''
    Create the 2D filtering image from the transfer function data
    -------------------------------------------------------------
    wn_as = vector of wave numbers in arcsec
    tf = transmission data
    side = one side length for the output image
    step = binning step
    -------------------------------
    RETURN: the (side x side) image
    '''
    f = interp1d(wn_as, tf, 'cubic', bounds_error=False, fill_value=tuple([tf[0], tf[-1]])) # tf interpolation
    kmax = 1/step
    karr = (dist(side)/side)*u.Unit('')
    karr /= karr.max()
    karr *= kmax
    return f(karr)

class SZ_data:
    '''
    Class for the SZ data required for the analysis
    -----------------------------------------------
    step = binning step
    kpc_as = kpc in arcsec
    convert = interpolation function for the temperature-dependent conversion Compton to mJy
    flux_data = radius (arcsec), flux density, statistical error
    beam_2d = 2D image of the beam
    radius = array of radii in arcsec
    sep = index of radius 0
    r_pp = radius in kpc used to compute the pressure profile
    d_mat = matrix of distances in kpc centered on 0 with given step
    filtering = transfer function matrix
    calc_integ = whether to include integrated Compton parameter in the likelihood (boolean, default is False)
    integ_mu = if calc_integ == True, prior mean
    integ_sig = if calc_integ == True, prior sigma
    '''
    def __init__(self, step, kpc_as, convert, flux_data, beam_2d, radius, sep, r_pp, d_mat, filtering, calc_integ=False,
                 integ_mu=None, integ_sig=None):
        self.step = step
        self.kpc_as = kpc_as
        self.convert = convert
        self.flux_data = flux_data
        self.beam_2d = beam_2d
        self.radius = radius
        self.sep = sep
        self.r_pp = r_pp
        self.d_mat = d_mat
        self.filtering = filtering
        self.calc_integ = calc_integ
        self.integ_mu = integ_mu
        self.integ_sig = integ_sig

def getEdges(infg, bands, units):
    '''
    Get edges of annuli in arcmin
    -----------------------------
    infg = foreground profiles file name
    bands = energy band in eV
    ---------------------------------
    RETURN: edges of annuli in arcmin
    '''
    data = np.loadtxt(infg % (bands[0][0].value, bands[0][1].value))*units
    radii = list(map(lambda x: x.value, data[:,0]))*units[0] # radii of centres of annuli in arcmin
    hws = list(map(lambda x: x.value, data[:,1]))*units[1] # half-width of annuli in arcmin
    return np.hstack((data[0,0]-data[0,1], radii+hws))

def loadBand(infg, inbg, bandE, rmf, arf, units_fg, units_bg):
    '''
    Load foreground and background profiles from file and construct band object
    ---------------------------------------------------------------------------
    infg = foreground profiles file name
    inbg = background profiles file name
    bandE = energy band in eV
    rmf = Redistribution Matrix File name
    arf = Auxiliary Response File name
    -------------------------------------
    RETURN: Band object 
    '''
    data = np.loadtxt(infg % (bandE[0].value, bandE[1].value))*units_fg # foreground profile
    radii = list(map(lambda x: x.value, data[:,0]))*units_fg[0] # radii of centres of annuli in arcmin
    hws = list(map(lambda x: x.value, data[:,1]))*units_fg[1] # half-width of annuli in arcmin
    cts = list(map(lambda x: x.value, data[:,2]))*units_fg[2] # number of counts (integer)
    areas = list(map(lambda x: x.value, data[:,3]))*units_fg[3] # areas of annuli, taking account of pixelization (arcmin^2)
    exps = list(map(lambda x: x.value, data[:,4]))*units_fg[4] # exposures (s)
    # note: vignetting can be input into exposure or areas, but background needs to be consistent
    geomareas = np.pi*((radii+hws)**2-(radii-hws)**2) # geometric area factor
    areascales = areas/geomareas # ratio between real pixel area and geometric area
    band = mb.Band(bandE[0]/1000, bandE[1]/1000, cts, rmf, arf, exps, areascales=areascales)
    backd = np.loadtxt(inbg % (bandE[0].value, bandE[1].value))*units_bg # background profile
    band.backrates = list(map(lambda x: x.value, backd[0:radii.size,4]))*units_bg[4] # rates in (cts/s/arcmin^2)
    lastmyrad = list(map(lambda x: x.value, backd[0:radii.size,0]))*units_bg[0]
    if (abs(lastmyrad[-1]-radii[-1]).value > .001):
         raise RuntimeError('Problem while reading bg file', lastmyrad[-1], radii[-1])
    return band

def add_param_unit():
    '''
    Adapt the param definition to include the unit measure
    ------------------------------------------------------
    '''
    def param_new_init(self, val, minval=-1e99, maxval=1e99, unit='', frozen=False):
        mb.ParamBase.__init__(self, val, frozen=frozen)
        self.minval = minval
        self.maxval = maxval        
        self.unit = unit
    def param_new_repr(self):
        return '<Param: val=%.3g, minval=%.3g, maxval=%.3g, unit=%s, frozen=%s>' % (
            self.val, self.minval, self.maxval, self.unit, self.frozen)
    mb.Param.__init__ = param_new_init
    mb.Param.__repr__ = param_new_repr
    def pargau_new_init(self, val, prior_mu, prior_sigma, unit='', frozen=False):
        mb.ParamBase.__init__(self, val, frozen=frozen)
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.unit = unit        
    def pargau_new_repr(self):
        return '<ParamGaussian: val=%.3g, prior_mu=%.3g, prior_sigma=%.3g, unit=%s, frozen=%s>' % (
            self.val, self.prior_mu, self.prior_sigma, self.unit, self.frozen)    
    mb.ParamGaussian.__init__ = pargau_new_init
    mb.ParamGaussian.__repr__ = pargau_new_repr

def Z_defPars(self):
    '''
    Update defpars function of metallicity object to include unit measure
    ---------------------------------------------------------------------
    '''
    return {self.name: mb.Param(self.defval, unit='solar', minval=self.minval, maxval=self.maxval)}

class CmptPressure(mb.Cmpt):
    '''
    Class to parametrize the pressure profile
    -----------------------------------------    
    '''
    def __init__(self, name, annuli):
        mb.Cmpt.__init__(self, name, annuli)

    def defPars(self):
        '''
        Default parameter values
        ------------------------
        P_0 = normalizing constant (keV.cm^{-3})
        a = rate of turnover between b and c
        b = logarithmic slope at r/r_p >> 1
        c = logarithmic slope at r/r_p << 1
        r_p = characteristic radius (kpc)
        '''        
        pars = {
            'P_0': mb.Param(0.4, minval=0., maxval=2., unit='keV cm-3'),
            'a': mb.Param(1.33, minval=0.1, maxval=20., unit=''),
            'b': mb.Param(4.13, minval=0.1, maxval=15., unit=''),
            'c': mb.Param(0.014, minval=0., maxval=3., unit=''),
            'r_p': mb.Param(300., minval=100., maxval=3000., unit='kpc')
        }
        return pars

    def press_fun(self, pars, r_kpc):
        '''
        Compute the gNFW pressure profile
        ---------------------------------
        pars = set of pressure parameters
        r_kpc = radius (kpc)
        '''
        P_0, r_p, a, b, c = map(lambda x: pars[x].val*u.Unit(pars[x].unit), 
                                ['P_0', 'r_p', 'a', 'b', 'c'])
        return P_0/((r_kpc/r_p)**c*(1+(r_kpc/r_p)**a)**((b-c)/a))

    def press_derivative(self, pars, r_kpc):
        '''
        Compute the gNFW pressure profile first derivative
        --------------------------------------------------
        pars = set of pressure parameters
        r_kpc = radius (kpc)
        '''
        P_0, r_p, a, b, c = map(lambda x: pars[x].val*u.Unit(pars[x].unit), 
                                ['P_0', 'r_p', 'a', 'b', 'c'])
        return -P_0*(c+b*(r_kpc/r_p)**a)/(r_p*(r_kpc/r_p)**(c+1)*(1+(r_kpc/r_p)**a)**((b-c+a)/a))

class CmptUPPTemperature(mb.Cmpt):
    '''
    Class to derive the temperature profile as pressure profile over density profile (ideal gas law)
    ------------------------------------------------------------------------------------------------
    '''    
    def __init__(self, name, annuli, press_prof, ne_prof):
        mb.Cmpt.__init__(self, name, annuli)
        self.press_prof = press_prof
        self.ne_prof = ne_prof

    def defPars(self):
        '''
        Default parameter T_ratio = T_X / T_SZ
        --------------------------------------
        '''
        pars = {'log(T_X/T_{SZ})': mb.Param(0., minval=-1., maxval=1., unit='')}
        return pars
    
    def temp_fun(self, pars, r_kpc, getT_SZ=False):
        '''
        Compute the temperature profile (X-ray by default, SZ alternatively).
        Please note that T_X = T_SZ if log(T_ratio) is not fitted but fixed to 0!
        -------------------------------------------------------------------------
        getT_SZ = whether to return T_SZ (boolean, default is False)
        '''
        pr = self.press_prof.press_fun(pars, r_kpc)
        ne = self.ne_prof.vikhFunction(pars, r_kpc)
        T_SZ = pr/ne
        if getT_SZ:
            return T_SZ
        else:
            T_XpT_SZ = 10**pars['log(T_X/T_{SZ})'].val
            T_X = T_SZ*T_XpT_SZ
            return T_X
        
    def computeProf(self, pars):
        return self.temp_fun(pars, self.annuli.midpt_kpc)

def mydens_defPars(self):
    '''
    Default density profile parameters.
    Copied from MBProj2 changing the parameter names for plotting reasons
    ---------------------------------------------------------------------
    n_0 = normalizing constant (cm^{-3})
    r_c = core radius (kpc)
    alpha = logarithmic slope at r/r_c << 1
    beta = shape parameter for the isothermal β-model
    r_s = scale radius (radius at which the density profile steepens with respect to the traditional β-model) (kpc)
    gamma = width of the transition region
    epsilon = change of slope near r_s
    #
    n_02 = additive constant (cm^{-3})
    r_c2 = small core radius (kpc)
    beta_2 = shape parameter
    '''
    pars = {
        'log(n_0)': mb.Param(-3., minval=-7., maxval=2., unit='cm-3'),
        r'\beta': mb.Param(2/3, minval=0., maxval=4., unit=''),
        'log(r_c)': mb.Param(2.3, minval=-1., maxval=3.7, unit='kpc'),
        'log(r_s)': mb.Param(2.7, minval=0., maxval=3.7, unit='kpc'),
        r'\alpha': mb.Param(0., minval=-1., maxval=2., unit=''),
        r'\epsilon': mb.Param(3., minval=0., maxval=5., unit=''),
        r'\gamma': mb.Param(3., minval=0., maxval=10., frozen=True, unit='')
        }
    if self.mode == 'double':
        pars.update({
            'log(n_{02})': mb.Param(-1., minval=-7., maxval=2., unit='cm-3'),
            r'\beta_2': mb.Param(0.5, minval=0., maxval=4., unit=''),
            'log(r_{c2})': mb.Param(1.7, minval=-1., maxval=3.7, unit='kpc')
            })
    return pars

def mydens_vikhFunction(self, pars, radii_kpc):
    '''
    Compute the Vikhlinin density profile.
    Copied from MBProj2 changing the parameter names for plotting reasons
    ---------------------------------------------------------------------
    '''
    n_0, r_c, r_s = map(lambda x: 10**pars[x].val*u.Unit(pars[x].unit), ['log(n_0)', 'log(r_c)', 'log(r_s)'])
    beta, alpha, epsilon, gamma = map(lambda x: pars[x].val*u.Unit(pars[x].unit), [r'\beta', r'\alpha', r'\epsilon', r'\gamma'])
    r = radii_kpc
    res_sq = n_0**2*(r/r_c)**(-alpha)/((1+(r/r_c)**2)**(3*beta-alpha/2)*(1+(r/r_s)**gamma)**(epsilon/gamma))
    if self.mode == 'double':
        n_02, r_c2 = map(lambda x: 10**pars[x].val*u.Unit(pars[x].unit), ['log(n_{02})', 'log(r_{c2})'])
        beta_2 = pars[r'\beta_2'].val*u.Unit(pars[r'\beta_2'].unit)
        res_sq += n_02**2/(1+(r/r_c2)**2)**(3*beta_2)
    return np.sqrt(res_sq)

def mydens_prior(self, pars):
    '''
    Density profile parameters prior.
    Copied from MBProj2 changing the parameter names for plotting reasons
    ---------------------------------------------------------------------
    '''   
    r_c = 10**pars['log(r_c)'].val
    r_s = 10**pars['log(r_s)'].val
    if r_c > r_s:
        return -np.inf
    return 0.

class CmptMyMass(mb.Cmpt):
    '''
    Class to compute the mass profile under the hydrostatic equilibrium assumption
    ------------------------------------------------------------------------------
    '''        
    def __init__(self, name, annuli, press_prof, ne_prof):
        mb.Cmpt.__init__(self, name, annuli)
        self.press_prof = press_prof
        self.ne_prof = ne_prof

    def defPars(self):
        '''
        Default parameter values from pressure and density profiles
        -----------------------------------------------------------
        '''
        pars = self.press_prof.defPars()
        pars.update(self.ne_prof.defPars())
        return pars
    
    def mass_fun(self, pars, r_kpc, mu_gas=0.61):
        '''
        Compute the mass profile
        ------------------------
        '''
        dpr_kpc = self.press_prof.press_derivative(pars, r_kpc)
        dpr_cm = dpr_kpc*keV_erg/kpc_cm
        ne = self.ne_prof.vikhFunction(pars, r_kpc)
        r_cm = r_kpc*kpc_cm
        return -dpr_cm*r_cm**2/(mu_gas*mu_g*ne*G_cgs)/solar_mass_g
    
def get_sz_like(self, output='ll'):
    '''
    Computes the log-likelihood on SZ data for the current parameters
    -----------------------------------------------------------------
    output = desired output
        'll' = log-likelihood
        'chisq' = Chi-Squared
        'pp' = pressure profile
        'bright' = surface brightness profile
        'integ' = integrated Compton parameter (only if calc_integ == True)
    ----------------------
    RETURN: desired output
    '''
    # pressure profile
    pp = self.press.press_fun(self.pars, self.data.sz.r_pp)
    if output == 'pp':
        return pp
    # abel transform
    ab = direct_transform(pp.value, r=self.data.sz.r_pp.value, direction='forward', backend='Python')*pp.unit*self.data.sz.r_pp.unit
    # Compton parameter
    y = (const.sigma_T/(const.m_e*const.c**2)*ab).to('')
    f = interp1d(np.append(-self.data.sz.r_pp, self.data.sz.r_pp), np.append(y, y), 'cubic', bounds_error=False, fill_value=(0., 0.))
    # Compton parameter 2D image
    y_2d = f(self.data.sz.d_mat)*u.Unit('')
    # Convolution with the beam
    conv_2d = fftconvolve(y_2d, self.data.sz.beam_2d, 'same')*self.data.sz.step**2
    # Convolution with the transfer function
    FT_map_in = fft2(conv_2d)
    map_out = np.real(ifft2(FT_map_in*self.data.sz.filtering))
    # Temperature-dependent conversion from Compton parameter to mJy/beam
    t_prof = self.model.T_cmpt.temp_fun(self.pars, self.data.sz.r_pp[:self.data.sz.sep], getT_SZ=True)
    h = interp1d(np.append(-self.data.sz.r_pp[:self.data.sz.sep], self.data.sz.r_pp[:self.data.sz.sep]),
                 np.append(t_prof, t_prof), 'cubic', bounds_error=False, fill_value=(t_prof[-1], t_prof[-1]))
    map_prof = map_out[conv_2d.shape[0]//2, 
                       conv_2d.shape[0]//2:]*self.data.sz.convert(np.append(h(0.)*t_prof.unit, t_prof))*self.data.sz.convert.unit[1]*self.pars['calibration'].val
    if output == 'bright':
        return map_prof.to(self.data.sz.flux_data[1].unit)
    g = interp1d(self.data.sz.radius[self.data.sz.sep:], map_prof, 'cubic', fill_value='extrapolate')
    # Log-likelihood calculation
    chisq = np.nansum(((self.data.sz.flux_data[1]-(g(self.data.sz.flux_data[0])*map_prof.unit).to(self.data.sz.flux_data[1].unit))/self.data.sz.flux_data[2])**2)
    log_lik = -chisq/2
    if self.data.sz.calc_integ:
        x = np.arange(0., (self.data.sz.r_pp[-1]/self.kpc_as+self.data.sz.step).to('arcmin').value, self.data.sz.step.to('arcmin').value)*u.arcmin
        cint = simps(np.concatenate((f(0.), y), axis=None)*x, x)*2*np.pi
        new_chi = np.nansum(((cint-self.data.sz.integ_mu)/self.data.sz.integ_sig)**2)
        log_lik -= new_chi/2
        if output == 'integ':
            return cint
    if output == 'll':
        return log_lik
    elif output == 'chisq':
        return chisq
    else:
        raise RuntimeError('Unrecognised output name (must be "ll", "chisq", "pp", "bright" or "integ")')

def mylikeFromProfs(self, predprofs):
    '''
    Computes the X-ray log-likelihood for the current parameters
    Copied from MBProj2, allowing the computation even in case of missing data
    --------------------------------------------------------------------------
    predprofs = input profiles
    '''  
    likelihood = 0.
    for band, predprof in zip(self.data.bands, predprofs):
        likelihood += mb.utils.cashLogLikelihood(band.cts[~np.isnan(band.cts)], predprof[~np.isnan(band.cts)])
    return likelihood

def getLikelihood(self, vals=None):
    '''
    Computes the joint X-SZ log-likelihood for the current parameters
    -----------------------------------------------------------------------------------------------------------------------------
    RETURN: the log-likelihood value or -inf in case of parameter values not included in the parameter space or in case of set of
            parameters yielding unphysical mass profiles
    '''
    # update parameters
    if vals is not None:
        self.updateThawed(vals)
    # prior on parameters (-inf if at least one parameter value is out of the parameter space)
    parprior = sum((self.pars[p].prior() for p in self.pars))
    if not np.isfinite(parprior):
        return -np.inf
    # exclude unphysical mass profiles
    if self.exclude_unphy_mass:
        m_prof = self.mass_cmpt.mass_fun(self.pars, self.data.sz.r_pp) 
        if not(all(np.gradient(m_prof, 1) > 0.)):
            return -np.inf
    # X-ray fitted profiles
    profs = self.calcProfiles()
    # X-ray log-likelihood
    if np.array(profs).min() > 0.:
        like = self.mylikeFromProfs(profs)
    else:
        like = -np.inf
    # SZ log-likelihood
    sz_like = self.get_sz_like()
    # prior on parameters 
    prior = self.model.prior(self.pars)+parprior
    # JoXSZ log-likelihood
    totlike = float(like+prior+sz_like)
    # save best fitting parameters
    if mb.fit.debugfit and (totlike-self.bestlike) > 0.1:
        self.bestlike = totlike
        with mb.utils.AtomicWriteFile("%s/fit.dat" % self.savedir) as fout:
            mb.utils.uprint("likelihood = %g + %g + %g = %g" % (like, sz_like, prior, totlike), file=fout)
            for p in sorted(self.pars):
                mb.utils.uprint("%s = %s" % (p, self.pars[p]), file=fout)
    return totlike

def mcmc_run(mcmc, nburn, nsteps, nthin=1, comp_time=True, autorefit=True, minfrac=0.2, minimprove=0.01):
    '''
    MCMC execution. Adapted from MBProj2
    ------------------------------------
    mcmc = MCMC object (adapted from MBProj2)
    nburn = number of burn-in iterations
    nsteps = number of iterations after burn-in
    nthin = thinning
    comp_time = shows the computation time (boolean, default is True)
    autorefit = refit position if new minimum is found during burn in (boolean, default is True)
    minfrac = minimum fraction of burn in to do if new minimum found
    minimprove = minimum improvement in fit statistic to do a new fit
    '''
    def _generateInitPars(mcmc):
        '''
        Generate initial set of parameters from fit
        -------------------------------------------
        '''
        thawedpars = np.array(mcmc.fit.thawedParVals())
        assert np.all(np.isfinite(thawedpars))
        # create enough parameters with finite likelihoods
        p0 = []
        _ = 0
        while len(p0) < mcmc.walkers:
            if mcmc.seed is not None:
                _ += 1
                np.random.seed(mcmc.seed*_)
            p = thawedpars*(1+np.random.normal(0., mcmc.initspread, size=mcmc.numpars))
            if np.isfinite(mcmc.fit.getLikelihood(p)):
                p0.append(p)
        return p0
    def innerburn():
        '''
        Return False if new minimum found and autorefit is set. Adapted from MBProj2
        ----------------------------------------------------------------------------
        '''
        bestfit = None
        bestprob = initprob = mcmc.fit.getLikelihood(mcmc.fit.thawedParVals())
        p0 = _generateInitPars(mcmc)
        mcmc.header['burn'] = nburn
        for i, result in enumerate(mcmc.sampler.sample(p0, thin=nthin, iterations=nburn, storechain=False)):
            if i%10 == 0:
                print(' Burn %i / %i (%.1f%%)' %(i, nburn, i*100/nburn))
            mcmc.pos0, lnprob, rstate0 = result[:3]
            if lnprob.max()-bestprob > minimprove:
                bestprob = lnprob.max()
                maxidx = lnprob.argmax()
                bestfit = mcmc.pos0[maxidx]
            if (autorefit and i > nburn*minfrac and bestfit is not None):
                print('Restarting burn as new best fit has been found (%g > %g)' % (bestprob, initprob))
                mcmc.fit.updateThawed(bestfit)
                mcmc.sampler.reset()
                return False
        mcmc.sampler.reset()
        return True
    time0 = time.time()
    print('Starting burn-in')
    while not innerburn():
        print('Restarting, as new mininimum found')
        mcmc.fit.doFitting()
    print('Finished burn-in')
    mcmc.header['length'] = nsteps
    if mcmc.pos0 is None:
        print(' Generating initial parameters')
        p0 = _generateInitPars(mcmc)
    else:
        print(' Starting from end of burn-in position')
        p0 = mcmc.pos0
    for i, result in enumerate(mcmc.sampler.sample(p0, thin=nthin, iterations=nsteps)):
        if i%10 == 0:
            print(' Sampling %i / %i (%.1f%%)' %(i, nsteps, i*100/nsteps))
    print('Finished sampling')
    time1 = time.time()
    if comp_time:
        h, rem = divmod(time1-time0, 3600)
        print('Computation time: '+str(int(h))+'h '+str(int(rem//60))+'m')
    print('Acceptance fraction: %s' %np.mean(mcmc.sampler.acceptance_fraction))
