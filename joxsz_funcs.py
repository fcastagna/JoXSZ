from astropy.io import fits
import numpy as np
from scipy import optimize
from scipy.stats import norm
import mbproj2 as mb
from mbproj2.physconstants import keV_erg, kpc_cm, mu_g, G_cgs, solar_mass_g, ne_nH, Mpc_cm, yr_s, mu_e, Mpc_km
from abel.direct import direct_transform
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.fftpack import fft2, ifft2
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import corner

plt.style.use('classic')

class SZ_data:
    '''
    Class for SZ data used in PreProFit
    -----------------------------------
    '''
    def __init__(self, phys_const, step, kpc_as, convert, flux_data, beam_2d, radius, sep, r_pp, ub, d_mat, filtering):
        self.phys_const = phys_const
        self.step = step
        self.kpc_as = kpc_as
        self.convert = convert
        self.flux_data = flux_data
        self.beam_2d = beam_2d
        self.radius = radius
        self.sep = sep
        self.r_pp = r_pp
        self.ub = ub
        self.d_mat = d_mat
        self.filtering = filtering

def read_xy_err(filename, ncol):
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
    return data[:ncol]

def read_beam(filename):
    '''
    Read the beam data from the specified file up to the first negative or nan value
    --------------------------------------------------------------------------------
    '''
    radius, beam_prof = read_xy_err(filename, ncol=2)
    if np.isnan(beam_prof).sum() > 0:
        first_nan = np.where(np.isnan(beam_prof))[0][0]
        radius = radius[:first_nan]
        beam_prof = beam_prof[:first_nan]
    if beam_prof.min() < 0:
        first_neg = np.where(beam_prof < 0)[0][0]
        radius = radius[:first_neg]
        beam_prof = beam_prof[:first_neg]
    return radius, beam_prof

def mybeam(step, maxr_data, approx=False, filename=None, normalize=True, fwhm_beam=None):
    '''
    Set the 2D image of the beam, alternatively from file data or from a normal distribution with given FWHM
    --------------------------------------------------------------------------------------------------------
    step = binning step
    maxr_data = highest radius in the data
    approx = whether to approximate or not the beam to the normal distribution (True/False)
    filename = name of the file including the beam data
    normalize = whether to normalize or not the output 2D image (True/False)
    fwhm_beam = Full Width at Half Maximum
    -------------------------------------------------------------------
    RETURN: the 2D image of the beam and his Full Width at Half Maximum
    '''
    if not approx:
        r_irreg, b = read_beam(filename)
        f = interp1d(np.append(-r_irreg, r_irreg), np.append(b, b), 'cubic', bounds_error=False, fill_value=(0, 0))
        inv_f = lambda x: f(x)-f(0)/2
        fwhm_beam = 2*optimize.newton(inv_f, x0=5) 
    maxr = (maxr_data+3*fwhm_beam)//step*step
    rad = np.arange(0, maxr+step, step)
    rad = np.append(-rad[:0:-1], rad)
    rad_cut = rad[np.where(abs(rad) <= 3*fwhm_beam)]
    beam_mat = centdistmat(rad_cut)
    if approx:
        sigma_beam = fwhm_beam/(2*np.sqrt(2*np.log(2)))
        beam_2d = norm.pdf(beam_mat, loc=0., scale=sigma_beam)
    else:
        beam_2d = f(beam_mat)
    if normalize:
        beam_2d /= beam_2d.sum()*step**2
    return beam_2d, fwhm_beam

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

def read_tf(filename, approx=False, loc=0, scale=0.02, c=0.95):
    '''
    Read the transfer function data from the specified file
    -------------------------------------------------------
    approx = whether to approximate or not the tf to the normal cdf (True/False)
    loc, scale, c = location, scale and normalization parameters for the normal cdf approximation
    ---------------------------------------------------------------------------------------------
    RETURN: the vectors of wave numbers and transmission values
    '''
    wn, tf = read_xy_err(filename, ncol=2) # wave number, transmission
    if approx:
        tf = c*norm.cdf(wn, loc, scale)
    return wn, tf

def dist(naxis):
    '''
    Returns a symmetric matrix in which the value of each element is proportional to its frequency 
    (https://www.harrisgeospatial.com/docs/DIST.html)
    ----------------------------------------------------------------------------------------------
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
    f = interp1d(wn_as, tf, bounds_error=False, fill_value=tuple([tf[0], tf[-1]])) # tf interpolation
    kmax = 1/step
    karr = dist(side)/side*kmax
    return f(karr)#(np.rot90(np.rot90(karr)))

def getEdges(infg, bands):
    '''
    Get edges of annuli in arcmin
    -----------------------------
    infg = foreground profiles file name
    bands = energy band in eV
    ---------------------------------
    RETURN: edges of annuli in arcmin
    '''
    data = np.loadtxt(infg %(bands[0][0], bands[0][1]))
    return np.hstack((data[0,0]-data[0,1], data[:,0]+data[:,1]))

def loadBand(infg, inbg, bandE, rmf, arf):
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
    data = np.loadtxt(infg %(bandE[0], bandE[1])) # foreground profile
    radii = data[:,0] # radii of centres of annuli in arcmin
    hws = data[:,1] # half-width of annuli in arcmin
    cts = data[:,2] # number of counts (integer)
    areas = data[:,3] # areas of annuli, taking account of pixelization (arcmin^2)
    exps = data[:,4] # exposures (s)
    # note: vignetting can be input into exposure or areas, but background needs to be consistent
    geomareas = np.pi*((radii+hws)**2-(radii-hws)**2) # geometric area factor
    areascales = areas/geomareas # ratio between real pixel area and geometric area
    band = mb.Band(bandE[0]/1000, bandE[1]/1000, cts, rmf, arf, exps, areascales=areascales)
    backd = np.loadtxt(inbg % (bandE[0], bandE[1])) # background profile
    band.backrates = backd[0:radii.size, 4] # rates in (cts/s/arcmin^2)
    lastmyrad = backd[0:radii.size, 0]
    if (abs(lastmyrad[-1]-radii[-1]) > .001):
         raise RuntimeError('Problem while reading bg file', lastmyrad[-1], radii[-1])
    return band

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
        P_0 = normalizing constant
        a = rate of turnover between b and c
        b = logarithmic slope at r/r_p >> 1
        c = logarithmic slope at r/r_p << 1
        r_p = characteristic radius
        '''        
        pars = {
            'P_0': mb.Param(0.4, minval=0, maxval=20),
            'a': mb.Param(1.33, minval=0.1, maxval=10),
            'b': mb.Param(4.13, minval=0.1, maxval=15),
            'c': mb.Param(0.014, minval=0, maxval=3),
            'r_p': mb.Param(300., minval=100., maxval=1000.)
            }
        return pars

    def press_fun(self, pars, r_kpc):
        '''
        Compute the gNFW pressure profile
        ---------------------------------
        pars = set of pressure parameters
        r_kpc = radius (kpc)
        '''
        P_0 = pars['P_0'].val
        r_p = pars['r_p'].val
        a = pars['a'].val
        b = pars['b'].val
        c = pars['c'].val
        return P_0/((r_kpc/r_p)**c*(1+(r_kpc/r_p)**a)**((b-c)/a))

    def press_derivative(self, pars, r_kpc):
        '''
        Compute the gNFW pressure profile first derivative
        --------------------------------------------------
        pars = set of pressure parameters
        r_kpc = radius (kpc)
        '''
        P_0 = pars['P_0'].val
        r_p = pars['r_p'].val
        a = pars['a'].val
        b = pars['b'].val
        c = pars['c'].val
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
        pars = {'log(T_{ratio})': mb.Param(0, minval=-1, maxval=1)}
        return pars
    
    def temp_fun(self, pars, r_kpc, getT_SZ=False):
        '''
        Compute the temperature profile (X-ray by default, SZ alternatively).
        Please note that T_X = T_SZ if log(T_ratio) is not fitted but fixed to 0!
        -------------------------------------------------------------------------
        getT_SZ = whether to return T_SZ (True/False, default is False)
        '''
        pr = self.press_prof.press_fun(pars, r_kpc)
        ne = self.ne_prof.vikhFunction(pars, r_kpc)
        T_SZ = pr/ne
        if getT_SZ:
            return T_SZ
        else:
            T_XpT_SZ = 10**pars['log(T_{ratio})'].val
            T_X = T_SZ*T_XpT_SZ
            return T_X
        
    def computeProf(self, pars):
        return self.temp_fun(pars, self.annuli.midpt_kpc)


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
    
def mydens_defPars(self):
    '''
    Default density profile parameters.
    Copied from MBProj2 changing the parameter names for plotting reasons
    ---------------------------------------------------------------------
    n_0 = normalizing constant 
    r_c = core radius
    alpha = logarithmic slope at r/r_c << 1
    beta = shape parameter for the isothermal β-model
    r_s = scale radius (radius at which the density profile steepens with respect to the traditional β-model)
    gamma = width of the transition region
    epsilon = change of slope near r_s
    #
    n_02 = additive constant
    r_c2 = small core radius
    beta_2 = shape parameter
    '''
    pars = {
        'log(n_0)': mb.Param(-3, minval=-7, maxval=2),
        r'\beta': mb.Param(2/3., minval=0., maxval=4.),
        'log(r_c)': mb.Param(2.3, minval=-1., maxval=3.7),
        'log(r_s)': mb.Param(2.7, minval=0, maxval=3.7),
        r'\alpha': mb.Param(0., minval=-1, maxval=2.),
        r'\epsilon': mb.Param(3., minval=0., maxval=5.),
        r'\gamma': mb.Param(3., minval=0., maxval=10, frozen=True),
        }
    if self.mode == 'double':
        pars.update({
            'log(n_{02})': mb.Param(-1, minval=-7, maxval=2),
            r'\beta_2': mb.Param(0.5, minval=0., maxval=4.),
            'log(r_{c2})': mb.Param(1.7, minval=-1., maxval=3.7),
            })
    return pars

def mydens_vikhFunction(self, pars, radii_kpc):
    '''
    Compute the Vikhlinin density profile.
    Copied from MBProj2 changing the parameter names for plotting reasons
    ---------------------------------------------------------------------
    '''
    n_0 = 10**pars['log(n_0)'].val
    beta = pars[r'\beta'].val
    r_c = 10**pars['log(r_c)'].val
    r_s = 10**pars['log(r_s)'].val
    alpha = pars[r'\alpha'].val
    epsilon = pars[r'\epsilon'].val
    gamma = pars[r'\gamma'].val
    r = radii_kpc
    res_sq = n_0**2*(r/r_c)**(-alpha)/((1+(r/r_c)**2)**(3*beta-alpha/2)*(1+(r/r_s)**gamma)**(epsilon/gamma))
    if self.mode == 'double':
        n_02 = 10**pars['log(n_{02})'].val
        r_c2 = 10**pars['log(r_{c2})'].val
        beta_2 = pars[r'\beta_2'].val
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
    return 0

def get_sz_like(self, output='ll'):
    '''
    Computes the log-likelihood on SZ data for the current parameters
    -----------------------------------------------------------------
    output = desired output
        'll' = log-likelihood
        'chisq' = Chi-Squared
        'pp' = pressure profile
        'bright' = surface brightness profile
    ----------------------
    RETURN: desired output
    '''
    # pressure profile
    pp = self.press.press_fun(self.pars, self.data.sz.r_pp)
    # abel transform
    ab = direct_transform(pp, r=self.data.sz.r_pp, direction='forward', backend='Python')[:self.data.sz.ub]
    # Compton parameter
    y = (kpc_cm*self.data.sz.phys_const[1]/self.data.sz.phys_const[0]*ab)
    f = interp1d(np.append(-self.data.sz.r_pp[:self.data.sz.ub], self.data.sz.r_pp[:self.data.sz.ub]),
                 np.append(y, y), 'cubic', bounds_error=False, fill_value=(0, 0))
    # Compton parameter 2D image
    y_2d = f(self.data.sz.d_mat) 
    # Convolution with the beam
    conv_2d = fftconvolve(y_2d, self.data.sz.beam_2d, 'same')*self.data.sz.step**2
    # Convolution with the transfer function
    FT_map_in = fft2(conv_2d)
    map_out = np.real(ifft2(FT_map_in*self.data.sz.filtering))
    # Temperature-dependent conversion from Compton parameter to mJy/beam
    t_prof = self.model.T_cmpt.temp_fun(self.pars, self.data.sz.r_pp[:self.data.sz.ub], getT_SZ=True)
    h = interp1d(np.append(-self.data.sz.r_pp[:self.data.sz.ub], self.data.sz.r_pp[:self.data.sz.ub]),
                 np.append(t_prof, t_prof), 'cubic', bounds_error=False, fill_value=(t_prof[-1], t_prof[-1]))
    map_prof = map_out[conv_2d.shape[0]//2, conv_2d.shape[0]//2:]*self.data.sz.convert(np.append(h(0), t_prof))+np.log10(self.pars['calibration'].val)
    g = interp1d(self.data.sz.radius[self.data.sz.sep:], map_prof, 'cubic', fill_value='extrapolate')
    # Log-likelihood calculation
    chisq = np.nansum(((self.data.sz.flux_data[1]-g(self.data.sz.flux_data[0]))/self.data.sz.flux_data[2])**2)
    log_lik = -chisq/2
    if output == 'll':
        return log_lik
    elif output == 'chisq':
        return chisq
    elif output == 'pp':
        return pp
    elif output == 'bright':
        return map_prof
    else:
        raise RuntimeError('Unrecognised output name (must be "ll", "chisq", "pp" or "bright")')

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
        if not(all(np.gradient(m_prof, 1) > 0)):
            return -np.inf
    # X-ray fitted profiles
    profs = self.calcProfiles()
    # X-ray log-likelihood
    like = self.mylikeFromProfs(profs)
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

    def innerburn():
        bestfit = None
        bestprob = initprob = mcmc.fit.getLikelihood(mcmc.fit.thawedParVals())
        p0 = mcmc._generateInitPars()
        mcmc.header['burn'] = nburn
        for i, result in enumerate(mcmc.sampler.sample(p0, thin=nthin, iterations=nburn, storechain=False)):
            if i%10 == 0:
                print(' Burn %i / %i (%.1f%%)' %(i, nburn, i*100/nburn))
            mcmc.pos0, lnprob, rstate0 = result[:3]
            if lnprob.max()-bestprob > minimprove:
                bestprob = lnprob.max()
                maxidx = lnprob.argmax()
                bestfit = mcmc.pos0[maxidx]
            if (autorefit and i > nburn*minfrac and bestfit is not None ):
                print('Restarting burn as new best fit has been found (%g > %g)' % (bestprob, initprob) )
                mcmc.fit.updateThawed(bestfit)
                mcmc.sampler.reset()
                return False
        mcmc.sampler.reset()
        return True
    import time
    time0 = time.time()
    print('Starting burn-in')
    while not innerburn():
        print('Restarting, as new mininimum found')
        mcmc.fit.doFitting()
    print('Finished burn-in')
    mcmc.header['length'] = nsteps
    if mcmc.pos0 is None:
        print(' Generating initial parameters')
        p0 = mcmc._generateInitPars()
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

def traceplot(mysamples, param_names, nsteps, nw, plotw=20, ppp=4, labsize=18, ticksize=10, plotdir='./'):
    '''
    Traceplot of the MCMC
    ---------------------
    mysamples = array of sampled values in the chain
    param_names = names of the parameters
    nsteps = number of steps in the chain (after burn-in) 
    nw = number of random walkers
    plotw = number of random walkers that we wanna plot (default is 20)
    ppp = number of plots per page
    labsize = label font size
    ticksize = ticks font size
    plotdir = directory where to place the plot
    '''
    plt.clf()
    nw_step = int(np.ceil(nw/plotw))
    param_latex = ['${}$'.format(i) for i in param_names]
    pdf = PdfPages(plotdir+'traceplot.pdf')
    for i in np.arange(mysamples.shape[1]):
        plt.subplot(ppp, 1, i%ppp+1)
        for j in range(nw)[::nw_step]:
            plt.plot(np.arange(nsteps)+1, mysamples[j::nw,i], linewidth=.2)
            plt.tick_params(labelbottom=False)
        plt.ylabel('%s' %param_latex[i], fontdict={'fontsize': labsize})
        plt.tick_params('y', labelsize=ticksize)
        if (abs((i+1)%ppp) < 0.01):
            plt.tick_params(labelbottom=True)
            plt.xlabel('Iteration number')
            pdf.savefig(bbox_inches='tight')
            if i+1 < mysamples.shape[1]:
                plt.clf()
        elif i+1 == mysamples.shape[1]:
            plt.tick_params(labelbottom=True)
            plt.xlabel('Iteration number')
            pdf.savefig(bbox_inches='tight')
    pdf.close()

def triangle(mysamples, param_names, labsize=25, titsize=15, plotdir='./'):
    '''
    Univariate and multivariate distribution of the parameters in the MCMC
    ----------------------------------------------------------------------
    mysamples = array of sampled values in the chain
    param_names = names of the parameters
    labsize = label font size
    titsize = titles font size
    plotdir = directory where to place the plot
    '''
    param_latex = ['${}$'.format(i) for i in param_names]
    plt.clf()
    pdf = PdfPages(plotdir+'cornerplot.pdf')
    corner.corner(mysamples, labels=param_latex, quantiles=np.repeat(.5, len(param_latex)), show_titles=True, 
                  title_kwargs={'fontsize': titsize}, label_kwargs={'fontsize': labsize})
    pdf.savefig(bbox_inches='tight')
    pdf.close()

def fitwithmod(data, lo, med, hi, geomareas, xfig, errxfig, flatchain, fit, ci, labsize=25, ticksize=20, textsize=30, plotdir='./'):
    plt.clf()
    pdf = PdfPages(plotdir+'fit_on_data.pdf')
    npanels = len(data.bands)+1
    f, ax = plt.subplots(int(np.ceil(npanels/3)), 3, figsize=(24, 6*np.ceil(npanels/3)))
    for i, (band, llo, mmed, hhi) in enumerate(zip(data.bands, lo, med, hi)):
        ax[i//3, i%3].set_xscale('log')
        ax[i//3, i%3].set_yscale('log')
        ax[i//3, i%3].axis([0.9*xfig.min(), 1.2*xfig.max(), 
                            1, 10**np.ceil(np.log10(np.max([np.max(band.cts/geomareas/band.areascales) for band in data.bands])))])
        ax[i//3, i%3].text(0.1, 0.1, '[%g-%g] keV' % (band.emin_keV, band.emax_keV), horizontalalignment='left', 
                           verticalalignment='bottom', transform=ax[i//3, i%3].transAxes, fontdict={'fontsize': textsize})	
        ax[i//3, i%3].errorbar(xfig, mmed/geomareas/band.areascales, color='r', label='_nolegend_')
        ax[i//3, i%3].fill_between(xfig, hhi/geomareas/band.areascales, llo/geomareas/band.areascales, color='gold', label='_nolegend_')
        ax[i//3, i%3].errorbar(xfig, band.cts/geomareas/band.areascales, xerr=errxfig, yerr=band.cts**0.5/geomareas/band.areascales, 
                               fmt='o', markersize=3, color='black', label='_nolegend_')
        ax[i//3, i%3].set_ylabel('$S_X$ (counts·arcmin$^{-2}$)', fontdict={'fontsize': labsize})
        ax[i//3, i%3].set_xlabel('Radius (arcmin)', fontdict={'fontsize': labsize})
        ax[i//3, i%3].tick_params(labelsize=ticksize, length=10, which='major')
        ax[i//3, i%3].tick_params(labelsize=ticksize, length=6, which='minor')
    [ax[j//3, j%3].axis('off') for j in np.arange(i+2, ax.size)]
    ax[i//3, i%3].errorbar(xfig, band.cts/geomareas/band.areascales, xerr=errxfig, yerr=band.cts**0.5/geomareas/band.areascales, 
                           color='black', fmt='o', markersize=3, label='X-ray data')
    med_xsz, lo_xsz, hi_xsz = best_fit_xsz(flatchain, fit, ci)
    sep = data.sz.radius.size//2
    r_am = data.sz.radius[sep:sep+med_xsz.size]/60
    ax[(i+1)//3, (i+1)%3].errorbar(data.sz.flux_data[0]/60, data.sz.flux_data[1], yerr=data.sz.flux_data[2], fmt='o', markersize=2, 
                                   color='black', label='SZ data')
    ax[(i+1)//3, (i+1)%3].errorbar(r_am, med_xsz, color='r', label='Best-fit')
    ax[(i+1)//3, (i+1)%3].fill_between(r_am, lo_xsz, hi_xsz, color='gold', label='95% CI')
    ax[(i+1)//3, (i+1)%3].set_xlabel('Radius (arcmin)', fontdict={'fontsize': labsize})
    ax[(i+1)//3, (i+1)%3].set_ylabel('$S_{SZ}$ (mJy·beam$^{-1}$)', fontdict={'fontsize': labsize})
    ax[(i+1)//3, (i+1)%3].set_xscale('linear')
    ax[(i+1)//3, (i+1)%3].set_xlim(0, np.ceil(data.sz.flux_data[0][-1]/60))
    ax[(i+1)//3, (i+1)%3].tick_params(labelsize=ticksize)
    hand_sz, lab_sz = ax[(i+1)//3, (i+1)%3].get_legend_handles_labels()
    hand_x, lab_x = ax[i//3, i%3].get_legend_handles_labels()
    f.legend([hand_sz[2], hand_sz[0], hand_x[0], hand_sz[1]], [lab_sz[2], lab_sz[0], lab_x[0], lab_sz[1]], 
             loc='lower center', ncol=4, fontsize=labsize, bbox_to_anchor=(.5, .99))
    plt.tight_layout()
    pdf.savefig(bbox_inches='tight')
    pdf.close()

def best_fit_xsz(flatchain, fit, ci):
    '''
    Computes the surface brightness profile (median and interval) for the best fitting parameters
    ---------------------------------------------------------------------------------------------
    flatchain = array of sampled values in the chain
    fit = Fit object
    ci = uncertainty level of the interval
    ------------------------------------------------
    RETURN: median and interval profiles
    '''
    profs = []
    for pars in flatchain[::10]:
        fit.updateThawed(pars)
        out_prof = fit.get_sz_like(output='bright')
        profs.append(out_prof)
    profs = np.row_stack(profs)
    med = np.median(profs, axis=0)
    lo, hi = np.percentile(profs, [50-ci/2, 50+ci/2], axis=0)
    return med, lo, hi

def plot_best_sz(sz, med_xz, lo_xz, hi_xz, ci, plotdir='./'):
    '''
    SZ surface brightness profile (points with error bars) and best fitting profile with uncertainty interval
    ---------------------------------------------------------------------------------------------------------
    sz = SZ data object
    lo_xz, med_xz, hi_xz = best (median) fitting profiles with uncertainties
    ci = uncertainty level of the interval
    plotdir = directory where to place the plot    
    '''
    plt.clf()
    sep = sz.radius.size//2
    pdf = PdfPages(plotdir+'best_sz.pdf')
    plt.title('Compton parameter profile - best fit with %s' % ci+'% CI')
    plt.plot(sz.radius[sep:sep+med_xz.size], med_xz, color='b')
    plt.plot(sz.radius[sep:sep+med_xz.size], lo_xz, ':', color='b', label='_nolegend_')
    plt.plot(sz.radius[sep:sep+med_xz.size], hi_xz, ':', color='b', label='_nolegend_')
    plt.scatter(sz.flux_data[0], sz.flux_data[1], color='black')
    plt.errorbar(sz.flux_data[0], sz.flux_data[1], yerr=sz.flux_data[2], fmt='o', markersize=4, color='black')
    plt.axhline(0, linestyle=':', color='black', label='_nolegend_')
    plt.xlabel('Radius (arcsec)')
    plt.ylabel('y * 10$^{4}$')
    plt.xlim(-2, 127)
    plt.legend(('(SZ + X) fit', 'SZ data'), loc='lower right')
    pdf.savefig(bbox_inches='tight')
    pdf.close()

def frac_int(edges):
    '''
    Fraction of mass of shell which is in the inner and outer halves of the midpoint
    Adapted from MBProj2
    --------------------------------------------------------------------------------
    edges = edges of the shells containing X-ray data (cm)
    ----------------------------------------------
    RETURN: the fraction of mass in the inner half
    '''
    low_r, hig_r = edges[:-1], edges[1:]    
    volinside = (low_r+hig_r)**3/24-low_r**3/3
    voloutside = hig_r**3/3-(low_r+hig_r)**3/24
    return volinside/(volinside+voloutside)

def my_rad_profs(vals, r_kpc, fit):
    ''' 
    Compute the thermodynamic radial profiles
    -----------------------------------------
    vals = parameter values
    r_kpc = radius (kpc)
    fit = Fit object
    --------------------------------------------------------------------------------
    RETURN: density(cm^{-3}), temperature (keV), pressure (keV*cm^{-3}), 
            entropy (keV*cm^2), cooling time (yr), cumulative gas mass (solar mass)
    '''
    fit.updateThawed(vals)
    pars = fit.pars
    # density
    dens = fit.model.ne_cmpt.vikhFunction(pars, r_kpc)
    # temperature (SZ)
    temp = fit.model.T_cmpt.temp_fun(pars, r_kpc, getT_SZ=True)
    # temperature (X-ray)
    tempx = fit.model.T_cmpt.temp_fun(pars, r_kpc)
    # pressure
    press = fit.press.press_fun(pars, r_kpc)
    # entropy
    entr = temp/dens**(2/3)
    # cooling time
    cool = (5/2)*dens*(1+1/ne_nH)*temp*keV_erg/(fit.data.annuli.ctrate.getFlux(temp, np.repeat(pars['Z'].val, temp.size), dens)*
                                                4*np.pi*(fit.data.annuli.cosmology.D_L*Mpc_cm)**2)/yr_s
    # cumulative gas mass
    edg_cm = np.append(r_kpc[0]/2, r_kpc+r_kpc[0]/2)*kpc_cm
    mgas = dens*mu_e*mu_g/solar_mass_g*4/3*np.pi*(edg_cm[1:]**3-edg_cm[:-1]**3)
    cmgas = mgas*frac_int(edg_cm)+np.concatenate(([0], np.cumsum(mgas)[:-1]))
    return dens, temp, press, entr, cool, cmgas, tempx

def plot_rad_profs(r_kpc, xmin, xmax, dens, temp, prss, entr, cool, gmss, tempx, plotdir='./'):
    '''
    Plot the thermodynamic radial profiles
    --------------------------------------
    r_kpc = radius (kpc)
    xmin, xmax = x-axis boundaries for the plot
    dens, temp, press, entr, cool, gmss, tempx = thermodynamic best fitting profiles (median and interval)
    plotdir = directory where to place the plot
    '''
    pdf = PdfPages(plotdir+'radial_profiles.pdf')
    plt.clf()
    f, ax = plt.subplots(3, 2, sharex=True)
    ind = np.where((r_kpc > xmin) & (r_kpc < xmax))
    e_ind = np.concatenate(([ind[0][0]-1], ind[0], [ind[0][-1]+1]), axis=0)
    prop = [dens, temp, prss, entr, cool, gmss]
    labs = ['Density (cm$^{-3}$)', 'Temperature (keV)', 'Pressure (keV cm$^{-3}$)', 'Entropy (keV cm$^2$)', 'Cooling time (Gyr)', 'Gas mass $(10^{12}\,\mathrm{M}_\Theta)$']
    for (i, j) in enumerate(zip(prop, labs)):
        ax[i//2, i%2].plot(r_kpc[e_ind], j[0][1, e_ind])
        ax[i//2, i%2].fill_between(r_kpc[e_ind], j[0][0, e_ind], j[0][2, e_ind], color='powderblue')
        ax[i//2, i%2].set_xlim(xmin, xmax)
        ax[i//2, i%2].set_xscale('log')
        ax[i//2, i%2].set_yscale('log')
        ax[i//2, i%2].set_ylabel(j[1])
    if temp[1][0] != tempx[1][0]:
        ax[0,1].plot(r_kpc[e_ind], tempx[1][e_ind]) # add X temperature
        ax[0,1].fill_between(r_kpc[e_ind], tempx[0, e_ind], tempx[2, e_ind], color='lightgreen', alpha=0.25)
    ax[0,1].set_yscale('linear')
    ax[2,0].set_xlabel('Radius (kpc)')
    ax[2,1].set_xlabel('Radius (kpc)')
    ax[0,1].yaxis.set_label_position('right')
    ax[1,1].yaxis.set_label_position('right')
    ax[2,1].yaxis.set_label_position('right')
    pdf.savefig(bbox_inches='tight')
    pdf.close()
    plt.close()
    
def mass_r_delta(r_kpc, cosmo, delta=500):
    '''
    Compute the mass profile in terms of the overdensity radius 
    (overdensity radius = radius within which the average density is Δ times the critical density at the cluster's redshift)
    ------------------------------------------------------------------------------------------------------------------------
    r_kpc = radius (kpc)
    cosmo = Cosmology object
    delta = overdensity (Δ)
    ------------------------
    RETURN: the mass profile
    '''
    # H0 (s^-1)
    H0_s = cosmo.H0/Mpc_km
    # H(z) (s^-1)
    HZ = H0_s*np.sqrt(cosmo.WM*(1+cosmo.z)**3+cosmo.WV)
    # critical density (g cm^-3)
    rho_c = 3*HZ**2/(8*np.pi*G_cgs)
    # radius (cm)
    r_cm = r_kpc*kpc_cm   
    # M(< r_delta) (g*solar masses)
    mass_r_delta = 4/3*np.pi*rho_c*delta*r_cm**3/solar_mass_g
    return mass_r_delta

def m_r_delta(pars, fit, r_kpc, cosmo, delta=500):
    '''
    Compute the overdensity radius and the overdensity mass 
    (overdensity radius = radius within which the average density is Δ times the critical density at the cluster's redshift)
    (overdensity mass = mass enclosed within the overdensity radius)
    ------------------------------------------------------------------------------------------------------------------------
    pars = parameter values
    fit = Fit object
    r_kpc = radius (kpc)
    cosmo = Cosmology object
    delta = overdensity (Δ)
    ---------------------------------------------------------------------
    RETURN: cumulative mass profile, overdensity radius, overdensity mass
    '''
    fit.updateThawed(pars)
    m_prof = fit.mass_cmpt.mass_fun(fit.pars, r_kpc)
    r_delta = optimize.newton(lambda r: fit.mass_cmpt.mass_fun(fit.pars, r)-mass_r_delta(r, cosmo, delta), 700)
    m_delta = fit.mass_cmpt.mass_fun(fit.pars, r_delta)
    return m_prof, r_delta, m_delta

def mass_plot(r_kpc, med_mass, low_mass, hig_mass, med_rd, low_rd, hig_rd, med_md, low_md, hig_md, m_vd, xmin=100, xmax=1000, 
              ymin=1e11, ymax=1e16, labsize=23, ticksize=20, textsize=23, plotdir='./'):
    '''
    Cumulative mass profile plot
    ----------------------------
    r_kpc = radius (kpc)
    med_mass, low_mass, hig_mass = median mass profile with CI boundaries
    med_rd, low_rd, hig_rd = overdensity radius with CI boundaries
    med_md, low_md, hig_md = overdensity mass with CI boundaries
    m_vd = cumulative mass profile in terms of volume and density
    xmin, xmax = limits on the X-axis
    ymin, ymax = limits on the Y-axis
    labsize = label font size
    ticksize = ticks font size
    textsize = text font size
    plotdir = directory where to place the plot
    '''
    pdf = PdfPages(plotdir+'mass_hse.pdf')
    plt.clf()
    plt.errorbar(r_kpc, med_mass)
    plt.fill_between(r_kpc, low_mass, hig_mass, color='powderblue')
    plt.errorbar(r_kpc, m_vd, color='g')
    plt.vlines([med_rd, low_rd, hig_rd], [0, 0, 0], [med_md, low_md, hig_md], linestyle=['--', ':', ':'], color='black')
    plt.hlines([med_md, low_md, hig_md], [0, 0, 0], [med_rd, low_rd, hig_rd], linestyle=['--', ':', ':'], color='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel('Radius (kpc)', fontdict={'fontsize': labsize})
    plt.ylabel('Total mass (M$_\odot$)', fontdict={'fontsize': labsize})
    plt.text(xmin-xmin/5, med_md, '$M_{500}$', fontdict={'fontsize': textsize})
    plt.text(low_rd, ymin-ymin/2, '$r_{500}$', fontdict={'fontsize': textsize})
    plt.tick_params(labelsize=ticksize, length=5, which='major')
    plt.tick_params(labelsize=ticksize, length=3, which='minor')
    pdf.savefig(bbox_inches='tight')
    pdf.close()
