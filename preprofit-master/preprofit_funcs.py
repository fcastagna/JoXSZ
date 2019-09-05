"""
Authors: Castagna Fabio, Andreon Stefano, Pranjal RS.
"""

import numpy as np
from astropy.io import fits
from scipy.stats import norm
from scipy.interpolate import interp1d
from abel.direct import direct_transform
from scipy import optimize
from scipy.integrate import simps
from scipy.signal import fftconvolve
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import corner

font = {'size': 8}
plt.rc('font', **font)
plt.style.use('classic')

class Param:
    '''
    Class for parameters
    --------------------
    val = value of the parameter
    minval, maxval = minimum and maximum allowed values
    frozen = whether the parameter is allowed to vary (True/False)
    '''
    def __init__(self, val, minval=-1e99, maxval=1e99, frozen=False):
        self.val = float(val)
        self.minval = minval       
        self.maxval = maxval
        self.frozen = frozen

    def __repr__(self):
        return '<Param: val=%.3g, minval=%.3g, maxval=%.3g, frozen=%s>' % (
            self.val, self.minval, self.maxval, self.frozen)

class Pressure:
    '''
    Class to parametrize the pressure profile
    -----------------------------------------    
    '''
    def __init__(self):
        pass
        
    def defPars(self):
        '''
        Default parameter values
        ------------------------
        P_0 = normalizing constant
        a = slope at intermediate radii
        b = slope at large radii
        c = slope at small radii
        r_p = characteristic radius
        '''
        pars = {
            'P_0': Param(0.4, minval=0., maxval=1.),
            'a': Param(1.33, minval=0.5, maxval=5.),
            'b': Param(4.13, minval=3., maxval=7.),
            'c': Param(0.014, minval=0., maxval=0.5),
            'r_p': Param(300., minval=100., maxval=1000.)
            }
        return pars

    def update_vals(self, pars, fit_pars, pars_val):
        '''
        Update the parameter values
        ---------------------------
        pars = set of pressure parameters
        fit_pars = name of the parameters to update
        pars_val = new parameter values
        '''
        for name, i in zip(fit_pars, range(len(fit_pars))):
            pars[name].val = pars_val[i] 

    def press_fun(self, pars, r_kpc):
        '''
        Compute the gNFW pressure profile
        ---------------------------------
        pars = set of pressure parameters
        r_kpc = radius (kpc)
        '''
        P_0 = pars['P_0'].val
        a = pars['a'].val
        b = pars['b'].val
        c = pars['c'].val
        r_p = pars['r_p'].val
        return P_0/((r_kpc/r_p)**c*(1+(r_kpc/r_p)**a)**((b-c)/a)) 

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
    return f(np.rot90(np.rot90(karr)))

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

def log_lik(pars_val, press, pars, fit_pars, r_pp, phys_const, radius, 
            d_mat, beam_2d, step, filtering, sep, ub, flux_data, compt_mJy_beam, output='ll'):
    '''
    Computes the log-likelihood for the current pressure parameters
    ---------------------------------------------------------------
    pars_val = array of free parameters
    press = pressure object of the class Pressure
    pars = set of pressure parameters
    fit_pars = name of the parameters to fit
    r_pp = radius used to compute the pressure profile
    phys_const = physical constants
    radius = radius (arcsec)
    d_mat = matrix of distances
    beam_2d = beam image
    step = radius[1]-radius[0]
    filtering = transfer function matrix
    sep = index of radius 0
    ub = index of the highest radius considered (ub=sep unless r500 is too low and then r_pp.size < sep)
    flux data:
        r_sec = radius (arcsec)
        y_data = flux density
        err = statistical errors of the flux density
    compt_mJy_beam = conversion rate from compton parameter to mJy/beam
    r500 = characteristic radius
    output = desired output
        'll' = log-likelihood
        'chisq' = Chi-Squared
        'pp' = pressure profile
        'flux' = flux profile
    --------------------------------------------------------------
    RETURN: log-posterior probability or -inf whether theta is out of the parameter space
    '''
    # update pars
    press.update_vals(pars, fit_pars, pars_val)
    if all([pars[i].minval < pars[i].val < pars[i].maxval for i in pars]):
        # pressure profile
        pp = press.press_fun(pars, r_pp)
        # abel transform
        ab = direct_transform(pp, r=r_pp, direction='forward', backend='Python')[:ub]
        # Compton parameter
        y = phys_const[2]*phys_const[1]/phys_const[0]*ab
        f = interp1d(np.append(-r_pp[:ub], r_pp[:ub]), np.append(y, y), 'cubic', fill_value=(0, 0), bounds_error=False)
        # Compton parameter 2D image
        y_2d = f(d_mat)
        # Convolution with the beam
        conv_2d = fftconvolve(y_2d, beam_2d, 'same')*step**2
        # Convolution with the transfer function
        FT_map_in = fft2(conv_2d)
        map_out = np.real(ifft2(FT_map_in*filtering))
        map_prof = map_out[conv_2d.shape[0]//2, conv_2d.shape[0]//2:]*compt_mJy_beam
        g = interp1d(radius[sep:], map_prof, 'cubic', fill_value='extrapolate')
        # Log-likelihood calculation
        chisq = np.sum(((flux_data[1]-g(flux_data[0]))/flux_data[2])**2)
        log_lik = -chisq/2
        if output == 'll':
            return log_lik
        elif output == 'chisq':
            return chisq
        elif output == 'pp':
            return pp
        elif output == 'flux':
            return map_prof
    else:
        # if some parameter is out of the parameter space
        return -np.inf

def mcmc_run(sampler, p0, nburn, nsteps, comp_time=True):
    '''
    Run the MCMC
    ------------
    sampler = emcee.EnsembleSampler object
    p0 = initial position of the walkers in the parameter space
    nburn = number of steps to burn
    nsteps = number of steps to run
    comp_time = whether to show or not the computation time (True/False)
    '''
    import time
    time0 = time.time()
    print('Starting burn-in')
    for i, result in enumerate(sampler.sample(p0, iterations=nburn, storechain=False)):
        if i%10 == 0:
            print(' Burn %i / %i (%.1f%%)' %(i, nburn, i*100/nburn))
        val = result[0]
    print('Finished burn-in \nStarting sampling')
    for i, result in enumerate(sampler.sample(val, iterations=nsteps)):
        if i%10 == 0:
            print(' Sampling %i / %i (%.1f%%)' %(i, nsteps, i*100/nsteps))
    print('Finished sampling')
    time1 = time.time()
    if comp_time:
        h, rem = divmod(time1-time0, 3600)
        print('Computation time: '+str(int(h))+'h '+str(int(rem//60))+'m')
    print('Acceptance fraction: %s' %np.mean(sampler.acceptance_fraction))

def traceplot(mysamples, param_names, nsteps, nw, plotw=20, ppp=4, plotdir='./'):
    '''
    Traceplot of the MCMC
    ---------------------
    mysamples = array of sampled values in the chain
    param_names = names of the parameters
    nsteps = number of steps in the chain (after burn-in) 
    nw = number of random walkers
    plotw = number of random walkers that we wanna plot (default is 20)
    ppp = number of plots per page
    plotdir = directory where to place the plot
    '''
    nw_step = int(np.ceil(nw/plotw))
    param_latex = ['${}$'.format(i) for i in param_names]
    pdf = PdfPages(plotdir+'traceplot.pdf')
    for i in np.arange(mysamples.shape[1]):
        plt.subplot(ppp, 1, i%ppp+1)
        for j in range(nw)[::nw_step]:
            plt.plot(np.arange(nsteps)+1, mysamples[j::nw,i], linewidth=.2)
            plt.tick_params(labelbottom=False)
        plt.ylabel('%s' %param_latex[i], fontdict={'fontsize': 20})
        if (abs((i+1)%ppp) < 0.01):
            plt.tick_params(labelbottom=True)
            plt.xlabel('Iteration number')
            pdf.savefig()
            if i+1 < mysamples.shape[1]:
                plt.clf()
        elif i+1 == mysamples.shape[1]:
            plt.xlabel('Iteration number')
            pdf.savefig()
    pdf.close()

def triangle(mysamples, param_names, plotdir='./'):
    '''
    Univariate and multivariate distribution of the parameters in the MCMC
    ----------------------------------------------------------------------
    mysamples = array of sampled values in the chain
    param_names = names of the parameters
    plotdir = directory where to place the plot
    '''
    param_latex = ['${}$'.format(i) for i in param_names]
    plt.clf()
    pdf = PdfPages(plotdir+'cornerplot.pdf')
    corner.corner(mysamples, labels=param_latex, quantiles=np.repeat(.5, len(param_latex)), show_titles=True, 
                  title_kwargs={'fontsize': 20}, label_kwargs={'fontsize': 30})
    pdf.savefig()
    pdf.close()
    
def plot_best(theta, fit_pars, mp_med, mp_lb, mp_ub, radius, sep, flux_data, ci=95, plotdir='./'):
    '''
    Plot of the Compton parameter profile compared to the flux density data
    -----------------------------------------------------------------------
    mp_med = Compton profile for the median parameter values
    mp_lb, mp_ub = CI of the Compton profile
    ci = confidence interval
    plotdir = directory where to place the plot
    '''
    r_sec, y_data, err = flux_data
    plt.clf()
    pdf = PdfPages(plotdir+'best_fit.pdf')
    plt.plot(radius[sep:sep+mp_med.size], mp_med)
    plt.fill_between(radius[sep:sep+mp_med.size], mp_lb, mp_ub, color='powderblue', label='_nolegend_')
    plt.errorbar(r_sec, y_data, yerr=err, fmt='o', fillstyle='none', color='r')
    plt.legend(('Model (%i%% CI)' %ci, 'Observed data'), loc='lower right')
    plt.xlabel('Radius (arcsec)')
    plt.ylabel('Surface brightness (mJy/beam)')
    plt.xlim(0, np.ceil(r_sec[-1]/60)*60*7/6)
    pdf.savefig()
    pdf.close()
