import sys
from astropy.io import fits
import numpy as np
from scipy import optimize
from scipy.stats import norm
import mbproj2 as mb
from mbproj2.physconstants import keV_erg, kpc_cm, mu_g, G_cgs, solar_mass_g
from abel.direct import direct_transform
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.fftpack import fft2, ifft2
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import corner

plt.style.use('classic')

class SZ_data:
    """Dataset class."""

    def __init__(self, phys_const, step, kpc_as, convert, flux_data, beam_2d,
                 radius, sep, r_pp, ub, d_mat, filtering):

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
    return f(np.rot90(np.rot90(karr)))

def getEdges(infg, bands):
    """Get edges of annuli in arcmin.
    There should be one more than the number of annuli.
    """
    data = np.loadtxt(infg % (bands[0][0], bands[0][1]))
    return np.hstack((data[0,0]-data[0,1], data[:,0]+data[:,1]))

def loadBand(infg, inbg, bandE, rmf, arf):
    """Load foreground and background profiles from file and construct
    Band object."""

    data = np.loadtxt(infg % (bandE[0], bandE[1]))

    # radii of centres of annuli in arcmin
    radii = data[:,0]
    # half-width of annuli in arcmin
    hws = data[:,1]
    # number of counts (integer)
    cts = data[:,2]
    # areas of annuli, taking account of pixelization (arcmin^2)
    areas = data[:,3]
    # exposures (s)
    exps = data[:,4]
    # note: vignetting can be input into exposure or areas, but
    # background needs to be consistent

    # geometric area factor
    geomareas = np.pi*((radii+hws)**2-(radii-hws)**2)
    # ratio between real pixel area and geometric area
    areascales = areas/geomareas

    # this is the band object fitted to the data
    band = mb.Band(bandE[0]/1000, bandE[1]/1000, cts, rmf, arf, exps, areascales=areascales)

    # this is the background profile
    # load rates in cts/s/arcmin^2
    backd = np.loadtxt(inbg % (bandE[0], bandE[1]))
    # band.backrates = backd[:,5]
    # to read, and shorter, a bkg file over a larger radial range
    # e' in col5
    band.backrates = backd[0:radii.size, 4]
    lastmyrad = backd[0:radii.size, 0]
    if (abs(lastmyrad[-1]-radii[-1]) > .001):
         print('Problem while reading bg file', lastmyrad[-1], radii[-1])
         sys.exit()
    return band

class CmptPressure(mb.Cmpt):

    def __init__(self, name, annuli):
        mb.Cmpt.__init__(self, name, annuli)

    def defPars(self):
        pars = {
            'P_0': mb.Param(0.4, minval=0, maxval=20),
            'a': mb.Param(1.33, minval=0.1, maxval=10),
            'b': mb.Param(4.13, minval=0.1, maxval=15),
            'c': mb.Param(0.014, minval=0, maxval=3),
            'r_p': mb.Param(500, minval=5, maxval=3000)
            }
        return pars

    def press_fun(self, pars, r_kpc):
        P_0 = pars['P_0'].val
        r_p = pars['r_p'].val
        a = pars['a'].val
        b = pars['b'].val
        c = pars['c'].val
        return P_0/((r_kpc/r_p)**c*(1+(r_kpc/r_p)**a)**((b-c)/a))

    def press_derivative(self, pars, r_kpc):
        P_0 = pars['P_0'].val
        r_p = pars['r_p'].val
        a = pars['a'].val
        b = pars['b'].val
        c = pars['c'].val
        return -P_0*(c+b*(r_kpc/r_p)**a)/(
                r_p*(r_kpc/r_p)**(c+1)*(1+(r_kpc/r_p)**a)**((b-c+a)/a))

    def computeProf(self, pars):
        return self.press_fun(pars, self.annuli.midpt_kpc)

class CmptUPPTemperature(mb.Cmpt):
    
    def __init__(self, name, annuli, press_prof, ne_prof):
        mb.Cmpt.__init__(self, name, annuli)
        self.press_prof = press_prof
        self.ne_prof = ne_prof

    def defPars(self):
        pars = self.press_prof.defPars()
        pars.update(self.ne_prof.defPars())
        return pars
    
    def temp_fun(self, pars, r_kpc):
        pr = self.press_prof.press_fun(pars, r_kpc)
        ne = self.ne_prof.vikhFunction(pars, r_kpc)
        return pr/ne

    def computeProf(self, pars):
        return self.temp_fun(pars, self.annuli.midpt_kpc)

class CmptMyMass(mb.Cmpt):
    
    def __init__(self, name, annuli, press_prof, ne_prof):
        mb.Cmpt.__init__(self, name, annuli)
        self.press_prof = press_prof
        self.ne_prof = ne_prof

    def defPars(self):
        pars = self.press_prof.defPars()
        pars.update(self.ne_prof.defPars())
        return pars
    
    def mass_fun(self, pars, r_kpc, mu_gas=0.61):
        dpr_kpc = self.press_prof.press_derivative(pars, r_kpc)
        dpr_cm = dpr_kpc*keV_erg/kpc_cm
        ne = self.ne_prof.vikhFunction(pars, r_kpc)
        r_cm = r_kpc*kpc_cm
        return -dpr_cm*r_cm**2/(mu_gas*mu_g*ne*G_cgs)/solar_mass_g
    
    def computeProf(self, pars):
        return self.mass_fun(pars, self.annuli.midpt_kpc)

def mydefPars(self):
    pars = {
        'n_0': mb.Param(-3., minval=-7., maxval=2.),
        'beta': mb.Param(2/3., minval=0., maxval=4.),
        'log(r_c)': mb.Param(2.3, minval=-1., maxval=3.7),
        'log(r_s)': mb.Param(2.7, minval=0, maxval=3.7),
        'alpha': mb.Param(0., minval=-1, maxval=2.),
        'epsilon': mb.Param(3., minval=0., maxval=5.),
        'gamma': mb.Param(3., minval=0., maxval=10, frozen=True),
        }
    return pars

def myvikhFunction(self, pars, radii_kpc):
    n_0 = 10**pars['n_0'].val
    beta = pars['beta'].val
    r_c = 10**pars['log(r_c)'].val
    r_s = 10**pars['log(r_s)'].val
    alpha = pars['alpha'].val
    epsilon = pars['epsilon'].val
    gamma = pars['gamma'].val

    r = radii_kpc

    retn_sqd = (
        n_0**2 *
        (r/r_c)**(-alpha) / (
            (1+r**2/r_c**2)**(3*beta-0.5*alpha) *
            (1+(r/r_s)**gamma)**(epsilon/gamma)
            )
        )
    if self.mode == 'double':
        n_02 = 10**pars['n_{02}'].val
        r_c2 = 10**pars['log(r_{c2})'].val
        beta_2 = pars['beta_2'].val

        retn_sqd += n_02**2 / (1 + r**2/r_c2**2)**(3*beta_2)

    return np.sqrt(retn_sqd)

def myprior(self, pars):
    r_c = 10**pars['log(r_c)'].val
    r_s = 10**pars['log(r_s)'].val
    if r_c > r_s:
        return -np.inf
    return 0

def get_sz_like(self, output='ll'):
    pp = self.press.press_fun(self.pars, self.data.sz.r_pp)
    ab = direct_transform(pp, r=self.data.sz.r_pp, direction='forward', 
                          backend='Python')[:self.data.sz.ub]
    y = (kpc_cm*self.data.sz.phys_const[1]/self.data.sz.phys_const[0]*ab)
    f = interp1d(np.append(-self.data.sz.r_pp[:self.data.sz.ub], self.data.sz.r_pp[:self.data.sz.ub]),
                 np.append(y, y), 'cubic', bounds_error=False, fill_value=(0, 0))
    y_2d = f(self.data.sz.d_mat) 
    conv_2d = fftconvolve(y_2d, self.data.sz.beam_2d, 'same')*self.data.sz.step**2
    FT_map_in = fft2(conv_2d)
    map_out = np.real(ifft2(FT_map_in*self.data.sz.filtering))
    t_prof = self.model.T_cmpt.temp_fun(self.pars, self.data.sz.r_pp[:self.data.sz.ub])
    h = interp1d(np.append(-self.data.sz.r_pp[:self.data.sz.ub], self.data.sz.r_pp[:self.data.sz.ub]),
                 np.append(t_prof, t_prof), 'cubic', bounds_error=False, fill_value=(t_prof[-1], t_prof[-1]))
    map_prof = map_out[conv_2d.shape[0]//2, conv_2d.shape[0]//2:]*self.data.sz.convert(np.append(h(0), t_prof))
    g = interp1d(self.data.sz.radius[self.data.sz.sep:], map_prof, 'cubic', fill_value='extrapolate')
    chisq = np.sum(((self.data.sz.flux_data[1]-g(self.data.sz.flux_data[0]))/
                    self.data.sz.flux_data[2])**2)
    log_lik = -chisq/2
    if output == 'll':
        return log_lik
    elif output == 'chisq':
        return chisq
    elif output == 'pp':
        return pp
    elif output == 'flux':
        return map_prof

def getLikelihood(self, vals=None):
    """Get likelihood for parameters given.

    Also include are the priors from the various components
    """

    if vals is not None:
        self.updateThawed(vals)

    # prior on parameters
    parprior = sum((self.pars[p].prior() for p in self.pars))
    if not np.isfinite(parprior):
        # don't want to evaluate profiles for invalid parameters
        return -np.inf

    m_prof = self.mass_cmpt.mass_fun(self.pars, self.data.sz.r_pp) 
    if not(all(np.gradient(m_prof, 1) > 0)):
        return -np.inf

    profs = self.calcProfiles()
    like = self.likeFromProfs(profs)
    prior = self.model.prior(self.pars)+parprior
    sz_like = self.get_sz_like()
    totlike = float(like+prior+sz_like)

    if mb.fit.debugfit and (totlike-self.bestlike) > 0.1:
        self.bestlike = totlike
        #print("Better fit %.1f" % totlike)
        with mb.utils.AtomicWriteFile("%s/fit.dat" % self.savedir) as fout:
            mb.utils.uprint("likelihood = %g + %g + %g = %g" % (like, sz_like, prior, totlike), file=fout)
            for p in sorted(self.pars):
                mb.utils.uprint("%s = %s" % (p, self.pars[p]), file=fout)

    return totlike

def prelimfit(data, myprofs, geomareas, xfig, errxfig, plotdir='./'):
    pdf = PdfPages(plotdir+'prelimfit.pdf')
    for i, (band, prof) in enumerate(zip(data.bands, myprofs)):
        plt.subplot(2, 3, i+1)
        plt.xscale('log')
        plt.yscale('log')
        plt.axis([0.08, 1.2*xfig.max(), 1, 2*(band.cts/geomareas/
                  band.areascales).max()])
        plt.xlabel('r [arcmin]')
        plt.ylabel('counts / area [cts arcmin$^{-2}$]')
        #plt.title('[ %g - %g] keV' % (band.emin_keV, band.emax_keV))
        plt.text(0.1, 1.2, '[%g-%g] keV' % (band.emin_keV, band.emax_keV))
        plt.plot(xfig, myprofs[i]/geomareas/band.areascales, color='r')
        plt.plot(xfig, band.backrates*band.exposures, linestyle=':', color='b')
        plt.scatter(xfig, band.cts/geomareas/band.areascales, color='darkblue')
        plt.errorbar(xfig, band.cts/geomareas/band.areascales, xerr=errxfig, 
                     yerr=band.cts**0.5/geomareas/band.areascales, fmt='o')
    plt.subplots_adjust(wspace=0.45, hspace=0.35)	
    pdf.savefig()
    pdf.close()
    plt.clf()

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
            plt.tick_params(labelbottom=True)
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

def fitwithmod(data, lo, med, hi, geomareas, xfig, errxfig, plotdir='./'):
    plt.clf()
    pdf = PdfPages(plotdir+'fitwithmod.pdf')
    for i, (band, llo, mmed, hhi) in enumerate(zip(data.bands, lo, med, hi)):
        plt.subplot(2, 3, i+1)
        plt.xscale('log')
        plt.yscale('log')
        plt.axis([0.08, 1.2*xfig.max(), 1, 1.2*(hhi/geomareas/band.areascales).max()])
        plt.xlabel('r [arcmin]')
        plt.ylabel('counts / area [cts arcmin$^{-2}$]')
        plt.text(0.1, 1.2, '[%g-%g] keV' % (band.emin_keV, band.emax_keV))	
        plt.errorbar(xfig, mmed/geomareas/band.areascales, color='red')	
        plt.fill_between(xfig, hhi/geomareas/band.areascales, llo/geomareas/band.areascales, color='gold')
        plt.plot(xfig, band.backrates*band.exposures, linestyle=':', color='b')
        plt.scatter(xfig, band.cts/geomareas/band.areascales, color='darkblue')
        plt.errorbar(xfig, band.cts/geomareas/band.areascales, xerr=errxfig, 
                     yerr=band.cts**0.5/geomareas/band.areascales, fmt='o')
    plt.subplots_adjust(wspace=0.45, hspace=0.35)	
    pdf.savefig()
    pdf.close()

def best_fit_xsz(sz, chain, fit, ci, plotdir='./'):
    profs = []
    for pars in chain[::10]:
        fit.updateThawed(pars)
        out_prof = fit.get_sz_like(output='flux')
        profs.append(out_prof)
    profs = np.row_stack(profs)
    med = np.median(profs, axis=0)
    lo, hi = np.percentile(profs, [50-ci/2, 50+ci/2], axis=0)
    return med, lo, hi

def plot_best_sz(sz, med_xz, lo_xz, hi_xz, ci, plotdir='./'):
    sep = sz.radius.size//2
    pdf = PdfPages(plotdir+'best_sz.pdf')
    plt.clf()
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
    pdf.savefig()
    pdf.close()

def extract_profiles(name, data, rname='r_kpc'):
    profs = data[name]
    r = data[rname][:,0]
    med = profs[:,0]
    low, hig = med+profs[:,1], med+profs[:,2]
    return r, med, low, hig

def plot_profiles(ax, name, data, ylab='', rname='r_kpc', nrow=None):
    r, med, low, hig = extract_profiles(name, data)
    if nrow == None:
        nrow = r.size
    ax.plot(r[:nrow], med[:nrow])
    ax.fill_between(r[:nrow], low[:nrow], hig[:nrow], color='powderblue')
    ax.set_xlim(100, 1100)
    ax.set_ylabel(ylab)
    ax.set_yscale('log')

def plot_rad_profs(data, plotdir='./'):
    pdf = PdfPages(plotdir+'radial_profiles.pdf')
    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex = True)
    plot_profiles(ax1, 'ne_pcm3', data, 'Density (cm$^{-3}$)', nrow=12)
    plot_profiles(ax2, 'T_keV', data, 'Temperature (keV)', nrow=12)
    ax2.set_yscale('linear')
    plot_profiles(ax3, 'Pe_keVpcm3', data, 'Pressure (keV cm$^{-3}$)', nrow=12)
    plot_profiles(ax4, 'Se_keVcm2', data, 'Entropy (keV cm$^2$)', nrow=12)
    plot_profiles(ax5, 'tcool_yr', data, 'Cooling time (Gyr)', nrow=12)
    plot_profiles(ax6, 'Mgascuml_Msun', data, 'Gas mass $(10^{12}\,\mathrm{M}_\Theta)$', nrow=12)
    ax2.yaxis.set_label_position('right')
    ax4.yaxis.set_label_position('right')
    ax6.yaxis.set_label_position('right')
    pdf.savefig()
    pdf.close()
