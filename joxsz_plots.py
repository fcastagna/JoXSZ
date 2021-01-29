import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import corner
from mbproj2.physconstants import keV_erg, kpc_cm, mu_g, G_cgs, solar_mass_g, ne_nH, Mpc_cm, yr_s, mu_e, Mpc_km
from scipy import optimize

plt.style.use('classic')
font = {'size': 14}
plt.rc('font', **font)

def traceplot(cube_chain, param_names, plotw=20, seed=None, ppp=4, labsize=18., ticksize=10., plotdir='./'):
    '''
    Traceplot of the MCMC
    ---------------------
    cube_chain = 3d array of sampled values (nw x niter x nparam)
    param_names = names of the parameters
    plotw = number of random walkers that we wanna plot (default is 20)
    seed = random seed (default is None)
    ppp = number of plots per page
    labsize = label font size
    ticksize = ticks font size
    plotdir = directory where to place the plot
    '''
    plt.clf()
    nw, nsteps = cube_chain.shape[:2]
    np.random.seed(seed)
    ind_w = np.random.choice(nw, plotw, replace=False)
    param_latex = ['${}$'.format(i) for i in param_names]
    pdf = PdfPages(plotdir+'traceplot.pdf')
    for i in np.arange(cube_chain.shape[2]):
        plt.subplot(ppp, 1, i%ppp+1)
        for j in ind_w:
            plt.plot(np.arange(nsteps)+1, cube_chain[j,:,i], linewidth=.2)
            plt.tick_params(labelbottom=False)
        plt.ylabel('%s' %param_latex[i], fontdict={'fontsize': labsize})
        plt.tick_params('y', labelsize=ticksize)
        if (abs((i+1)%ppp) < 0.01):
            plt.tick_params(labelbottom=True)
            plt.xlabel('Iteration number')
            pdf.savefig(bbox_inches='tight')
            if i+1 < cube_chain.shape[1]:
                plt.clf()
        elif i+1 == cube_chain.shape[1]:
            plt.tick_params(labelbottom=True)
            plt.xlabel('Iteration number')
            pdf.savefig(bbox_inches='tight')
    pdf.close()

def triangle(mat_chain, param_names, show_lines=True, col_lines='r', ci=95, labsize=25., titsize=15., plotdir='./'):
    '''
    Univariate and multivariate distribution of the parameters in the MCMC
    ----------------------------------------------------------------------
    mat_chain = 2d array of sampled values ((nw x niter) x nparam)
    param_names = names of the parameters
    show_lines = whether to show lines for median and uncertainty interval (boolean, default is True)
    col_lines = line colour (default is red)
    ci = uncertainty level of the interval
    labsize = label font size
    titsize = titles font size
    plotdir = directory where to place the plot
    '''
    plt.clf()
    pdf = PdfPages(plotdir+'cornerplot.pdf')
    param_latex = ['${}$'.format(i) for i in param_names]
    fig = corner.corner(mat_chain, labels=param_latex, title_kwargs={'fontsize': titsize}, label_kwargs={'fontsize': labsize})
    axes = np.array(fig.axes).reshape((len(param_names), len(param_names)))
    plb, pmed, pub = get_equal_tailed(mat_chain, ci=ci)
    for i in range(len(param_names)):
        l_err, u_err = pmed[i]-plb[i], pub[i]-pmed[i]
        axes[i,i].set_title('%s = $%.2f_{-%.2f}^{+%.2f}$' % (param_latex[i], pmed[i], l_err, u_err))
        if show_lines:
            axes[i,i].axvline(pmed[i], color=col_lines, linestyle='--', label='Median')
            axes[i,i].axvline(plb[i], color=col_lines, linestyle=':', label='%i%% CI' % ci)
            axes[i,i].axvline(pub[i], color=col_lines, linestyle=':', label='_nolegend_')
            for yi in range(len(param_names)):
                for xi in range(yi):
                    axes[yi,xi].axvline(pmed[xi], color=col_lines, linestyle='--')
                    axes[yi,xi].axhline(pmed[yi], color=col_lines, linestyle='--')
                    axes[yi,xi].plot(plb[xi], plb[yi], marker=1, color=col_lines)
                    axes[yi,xi].plot(plb[xi], plb[yi], marker=2, color=col_lines)
                    axes[yi,xi].plot(plb[xi], pub[yi], marker=1, color=col_lines)
                    axes[yi,xi].plot(plb[xi], pub[yi], marker=3, color=col_lines)
                    axes[yi,xi].plot(pub[xi], plb[yi], marker=0, color=col_lines)
                    axes[yi,xi].plot(pub[xi], plb[yi], marker=2, color=col_lines)
                    axes[yi,xi].plot(pub[xi], pub[yi], marker=0, color=col_lines)
                    axes[yi,xi].plot(pub[xi], pub[yi], marker=3, color=col_lines)
            fig.legend(('Median', '%i%% CI' % ci), loc='lower center', ncol=2, bbox_to_anchor=(0.55, 0.95), 
                       fontsize=titsize+len(param_names))
    pdf.savefig(bbox_inches='tight')
    pdf.close()

def get_equal_tailed(data, ci=95):
    '''
    Computes the median and lower/upper limits of the equal tailed uncertainty interval
    -----------------------------------------------------------------------------------
    ci = uncertainty level of the interval
    ----------------------------------------
    RETURN: lower bound, median, upper bound
    '''
    low, med, upp = map(np.atleast_1d, np.percentile(data, [50-ci/2, 50, 50+ci/2], axis=0))
    return np.array([low, med, upp])

def best_fit_prof(cube_chain, fit, num='all', seed=None, ci=95):
    '''
    Computes the surface brightness profile (median and uncertainty interval) for the best fitting parameters
    ---------------------------------------------------------------------------------------------------------
    cube_chain = 3d array of sampled values (nw x niter x nparam)
    fit = Fit object
    num = number of set of parameters to include (default is 'all', i.e. nw x niter parameters)
    seed = random seed (default is None)
    ci = uncertainty level of the interval
    ------------------------------------------------
    RETURN: median and uncertainty interval profiles
    '''
    nw = cube_chain.shape[0]
    if num == 'all':
        num = nw*cube_chain.shape[1]
    w, it = np.meshgrid(np.arange(nw), np.arange(cube_chain.shape[1]))
    w = w.flatten()
    it = it.flatten()
    np.random.seed(seed)
    rand = np.random.choice(w.size, num, replace=False)
    profs_x, profs_sz = [], []
    for j in rand:
        fit.updateThawed(cube_chain[w[j],it[j],:])
        profs_x.append(fit.calcProfiles())
        out_prof = fit.get_sz_like(output='bright')
        profs_sz.append(out_prof)
    perc_x = get_equal_tailed(profs_x, ci)
    perc_sz = get_equal_tailed(profs_sz, ci)
    return perc_x, perc_sz

def fitwithmod(data, perc_x, perc_sz, ci=95, labsize=25., ticksize=20., textsize=30., plotdir='./'):
    '''
    Surface brightness profiles (points with error bars) and best fitting profiles with uncertainties
    -------------------------------------------------------------------------------------------------
    data = Data object containing information on the X-ray bands and SZ data
    perc_x = best (median) X-ray fitting profiles with uncertainties
    perc_sz = best (median) SZ fitting profiles with uncertainties
    ci = uncertainty level of the interval
    labsize = label font size
    ticksize = ticks font size
    textsize = text font size
    plotdir = directory where to place the plot
    '''
    plt.clf()
    pdf = PdfPages(plotdir+'fit_on_data.pdf')
    lx, mx, ux = perc_x
    lsz, msz, usz = perc_sz
    edges = data.annuli.edges_arcmin
    xfig = 0.5*(edges[1:]+edges[:-1]) # radii of X-ray data
    errxfig = 0.5*(edges[1:]-edges[:-1]) # errors
    geomareas = np.pi*(edges[1:]**2-edges[:-1]**2) # annuli areas for X-ray data
    npanels = len(data.bands)+1
    f, ax = plt.subplots(int(np.ceil(npanels/3)), 3, figsize=(24., 6.*np.ceil(npanels/3)))
    for i, (band, llo, mmed, hhi) in enumerate(zip(data.bands, lx, mx, ux)):
        ax[i//3, i%3].set_xscale('log')
        ax[i//3, i%3].set_yscale('log')
        ax[i//3, i%3].axis([0.9*xfig.min(), 1.2*xfig.max(), 
                            1., 10.**np.ceil(np.log10(np.max([np.max(band.cts/geomareas/band.areascales) for band in data.bands])))])
        ax[i//3, i%3].text(0.1, 0.1, '[%g-%g] keV' % (band.emin_keV, band.emax_keV), horizontalalignment='left', 
                           verticalalignment='bottom', transform=ax[i//3, i%3].transAxes, fontdict={'fontsize': textsize})	
        ax[i//3, i%3].errorbar(xfig, mmed/geomareas/band.areascales, color='r', label='_nolegend_')
        ax[i//3, i%3].fill_between(xfig, hhi/geomareas/band.areascales, llo/geomareas/band.areascales, color='gold', label='_nolegend_')
        ax[i//3, i%3].errorbar(xfig, band.cts/geomareas/band.areascales, xerr=errxfig, yerr=band.cts**0.5/geomareas/band.areascales, 
                               fmt='o', markersize=3., color='black', label='_nolegend_')
        ax[i//3, i%3].set_ylabel('$S_X$ (counts·arcmin$^{-2}$)', fontdict={'fontsize': labsize})
        ax[i//3, i%3].set_xlabel('Radius (arcmin)', fontdict={'fontsize': labsize})
        ax[i//3, i%3].tick_params(labelsize=ticksize, length=10., which='major')
        ax[i//3, i%3].tick_params(labelsize=ticksize, length=6., which='minor')
    [ax[j//3, j%3].axis('off') for j in np.arange(i+2, ax.size)]
    ax[i//3, i%3].errorbar(xfig, band.cts/geomareas/band.areascales, xerr=errxfig, yerr=band.cts**0.5/geomareas/band.areascales, 
                           color='black', fmt='o', markersize=3., label='X-ray data')
    sep = data.sz.radius.size//2
    r_am = data.sz.radius[sep:sep+msz.size]/60
    ax[(i+1)//3, (i+1)%3].errorbar(data.sz.flux_data[0]/60, data.sz.flux_data[1], yerr=data.sz.flux_data[2], fmt='o', markersize=2., 
                                   color='black', label='SZ data')
    ax[(i+1)//3, (i+1)%3].errorbar(r_am, msz, color='r', label='Best-fit')
    ax[(i+1)//3, (i+1)%3].fill_between(r_am, lsz, usz, color='gold', label='%i%% CI' % ci)
    ax[(i+1)//3, (i+1)%3].set_xlabel('Radius (arcmin)', fontdict={'fontsize': labsize})
    ax[(i+1)//3, (i+1)%3].set_ylabel('$S_{SZ}$ (mJy·beam$^{-1}$)', fontdict={'fontsize': labsize})
    ax[(i+1)//3, (i+1)%3].set_xscale('linear')
    ax[(i+1)//3, (i+1)%3].set_xlim(0., np.ceil(data.sz.flux_data[0][-1]/60))
    ax[(i+1)//3, (i+1)%3].tick_params(labelsize=ticksize)
    hand_sz, lab_sz = ax[(i+1)//3, (i+1)%3].get_legend_handles_labels()
    hand_x, lab_x = ax[i//3, i%3].get_legend_handles_labels()
    f.legend([hand_sz[2], hand_sz[0], hand_x[0], hand_sz[1]], [lab_sz[2], lab_sz[0], lab_x[0], lab_sz[1]], 
             loc='lower center', ncol=4, fontsize=labsize, bbox_to_anchor=(.5, .99))
    plt.tight_layout()
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

def cum_gas_mass(r_kpc, dens):
    '''
    Cumulative gas mass profile computation, given the density profile
    ------------------------------------------------------------------
    r_kpc = radius (kpc)
    dens = density profile
    '''
    edg_cm = np.append(r_kpc[0]/2, r_kpc+r_kpc[0]/2)*kpc_cm
    mgas = dens*mu_e*mu_g/solar_mass_g*4/3*np.pi*(edg_cm[1:]**3-edg_cm[:-1]**3)
    return mgas*frac_int(edg_cm)+np.concatenate(([0], np.cumsum(mgas)[:-1]))

def thermodynamic_profs(vals, r_kpc, fit):
    ''' 
    Thermodynamic radial profiles computation
    -----------------------------------------
    vals = parameter values
    r_kpc = radius (kpc)
    fit = Fit object
    -----------------------------------------------------------------------------------------------------------
    RETURN: density(cm^{-3}), gass mass weighted temperature (keV), pressure (keV*cm^{-3}), entropy (keV*cm^2),
            cooling time (yr), cumulative gas mass (solar mass), X-ray temperature (keV)
    '''
    fit.updateThawed(vals)
    pars = fit.pars
    # density
    dens = fit.model.ne_cmpt.vikhFunction(pars, r_kpc)
    # pressure
    press = fit.press.press_fun(pars, r_kpc)
    # temperature (SZ)
    temp = press/dens
    # temperature (X-ray)
    tempx = temp*10**pars['log(T_X/T_{SZ})'].val
    # entropy
    entr = temp/dens**(2/3)
    # cooling time
    cool = (5/2)*dens*(1.+1/ne_nH)*temp*keV_erg/(fit.data.annuli.ctrate.getFlux(temp, np.repeat(pars['Z'].val, temp.size), dens)*
                                                 4.*np.pi*(fit.data.annuli.cosmology.D_L*Mpc_cm)**2)/yr_s
    # cumulative gas mass
    cmgas = cum_gas_mass(r_kpc, dens)
    return dens, temp, press, entr, cool, cmgas, tempx

def comp_rad_profs(cube_chain, fit, num='all', seed=None, ci=95):
    '''
    Compute all the thermodynamic profiles from the chain of sampled values
    -----------------------------------------------------------------------
    cube_chain = 3d array of sampled values (nw x niter x nparam)
    fit = Fit object
    num = number of set of parameters to include (default is 'all', i.e. nw x niter parameters)
    seed = random seed (default is None)
    ci = uncertainty level of the interval
    ---------------------------------------------------------------------------------------
    RETURN: median and equal-tailed uncertainty interval of the main thermodynamic profiles
    '''
    nw = cube_chain.shape[0]
    if num == 'all':
        num = nw*cube_chain.shape[1]
    w, it = np.meshgrid(np.arange(nw), np.arange(cube_chain.shape[1]))
    w = w.flatten()
    it = it.flatten()
    np.random.seed(seed)
    rand = np.random.choice(w.size, num, replace=False)    
    td, tt, tp, te, tc, tg, tx = [np.zeros((num, fit.data.sz.r_pp.size)) for _ in range(7)]
    for (i, j) in enumerate(rand):
        td[i], tt[i], tp[i], te[i], tc[i], tg[i], tx[i] = thermodynamic_profs(cube_chain[w[j],it[j],:], fit.data.sz.r_pp, fit)
    dens, temp, prss, entr, cool, gmss, xtmp = map(lambda x: get_equal_tailed(x, ci), [td, tt, tp, te, tc, tg, tx])
    return dens, temp, prss, entr, cool, gmss, xtmp

def plot_rad_profs(r_kpc, dens, temp, prss, entr, cool, gmss, tempx, xmin=np.nan, xmax=np.nan, ci=95, labsize=10., plotdir='./'):
    '''
    Plot the thermodynamic radial profiles
    --------------------------------------
    r_kpc = radius (kpc)
    dens, temp, press, entr, cool, gmss, tempx = thermodynamic best fitting profiles (median and interval)
    xmin, xmax = x-axis boundaries for the plot (by default, they are obtained based on r_kpc)
    ci = uncertainty level of the interval
    labsize = label font size
    plotdir = directory where to place the plot
    '''
    pdf = PdfPages(plotdir+'radial_profiles.pdf')
    plt.clf()
    f, ax = plt.subplots(3, 2, sharex=True)
    xmin, xmax = np.nanmax([r_kpc[0], xmin]), np.nanmin([r_kpc[-1], xmax])
    ind = np.where((r_kpc > xmin) & (r_kpc < xmax))
    e_ind = np.concatenate(([ind[0][0]-1], ind[0], [ind[0][-1]+1]), axis=0)
    prop = [dens, temp, prss, entr, cool/1e9, gmss/1e12]
    labs = ['Density (cm$^{-3}$)', 'Temperature (keV)', 'Pressure (keV cm$^{-3}$)', 'Entropy (keV cm$^2$)', 
            'Cooling time (Gyr)', 'Gas mass $(10^{12}\,\mathrm{M}_\Theta)$']
    for (i, j) in enumerate(zip(prop, labs)):
        ax[i//2,i%2].plot(r_kpc[e_ind], j[0][1,e_ind])
        ax[i//2,i%2].fill_between(r_kpc[e_ind], j[0][0,e_ind], j[0][2,e_ind], color='powderblue')
        ax[i//2,i%2].set_xlim(xmin, xmax)
        ax[i//2,i%2].set_xscale('log')
        ax[i//2,i%2].set_yscale('log')
        ax[i//2,i%2].set_ylabel(j[1], fontsize=labsize)
    if temp[1][0] != tempx[1][0]:
        ax[0,1].plot(r_kpc[e_ind], tempx[1][e_ind]) # add X temperature
        ax[0,1].fill_between(r_kpc[e_ind], tempx[0,e_ind], tempx[2,e_ind], color='lightgreen', alpha=0.25)
        ax[0,1].legend(('$T_{SZ}$ (%i%% CI)' % ci, '$T_X$ (%i%% CI)' % ci), fontsize=labsize)#, loc='lower left')
    ax[0,1].set_yscale('linear')
    ax[2,0].set_xlabel('Radius (kpc)', fontdict={'fontsize': labsize})
    ax[2,1].set_xlabel('Radius (kpc)', fontdict={'fontsize': labsize})
    ax[0,1].yaxis.set_label_position('right')
    ax[1,1].yaxis.set_label_position('right')
    ax[2,1].yaxis.set_label_position('right')
    pdf.savefig(bbox_inches='tight')
    pdf.close()
    plt.close()

def hydro_mass(pars, fit, r_kpc, cosmo, overdens=True, delta=500, start_opt=700.):
    '''
    Compute the cumulative mass profile under hydrostatic equilibrium assumption and optionally overdensity radius and overdensity mass 
    (overdensity radius = radius within which the average density is Δ times the critical density at the cluster's redshift)
    (overdensity mass = mass enclosed within the overdensity radius)
    -----------------------------------------------------------------------------------------------------------------------------------
    pars = parameter values
    fit = Fit object
    r_kpc = radius (kpc)
    cosmo = Cosmology object (adapted from MBProj2)
    overdens = whether to compute overdensity measures (boolean, default is True)
    delta = overdensity (Δ)
    start_opt = starting value for optimization (kpc)
    -------------------------------------------------------------------------------
    RETURN: cumulative mass profile, and optionally the overdensity radius and mass
    '''
    fit.updateThawed(pars)
    m_prof = fit.mass_cmpt.mass_fun(fit.pars, r_kpc)
    if overdens:
        r_delta = optimize.newton(lambda r: fit.mass_cmpt.mass_fun(fit.pars, r)-mass_overdens(r, cosmo, delta), start_opt)
        m_delta = fit.mass_cmpt.mass_fun(fit.pars, r_delta)
        return m_prof, r_delta, m_delta
    else:
        return m_prof

def comp_mass_prof(cube_chain, fit, num='all', seed=None, overdens=True, delta=500, start_opt=700., ci=95):
    '''
    Compute the hydrostatic mass profile from the chain of sampled values
    ---------------------------------------------------------------------
    cube_chain = 3d array of sampled values (nw x niter x nparam)
    fit = Fit object
    num = number of set of parameters to include (default is 'all', i.e. nw x niter parameters)
    seed = random seed (default is None)
    overdens = whether to compute overdensity measures (boolean, default is True)
    delta = overdensity (Δ)
    start_opt = starting value for optimization (kpc)
    ci = uncertainty level of the interval
    ---------------------------------------------------------------------------------------------------------------------------
    RETURN: median and equal-tailed uncertainty interval of the mass profile, and optionally of the overdensity radius and mass
    '''    
    nw = cube_chain.shape[0]
    if num == 'all':
        num = nw*cube_chain.shape[1]
    w, it = np.meshgrid(np.arange(nw), np.arange(cube_chain.shape[1]))
    w = w.flatten()
    it = it.flatten()
    np.random.seed(seed)
    rand = np.random.choice(w.size, num, replace=False)    
    m_prof, r_d, m_d = [], [], []
    for j in rand:
        res = hydro_mass(cube_chain[w[j],it[j],:], fit, fit.data.sz.r_pp, fit.data.annuli.cosmology, delta=delta, start_opt=start_opt)
        m_prof.append(res[0])
        if overdens:
            r_d.append(res[1]), m_d.append(res[2])
    mass = get_equal_tailed(m_prof, ci)
    if overdens:
        r_delta = get_equal_tailed(r_d, ci)
        m_delta = get_equal_tailed(m_d, ci)
        return mass, r_delta, m_delta
    else:
        return mass

def mass_overdens(r_kpc, cosmo, delta=500):
    '''
    Compute the mass profile (volume x density) in terms of the overdensity radius 
    (overdensity radius = radius within which the average density is Δ times the critical density at the cluster's redshift)
    ------------------------------------------------------------------------------------------------------------------------
    r_kpc = radius (kpc)
    cosmo = Cosmology object (adapted from MBProj2)
    delta = overdensity (Δ)
    ------------------------
    RETURN: the mass profile
    '''
    # H0 (s^-1)
    H0_s = cosmo.H0/Mpc_km
    # H(z) (s^-1)
    HZ = H0_s*np.sqrt(cosmo.WM*(1.+cosmo.z)**3+cosmo.WV)
    # critical density (g cm^-3)
    rho_c = 3.*HZ**2/(8.*np.pi*G_cgs)
    # radius (cm)
    r_cm = r_kpc*kpc_cm   
    # M(< r_delta) (solar masses)
    mass_r_delta = 4/3*np.pi*rho_c*delta*r_cm**3/solar_mass_g
    return mass_r_delta
    
def mass_plot(r_kpc, mass_prof, cosmo, overdens=True, delta=500, r_delta=None, m_delta=None, xmin=np.nan, xmax=np.nan, 
              labsize=23., ticksize=20., textsize=23., plotdir='./'):
    '''
    Cumulative mass profile plot
    ----------------------------
    r_kpc = radius (kpc)
    mass_prof = median mass profile with uncertainty interval
    cosmo = Cosmology object (adapted from MBProj2)
    overdens = whether to compute overdensity measures (boolean, default is True)
    delta = overdensity (Δ)
    r_delta = overdensity radius with uncertainty interval
    m_delta = overdensity mass with uncertainty interval
    xmin, xmax = x-axis boundaries for the plot (by default, they are obtained based on r_kpc)
    labsize = label font size
    ticksize = ticks font size
    textsize = text font size
    plotdir = directory where to place the plot
    '''
    m_vol_dens = mass_overdens(r_kpc, cosmo, delta=delta)
    pdf = PdfPages(plotdir+'mass_hse.pdf')
    plt.clf()
    f, ax = plt.subplots(1, 1)
    xmin, xmax = np.nanmax([r_kpc[0], xmin]), np.nanmin([r_kpc[-1], xmax])
    ind = np.where((r_kpc > xmin) & (r_kpc < xmax))
    e_ind = np.concatenate(([ind[0][0]-1], ind[0], [ind[0][-1]+1]), axis=0)
    plt.errorbar(r_kpc[e_ind], mass_prof[1][e_ind])
    plt.fill_between(r_kpc[e_ind], mass_prof[0][e_ind], mass_prof[2][e_ind], color='powderblue')
    plt.errorbar(r_kpc[e_ind], m_vol_dens[e_ind], color='g')
    if overdens:
        plt.vlines([r_delta], [0, 0, 0], [m_delta], linestyle=[':', '--', ':'], color='black')
        plt.hlines([m_delta], [0, 0, 0], [r_delta], linestyle=[':', '--', ':'], color='black')
        mag = int(np.log10(m_delta[1]))
        plt.text(0., 1.05, r'$\rm{M}_{%i}=%.2f^{+%.2f}_{-%.2f} \times 10^{%i}\rm{M}_{\odot}$' % 
                 (delta, m_delta[1]/10**mag, (m_delta[2]-m_delta[1])/10**mag, (m_delta[1]-m_delta[0])/10**mag, mag), 
                 transform=ax.transAxes, fontdict={'fontsize': textsize})
        plt.text(0., 1.15, r'$r_{%i}=%.f^{+%.f}_{-%.f}\,\rm{kpc}$' % (delta, r_delta[1], r_delta[2]-r_delta[1], r_delta[1]-r_delta[0]),
                 transform=ax.transAxes, fontdict={'fontsize': textsize})
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xmin, xmax)
    ymin = 10**int(np.log10(mass_prof[0][e_ind].min()))
    ymax = 10**np.ceil(np.log10(mass_prof[2][e_ind].max())) 
    plt.ylim(ymin, ymax)
    plt.xlabel('Radius (kpc)', fontdict={'fontsize': labsize})
    plt.ylabel('Total mass (M$_\odot$)', fontdict={'fontsize': labsize})
    plt.tick_params(labelsize=ticksize, length=5., which='major')
    plt.tick_params(labelsize=ticksize, length=3., which='minor')
    pdf.savefig(bbox_inches='tight')
    pdf.close()
    
def frac_gas_prof(cube_chain, fit, num='all', seed=None, ci=95):
    '''
    Computes the gas fraction profile (median and uncertainty interval)
    -------------------------------------------------------------------
    cube_chain = 3d array of sampled values (nw x niter x nparam)
    fit = Fit object
    num = number of set of parameters to include (default is 'all', i.e. nw x niter parameters)
    seed = random seed (default is None)
    ci = uncertainty level of the interval
    '''
    nw = cube_chain.shape[0]
    if num == 'all':
        num = nw*cube_chain.shape[1]
    w, it = np.meshgrid(np.arange(nw), np.arange(cube_chain.shape[1]))
    w = w.flatten()
    it = it.flatten()
    np.random.seed(seed)
    rand = np.random.choice(w.size, num, replace=False)
    f_gas = []
    for j in rand:
        fit.updateThawed(cube_chain[w[j],it[j],:])
        pars = fit.pars
        dens = fit.model.ne_cmpt.vikhFunction(pars, fit.data.sz.r_pp)
        m_gas = cum_gas_mass(fit.data.sz.r_pp, dens)        
        m_tot = hydro_mass(cube_chain[w[j],it[j],:], fit, fit.data.sz.r_pp, fit.data.annuli.cosmology, overdens=False)
        f_gas.append(m_gas/m_tot)
    frac_gas = get_equal_tailed(f_gas, ci)
    return frac_gas   
    
def frac_gas_plot(r_kpc, f_gas, xmin=np.nan, xmax=np.nan, ci=95, labsize=23., plotdir='./'):
    '''
    Gas fraction profile plot
    ----------------------------
    r_kpc = radius (kpc)
    f_gas = median gas fraction profile with uncertainty interval
    xmin, xmax = x-axis boundaries for the plot (by default, they are obtained based on r_kpc)
    ci = uncertainty level of the interval
    labsize = label font size
    plotdir = directory where to place the plot
    '''
    pdf = PdfPages(plotdir+'frac_gas.pdf')
    plt.clf()
    plt.title('Gas fraction profile (median + %i%% error)' % ci, fontdict={'fontsize': labsize})
    xmin, xmax = np.nanmax([r_kpc[0], xmin]), np.nanmin([r_kpc[-1], xmax])
    ind = np.where((r_kpc > xmin) & (r_kpc < xmax))
    e_ind = np.concatenate(([ind[0][0]-1], ind[0], [ind[0][-1]+1]), axis=0)
    plt.errorbar(r_kpc[e_ind], f_gas[1][e_ind])
    plt.fill_between(r_kpc[e_ind], f_gas[0][e_ind], f_gas[2][e_ind], color='powderblue')
    plt.xscale('log')
    plt.xlim(xmin, xmax)
    plt.xlabel('Radius (kpc)', fontdict={'fontsize': labsize})
    plt.ylabel('Gas fraction', fontdict={'fontsize': labsize})
    pdf.savefig(bbox_inches='tight')
    pdf.close()
