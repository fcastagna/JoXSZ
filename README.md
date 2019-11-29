# JoXSZ
# Joint X-ray and SZ fitter for galaxy clusters in Python
*Castagna Fabio, Andreon Stefano*

`JoXSZ` is a Python program that allows to jointly fit the pressure profile of galaxy clusters from both SZ and X-ray data using MCMC.
`JoXSZ` is the enhanced version of [`preprofit`](https://github.com/fcastagna/preprofit), which only fits SZ data. 

As an example, we show the application of `JoXSZ` on the high-redshift cluster of galaxies CL J1226.9+3332 (z = 0.89).

Beam data and transfer function data come from the [NIKA data release](http://lpsc.in2p3.fr/NIKA2LPSZ/nika2sz.release.php), while X-ray data come from Chandra (reference...).

### Requirements
`JoXSZ` requires the following:
- [mbproj2](https://github.com/jeremysanders/mbproj2)
- [PyAbel](https://github.com/PyAbel/PyAbel)
- [numpy](http://www.numpy.org/)
- [scipy](http://www.scipy.org/)
- [astropy](http://www.astropy.org/)
- [six](https://pypi.org/project/six/)
- [matplotlib](https://matplotlib.org/)
- [corner](https://pypi.org/project/corner/)

### Credits
`preprofit` Castagna F, Andreon S. [PreProFit: Pressure Profile Fitter for galaxy clusters](https://ui.adsabs.harvard.edu/abs/2019A%26A...632A..22C/abstract).

`MBProj2` Sanders J et al. [Hydrostatic Chandra X-ray analysis of SPT-selected galaxy clusters - I. Evolution of profiles and core properties](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.1065S/abstract).
