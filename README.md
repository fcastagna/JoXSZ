# JoXSZ
# Joint X-ray and SZ fitting code for galaxy clusters in Python
*Castagna Fabio, Andreon Stefanoour 

`JoXSZ` is a Python program that allows to jointly fit the thermodynamic profiles of galaxy clusters from both SZ and X-ray data using MCMC.
`JoXSZ` is the enhanced version of [`preprofit`](https://github.com/fcastagna/preprofit), which only fits SZ data. 

As an example, we show the application of `JoXSZ` on the high-redshift cluster of galaxies CL J1226.9+3332 (z = 0.89).

Beam data and transfer function data come from the [NIKA data release](http://lpsc.in2p3.fr/NIKA2LPSZ/nika2sz.release.php), while X-ray data come from *Chandra* observations, reduced following the standard procedures (e.g. [Andreon et al. 2019](https://ui.adsabs.harvard.edu/abs/2019A%26A...630A..78A/abstract)).

PLEASE NOTE that this version of the code is only compatible with version 2 of the `emcee` package. It is the version of the code that we presented in the paper linked below. For an updated version of the code, compatible with both version 2 and version 3 of the `emcee` package, please consider using the [master](https://github.com/fcastagna/JoXSZ) branch.

For more details, see [Castagna and Andreon, 2020](https://ui.adsabs.harvard.edu/abs/2020arXiv200512315C/abstract).

### Requirements
`JoXSZ` requires the following:
- [mbproj2](https://github.com/jeremysanders/mbproj2)
- [PyAbel](https://github.com/PyAbel/PyAbel)
- [numpy](http://www.numpy.org/)
- [scipy](http://www.scipy.org/)
- [astropy](http://www.astropy.org/)
- [emcee](https://emcee.readthedocs.io/en/stable/)
- [six](https://pypi.org/project/six/)
- [matplotlib](https://matplotlib.org/)
- [corner](https://pypi.org/project/corner/)

### Credits
`preprofit` See [Castagna and Andreon 2019](https://ui.adsabs.harvard.edu/abs/2019A%26A...632A..22C/abstract) and [GitHub](https://github.com/fcastagna/preprofit).

`MBProj2` See [Sanders et al. 2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.1065S/abstract) and [GitHub](https://github.com/jeremysanders/mbproj2).
