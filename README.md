# JoXSZ
# Joint X-ray and SZ pressure profile fitter for galaxy clusters in Python
*Castagna Fabio, Andreon Stefano, Pranjal RS.*

`JoXSZ` is a Python program that allows to jointly fit the pressure profile of galaxy clusters from both SZ and X-ray data using MCMC.

`JoXSZ` is the enhanced version of `preprofit` (https://github.com/fcastagna/preprofit), whose fit is suitable for SZ data only. 

As an example, we show the application of `JoXSZ` on the high-redshift cluster of galaxies CL J1226.9+3332 (z = 0.89).

Beam data and transfer function data come from the NIKA data release (http://lpsc.in2p3.fr/NIKA2LPSZ/nika2sz.release.php).
X-ray data come from (...)

### Requirements
`preprofit` requires the following:
- mbproj2 https://github.com/jeremysanders/mbproj2
- PyAbel https://github.com/PyAbel/PyAbel
- numpy http://www.numpy.org/
- scipy http://www.scipy.org/
- astropy http://www.astropy.org/
- six https://pypi.org/project/six/
- matplotlib https://matplotlib.org/
- corner https://pypi.org/project/corner/

### Credits
Castagna Fabio, Andreon Stefano, Pranjal RS.

For more details, see Castagna _et al_. (in preparation).
