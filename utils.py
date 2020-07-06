import numpy as np
from scipy import interpolate

def gamma_lxe_cross_section(xs):
    """
    Interpolated gamma-ray interaction cross sections in Xenon
    
    Parameters
    ----------
    
    xs: str
        path to .txt file with xs data (xcom)
    """
    lxe_density = 2.78
    xs = np.genfromtxt(xs, skip_header=2)
    
    compton = interpolate.interp1d(xs[:, 0] * 1000, xs[:, 1] * lxe_density, fill_value='extrapolate', kind='slinear')
    photo = interpolate.interp1d(xs[:, 0] * 1000, xs[:, 2] * lxe_density, fill_value='extrapolate', kind='slinear')
    total = interpolate.interp1d(xs[:, 0] * 1000, (xs[:, 1] + xs[:, 2] )* lxe_density, fill_value='extrapolate', kind='slinear')

    return compton, photo, total