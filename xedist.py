import numpy as np
import utils

class TPC():
    """
    Cylindrical TPC
    
    Parameters
    ----------
    
    radius: cm
    
    height: cm
        excluding gas gap and reverse field region
    
    n_mesh_points: int
        number of points to use for discretised LXe volume. Smaller == faster but produces coarse predictions
    """
    
    def __init__(self, radius, height, n_mesh_points = 40):
        self.radius = radius
        self.height = height
        self.n_mesh_points = n_mesh_points
        self.r2, self.phi, self.z = self.coordinates()
        
    def coordinates(self):
        """
        Get cylindrical mesh coordinates
        """
        r, phi, z = np.mgrid[0:self.radius**2:self.n_mesh_points*1j, 0:2*np.pi:self.n_mesh_points*1j, 0:self.height:self.n_mesh_points*1j]
        return r, phi, z
    
class GammaSource():
    """
    Gamma-ray source base class
    """
    def __init__(self, energy, n):
        self.energy = energy
        self.n = int(n)


class CylindricalGammaSource(GammaSource):
    """
    Gamma-ray source uniformly distributed on a cylindrical surface
    
    Parameters
    ----------
    
    radius: cm
    
    height: cm
    
    half_z: cm
    
    energy: float
        gamma-ray energy
    
    n: int
        number of events
    """
    def __init__(self, radius, height, half_z, energy, n):
        super().__init__(energy, n)
        self.radius = radius
        self.height = height
        self.half_z = half_z
        
    def generate(self):
        """
        Generate sources at random positions
        """
        phi = np.random.uniform(0, 2*np.pi, self.n)
        x, y = self.radius*np.cos(phi), self.radius*np.sin(phi)
        z = np.random.uniform(self.half_z-self.height/2, self.half_z+self.height/2., self.n)
        return x, y, z
    

class DiskGammaSource(GammaSource):
    """
    Gamma-ray source uniformly distributed on disk
    
    Parameters
    ----------
    
    radius: cm
            
    energy: float
        gamma-ray energy
    
    n: int
        number of events
    """
    def __init__(self, radius, z, energy, n):
        super().__init__(energy, n)
        self.radius = radius
        self.z = z
        
    def generate(self):
        """
        Generate sources at random positions
        """
        phi = np.random.uniform(0, 2*np.pi, self.n)
        r = np.sqrt(np.random.uniform(0, self.radius**2, self.n))
        x, y = r*np.cos(phi),r*np.sin(phi)
        z = np.full(self.n, self.z)
        return x, y, z     

    
def spatial_distribution(tpc, *sources):
    """
    Calculate 2d-spatial distribution (r, z) of gamma-ray interactions in the TPC. 
    Only photoelectric absorption interactions are assumed to occur. 
    
    Warning: only accurate for gamma-sources in close proximity to the tpc, i.e the PMTs and TPC walls.
    It is assumed that the path length of gamma-rays in LXe is the same as the distance 
    from the source to the location in the TPC. Results obtained for sources far from the TPC will therefore be 
    innacurate as the dead space outside of the TPC will be treated as LXe. 
    To do this correctly, need to do simple ray tracing and find length of path in LXe
    
    Parameters
    ----------
    
    tpc: TPC 
        TPC instnace
    
    sources: 
        variable number of GammaSources
        
    returns: 
        spatial probability distribution of events across discretised TPC volume, marginalised over azimuthal coordinate. 
    """
    
    compton, photo, total = utils.gamma_lxe_cross_section("data/xe_gamma_xs.txt")
    
    dp_drdz = 0
    for source in sources:
        
        photoelectric_attenuation_length = 1/photo(source.energy).astype(np.float32)
        attenuation_length = 1/total(source.energy).astype(np.float32)
        
        x_source, y_source, z_source = np.float32(source.generate())
        x_tpc, y_tpc = np.float32(np.sqrt(tpc.r2)*np.cos(tpc.phi)), np.float32(np.sqrt(tpc.r2)*np.sin(tpc.phi))
        
        distance = np.sqrt((x_tpc.ravel() - x_source[:, np.newaxis])**2 + (y_tpc.ravel() - y_source[:, np.newaxis])**2 + (np.float32(tpc.z.ravel()) - z_source[:, np.newaxis])**2)
        dp_dV = 1 / (photoelectric_attenuation_length) * np.exp(-distance / attenuation_length) / (4*np.pi*distance**2)
        # integrate azimuthal dimension
        dp_drdz += dp_dV.sum(0).reshape(tpc.r2.shape).sum(1)
    return dp_drdz


def scale_rate(rate, tpc, lz_integrated_rate=18):
    """
    Normalise and scale rate to total 0nBB BG rate in LZ, approx. 18 events/day. 
    Result is scaled by increase in surface area compared to LZ.
    
    Parameters
    ----------
    rate: np.ndarray
        output of spatial_distribution
        
    tpc: TPC
        tpc instance
    
    lz_integrated_rate: float (default=18)
        total background rate (events/day) in 0nBB ROI of LZ
    """
    lz_surface_area = 2*np.pi*72.8**2 + 2*np.pi*72.8*145.6
    tpc_surface_area = 2*np.pi*tpc.radius**2 + 2*np.pi*tpc.radius*tpc.height
    return tpc_surface_area / lz_surface_area * lz_integrated_rate / rate.sum() * rate


def get_cumulative_rate(rate, tpc):
    """
    Calculate rate [events/year] as function of fiducial mass, integrating from center outwards
    
    Parameters
    ----------
    
    rate: np.ndarray
        output of spatial_distribution
        
    tpc: TPC
        tpc instance
    """
    rate = scale_rate(rate, tpc) * 365
    r = np.sqrt(tpc.r2[:, 0, 0])
    n_bins = len(r)-1
    z = tpc.z[0, 0, :]
    rate =  np.array([[fiducial_mass(r[n_bins-2*i], z[i], z[n_bins-i]), rate[:n_bins-2*i, i:n_bins-i].sum()] for i in range(n_bins//2)])
    return rate[:,0], rate[:,1]


def fiducial_mass(radius, z_low, z_high):
    """
    Get mass of a fiducial volume in tonnes
    
    Parameters
    ----------
    
    radius: cm
    
    z_low: cm
    
    z_high: cm
    """
    lxe_density = 0.00289 / 1000 # tonne/cm3
    return np.pi * lxe_density * radius**2 * (z_high - z_low)