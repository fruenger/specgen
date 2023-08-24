### DigiCone v1.3 - 27.07.2022 - Florian Ruenger ###

# v1.0
# first implementation of everything

# v1.1
# major performance improments for the cone search and radial digitizing
# implemented line-of-sight geometry

# v1.2
# implement the generation of synthetic spectra by querying the required informatrio from the NIST database

# v1.3
# add some utility functions or the line of sight analysis, in the scope of easing the process of generating line-of-sight spectra
# now the generate spectra routine allows for dynamic smaplic as well and the hard-coded dl-parameter has been abolished

DEBUG_MODE = True

import numpy as np
from scipy import spatial # needed for the line of sight analysis
from scipy.special import voigt_profile
import astropy.units as u
import astropy.constants as c
from astroquery.nist import Nist
from time import time

def debug(*args):
    """
    A function that is supposed to give an output only if debug mode is enabled
    """
    if DEBUG_MODE:
        print("[DEBUG]", *args)

def magnitude(vector):
    """
    Computes the magnitude of a multidimensional vector in euclidean space
    """
    return np.sqrt(np.sum(np.asarray(vector, dtype=float)**2))







class cone_bin: # single sheet/bin of which multiple ones stack, make a cone
    """
    This class represents a radial bin of a cone with a defined inner and outer bound. It is the fundament of the digitized_cone object.
    ## Input:
     - bounds: radial bounds of the bin. Must be a array-like, containing TWO floats (upper and lower bound, respectively.
     - opening angle: float (angle in radians). This angle is HALF the total opening angle of the cone, i.e. the full sphere would have an appropriate value of PI (180 deg)
     - direction: Tuple / Array of floats, contains THREE coordinates, corresponding to the coordinates of the center-of-cone direction. Magnitude will be automaticlly normalized to 1.0
    """
    def __init__(self, bounds, opening_angle=np.pi, direction=(0., 0., 1.)): # initialize the properties of the radial cone bin
        self.bounds     = np.array(bounds)
        self.theta      = opening_angle
        self.direction  = direction / magnitude(direction) # compute normalized direction vector
        self.solid_angle= self.get_solid_angle()
        self.volume     = self.get_volume()
        self.center_dist= np.sum(self.bounds)/2.
    
    def get_solid_angle(self) -> float:
        return 2. * np.pi * (1. - np.cos(self.theta))

    def get_volume(self) -> float:
        return self.solid_angle / 3. * (np.max(self.bounds)**3 - np.min(self.bounds)**3) # computes the volume of the bin
    
    def set_bounds(self, bounds):
        self.bounds = np.array(bounds)
        self.volume = self.get_volume() # update bin volume

    def set_direction(self, direction):
        self.direction = direction / magnitude(direction)
    
    def set_opening_angle(self, opening_angle):
        self.theta = opening_angle
        self.solid_angle = self.get_solid_angle()
        self.volume = self.get_volume()







class digitized_cone:
    """
    This class represents a cone that can generate masks on position arrays such that particles can digitized into radial bins
    ## Input:
     - radial_shell_bounds (defines the radial bin sampling)
     - origin_direction (optional, defaulting to (0,0,1))
     - opening_angle (optional, defaulting to PI i.e. a full sphere)
    """
    def __init__(self, radial_shell_bounds, **kwargs):

        self.bin_count = len(radial_shell_bounds) - 1
        self.radial_shell_bounds = radial_shell_bounds

        if "opening_angle" in kwargs:
            self.theta = kwargs["opening_angle"]
        else:
            self.theta = np.pi
        
        if "origin_direction" in kwargs:
            self.direction = np.array(kwargs["origin_direction"]) / magnitude(np.array(kwargs["origin_direction"])) # NEEDS to be normalized for calculating angles later on
        else:
            self.direction = np.array((0., 0., 1.))

        if len(radial_shell_bounds) <= 1:
            raise Exception("Not sufficiently enough radial shell bounds defined. List of boundaries must at least contain 2 elements.") # throw an exception if not enough binning boundaries are provided
        
        ###### bins are defined below ######

        self.bins = []
        for i in range(self.bin_count):
            self.bins.append(cone_bin(bounds=radial_shell_bounds[[i, i+1]], opening_angle=self.theta, direction=self.direction)) # adopt bin parameters from paramters of cone
    
        self.solid_angle = self.bins[0].solid_angle # since all bins cover the same solid angle, we can infer this propery from retrieving it from an ARBITRARY bin in the list of all bins

    def get_bin_volumes(self) -> np.ndarray: # return all bin volumes in a list
        return np.array([b.volume for b in self.bins])
    
    def get_bin_centers(self) -> np.ndarray: # return the center raddi of all bins
        return np.array([b.center_dist for b in self.bins])
    
    def mask_datapoints_to_direction(self, positions):
        """
        Takes a set of datapoints and calculates a mask on then, setting those to True which are in the direction of the cone element, regardless of their distance.
        """
        positions = np.asarray(positions) # data set: Positions of the particles
        # artificially create some arrays to ease the array-wise multiplication of coordinates
        origin_direction_array = np.array(np.ones_like(positions))
        for i in range(3): origin_direction_array[:,i] = self.direction[i] # fill in the array with the respective numbers
        # calculate the dot products
        cos_angles = np.sum(origin_direction_array * positions, axis=1) / (np.sqrt(np.sum(positions**2, axis=1)))

        # mask the data points that are within the defined cone:
        direction_mask = np.where(
            np.cos(self.theta) <= cos_angles, # cosine functions is a decreasing function over the interval of [0, pi] hence, even though we require that all angle should be *smaller* than theta, taking the cosine on both siedes yields the equation cos theta < cos angles
            True, # return True statement if the data point is inside the cone
            False # return False statement if the data point is outside the cone
        )
        return direction_mask

    def mask_datapoints_to_bin(self, bin_number, positions) -> np.ndarray:
        """
        Positions array must be array-like and have shape (N, 3). This function returns a mask ( = list of True and False with the same length as the first dimension of the positions array) separating the particles as described by the positions array into two groups: Particles, belonging to the defined cone, and ones which do not.
        """
        positions = np.asarray(positions) # data set: Positions of the particles
        direction_mask = self.mask_datapoints_to_direction(positions) # create the direction mask for the dtaa set first
        radial_limits = self.bins[bin_number].bounds # get the radial limits for the bin of interest

        # mask the data according to the respective bin, they are belonging to
        radii = np.sqrt(np.sum(positions**2, axis=1)) # center distance of all particles within the data set
        radii_mask = np.where(
            ((radii >= np.min(radial_limits)) & (radii <= np.max(radial_limits))),
            True, # return True statement if the data point is outside the radial bin
            False # return False statement if the data point is inside the radial bin
        )
        return direction_mask & radii_mask # the intersection of both masks, i.e. particles that satisfy the correct directio AND distance to the origin
    
    def mask_datapoints_to_all_bins(self, positions):
        """
        An optimized routine for the routine to the set of bins, comprising the cone, instead of just one.
        """
        # direction masking
        positions = np.asarray(positions)
        direction_mask = self.mask_datapoints_to_direction(positions)
        
        radii = np.sqrt(np.sum(positions**2, axis=1)) # center distance of all particles within the data set
        digitized_array = np.digitize(radii, self.radial_shell_bounds)

        return [direction_mask & np.where(digitized_array == i, True, False) for i in np.arange(self.bin_count) + 1]







class line_of_sight():
    """
    Defines the line-of-sight of a certain direction as a geometrical object.
    ## Input:
     - origin_direction (optional, defaulting to (0,0,1))
    """
    def __init__(self, **kwargs):
        if "origin_direction" in kwargs: # check wheteher there is a 
            self.direction = np.array(kwargs["origin_direction"]) / magnitude(np.array(kwargs["origin_direction"])) # NEEDS to be normalized for calculating angles later on
        else:
            self.direction = np.array((0., 0., 1.))

    def get_xyz_coords(self, distances):
        return np.outer(distances, self.direction)
    
    def get_cell_chords(self, max_dist, divisions=10000, **kwargs):
        """
        ### Arguments
        As an argument, specify the distances along the line of sight at which the nearest neighbor should be evaluated. Furthermore, you have to pass EXACTLY ONE of the following two arguments to the function:
         - Feed in a spatial data set with an array of dimension (n, 3). You can add this argument with the keyword "GasPos"
         - This function utilizes a cKDTree to find the nearest neightbor within a data set. Generating this tree is really the bottleneck of the computation (in terms of computation time). If the cKDTree already is available from another step, it can be passed as an argument as well to speed up computation time severely. It can be fed into the function with the "tree" argument.
        
        ### Output
        This function returns a tuple with ...
         - in its first entry, the cell ids which the line of sight crosses with at least one grid point
         - in its second entry, the phyical chord length within the cells as determined from the grid points belonging to the respective cell
        
        NOTE: The cells and their IDs are *NOT* sorted in any manner, meaning that their order is arbitrary and determined with the order in the underlying spatial data set itself.
        """
        
        if ("GasPos" in kwargs) and ("tree" in kwargs):
            raise Exception("Overdescription error! Too many parameters are specified. EITHER define the 'tree' parameter OR 'GasPos', but NOT both!")

        if "tree" in kwargs:
            tree = kwargs["tree"]
        elif "GasPos" in kwargs:
            tree = spatial.cKDTree(np.asarray(kwargs["GasPos"]))
        else:
            raise Exception("Underdescription error! No spatial parameters are specified, neither the tree, nor a positional array of the gas cells! EITHER define the 'tree' parameter OR 'GasPos'!")

        distances = np.linspace(0., max_dist, divisions + 1)[1:] # do not include the origin ...
        dist_step = distances[0] # first step = step size
        debug("The step size along the line of sight has been set to", dist_step, "kpc")
        
        deviations, cells = tree.query(self.get_xyz_coords(distances))
        
        cell_ids = np.arange(tree.n)# cell count, the i'th entry corresponds to the i'th cell in the simulation
        counts = np.bincount(cells, minlength=tree.n)
        
        mask_zeros = np.argwhere(counts > 0) # only consider cells through we we actually probe through
        
        cell_ids = cell_ids[mask_zeros]
        counts = counts[mask_zeros]
                
        
        return (np.ravel(cell_ids), np.ravel(counts * dist_step))

    def get_gridpoint_IDs(self, distances, **kwargs): #Define a sequence of distances (either a list or a one-dimensional array of floats) at which physical parameters like the "GasPotential", "GasTemp" etc. can be evaluated.
        """
        Define a sequence of distances (either a list or a one-dimensional array of floats) at which a gridpoint (direction & distance) is unambiguously defined. In addition, the position array must be specified on which basis the closest particles are being identified.
        """
        # makes sure that the input is proper as given by the user
        distances = np.asarray(distances)
        
        points = np.tile(self.direction, (len(distances),1)) # repeat the sightline direction and stack the array
        points *= np.tile(distances, (3, 1)).T # multiply the unit vector with the corresponding array of distances to get the xyz coordinates

        if "tree" in kwargs:
            tree = kwargs["tree"]
        elif "GasPos" in kwargs:
            tree = spatial.cKDTree(np.asarray(kwargs["GasPos"]))
        else:
            raise Exception("Underdescription error! No spatial parameters are specified, neither the tree, nor a positional array of the gas cells! EITHER define the 'tree' parameter OR 'GasPos'!")
        
        #We initialize Kdtree, and look up nearest gas cell to each of the above sightline points:
        _,NearestID = tree.query(points)
        return NearestID





def create_spectrum(dl, number_density, temp, rad_vel, spectrum, ion_mass, wavelength_lims=(1000., 2000.) * u.AA, spec_resolution = 2e4, h=0.7, **kwargs):
    """
    Generates and integrates the spectrum along a given line-of sight on a regular-spaced grid
    - dist_lim          = maximal distance up to which the gas and its optical depth is integrated over (astropy quantity)
    - dl                = interval size over which is being integrated (astropy quantity)
    - number_density    = particle number density of in each cell
    - temp              = quantity array containing the temperatures of the medium at the gridpoints
    - rad_vel           = radial_velocities of the particles
    - spectrum          = which particle spectrum to account for. Should be string with the respective ion (e. g. "O VI")
    - ion_mass          = mass of the ion given as an astropy quantity with dimesion mass
    - wavelength_lims   = defines the wavelengths limits in between which the spectrum is to be generated
    - spec_resolution   = defines the spectral resolution at which the artificial spectrum is being generated. Refers to the R number = Delta lambda / lambda
    """

    tic = time()
    H_0 = h * 100. * u.km / u.s / u.Mpc
    #debug("dl =", dl)
    distances = np.cumsum(dl) # cululative sum of all integration intervals
    N_spatial_stencils = len(dl)
    #debug("Distances:", distances)

    # build the wavelength axis
    wavelengths = np.linspace(
        *wavelength_lims.value, # defines the wavelength interval boundas
        int(spec_resolution * (np.max(wavelength_lims.value) - np.min(wavelength_lims.value)) / np.mean(wavelength_lims.value)) # defines the spectral resolution, wavelengths have to be given in Angstroms
    )

    # convert to frequencies in EHz
    frequencies = (c.c / (wavelengths * u.AA)).to(u.THz).value # transform to frequencies given in exahertz, this gives numbers at the order of a few in the optical regime

    def debug(*args):
        if ("debug" in kwargs):
            if kwargs["debug"] == True:
                print("[DEBUG]", *args)
    
    if "Nist_transitions" in kwargs:
        Nist_transitions = kwargs["Nist_transitions"]
    else:
        Nist_transitions = Nist.query(*wavelength_lims, linename=spectrum, wavelength_type='vacuum') # wavelengths are *vacuum* wavelengths!

    try: # see whether a "spectrum" column has been generated ...
        Nist_transitions = Nist_transitions["Spectrum", "Ritz", "Observed", "Rel.", "Aki", "fik"] # only keep interesting columns
    except: # if not, then proceed without it
        Nist_transitions = Nist_transitions["Observed", "Ritz", "Rel.", "Aki", "fik"] # only keep interesting columns
        
    # only keep those for which also the gamma parameter and the oscillator strength are known
    good_rows = np.squeeze(np.argwhere(~np.isnan(Nist_transitions["fik"]) & ~np.isnan(Nist_transitions["Aki"])))
    Nist_transitions = Nist_transitions[good_rows] 
    #debug(Nist_transitions)

    optical_depth = np.pi * c.e.esu**2 / c.m_e / c.c # t othis point, just a number
    
    
    
    
    nu_0 = (np.expand_dims(c.c / (Nist_transitions["Ritz"] * u.AA), axis=1) * np.expand_dims((1 - (distances * H_0 + rad_vel) / c.c), axis=0)).to(u.THz)
    all_nu_0 = np.expand_dims(nu_0, axis=2)
    debug("shape of all_nu_0: ", all_nu_0.shape)

    
    sigma = (nu_0 * np.sqrt(6*np.pi**2 * c.k_B * np.expand_dims(temp, axis=0) / (2 * ion_mass * c.c**2))).to(u.THz).value # ADJUST THE PROTON MASS WITH THE PROPER MASS OF THE ION
    all_sigma = np.expand_dims(sigma, axis=2)
    debug("shape of all_sigma", all_sigma.shape)

    
    gamma = np.expand_dims((Nist_transitions["Aki"] / u.s / 2.).to(u.THz).value, axis=1)
    all_gamma = np.expand_dims(gamma, axis=2)
    debug("shape of all_gamma", all_gamma.shape)

    
    all_frequencies = np.expand_dims(frequencies, axis=(0, 1)) # create a 3d cube of frequencies over which sides then can be integrated
    debug("shape of all_frequencies", all_frequencies.shape)
    
    
    all_column_densities = np.expand_dims(number_density * dl, axis=(0, 2))
    debug("shape of all_column_densities", all_column_densities.shape)


    optical_depth = (optical_depth * all_column_densities * np.expand_dims(Nist_transitions["fik"], axis=(1, 2))).to(u.THz)
    debug("shape of optical_depth", optical_depth.shape)

    
    # now multiply with the Voigt profile
    optical_depth = optical_depth * voigt_profile(
        all_frequencies - all_nu_0.to(u.THz).value, 
        all_sigma, 
        all_gamma
    ) / u.THz

    if ("cube_out" in kwargs):
        if kwargs["cube_out"] == True:
            
            debug("The calculation of the spectral cube took %.2f sec." % (time() - tic))
            return (wavelengths, optical_depth)
        
    #print("shape of optical_depth", optical_depth.shape)
    optical_depth = np.sum(optical_depth, axis=0) # sum over all spectral lines that have been found
    optical_depth = np.sum(optical_depth, axis=0) # integrate over all spatial grindpoints
    #print("shape of optical_depth", optical_depth.shape)
    optical_depth = optical_depth.to(1).value

    debug("The calculation of the spectral cube took %.2f sec." % (time() - tic))
    return (wavelengths, optical_depth)
