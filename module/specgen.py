import numpy as np
from time import time
from scipy.special import voigt_profile as vp
import astropy.units as u
import astropy.constants as c
from astroquery.nist import Nist
from .config import load_config

CONFIG = load_config()["specgen"]

def create_spectrum(dN, temp, rad_vel, wavelength_lims, spec_resolution, ion_mass, **kwargs):
    """
    Generates and integrates the spectrum along a given line-of sight on a regular-spaced grid
    
    Args:
    
        - dN (astropy.units.Quantity): array of column densities
        - temp (astropy.units.Quantity): quantity array containing the temperatures of the absorber
        - rad_vel (astropy.units.Quantity): radial_velocities of the absorbers
        - wavelength_lims (astropy.units.Quantity): defines the wavelengths limits in between which the spectrum is to be generated
        - spec_resolution (float): defines the spectral resolution at which the artificial spectrum is being generated. Refers to the R number = Delta lambda / lambda
        - ion_mass (astropy.units.Quantity): mass of the ion given as an astropy quantity with dimension mass
    
    
    Kwargs:
    
        - debug (boolean, optional) : Enables the debug mode in set true
        - nist_args (dict, optional) : A dictionary for a nist query. The dictionary must contain the following entries:
            
            - "linename" : Name of the ion (e.g. "Na I")
            - "wavelength_type" : The type of wavelength to retrieve. You may choose between "vacuum" or "air+vac"
            - "wavelength_lims" : The wavelength range to explore. This must be an astropy Quantity or an iterable of length 2 containing astropy quantities. It must be compatible with a length.
        
        - line_dict (dict, optional) : A dictionary that contains information regarding the quantum transitions. It must contains the following entries:
            
            - "Wavelength" (float or astropy.units.Quantity): The central wavelength of the quantum transition. If a floating point value or an array of floating point values is provided they are assumed to be given in units of angstrom.
            - "Aki" (float or astropy.units.Quantity): The transition's Einstein coefficient. If a floating point value or an array of floating point values is provided they are assumed to be given in units of inverse seconds.
            - "fik" (float): The transition's oscillator strength.
            
        - cube_out (boolean): Specify whether the optical depths of all transitions and absorbing systems should be summed up before output. If enabled, this function yields the uncollapsed data output. (defaults to: False)
    
    
    Returns:
    
        2-tuple of np.ndarray. The first entry gives a wavelength range. The second entry yields the cube/spectral information of the computed optical depths. If the cube option is enabled, the cube encodes the following quantities:
        
        Axes:
        
            [0] individual spectral lines
            [1] individual absorbing systems
            [2] spectral axis (varying wavelengths)
    """
    
    if np.asarray(dN).ndim == 0:
        dN = np.expand_dims(dN, axis=0)
        
    if np.asarray(temp).ndim == 0:
        temp = np.expand_dims(temp, axis=0)
        
    if np.asarray(rad_vel).ndim == 0:
        rad_vel = np.expand_dims(rad_vel, axis=0)

    def debug(*args):
        if ("debug" in kwargs):
            if kwargs["debug"] == True:
                print("[DEBUG]", *args)

    start_timer = time()

    # build the wavelength axis
    wavelengths = np.linspace(
        *wavelength_lims.value, # defines the wavelength interval boundas
        int(spec_resolution * (np.max(wavelength_lims.value) - np.min(wavelength_lims.value)) / np.mean(wavelength_lims.value)) # defines the spectral resolution, wavelengths have to be given in Angstroms
    )

    # convert to frequencies in THz
    frequencies = (c.c / (wavelengths * u.AA)).to_value(u.THz) # transform to frequencies given in terahertz, this gives numbers at the order of a few in the optical regime
    
    if "nist_args" in kwargs:
        nist_args = kwargs["nist_args"]
    
        NIST_transitions = Nist.query(
            *nist_args['wavelength_lims'], 
            linename=nist_args['linename'], 
            wavelength_type=nist_args['wavelength_type']
        ) # wavelengths are *vacuum* wavelengths!

        try: # see whether a "spectrum" column has been generated ...
            transitions = NIST_transitions["Spectrum", "Observed", "Rel.", "Aki", "fik"] # only keep interesting columns
            transitions["Wavelength"] = NIST_transitions["Ritz"]
        
        except: # if not, then proceed without it
            transitions = NIST_transitions["Observed", "Rel.", "Aki", "fik"] # only keep interesting columns
            transitions["Wavelength"] = NIST_transitions["Ritz"]
        
        # only keep those for which also the gamma parameter and the oscillator strength are known
        good_rows = np.squeeze(np.argwhere(~np.isnan(transitions["fik"]) & ~np.isnan(transitions["Aki"])))
        transitions = transitions[good_rows] 
        debug(NIST_transitions)
            
    elif "line_dict" in kwargs:
        """
        line_dict = {
            "Wavelength" : array of central wavelengths,
            "fik" : array of oscillator strengths,
            "Aki" : array of einstein coefficients
        }
        """
        transitions = kwargs["line_dict"]
        
        if isinstance(transitions["Wavelength"], u.Quantity):
            transitions["Wavelength"] = transitions["Wavelength"].to_value(u.AA)
        
        if isinstance(transitions["Aki"], u.Quantity):
            transitions["Aki"] = transitions["Aki"].to_value(u.s**-1)
        
        if np.asarray(transitions["Wavelength"]).ndim == 0:
            transitions["Wavelength"] = np.expand_dims(transitions["Wavelength"], axis=0)
        
        if np.asarray(transitions["Aki"]).ndim == 0:
            transitions["Aki"] = np.expand_dims(transitions["Aki"], axis=0)
        
        if np.asarray(transitions["fik"]).ndim == 0:
            transitions["fik"] = np.expand_dims(transitions["fik"], axis=0)
    
    else:
        raise Exception("[ERROR] You need to either specify the input parameters for a nist query or you need to plug in the respective parameters manually.")

    optical_depth = np.pi * c.e.esu**2 / c.m_e / c.c # to this point, just a number
    
    
    nu_0 = (np.expand_dims(c.c / (transitions["Wavelength"] * u.AA), axis=1) * np.expand_dims((1 - rad_vel / c.c), axis=0)).to(u.THz)
    all_nu_0 = np.expand_dims(nu_0, axis=2)
    debug("shape of all_nu_0: ", all_nu_0.shape)

    
    sigma = (nu_0 * np.sqrt(6*np.pi**2 * c.k_B * np.expand_dims(temp, axis=0) / (2 * ion_mass * c.c**2))).to(u.THz).value
    all_sigma = np.expand_dims(sigma, axis=2)
    debug("shape of all_sigma", all_sigma.shape)

    
    gamma = np.expand_dims((transitions["Aki"] / u.s / 2.).to(u.THz).value, axis=1)
    all_gamma = np.expand_dims(gamma, axis=2)
    debug("shape of all_gamma", all_gamma.shape)

    
    all_frequencies = np.expand_dims(frequencies, axis=(0, 1)) # create a 3d cube of frequencies over which sides then can be integrated
    debug("shape of all_frequencies", all_frequencies.shape)
    
    
    all_column_densities = np.expand_dims(dN, axis=(0, 2))
    debug("shape of all_column_densities", all_column_densities.shape)


    optical_depth = (optical_depth * all_column_densities * np.expand_dims(transitions["fik"], axis=(1, 2))).to(u.THz)
    debug("shape of optical_depth", optical_depth.shape)

    
    # now multiply with the Voigt profile
    optical_depth = optical_depth * vp(
        all_frequencies - all_nu_0.to_value(u.THz), 
        all_sigma, 
        all_gamma
    ) / u.THz

    if ("cube_out" in kwargs):
        if kwargs["cube_out"] == True:
            
            debug("The calculation of the spectral cube took %.2f sec." % (time() - start_timer))
            return (wavelengths, optical_depth)
        
    optical_depth = np.sum(optical_depth, axis=0) # sum over all spectral lines that have been found
    optical_depth = np.sum(optical_depth, axis=0) # integrate over all individual systems
    optical_depth = optical_depth.to(1).value

    debug("The calculation of the spectral cube took %.2f sec." % (time() - start_timer))
    
    return (wavelengths, optical_depth)



def make_CoG(Nrange, temp, line_dict, ion_mass, **kwargs):
    """This function generates a curve of growth for a given set of atomic parameters. They may be either the result of a query from NIST or fed in by the user manually.
    """
    Nrange = np.asarray(Nrange)
    temp = np.full_like(Nrange, temp)
    rad_vel = np.zeros_like(Nrange)
    
    if not isinstance(line_dict, dict):
        raise IOError("[ERROR] You must provide a dictionary containing all the quantum transition's information. Instead, you provided: ", type(line_dict))
    
    if "Wavelength" not in line_dict:
        raise IOError("[ERROR] It seems like you did not specify the central wavelength of the transition you want to calculate the curve-of-growth for. You need to specify the wavelength with the 'Wavelength' entry in the dictionary. Your dictionary has the following entries: ", line_dict.keys())
    
    if "fik" not in line_dict:
        raise IOError("[ERROR] It seems like you did not specify the oscillator strength of the transition you want to calculate the curve-of-growth for. You need to specify the oscillator strength with the 'fik' entry in the dictionary. Your dictionary has the following entries: ", line_dict.keys())
    
    if "Aki" not in line_dict:
        raise IOError("[ERROR] It seems like you did not specify the transition's Einstein coefficient of the line that you want to calculate the curve-of-growth for. You need to specify the Einstein coefficient with the 'Aki' entry in the dictionary. Your dictionary has the following entries: ", line_dict.keys())
    
    central_wavelength = line_dict["Wavelength"]
    
    wavelength, spectrum = create_spectrum(
        Nrange[0], 
        temp, 
        rad_vel, 
        [
            central_wavelength * (1. - CONFIG["cog"]["integration_range_factor"]),
            central_wavelength * (1. + CONFIG["cog"]["integration_range_factor"])
        ] * u.AA, 
        central_wavelength / CONFIG["cog"]["integration_resolution"], 
        ion_mass, 
        line_dict=line_dict
    )
    
    wavelength_increment = wavelength[1] - wavelength[0]
    
    all_spectra = np.tile(spectrum, (1, Nrange.size))
    column_dens = np.expand_dims(Nrange, axis=1)
    
    ews = np.sum(1. - np.exp(all_spectra * column_dens / Nrange[0]), axis=0) * wavelength_increment
    
    return ews