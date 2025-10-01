### helper functions for radiation calculations

import numpy as np

import scipy as sp
from scipy.constants import speed_of_light as c
import scipy.constants
from scipy.constants import sigma as SB_sigma
import metpy.calc as mpcalc
from metpy.units import units


from importlib import reload

# create idealized moist-adiabatic profile
def create_profile_moist(T_s, rh, p):
    ''' 
    create an idealized profile with given surface temperature that follows
    a moist adiabatic lapse rate until a 200K isothermal stratosphere

    IN:
        T_s [K]: given surface temperature
        rh [%]: given relative humidity, constant throughout atmosphere
        
    OUT:
        T_profile [K]: temperature profile
        vmr [unitless]: water vapor volume mixing ratio
        q_star_s [kg/kg]: saturation mass mixing ratio of water vapor at the surface        
    
    '''
    T_s = units.Quantity(T_s, 'K')
    #rh = units.Quantity(rh, 'dimensionless')
    # create pressure, temperature, and mixing ratio profiles for each initial temp
    pressure_profile = units.Quantity(p, 'Pa')
    pressure_profile_temp = p[np.where(p > 100)[0]]
    pressure_profile_temp = units.Quantity(pressure_profile_temp, 'Pa')
    p_s = units.Quantity(p[0], 'Pa')

    temperature = mpcalc.moist_lapse(pressure_profile_temp, T_s, p_s)
    idx = np.where(temperature > units.Quantity(200, 'K'))[0]
    T_profile = units.Quantity(200, 'K')*np.ones(len(p))
    T_profile[:idx[-1] + 1] = temperature[idx]

    mass_mixing_ratio = mpcalc.mixing_ratio_from_relative_humidity(pressure_profile, T_profile, rh).to('kg/kg') 
    q_star_s = mpcalc.saturation_mixing_ratio(p_s, T_profile[0]).to('kg/kg')

    # mmr = vmr * molar mass / molar mass of dry air
    vmr = mass_mixing_ratio * (29 / 18.015)       
    
    return T_profile, vmr, q_star_s


# Planck function
def B_nu(T, nu):
    """
    Planck function
    
    In:
    T [K]: temperature
    nu [cm-1]: multiply all nu's by 100 to convert to m-1 in formula, then multiply
               B by 100 for units of W/m^2/sr/cm-1
        
    Out:
        Planck function in units of W/m^2/sr/cm-1
    """
    k_B = scipy.constants.k # Boltzmann constant
    h = scipy.constants.h # Planck constant
    c = scipy.constants.speed_of_light # speed of light in a vacuum
    
    return ((2*h*(c**2)*((100*nu)**3))/(np.exp((h*c*(100*nu))/(k_B*T)) - 1))*100


def dPlanckdT_nu(nu,T):
    """
    Helper for dPlanckdT_n
    Derivative of Planck function with respect to temperature
    In:
        nu [m^-1]: frequency
        T [K]: temperature
    Out:
        dB_nu/dT [W/m^2/sr/m^-1/K]
    
    """
    k_B = scipy.constants.k # Boltzmann constant
    h = scipy.constants.h # Planck constant
    c = scipy.constants.speed_of_light # speed of light in a vacuum


    u = h*nu/(k_B*T)
    u[u>500.] = 500. #To prevent overflow
    return 2.*h**2/(k_B*c**2) * (nu**4/T**2) * np.exp(u)/( (np.exp(u)-1.)**2 )



def dPlanckdT_n(n,T,unit="cm^-1"):
    """
    ### From Koll et al., 2023
    # note: assume 'n' is in cm^-1!

    Derivative of Planck function with respect to temperature

    In:
        n [cm^-1]: wavenumber
        T [K]: temperature
    """
    if unit=="m^-1":
        k = 1.
    elif unit=="cm^-1":
        k = 100.
    else:
        print( "(Planck_n) Error: unit not recognized!")

    k_B = scipy.constants.k # Boltzmann constant
    h = scipy.constants.h # Planck constant
    c = scipy.constants.speed_of_light # speed of light in a vacuum


    return (k*c) * dPlanckdT_nu( c*n*k ,T)

def BT(F, nu):
    """
    Brightness temperature from flux

    In:
        F [W/m^2/cm-1]: flux
        nu [cm-1]: wavenumber
    Out:
        Brightness temperature [K]
    """
    k_B = scipy.constants.k # Boltzmann constant
    h = scipy.constants.h # Planck constant
    c = scipy.constants.speed_of_light # speed of light in a vacuum
    return ((k_B/(h*c*100*nu))*np.log(((100*np.pi*2*h*(c**2)*(100*nu)**3)/(F)) + 1))**(-1)


def BT_from_OLR(OLR):
    """
    broadband brightness temperature from OLR
    In:
        OLR [W/m^2]: outgoing longwave radiation
    Out:        
        T [K]: brightness temperature
    """
    # compute brightness temperature
    T = (OLR/(SB_sigma))**(1/4)
    return T


# draw idealized exponential absorption coefficient
def k(k_0, nu_0, l, wavenumbers):
    """
    calculate idealized exponential absorption coefficient
    
    In:
        k_0 [m^2/kg]: height of peak of the absorption band
        nu_0 [cm-1]: center of the band or where peak is, in wavenumber
        l [cm-1]: exponential slope of the coefficient
        wavenumbers [cm-1]: wavenumber range across which to calculate

    Out:
        k [m^2/kg]
    """
    return k_0 * np.exp(-(np.abs(wavenumbers - nu_0*np.ones(len(wavenumbers)))/l))
