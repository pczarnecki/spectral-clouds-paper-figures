"""
This script will run the line-by-line radiative transfer model ARTS to compute fluxes up and down through an idealized atmosphere defined by a moist adiabat below an isothermal stratosphere at 200 K. It relies on the following packages/modules:
- konrad: to define a standard pressure grid and interface with ARTS
- metpy_calc: to create the moist adiabat
- FluxSimulator: to run the radiative transfer simulations
- pyarts: the ARTS python interface
"""

### ARTS as the line-by-line code
import os
# set global variables BEFORE importing ARTS
os.environ['ARTS_DATA_PATH'] = 'path/to/arts-cat-data'

import pyarts
from pyarts import xml
from pyarts.arts import GriddedField4
from pyarts.arts import convert

# the fluxsim module
import FluxSimulator as fsm

# helper packages
import numpy as np
import xarray as xr

import absorption_module as am

import konrad

def define_atmosphere_interp(T_s, col_RH, CTT, LWC=0, IWC=0):
    """
    define an idealized atmosphere with a water or ice cloud at the specified temperature level
    IN:
        T_s [K]: surface temperature
        col_RH [%]: column relative humidity
        CTT [K]: cloud top temperature
        LWC [kg/m^3]: liquid water content
        IWC [kg/m^3]: ice water content
    OUT:
        atm_field: atmosphere field for ARTS
        true_CTT: exact cloud tp temperature from the interpolated profile
    """
    import scipy.interpolate as interp

    # Define standard grids
    plev, phlev = konrad.utils.get_pressure_grids(1000e2, 1e2, 128)

    # Get temperature and water vapor profiles
    T, h2o_vmr, q = am.create_profile_moist(T_s, col_RH, plev)

    # Interpolate to find exact pressure level corresponding to desired CTT
    T_profile = T.magnitude
    plev_log = np.log(plev)  # use log-pressure for better interpolation stability

    # Interpolate pressure at exact CTT
    p_CTT_log = interp.interp1d(T_profile[::-1], plev_log[::-1], kind='linear')(CTT)
    p_CTT = np.exp(p_CTT_log)

    # Extend the pressure and temperature profiles to include this exact level
    plev_ext = np.sort(np.append(plev, p_CTT))
    T_interp = interp.interp1d(plev, T_profile, fill_value="extrapolate")
    T_ext = T_interp(plev_ext)

    h2o_interp = interp.interp1d(plev, h2o_vmr.magnitude, fill_value="extrapolate")
    h2o_ext = h2o_interp(plev_ext)

    # Set LWC and IWC only at the interpolated CTT level
    LWC_profile = np.zeros_like(plev_ext)
    IWC_profile = np.zeros_like(plev_ext)
    idx_CTT = np.argmin(np.abs(plev_ext - p_CTT))
    LWC_profile[idx_CTT] = LWC
    IWC_profile[idx_CTT] = IWC

    atm_field = fsm.generate_gridded_field_from_profiles(
        plev_ext[::-1].tolist(),
        T_ext[::-1],
        gases={
            "H2O": h2o_ext[::-1],
            "CO2": 420e-6,
        },
        particulates={
            "LWC-mass_density": LWC_profile[::-1],
            "IWC-mass_density": IWC_profile[::-1],
        },
        z_field=None
    )

    return atm_field, T_ext[idx_CTT]


### define parameter ranges for which to run the line-by-line calculations
Ts_array = np.array([290]) # surface temperatures
colRH_array = np.array([10e-10, 0.25, 0.5, 0.75, 0.99]) # column relative humidities
CTT_array = np.array([210, 220, 230, 240, 250, 260, 270, 280]) # cloud top temperatures
nu = np.arange(0.1, 1500, 0.01) # wavenumbers

# containers for ARTS atmospheres
atms = []
true_CTTs = []
altitudes = []
T_s = []

# containers for fluxes
cs_spec_up = np.empty((len(Ts_array), len(colRH_array), len(CTT_array), 129, len(nu)))
cs_spec_down = np.empty((len(Ts_array), len(colRH_array), len(CTT_array), 129, len(nu)))
cs_up = np.empty((len(Ts_array), len(colRH_array), len(CTT_array), 129))
cs_down = np.empty((len(Ts_array), len(colRH_array), len(CTT_array), 129))

as_spec_up = np.empty((len(Ts_array), len(colRH_array), len(CTT_array), 129, len(nu)))
as_spec_down = np.empty((len(Ts_array), len(colRH_array), len(CTT_array), 129, len(nu)))
as_up = np.empty((len(Ts_array), len(colRH_array), len(CTT_array), 129))
as_down = np.empty((len(Ts_array), len(colRH_array), len(CTT_array), 129))


## loop through parameters
for colRH_idx, colRH in enumerate(colRH_array):
    for CTT_idx, CTT in enumerate(CTT_array):
        for Ts_idx, Ts in enumerate(Ts_array):
            # check if cloud is colder than surface
            if Ts > CTT:

                ### define ARTS atmospheres
                atm, true_CTT = define_atmosphere_interp(Ts, colRH, CTT, 0, 1) # very optically-thick cloud
                atms.append(atm)
                true_CTTs.append(true_CTT)
                altitudes.append(atm[1][0][0][0])
                T_s.append(Ts)

                # standard longwave values
                surface_reflectivity_lw = 0.0
                geographical_positions_lw = [0, 0]
                sun_position = [1.495978707e11, 0.0, 0.0]

                # run FluxSimulator Module
                FluxSimulator_LW = fsm.FluxSimulator("lw")
                FluxSimulator_LW.set_paths(lut_path='/../data/pc2943/allsky/') #set path to existing LUT

                # convert wavenumber to frequency
                FluxSimulator_LW.ws.f_grid = convert.kaycm2freq(nu)

                # set the gas species
                FluxSimulator_LW.set_species(
                    [
                        "H2O, H2O-SelfContCKDMT400",
                        "CO2, CO2-CKDMT252",
                    ]
                )

                # point FluxSimulator to scattering data (available with ARTS release)
                FluxSimulator_LW.set_paths(basename_scatterer="/data/pc2943/scattering_data/")

                # Setup scatterers for the longwave simulation
                FluxSimulator_LW.define_particulate_scatterer(
                    "LWC", "pnd_agenda_CGLWC", "MieSpheres_H2O_liquid", ["mass_density"]
                )

                FluxSimulator_LW.define_particulate_scatterer(
                    "IWC", "pnd_agenda_CGIWC", "MieSpheres_H2O_ice", ["mass_density"]
                
                )

                # define sun (not needed for longwave, but required by ARTS)
                FluxSimulator_LW.sunspectrumtype = 'arts-xml-data/star/Sun/solar_spectrum_July_2008.xml'

                # run FluxSimulator
                results_lw = FluxSimulator_LW.flux_simulator_single_profile(
                    atm,
                    Ts,
                    atm[1][0][0][0],  # surface altitude
                    surface_reflectivity_lw,
                    geographical_positions_lw,
                )

                # assign dictionary values to more convenient output
                cs_spec_up[Ts_idx, colRH_idx, CTT_idx, :, :], cs_spec_down[Ts_idx, colRH_idx, CTT_idx, :, :], cs_up[Ts_idx, colRH_idx, CTT_idx, :], cs_down[Ts_idx, colRH_idx, CTT_idx, :] = results_lw['spectral_flux_clearsky_up'].T, results_lw['spectral_flux_clearsky_down'].T, results_lw['flux_clearsky_up'].T, results_lw['flux_clearsky_down'].T

                as_spec_up[Ts_idx, colRH_idx, CTT_idx, :, :], as_spec_down[Ts_idx, colRH_idx, CTT_idx, :, :], as_up[Ts_idx, colRH_idx, CTT_idx, :], as_down[Ts_idx, colRH_idx, CTT_idx, :] = results_lw['spectral_flux_allsky_up'].T, results_lw['spectral_flux_allsky_down'].T, results_lw['flux_allsky_up'].T, results_lw['flux_allsky_down'].T

### save as xarray
results = xr.Dataset({
    "cs_spec_up": (("Ts", "colRH", "CTT", "level", "nu"), cs_spec_up),
    "cs_spec_down": (("Ts", "colRH", "CTT", "level", "nu"), cs_spec_down),
    "cs_up": (("Ts", "colRH", "CTT", "level"), cs_up),
    "cs_down": (("Ts", "colRH", "CTT", "level"), cs_down),
    "as_spec_up": (("Ts", "colRH", "CTT", "level", "nu"), as_spec_up),
    "as_spec_down": (("Ts", "colRH", "CTT", "level", "nu"), as_spec_down),
    "as_up": (("Ts", "colRH", "CTT", "level"), as_up),
    "as_down": (("Ts", "colRH", "CTT", "level"), as_down),
},
    coords={
        "Ts": Ts_array,
        "colRH": (("colRH"), colRH_array),
        "true_CTT": (("true_CTT"), true_CTTs),
        "CTT": (("CTT"), CTT_array),
        "level": np.arange(129),
        "nu": (("nu"), nu),
    },
)

results.to_netcdf("save/to/file")
