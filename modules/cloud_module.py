## idealized model functions for cloudy-sky calculations

import numpy as np
import cs_module as cs
import absorption_module as am
from importlib import reload
reload(cs)
reload(am)

def get_Trad_total_cloud(nu, Ts, Tstrat, T_cloud, gammaLR, RH, qco2, T0=300.):
    """
    calculate overall radiating temperature, including cloud
    -> defined as minimum at each wavenumber
    In:
        nu [cm-1]: wavenumber array
        Ts [K]: surface temperature
        Tstrat [K]: stratospheric temperature
        gammaLR [K/km]: bulk atmospheric lapse rate
        RH [-]: relative humidity (0-1)
        qco2 [-]: molar mixing ratio of CO2
        T0 [K]: reference temperature
    Out:
        Trad [K]: radiating temperature
    """
    Tco2 = cs.get_Trad_co2(nu, Ts, gammaLR, qco2)
    Th2o = cs.get_Trad_h2o(nu, Ts, gammaLR, RH, T0=T0)
    Tcntm = cs.get_Trad_cntm(nu, gammaLR, RH, T0=T0)        
    T_cld = T_cloud * np.ones_like(nu)
    
    Tatm = np.minimum(np.minimum(np.minimum(Tco2,Th2o),Tcntm), T_cld)
    Trad = np.maximum(np.minimum(Ts,Tatm),Tstrat)
    return Trad

# get intersections of gas emission with cloud/surface emission
# adapted from Koll et al., 2023

def constrain_nu(nu, nu_em):
    """
    ensure wavenumbers are in bounds
    In:
        nu [cm-1]: wavenumber array
        nu_em [cm-1]: wavenumber to constrain
    Out:
        nu_em [cm-1]: constrained wavenumber
    """
    return np.maximum(np.minimum(nu_em, nu[-1]), nu[0])

def nu_em_H2O(nu_0, l, k, T_em, T_s, gammaLR, rh, T0=300):
    """
    get intersection wavenumbers of H2O line emission with cloud/surface emission
    In:
        nu_0 [cm-1]: center of the band or where peak is, in wavenumber
        l [cm-1]: exponential slope of the coefficient
        k [m^2/kg]: height of peak of the absorption band
        T_em [K]: emission temperature where we want to find the intersection
        T_s [K]: surface temperature
        gammaLR [K/km]: bulk atmospheric lapse rate
        rh [-]: relative humidity (0-1)
        T0 [K]: reference temperature
    Out:
        R [cm-1]: right intersection wavenumber
        L [cm-1]: left intersection wavenumber
    """
    esat0 = cs.esat(T0)
    gammaWV = cs.Lvap/(cs.Rv * T0)

    # calculate reference optical depth
    tau_star = (cs.R/cs.Rv) * k*esat0/(cs.g*cs.D)
    
    # invert for intersection
    term = (rh*tau_star/(1 + gammaWV*gammaLR)) * (T0/T_s)**(1/gammaLR) * (T_em/T0)**((1 + gammaWV*gammaLR)/gammaLR)
    
    R = nu_0 + l*np.log(term)
    L = nu_0 - l*np.log(term)
    
    return np.maximum(0, R), np.maximum(0, L)

def nu_em_CO2(q, T_em, T_s, gammaLR):
    """
    get intersection wavenumbers of CO2 line emission with cloud/surface emission
    In:
        q [-]: molar mixing ratio of CO2
        T_em [K]: emission temperature where we want to find the intersection
        T_s [K]: surface temperature
        gammaLR [K/km]: bulk atmospheric lapse rate
        rh [-]: relative humidity (0-1)
        T0 [K]: reference temperature
    Out:
        R [cm-1]: right intersection wavenumber
        L [cm-1]: left intersection wavenumber
    """
    nu_0 = 667.5
    tau_star = cs.kappa_co2*cs.ps_dry/(2*cs.g*cs.D)
    qco2 = cs.convert_molar_to_mass_ratio(q, cs.R_co2, cs.R)
    term = qco2 * tau_star * (T_em/T_s)**(2/gammaLR)
    
    R = nu_0 + cs.l_co2*np.log(term)
    L = nu_0 - cs.l_co2*np.log(term)
    
    return np.maximum(0, R), np.maximum(0, L)

def nu_em_cont(T_em, gammaLR, rh, T0=300):
    """
    get intersection wavenumbers of H2O self-continuum emission with cloud/surface emission 
    In:
        T_em [K]: emission temperature where we want to find the intersection
        gammaLR [K/km]: bulk atmospheric lapse rate
        rh [-]: relative humidity (0-1)
        T0 [K]: reference temperature
    
    Out:    
        R [cm-1]: right intersection wavenumber
        L [cm-1]: left intersection wavenumber
    """

    ## get continuum parameters
    k = cs.kappaX0
    esat0 = cs.esat(T0)
    a = 7.
    gammaWV = cs.Lvap/(cs.Rv * T0)
    l = cs.lX
    nu_0 = 700

    # calculate reference optical depth
    tau_star = (cs.R/cs.Rv) * k*esat0/(cs.g*cs.D)
    
    # perform inversion for intersection
    term = (T_em/T0)**(2*gammaWV - a) * (rh**2 * tau_star)/((2*gammaWV - a)*gammaLR)
    
    R = nu_0 + l * np.log(term)
    L = nu_0 - l * np.log(term)
    
    return np.maximum(0, R), np.maximum(0, L)


def calc_model_CRE(nu, T_s, T_em_cloud, gammaLR, RH, qco2):
    """
    calculate cloud radiative effect (CRE) at TOA using geometric view
    In:
        nu [cm-1]: wavenumber array
        T_s [K]: surface temperature
        T_em_cloud [K]: cloud top temperature
        gammaLR [K/km]: bulk atmospheric lapse rate
        RH [-]: relative humidity (0-1)
        qco2 [-]: molar/volume mixing ratio of CO2
    Out:
        CRE [W/m^2]: cloud radiative effect    
    """

    # get H2O line emission temperature
    T_em_H2O_all = np.minimum(cs.get_Trad_h2o(nu, T_s, gammaLR, RH), T_s)

    # get left and right intersection of rotational band with the cloud
    nu_H2O_1_R, nu_H2O_1_L = nu_em_H2O(150, cs.ln0, cs.kappa0, T_em_cloud, T_s, gammaLR, RH)
    nu_H2O_1_R, nu_H2O_1_L = constrain_nu(nu, nu_H2O_1_R), constrain_nu(nu, nu_H2O_1_L)

    # get left and right intersection of rotational band with the surface
    nu_H2O_1_R_s, nu_H2O_1_L_s = nu_em_H2O(150, cs.ln0, cs.kappa0, T_s, T_s, gammaLR, RH)
    nu_H2O_1_R_s = np.maximum(150, nu_H2O_1_R_s)
    nu_H2O_1_R_s, nu_H2O_1_L_s = constrain_nu(nu, nu_H2O_1_R_s), constrain_nu(nu, nu_H2O_1_L_s)
    
    # get left and right intersection of CO2 band with the cloud
    nu_CO2_R, nu_CO2_L = nu_em_CO2(qco2, T_em_cloud, T_s, gammaLR)
    nu_CO2_R, nu_CO2_L = constrain_nu(nu, nu_CO2_R), constrain_nu(nu, nu_CO2_L)

    # get left and right intersection of CO2 band with the surface
    nu_CO2_R_s, nu_CO2_L_s = nu_em_CO2(qco2, T_s, T_s, gammaLR)
    nu_CO2_R_s, nu_CO2_L_s = constrain_nu(nu, nu_CO2_R_s), constrain_nu(nu, nu_CO2_L_s)
    
    # figure out where to cut off the continuum
    co2_idx = np.where(nu > nu_CO2_R)[0][0]

    T_em_co2_all = np.minimum(cs.get_Trad_co2(nu, T_s, gammaLR, qco2), T_s)
    T_em_min = np.minimum(np.minimum(T_em_H2O_all[:co2_idx], T_em_co2_all[:co2_idx]), T_s)
    T_em_co2_L = np.max(T_em_min)

    # calculate A1 region (appendix B in Czarnecki and Pincus)
    A1 = (1/2) * (np.pi * am.B_nu(T_em_co2_L, np.mean([nu_CO2_L_s, nu_H2O_1_R_s])) - np.pi * am.B_nu(T_em_cloud, np.mean([nu_CO2_L, nu_H2O_1_R]))) * (np.max([0, nu_CO2_L_s - nu_H2O_1_R_s]) + np.max([0, nu_CO2_L - nu_H2O_1_R]))


    # find continuum emission temperature
    T_em_cont_all = np.minimum(cs.get_Trad_cntm(nu, gammaLR, RH)[co2_idx:], T_s)
    T_em_H2O_2_all = np.minimum(cs.get_Trad_h2o(nu, T_s, gammaLR, RH)[co2_idx:], T_s)

    # find right intersection of continuum with the surface
    nu_cont_R = nu_em_cont(T_s, T_s, gammaLR, RH)[0]
    nu_cont_R = constrain_nu(nu, nu_cont_R)

    # is continuum entirely optically thick? or do we have to account for overlap between the
    # continuum and the v-r band?
    T_em_min = np.minimum(np.minimum(T_em_cont_all, T_em_H2O_2_all), T_s)
    T_em_cont_R = np.max(T_em_min)
    
    if np.max(T_em_cont_all) == T_s:
        nu_cont_R = nu_em_H2O(1500, cs.ln1, cs.kappa1, T_s, T_s, gammaLR, RH)[1]
        nu_cont_R = constrain_nu(nu, nu_cont_R)
        
    # calculate the intersections of the v-r band with the cloud
    nu_H2O_2_R, nu_H2O_2_L = nu_em_H2O(1500, cs.ln1, cs.kappa1, T_em_cloud, T_s, gammaLR, RH)
    nu_H2O_2_R, nu_H2O_2_L = constrain_nu(nu, nu_H2O_2_R), constrain_nu(nu, nu_H2O_2_L)

    T_em_min = np.minimum(np.minimum(T_em_cont_all, T_em_co2_all[co2_idx:]), T_s)
    T_em_co2_R = np.min(T_em_min)
    T_em_cont_L = np.min(np.minimum(T_em_cont_all, T_s))

    # calculate A2_1 and A2_2 regions (appendix B in Czarnecki and Pincus)
    A2_1 = (1/2)*(np.pi* am.B_nu(T_em_cont_L, nu_CO2_R) - np.pi * am.B_nu(T_em_cloud, nu_CO2_R)) * np.max([0, nu_cont_R - nu_CO2_R])
    A2_2 = (1/2)*(np.pi * am.B_nu(T_em_cont_R, nu_cont_R) - np.pi * am.B_nu(T_em_cloud, nu_cont_R)) * np.max([0, nu_H2O_2_L - nu_CO2_R])

    A2 = A2_1 + A2_2    
    
    CRE = A1 + A2
    
    return CRE

def OLR_from_Tem(nu, T_em):
    spectral_OLR = np.pi*am.B_nu(T_em, nu)

    int_OLR = np.trapz(spectral_OLR, nu)

    return spectral_OLR, int_OLR

### clear-sky feedbacks in cloudy columns 
# adapted from Koll et al., 2023

## constants 
c_surf = 0.82
c_co2 = 0.7
c_h2o = 0.56
c_cnt = 0.41

def get_lambda_h2o(nu, Ts, Tstrat, Tcloud, RH, qco2, T0=300., gammaLR=None, dgammaLRdTs=None):
    """
    water vapor feedback above cloud top
    In:
        nu [cm-1]: wavenumber array
        Ts [K]: surface temperature
        Tstrat [K]: stratospheric temperature
        Tcloud [K]: cloud top temperature
        RH [-]: relative humidity (0-1)
        qco2 [-]: molar mixing ratio of CO2
        T0 [K]: reference temperature
        gammaLR [K/km]: bulk atmospheric lapse rate (if None, compute)
        dgammaLRdTs [K/km/K]: derivative of bulk atmospheric lapse rate w.r.t. surface temperature (if None, compute)
    Out:
        lambda_h2o [W/m^2/K]: water vapor feedback above cloud top
    
    """
    #
    Ts = np.atleast_1d(Ts)

    # ---
    # get gammaLR
    if gammaLR is None:
        gammaLR = cs.get_gammaLR(Ts, Tstrat)  # if not defined, use analytical approx
    else:
        pass   # take as input...

    # get d(gammaLR)/dTs
    if dgammaLRdTs is None:
        dTs = 1.   # compute derivative numerically for now -> assumes a moist-adiabatic response!
        dgammaLRdTs = (cs.get_gammaLR(Ts+dTs, Tstrat) - cs.get_gammaLR(Ts, Tstrat))/dTs   # should be <0
    else:
        pass   # take as input...
    
    # get intersection of h2o rot band with cloud top
    nu_H2O_1_R, nu_H2O_1_L = nu_em_H2O(150, cs.ln0, cs.kappa0, Tcloud, Ts, gammaLR, RH)
    nu_H2O_1_R, nu_H2O_1_L = constrain_nu(nu, nu_H2O_1_R), constrain_nu(nu, nu_H2O_1_L)

    # get intersection of co2 band with cloud top
    nu_CO2_R, nu_CO2_L = nu_em_CO2(qco2, Tcloud, Ts, gammaLR)
    nu_CO2_R, nu_CO2_L = constrain_nu(nu, nu_CO2_R), constrain_nu(nu, nu_CO2_L)
    
    # where to evaluate the Planck function
    nu_mid_1 = np.mean([nu_H2O_1_R, nu_H2O_1_L])
    dnu_1 = np.maximum(0, nu_H2O_1_R - nu_H2O_1_L)
    
    gammaWV = cs.Lvap/(cs.Rv * T0)
    esat0 = cs.esat(T0)
     
    # ...
    # get H2O emission temperature (rot band)
    kappa_1 = cs.get_kappa_h2o(150, cs.ps_dry)
    tau0_star = cs.R/cs.Rv * kappa_1*esat0/(cs.g*cs.D)

    Th2o = np.minimum(cs.get_Trad_h2o(nu_mid_1, Ts, gammaLR, RH), Tcloud)
    dTh2odTs_ts = 1./(1. + gammaLR*gammaWV) * Th2o/Ts
    dTh2odTs_gamma = (gammaLR*gammaWV -gammaWV*np.log(Ts/T0) + np.log((1.+gammaLR*gammaWV)/(RH*tau0_star))) / ((1. + gammaLR*gammaWV)**2) * Th2o

    # get H2O emission temperature (v-r band)
    kappa_2 = cs.get_kappa_h2o(1500, cs.ps_dry)
    tau1_star = cs.R/cs.Rv * kappa_2 *esat0/(cs.g*cs.D) 

    # get intersection of v-r band with cloud top
    nu_H2O_2_R, nu_H2O_2_L = nu_em_H2O(1500, cs.ln1, cs.kappa1, Tcloud, Ts, gammaLR, RH)
    nu_H2O_2_R, nu_H2O_2_L = constrain_nu(nu, nu_H2O_2_R), constrain_nu(nu, nu_H2O_2_L)

    # where to evaluate Planck function
    nu_mid_2 = np.mean([nu_H2O_2_R, nu_H2O_2_L])
    dnu_2 = np.maximum(0, nu_H2O_2_R - nu_H2O_2_L)
    Th2o_2 = np.minimum(cs.get_Trad_h2o(nu_mid_2, Ts, gammaLR, RH), Tcloud)

    # derivatives at v-r band (new from Koll et al, 2023)
    dTh2odTs_ts_2 = 1./(1. + gammaLR*gammaWV) * Th2o_2/Ts
    dTh2odTs_gamma_2 = (gammaLR*gammaWV -gammaWV*np.log(Ts/T0) + np.log((1.+gammaLR*gammaWV)/(RH*tau1_star))) / ((1. + gammaLR*gammaWV)**2) * Th2o_2

    # feedbacks in rot. and v-r bands
    lambda_h2o_1 = c_h2o*np.pi*am.dPlanckdT_n(nu_mid_1, Th2o) * (dnu_1) * (dTh2odTs_ts + dTh2odTs_gamma * dgammaLRdTs)
    lambda_h2o_2 = c_cnt*np.pi*am.dPlanckdT_n(nu_mid_2, Th2o_2) * (dnu_2) * (dTh2odTs_ts_2 + dTh2odTs_gamma_2 * dgammaLRdTs)
    
    return lambda_h2o_1 + lambda_h2o_2

def get_lambda_cntm(nu, Ts, Tstrat, Tcloud, RH, qco2, T0=300., gammaLR=None, dgammaLRdTs=None):
    """
    water vapor continuum feedback above cloud top
    In:
        nu [cm-1]: wavenumber array
        Ts [K]: surface temperature
        Tstrat [K]: stratospheric temperature
        Tcloud [K]: cloud top temperature
        RH [-]: relative humidity (0-1)
        qco2 [-]: molar mixing ratio of CO2
        T0 [K]: reference temperature
        gammaLR [K/km]: bulk atmospheric lapse rate (if None, compute)
        dgammaLRdTs [K/km/K]: derivative of bulk atmospheric lapse rate w.r.t. surface temperature (if None, compute)
    Out:
        lambda_cntm [W/m^2/K]: water vapor continuum feedback above cloud top
    """
    Ts = np.atleast_1d(Ts)

    # set thermodynamic parameters, based on approximation of CC around T=T0
    gammaWV = cs.Lvap/(cs.Rv * T0)
    esat0 = cs.esat(T0)

    # compute lapse rate/derivatives
    if gammaLR is None:
        gammaLR = cs.get_gammaLR(Ts, Tstrat)  # if not defined, use analytical approx
    else:
        pass   # take as input...

    # get d(gammaLR)/dTs
    if dgammaLRdTs is None:
        dTs = 1.   # compute derivative numerically for now -> assumes a moist-adiabatic response!
        dgammaLRdTs = (cs.get_gammaLR(Ts+dTs, Tstrat) - cs.get_gammaLR(Ts, Tstrat))/dTs   # should be <0
    else:
        pass   # take as input...

    # get continuum optical thickness
    tau_cntm_star = cs.R/cs.Rv * cs.kappaX*esat0/(cs.g*cs.D)

    # get intersection of co2 band with cloud top
    nu_CO2_R, nu_CO2_L = nu_em_CO2(qco2, Tcloud, Ts, gammaLR)
    nu_CO2_R, nu_CO2_L = constrain_nu(nu, nu_CO2_R), constrain_nu(nu, nu_CO2_L)

    co2_idx = np.where(nu > nu_CO2_R)[0][0]

    # determine how much of continuum is optically thick
    T_em_cont_all = np.minimum(cs.get_Trad_cntm(nu, gammaLR, RH)[co2_idx:], Tcloud)

    nu_cont_R = nu_em_cont(Tcloud, Ts, gammaLR, RH)[0]
    nu_cont_R = constrain_nu(nu, nu_cont_R)
    if np.max(T_em_cont_all) == Tcloud:
        nu_cont_R = nu_em_H2O(1500, cs.ln1, cs.kappa1, Tcloud, Ts, gammaLR, RH)[1]
        nu_cont_R = constrain_nu(nu, nu_cont_R)

    dnu = np.maximum(0, nu_cont_R - nu_CO2_R)

    # calculate feedback
    # average optical thickness across continuum region (different from Koll et al., 2023,
    #   due to logarithmic absorption coefficient in window)
    dTcntdTs_gamma = T_em_cont_all/( (2.*gammaWV-cs.cntm_alpha)*gammaLR )
    tau_cnt_cld = tau_cntm_star[co2_idx:] * RH**2 * 1./( (2.*gammaWV-cs.cntm_alpha)*gammaLR) *(Tcloud/T0)**(2.*gammaWV-cs.cntm_alpha)
    lambda_cntm = (0.5)*c_cnt*np.pi*am.dPlanckdT_n(nu_CO2_R, T_em_cont_all[0]) *(dTcntdTs_gamma[0] * dgammaLRdTs) *(dnu) *(1.-np.exp(-np.mean(tau_cnt_cld)))

    return lambda_cntm

def get_lambda_co2(Ts, Tstrat, Tcloud, q, gammaLR=None, dgammaLRdTs=None, Ts0 = 310):
    """
    CO2 feedback above cloud top
    In:
        Ts [K]: surface temperature
        Tstrat [K]: stratospheric temperature
        Tcloud [K]: cloud top temperature
        RH [-]: relative humidity (0-1)
        qco2 [-]: molar mixing ratio of CO2
        gammaLR [K/km]: bulk atmospheric lapse rate (if None, compute)
        dgammaLRdTs [K/km/K]: derivative of bulk atmospheric lapse rate w.r.t. surface temperature (if None, compute)
        Ts0 [K]: threshold surface temperature for CO2 feedback to be active (otherwise zero)
    Out:
        lambda_co2 [W/m^2/K]: CO2 feedback above cloud top
    """

    if gammaLR is None:
        gammaLR = cs.get_gammaLR(Ts, Tstrat)  # if not defined, use analytical approx
    else:
        pass   # take as input...

    # get d(gammaLR)/dTs
    if dgammaLRdTs is None:
        dTs = 1.   # compute derivative numerically for now -> assumes a moist-adiabatic response!
        dgammaLRdTs = (cs.get_gammaLR(Ts+dTs, Tstrat) - cs.get_gammaLR(Ts, Tstrat))/dTs   # should be <0
    else:
        pass   # take as input...

    # get CO2 optical thickness
    nu0 = 667.5
    qco2 = cs.convert_molar_to_mass_ratio(q, cs.R_co2, cs.R)
    tau_co2_star = cs.get_kappa_co2(nu0, cs.ps_dry) * cs.ps_dry/(2.*cs.g*cs.D)          # CO2 column optical thickness in center of CO2 band;

    lambda_co2 = np.zeros_like(Ts) + np.nan
    Thot = np.zeros_like(Ts) + np.nan
    Tcold = np.zeros_like(Ts) + np.nan
    mask = Ts <= Ts0
    
    # find left and right edge of the CO2 band at cloud top
    Tcold = np.where(mask,np.zeros_like(Ts)+Tstrat,Tcold)
    Thot = np.where(mask,Tcloud,Thot)

    nuHot = np.zeros_like(Ts)                          # (needs correct array shape..)
    nuHot = nu0 + cs.l_co2 *np.log( qco2* tau_co2_star )
    nuCold = nu0 + cs.l_co2 *np.log( qco2*tau_co2_star*(Tcold/Tcloud)**(2./gammaLR) )
    dnuCold_dTs = - 2.*cs.l_co2/(Ts*gammaLR) + 2.*cs.l_co2/(gammaLR**2)*np.log(Tcloud/Tcold)*dgammaLRdTs
    
    # calculate feedback
    lambda_co2_cool = np.pi*am.dPlanckdT_n(np.array([nu0]),Thot) *( nuHot - nuCold ) + \
                      ( np.pi*am.B_nu(nu0,Thot) - np.pi*am.B_nu(nu0,Tcold) ) *(-1) *dnuCold_dTs

    lambda_co2 = c_co2*lambda_co2_cool[0]

    return lambda_co2


### change in cloud top temperature/local adiabatic temperature

def dTclouddTs(T_s, Tstrat, T_em_cloud, gammaLR=None, dgammaLRdTs=None):
    """
    change in cloud top temperature with surface temperature
    In:
        T_s [K]: surface temperature
        Tstrat [K]: stratospheric temperature
        T_em_cloud [K]: cloud top temperature
        RH [-]: relative humidity (0-1)
        gammaLR [K/km]: bulk atmospheric lapse rate (if None, compute)
        dgammaLRdTs [K/km/K]: derivative of bulk atmospheric lapse rate w.r.t. surface temperature (if None, compute)
    Out:
        dTclddTs [K/K]: change in cloud top temperature with surface temperature
    
    """
    T_s = np.atleast_1d(T_s)

    ## calculate numerically if not given
    # ---
    # get gammaLR
    if gammaLR is None:
        gammaLR = cs.get_gammaLR(T_s, Tstrat)  # if not defined, use analytical approx
    else:
        pass   # take as input...

    # get d(gammaLR)/dTs
    if dgammaLRdTs is None:
        dTs = 1.   # compute derivative numerically for now -> assumes a moist-adiabatic response!
        dgammaLRdTs = (cs.get_gammaLR(T_s+dTs, Tstrat) - cs.get_gammaLR(T_s, Tstrat))/dTs   # should be <0
    else:
        pass   # take as input...

    ## ### Eqn. 4 in Czarnecki and Pincus
    dTclddTs = (T_em_cloud/T_s)*(T_s * np.log((T_em_cloud/T_s)**(1./gammaLR))*dgammaLRdTs + 1)

    return dTclddTs


def get_lambda_cloud(nu, T_s, Tstrat, T_em_cloud, RH, qco2, T0=300., gammaLR=None, dgammaLRdTs=None):
    """
    cloud temperature feedback (adapted from surface temperature feedback in Koll et al., 2023)
    In:
        nu [cm-1]: wavenumber array
        T_s [K]: surface temperature
        Tstrat [K]: stratospheric temperature
        T_em_cloud [K]: cloud top temperature
        RH [-]: relative humidity (0-1)
        qco2 [-]: molar mixing ratio of CO2
        T0 [K]: reference temperature
        gammaLR [K/km]: bulk atmospheric lapse rate (if None, compute)
        dgammaLRdTs [K/km/K]: derivative of bulk atmospheric lapse rate w.r.t. surface temperature (if None, compute)
    Out:
        lambda_cloud [W/m^2/K]: cloud temperature feedback
    """
    T_s = np.atleast_1d(T_s)

    ### compute numerical derivatives
    # ---
    # get gammaLR
    if gammaLR is None:
        gammaLR = cs.get_gammaLR(T_s, Tstrat)  # if not defined, use analytical approx
    else:
        pass   # take as input...

    # get d(gammaLR)/dTs
    if dgammaLRdTs is None:
        dTs = 1.   # compute derivative numerically for now -> assumes a moist-adiabatic response!
        dgammaLRdTs = (cs.get_gammaLR(T_s+dTs, Tstrat) - cs.get_gammaLR(T_s, Tstrat))/dTs   # should be <0
    else:
        pass   # take as input...

    # get intersection of h2o rot band with cloud top
    nu_H2O_1_R, nu_H2O_1_L = nu_em_H2O(150, cs.ln0, cs.kappa0, T_em_cloud, T_s, gammaLR, RH)
    nu_H2O_1_R, nu_H2O_1_L = constrain_nu(nu, nu_H2O_1_R), constrain_nu(nu, nu_H2O_1_L)
    
    # get intersection of co2 band with cloud top
    nu_CO2_R, nu_CO2_L = nu_em_CO2(qco2, T_em_cloud, T_s, gammaLR)
    nu_CO2_R, nu_CO2_L = constrain_nu(nu, nu_CO2_R), constrain_nu(nu, nu_CO2_L)
    
    co2_idx = np.where(nu > nu_CO2_R)[0][0]
    dnu_1 = np.maximum(0, nu_CO2_L - nu_H2O_1_R) # where to evaluate Planck function btw. CO2/rot band

    # get intersection of v-r band with cloud top        
    nu_H2O_2_R, nu_H2O_2_L = nu_em_H2O(1500, cs.ln1, cs.kappa1, T_em_cloud, T_s, gammaLR, RH)
    nu_H2O_2_R, nu_H2O_2_L = constrain_nu(nu, nu_H2O_2_R), constrain_nu(nu, nu_H2O_2_L)

    # where to evaluate Planck function in window
    dnu_2 = np.maximum(0, nu_H2O_2_L - nu_CO2_R)
    nu_mid = np.mean([nu_H2O_1_R, nu_H2O_2_L])

    # calculate continuum optical thickness
    gammaWV = cs.Lvap/(cs.Rv * T0)
    esat0 = cs.esat(T0)
    tau_cntm_star = cs.R/cs.Rv * cs.kappaX*esat0/(cs.g*cs.D)
    tau_cnt_cld = tau_cntm_star[co2_idx:] * RH**2 * 1./( (2.*gammaWV-cs.cntm_alpha)*gammaLR) *(T_em_cloud/T0)**(2.*gammaWV-cs.cntm_alpha)

    # calculuate feedback
    trans_cntm = np.exp(-np.mean(tau_cnt_cld))
    dTcld_dTs = dTclouddTs(T_s, Tstrat, T_em_cloud, gammaLR=gammaLR, dgammaLRdTs=dgammaLRdTs)
    lambda_cloud = dTcld_dTs * (c_surf*np.pi*am.dPlanckdT_n(np.array([nu_mid]), T_em_cloud) * dnu_1 + c_surf*np.pi*am.dPlanckdT_n(np.array([nu_mid]), T_em_cloud) * trans_cntm * dnu_2)

    return lambda_cloud
