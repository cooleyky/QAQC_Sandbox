# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np


class WaterVapor():
    """Collection of functions for calculating water vapor pressure."""
    
    def saturation_pressure(self, temp, pressure):
        """Calculate dry-bulb water vapor saturation pressure.

        Calculate the dry-bulb water vapor saturation pressure [hPa]
        from the measured temperature [C], correcting for the effect
        of atmospheric pressure [hPa].

        Parameters
        ----------
        pressure: (float)
            Atmospheric pressure of moist air [hPa]
        temp: (float)
            Dry-bulb temperature [C]

        Returns
        -------
        ew_sat: (float)
            The saturation vapor pressure [hPa] of water vapor corrected
            for atmospheric pressure

        References
        ----------
        JCOMM, 2015. Recommended Algorithms for the Computation of Marine
            Meterological Variables. JCOMM Technical Report No. 63
        WMO, 2010. Guide to Meterological Instrument and Methods of Observation
            WMO-No 8 (2008 edition, updated 2010)
        """
        # Calculate the moist-air vapor saturation value
        ew = 6.112*np.exp((17.62*temp)/(243.12 + temp))

        # Calculate the atmospheric pressure correction
        fp = self.pressure_correction(pressure)

        # Apply atmoshperic pressure correction
        ew_sat = fp * ew

        return ew_sat
    
    
    def pressure_correction(self, pressure):
        """Correct water vapor saturation pressure for atmospheric pressure.

        This function adjusts the saturation pressure of water vapor for the
        effect of atmospheric pressure. Ref: WMO-8 [20145]

        Parameters
        ----------
        pressure: (float or numpy array)
            Atmospheric pressure of moist air [hPa]

        Returns
        -------
        fp: (float or numpy array)
            A float or array of floats with the pressure correction for
            water vapor saturation pressure

        References
        ----------
        JCOMM, 2015. Recommended Algorithms for the Computation of Marine
            Meterological Variables. JCOMM Technical Report No. 63
        WMO, 2010. Guide to Meterological Instrument and Methods of Observation
            WMO-No 8 (2008 edition, updated 2010)
        """

        fp = 1.0016 + 3.15E-6*pressure - 0.074/pressure

        return fp
    
    
    def RH_to_vapor_pressure(self, RH, temp, pressure):
        """Convert Relative Humidity (%) to water vapor pressure [hPa]
        
        Parameters
        ----------
        RH: (float)
            Relative humidity (%)
        temp: (float)
            Dry-bulb temperature [C]
        pressure: (float)
            Atmospheric pressure of moist air [hPa]
        
        Returns
        -------
        ew: (float)
            Water vapor pressure [hPa]
            
        References
        ----------
        JCOMM, 2015. Recommended Algorithms for the Computation of Marine
            Meterological Variables. JCOMM Technical Report No. 63
        WMO, 2010. Guide to Meterological Instrument and Methods of Observation
            WMO-No 8 (2008 edition, updated 2010)
        """
        # Calculate the saturation pressure of water vapor
        ew_sat = self.saturation_pressure(temp, pressure)
        
        # Calculate the water vapor pressure from RH and the saturation pressure
        ew = (RH/100)*ew_sat
        
        return ew


# +
def O2sol(temp, sal):
    """Calculate the oxygen solubility in seawater.
    
    Saturation solubility of oxygen in seawater at 1-atmosphere
    moist air. Adapted from MatLab code by Roberta C. Hamme (U. Vic)
    
    Parameters
    ----------
    temp: (float)
        temperature [C]
    sal: (float)
        salinity [psu]
        
    Returns
    -------
    cO2: (float)
        Solubility of oxygen in seawater [umol/kg]
        
    References
    ----------
    Garcia, Hernan E., and Louis I. Gordon, 1992. Oxygen Solubility
        in seawater: Better fitting equations. Limnology & 
        Oceanography, 37: 1307-1312
    Hamme, Roberta C. 2005. Solubility of O2 in sea water [Computer software]    
    """
    # Scale the temperature
    temp = np.log((298.15 - temp)/(273.15 + temp))
    
    # Garcia & Gordon Table 1 coefficients
    A0 = 5.80871 
    A1 = 3.20291
    A2 = 4.17887
    A3 = 5.10006
    A4 = -9.86643E-2
    A5 = 3.80369
    B0 = -7.01577E-3
    B1 = -7.70028E-3
    B2 = -1.13864E-2
    B3 = -9.51519E-3
    C0 = -2.75915E-7
    
    # Equation (8) from Garcia & Gordon
    cO2 = np.exp(A0 + A1*temp + A2*temp**2 + A3*temp**3 + A4*temp**4 + A5*temp**5 +
              sal*(B0 + B1*temp + B2*temp**2 + B3*temp**3) + C0*sal**2)
    
    return cO2


def N2sol(temp, sal):
    """Calculate the nitrogen solubility in seawater.
    
    Saturation solubility of nitrogen in seawater at 1-atmosphere
    moist air. Adapted from MatLab code by Roberta C. Hamme (U. Vic)
    
    Parameters
    ----------
    temp: (float)
        temperature [C]
    sal: (float)
        salinity [psu]
        
    Returns
    -------
    cN2: (float)
        Solubility of nitrogen in seawater [umol/kg]
        
    References
    ----------
    Hamme, Roberta, and Steve Emerson, 2004. The solubility of neon, nitrogen
        and argon in distilled water and seawater. Deep-Sea Research I, 51(11):
        1517-1528
    Hamme, Roberta C. 2005. Solubility of N2 in sea water [Computer software]    
    """
    
    # Scale the temperature
    temp = np.log((298.15 - temp)/(273.15 + temp))

    # constants from Table 4 of Hamme and Emerson 2004
    A0 = 6.42931
    A1 = 2.92704
    A2 = 4.32531
    A3 = 4.69149
    B0 = -7.44129E-3
    B1 = -8.02566E-3
    B2 = -1.46775E-2

    # Equation (1) of Hamme and Emerson 2004
    cN2 = np.exp(A0 + A1*temp + A2*temp**2 + A3*temp**3 + sal*(B0 + B1*temp + B2*temp**2))

    return cN2


def Arsol(temp, sal):
    """Calculate the argon solubility in seawater.
    
    Saturation solubility of argon in seawater at 1-atmosphere
    moist air. Adapted from MatLab code by Roberta C. Hamme (U. Vic)
    
    Parameters
    ----------
    temp: (float)
        temperature [C]
    sal: (float)
        salinity [psu]
        
    Returns
    -------
    cAr: (float)
        Solubility of nitrogen in seawater [umol/kg]
        
    References
    ----------
    Hamme, Roberta, and Steve Emerson, 2004. The solubility of neon, nitrogen
        and argon in distilled water and seawater. Deep-Sea Research I, 51(11):
        1517-1528
    Hamme, Roberta C. 2005. Solubility of Ar in sea water [Computer software]    
    """
    # Scale the temperature
    temp = np.log((298.15 - temp)/(273.15 + temp))

    # Constants from Table 4 of Hamme and Emerson 2004
    A0 = 2.79150
    A1 = 3.17609
    A2 = 4.13116
    A3 = 4.90379
    B0 = -6.96233e-3
    B1 = -7.66670e-3
    B2 = -1.16888e-2
    
     # Equation (1) of Hamme and Emerson 2004
    cAr = np.exp(A0 + A1*temp + A2*temp**2 + A3*temp**3 + sal*(B0 + B1*temp + B2*temp**2))

    return cAr


def Nesol(temp, sal):
    """Calculate the neon solubility in seawater.
    
    Saturation solubility of neon in seawater at 1-atmosphere
    moist air. Adapted from MatLab code by Roberta C. Hamme (U. Vic)
    
    Parameters
    ----------
    temp: (float)
        temperature [C]
    sal: (float)
        salinity [psu]
        
    Returns
    -------
    cNe: (float)
        Solubility of neon in seawater [umol/kg]
        
    References
    ----------
    Hamme, Roberta, and Steve Emerson, 2004. The solubility of neon, nitrogen
        and argon in distilled water and seawater. Deep-Sea Research I, 51(11):
        1517-1528
    Hamme, Roberta C. 2005. Solubility of Ar in sea water [Computer software]    
    """
    # Scale the temperature
    temp = np.log((298.15 - temp)/(273.15 + temp))

    # Constants from Table 4 of Hamme and Emerson 2004
    A0 = 2.18156
    A1 = 1.29108
    A2 = 2.12504
    B0 = -5.94737e-3
    B1 = -5.13896e-3

    # Equation (1) of Hamme and Emerson 2004
    cNe = np.exp(A0 + A1*temp + A2*temp**2 + A3*temp**3 + sal*(B0 + B1*temp + B2*temp**2))

    # Convert from nmol/kg to umol/kg
    cNe = cNe/1000;

    return cNe


def Hesol(temp, sal):
    """Calculate the helium solubility in seawater.
    
    Saturation solubility of helium in seawater at 1-atmosphere
    moist air. Adapted from MatLab code by Roberta C. Hamme (U. Vic)
    
    Parameters
    ----------
    temp: (float)
        temperature [C]
    sal: (float)
        salinity [psu]
        
    Returns
    -------
    cHe: (float)
        Solubility of helium in seawater [umol/kg]
        
    References
    ----------
    Weiss, Ray F., 1971. Solubility of Helium and Neon in Water and Seawater.
        Journal of Chemical and Engineering Data, 16(2): 235-241
    Hamme, Roberta C. 2005. Solubility of Ar in sea water [Computer software]
    """    
    # convert T to absolute temperature
    temp_abs = temp + 273.15;

    # Coefficients
    A1 = -167.2178;
    A2 = 216.3442;
    A3 = 139.2032;
    A4 = -22.6202;
    B1 = -0.044781;
    B2 = 0.023541;
    B3 = -0.0034266;

    # Equation (2) of Weiss and Kyser
    cHe = np.exp(A1 + (A2*100/temp_abs) + (A3*np.log(temp_abs/100)) + (A4*temp_abs/100) + sal*(B1 + (B2*temp_abs/100) + (B3*(temp_abs/100)**2)))

    # Convert concentration from mL/kg to umol/kg
    # Molar volume at STP is calculated from Dymond and Smith (1980) "The virial coefficients of pure gases and mixtures", Clarendon Press, Oxford.
    cHe = cHe / 22.44257E-3;

    return cHe


def Krsol(temp, sal):
    """Calculate the krypton solubility in seawater.
    
    Saturation solubility of krypton in seawater at 1-atmosphere
    moist air. Adapted from MatLab code by Roberta C. Hamme (U. Vic)
    
    Parameters
    ----------
    temp: (float)
        temperature [C]
    sal: (float)
        salinity [psu]
        
    Returns
    -------
    cKr: (float)
        Solubility of krypton in seawater [umol/kg]
        
    References
    ----------
    Weiss, Ray F. and T. Kurt Kyser, 1978. Solubility of Krypton in Water and Seawater.
        Journal of Chemical Thermodynamics, 16(2): 235-241
    Hamme, Roberta C. 2005. Solubility of Ar in sea water [Computer software]
    """    
    # convert T to absolute temperature
    temp = temp + 273.15;

    # constants from Table 2 Weiss and Kyser for mL/kg
    A1 = -112.6840
    A2 = 153.5817
    A3 = 74.4690
    A4 = -10.0189
    B1 = -0.011213
    B2 = -0.001844
    B3 = 0.0011201

    # Eqn (7) of Weiss and Kyser
    cKr = np.exp(A1 + A2*100/temp + A3*np.log(temp/100) + A4*temp/100 + sal*(B1 + B2*temp/100 + B3*(temp/100)**2))

    # Convert concentration from mL/kg to umol/kg
    # Molar volume at STP is calculated from Dymond and Smith (1980) 
    # "The virial coefficients of pure gases and mixtures", Clarendon Press, Oxford.
    cKr = cKr / 22.3511E-3;

    return cKr


def seawater_vapor_pressure(S, T, unit="atm"):
    """Calculate vapor pressure of seawater.
    
    Parameters
    ----------
    S: (float)
        Salinity [psu]
    T: (float)
        Temperature [degree_C]
    unit: (str)
        A string indicating the units to return
            atm = atmosphere (default)
            mbar = mbar
        
    Returns
    -------
    vapor_press_atm: (float)
        Seawater vapor pressure [mbar]
        
    Reference
    ---------
    Dickson, A.G., C.L. Sabine, J.R. Christian, 2007. Guide to Best Practices
        for Ocean CO2 Measurements. PICES Special Publication 3: 191
    Wagner, W., and A. Pruss, 2002. The IAPWS formulation 1995 for the 
        thermodynamic properties of ordinary water substance for general and
        scientific use. Journal of Physical and Chemical Reference Data:
        31: 387-535
    Millero, F.J., 1974. Seawater as a multicomponent electrolyte solution.
        The Sea (5): 3-80. E.D. Goldberg Ed.
    Hamme, Roberta C. 2005. Solubility of Ar in sea water [Computer software]
    """
    # Calculate temperature in Kelvin and modified temperature for Chebyshev polynomial
    K = T + 273.15
    K_mod = 1 - K/647.096

    # Calculate value of Wagner polynomial
    Wagner = (-7.85951783*K_mod) + (1.84408259*K_mod**1.5) - (11.7866497*K_mod**3) + (22.6807411*K_mod**3.5) - (15.9618719*K_mod**4) + 1.80122502*K_mod**7.5

    # Vapor pressure of pure water in kiloPascals and mm of Hg
    vapor_0sal_kPa = np.exp(Wagner * 647.096/K) * 22.064 * 1000

    # Correct vapor pressure for salinity
    molality = 31.998 * S / (1E3 - 1.005*S)
    osmotic_coef = 0.90799 - 0.08992*(0.5*molality) + 0.18458*(0.5*molality)**2 - 0.07395*(0.5*molality)**3 - 0.00221*(0.5*molality)**4
    vapor_press_kPa = vapor_0sal_kPa * np.exp(-0.018 * osmotic_coef * molality)
    
    if unit=="atm":
        vapor_press = vapor_press_kPa/101.32501
    elif unit=="mbar":
        vapor_press = vapor_press_kPa*10
    else:
        raise Exception(f"{unit} not valid units.")

    # Convert to atm
    # vapor_press_atm = vapor_press_kPa/101.32501
    
    return vapor_press
