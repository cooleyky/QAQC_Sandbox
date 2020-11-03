import re
import numpy as np
import pandas as pd
import gsw


def parse_optode(data):
    """Parse the loaded optode data into a dataframe with column headers."""
    columns = ["timestamp", "model", "serial number", "oxygen concentration",
               "oxygen saturation", "temperature", "calibrated phase",
               "temp-compensated calibrated phase", "blue phase", "red phase",
               "blue amplitude", "red amplitude", "raw temperature"]
    df = pd.DataFrame(columns=columns)
    for line in data.splitlines():
        # Now need to parse the lines
        # Get the timestamp of the line
        timeindex = re.search("\[[^\]]*\]", line)
        timestamp = pd.to_datetime(line[timeindex.start()+1:timeindex.end()-1])
        line = line[timeindex.end():].strip()
        # Next, split the data
        model, sn, o2con, o2sat, temp, cal_phase, tcal_phase, blue_phase, red_phase, blue_amp, red_amp, raw_temp = line.split(
            "\t")
        # Put the data into a dataframe
        df = df.append({
            "timestamp": timestamp,
            "model": int(model.strip("!")),
            "serial number": int(sn),
            "oxygen concentration": float(o2con),
            "oxygen saturation": float(o2sat),
            "temperature": float(temp),
            "calibrated phase": float(cal_phase),
            "temp-compensated calibrated phase": float(tcal_phase),
            "blue phase": float(blue_phase),
            "red phase": float(red_phase),
            "blue amplitude": float(blue_amp),
            "red amplitude": float(red_amp),
            "raw temperature": float(raw_temp)
        }, ignore_index=True)

    return df


def load_optode(file):
    """Open, load, and parse a file from the Aanderaa oxygen optode."""
    # Open and load the optode file
    with open(file) as f:
        data = f.read()
        data = data.strip("\n")
        data = data.strip("!")

    # Parse the data into a dataframe with column headers
    df = parse_optode(data)

    return df


def parse_barometer(data):
    """Parse the barometric info into a pandas dataframe."""
    columns = ["timestamp", "temperature", "pressure", "relative humidity"]
    df = pd.DataFrame(columns=columns)

    for line in data:
        # Skip the lines without actual data
        if "+" not in line:
            continue

        # If the line has data, split into its different measurements
        line = line.strip("\x00")
        line = line.replace("+", "")
        rh, temp, pres, time, date = line.split(",")

        # Make a datetime object out of the date and time
        datetime = " ".join((date, time))
        datetime = pd.to_datetime(datetime)

        # Parse the measurements into a dataframe
        df = df.append({
            "timestamp": datetime,
            "temperature": float(temp),
            "pressure": float(pres),
            "relative humidity": float(rh)
        }, ignore_index=True)

    return df


def load_barometer(file):
    """Open, read, and parse barometric data into a dataframe."""
    # Open and read in the barometric data
    with open(file) as f:
        data = f.read()
        data = data.strip("\n")
        data = data.splitlines()

    # Parse the data
    df = parse_barometer(data)

    return df


def do2_SVU(calphase, temp, csv, conc_coef=np.array([0.0, 1.0])):
    """
    Description:
        Calculates the DOCONCS_L1 data product from autonomously-operated DOSTA
        (Aanderaa) instruments (on coastal surface moorings and gliders) using
        the Stern-Volmer-Uchida equation for calculating temperature corrected
        dissolved oxygen concentration.
    Usage:
        DO = do2_SVU(calphase, temp, csv, conc_coef)
            where
        DO = dissolved oxygen [micro-mole/L], DOCONCS_L1. see Notes.
        calphase = calibrated phase from an Oxygen sensor [deg], DOCONCS-DEG_L0
            (see DOCONCS DPS)
        temp = oxygen sensor foil temperature T(optode) [deg C], (see DOCONCS DPS)
        csv = Stern-Volmer-Uchida Calibration Coefficients array.
            7 element float array, (see DOCONCS DPS)
        conc_coef = 'secondary' calibration coefficients: an array of offset and slope
            coefficients to apply to the result of the SVU equation. See Notes.
            conc_coef[0, 0] = offset
            conc_coef[0, 1] = slope
    Example:
        csv = np.array([0.002848, 0.000114, 1.51e-6, 70.42301, -0.10302,
                        -12.9462, 1.265377])
        calphase = 27.799
        temp = 19.841
        DO = do2_SVU(calphase, temp, csv)
        print DO
        > 363.900534505
    Implemented by:
        2013-04-26: Stuart Pearce. Initial Code.
        2015-04-10: Russell Desiderio. Revised code to work with CI implementation
                    of calibration coefficients: they are to be implemented as time-
                    vectorized arguments (tiled in the time dimension to match the
                    number of data packets). Fix for "blocker #2972".
        2015-08-04: Russell Desiderio. Added documentation.
        2015-08-10: Russell Desiderio. Added conc_coef calibration array to argument list.
                    Required to be a 2D row vector for broadcasting purposes.
        2015-10-28: Russell Desiderio. Added conc_coef = np.atleast_2d(conc_coef) line so
                    that function will now accept conc_coef as a 1D array (so that 1D array
                    entries in Omaha cal sheets won't result in DPA exceptions being raised).
                    So. Also changed default value for conc_coef in argument list to be
                    the 1D array [0.0, 1.0].
    Notes:
        General:
            The DOCONCS_L1 data product has units of micromole/liter; SAF incorrectly
            lists the units for this L1 product as micromole/kg. (To change units from
            mmole/L to mmole/kg, salinity is required, making the result an L2 data
            product).
            The DOCONCS_L1 data product is uncorrected for salinity and pressure.
        Temperature dependence:
            The optode sensor's thermistor temperature should be used whenever possible
            because for the OOI DOSTAs (model 4831) it is situated directly at the sensor
            foil and the SVU cal coefficients are derived in part to compensate for the
            change in oxygen permeability through the foil as a function of its temperature.
            The time constant of the model 4831 thermistor is < 2 seconds. Because the foil
            and therefore the calphase response time itself is 8 sec or 24 sec depending on
            the particular optode, there is little or no advantage to be gained by using a
            temperature sensor (eg, from a CTD) with a faster response. It is better to make
            sure that the temperature used most accurately reflects the foil temperature.
            On gliders, there is often a difference in CTD and optode temperature readings of
            1 degree Celsius, which translates to about a 5% difference in calculated oxygen
            concentration for a range of typical water column conditions.
        Conc_coef (this information is not currently in the DPS):
            Aanderaa uses two calibration procedures for the 4831 optode. The primary 'multi-point'
            calibration, done in Norway, determines the SVU foil coefficients (variable csv in the
            DPA). The secondary two-point calibration, done in Ohio, corrects the multi-point
            calibration calculation against 0% oxygen and 100% oxygen data points to provide the
            conc_coef values. (Aanderaa is in the process of changing the secondary cal to a one
            point cal, using just the 100% oxygen data point, but the result will still be expressed
            as offset and slope conc_coef values.) For standard optode refurbishment Aanderaa recommends
            a secondary calibration instead of a new multi-point SVU foil calibration.
            Secondary calibrations are not done on new optodes nor on optodes with new determinations
            of the SVU foil coefficients; in these cases Aanderaa sets the conc_coef values to 0 (offset)
            and 1 (slope) in the optode firmware by default. Conc_coef determinations resulting from the
            secondary calibration procedure are also incorporated into the optode firmware and are also
            listed on the Aanderaa Form No. 710 calibration certificate, although they are currently
            mislabelled on this form as "PhaseCoef".
            The conc_coef correction to optode-calculated values for oxygen concentration is automatically
            applied by the optode firmware. However, this correction must be done manually when oxygen
            concentration is calculated from calphase and optode temperature external to the optode, as in
            this DPA do2_SVU.
    References:
        OOI (2012). Data Product Specification for Oxygen Concentration
            from "Stable" Instruments. Document Control Number
            1341-00520. https://alfresco.oceanobservatories.org/ (See:
            Company Home >> OOI >> Controlled >> 1000 System Level
            >> 1341-00520_Data_Product_SPEC_DOCONCS_OOI.pdf)
        Aanderaa Data Instruments (August 2012). TD 269 Operating Manual Oxygen Optode 4330, 4831, 4835.
        August 2015. Shawn Sneddon, Xylem-Aanderaa technical support, MA, USA, 800-765-4974
    """
    conc_coef = np.atleast_2d(conc_coef)
    # this will work for both old and new CI implementations of cal coeffs.
    csv = np.atleast_2d(csv)

    # Calculate DO using Stern-Volmer:
    Ksv = csv[:, 0] + csv[:, 1]*temp + csv[:, 2]*(temp**2)
    P0 = csv[:, 3] + csv[:, 4]*temp
    Pc = csv[:, 5] + csv[:, 6]*calphase
    DO = ((P0/Pc) - 1) / Ksv

    # apply refurbishment calibration
    # conc_coef can be a 2D array of either 1 row or DO.size rows.
    DO = conc_coef[:, 0] + conc_coef[:, 1] * DO
    return DO


def do2_salinity_correction(DO, P, T, SP, lat, lon, pref=0):
    """
    Description:
        Calculates the data product DOXYGEN_L2 (renamed from DOCONCS_L2) from DOSTA
        (Aanderaa) instruments by correcting the the DOCONCS_L1 data product for
        salinity and pressure effects and changing units.
    Usage:
        DOc = do2_salinity_correction(DO,P,T,SP,lat,lon, pref=0)
            where
        DOc = corrected dissolved oxygen [micro-mole/kg], DOXYGEN_L2
        DO = uncorrected dissolved oxygen [micro-mole/L], DOCONCS_L1
        P = PRESWAT water pressure [dbar]. (see
            1341-00020_Data_Product_Spec_PRESWAT). Interpolated to the
            same timestamp as DO.
        T = TEMPWAT water temperature [deg C]. (see
            1341-00010_Data_Product_Spec_TEMPWAT). Interpolated to the
            same timestamp as DO.
        SP = PRACSAL practical salinity [unitless]. (see
            1341-00040_Data_Product_Spec_PRACSAL)
        lat, lon = latitude and longitude of the instrument [degrees].
        pref = pressure reference level for potential density [dbar].
            The default is 0 dbar.
    Example:
        DO = 433.88488978325478
        do_t = 1.97
        P = 5.4000000000000004
        T = 1.97
        SP = 33.716000000000001
        lat,lon = -52.82, 87.64
        DOc = do2_salinity_correction(DO,P,T,SP,lat,lon, pref=0)
        print DO
        > 335.967894709
    Implemented by:
        2013-04-26: Stuart Pearce. Initial Code.
        2015-08-04: Russell Desiderio. Added Garcia-Gordon reference.
    References:
        OOI (2012). Data Product Specification for Oxygen Concentration
            from "Stable" Instruments. Document Control Number
            1341-00520. https://alfresco.oceanobservatories.org/ (See:
            Company Home >> OOI >> Controlled >> 1000 System Level
            >> 1341-00520_Data_Product_SPEC_DOCONCS_OOI.pdf)
        "Oxygen solubility in seawater: Better fitting equations", 1992,
        Garcia, H.E. and Gordon, L.I. Limnol. Oceanogr. 37(6) 1307-1312.
        Table 1, 5th column.
    """

    # density calculation from GSW toolbox
    SA = gsw.sa_from_sp(SP, P, lon, lat)
    CT = gsw.ct_from_t(SA, T, P)
    pdens = gsw.rho(SA, CT, pref)  # potential referenced to p=0

    # Convert from volume to mass units:
    DO = ne.evaluate('1000*DO/pdens')

    # Pressure correction:
    DO = ne.evaluate('(1 + (0.032*P)/1000) * DO')

    # Salinity correction (Garcia and Gordon, 1992, combined fit):
    S0 = 0
    ts = ne.evaluate('log((298.15-T)/(273.15+T))')
    B0 = -6.24097e-3
    B1 = -6.93498e-3
    B2 = -6.90358e-3
    B3 = -4.29155e-3
    C0 = -3.11680e-7
    Bts = ne.evaluate('B0 + B1*ts + B2*ts**2 + B3*ts**3')
    DO = ne.evaluate('exp((SP-S0)*Bts + C0*(SP**2-S0**2)) * DO')
    return DO
