# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # PCO2W Data Analysis
#
# ---
# ### Purpose
# The purpose of this notebook is to analyze the performance of the Sunburst Sensors, LLC. SAMI-CO<sub>2</sub> (PCO2W) pCO2 seawater measurements at the Pioneer Array. This is done on a deployment-by-deployment, site-by-site comparison with the pCO<sub>2</sub> calculated from the Total Alkalinity and DIC measurements from discete water samples collected by Niskin Bottle casts during deployment and recovery of the instrumentation during mooring maintainence. 
#
# ---
# ### Datasets
# There are three main sources of data sources:
# * **Deployments**: These are the deployment master sheets from OOI Asset Management. They contain the deployment numbers, deployment start times and cruise, and recovery times and cruise, for all of the instrumentation deployed. 
# * **PCO2W**: This is the Sunburst Sensors, LLC. SAMI-CO<sub>2</sub> sensor. It is calibrated for pCO<sub>2</sub> concentrations from 200 - 1000 ppm. Manufacturers stated accuracy of 3 ppm, precision < 1 ppm, and long-term drift of < 1 ppm / 6 months. The data is downloaded from the Ocean Observatories data portal (OOINet) as netCDF files.
# * **CTDBP**: This is the collocated SeaBird CTD with the PCO2W sensor. The data is downloaded from the Ocean Observatories data portal (OOINet) as netCDF files. These data are needed since the PCO2W datasets do not contain either temperature (T), salinity (S), pressure (P), or density ($\rho$) data needed to compare with the discrete sampling.
# * **Discrete Water Samples**: These are discrete water samples collected via Niskin Bottle casts during deployment and recovery of the moored instrumentation. The data is downloaded from OOI Alfresco website as excel files. Parameters sampled include oxygen, salinity, nutrient concentrations (phosphate, nitrate, nitrite, ammonium, silicate), chlorophyll concentrations, and the carbon system. The carbon system parameters sampled are Total Alkalinity (TA), Dissolved Inorganic Carbon (DIC), and pH. 
#
# ---
# ### Method
# #### PCO2W Processing
# Verifying the in-situ pCO<sub>2</sub> measured by the PCO2W against the pCO<sub>2</sub> calculated from the discrete water samples TA and DIC requires several preprocessing steps of the PCO2W datasets. First, the netCDF datasets are opened using ```xarray``` into an xarray ```dataset``` object and the primary dimension set to 'time'. Next, T, S, P, and $\rho$ are interpolated to the PCO2W time base using xarray ```ds.interp_like``` from the dataset from the collocated CTDBP. Next, the pCO2 is corrected for hydrostatic pressure using a correction of 15% per 1000 dbar pressure (Enns 1965, Reed et al. 2018). Then the first and last four days of PCO2W data are selected. The standard deviation of the selected pCO2 is calculated using the first-order differencing with a time-lag of one, in order to arrive at a quasi-stationary time series.
#
# #### CTDBP Processing
# The associated CTD datasets to the PCO2W are opened using ```xarray``` into an xarray ```dataset``` object and the primary dimension set to 'time'. The CTD dataset T, S, P, and $\rho$ are interpolated to the PCO2W time base and merged into the PCO2W dataset using ```ds.interp_like```. 
#
# #### Discrete Water Samples Processing
# The relevant deployment and recovery cruise data for comparison with the PCO2W dataset(s) are opened and loaded into a pandas ```DataFrame``` object. Next, the pCO<sub>2</sub> concentrations are calculated using the ```CO2SYS``` package from the associated TA and DIC concentrations. The bottle samples are then filtered by cruise, time, and depth to identify the samples associated with the deployment and recovery of the PCO2W dataset being analyzed.
#
# #### Verification
# There are several ways that we check on the precision and accuracy 

# ---

# Import libraries
import os, shutil, sys, time, re, requests, csv, datetime, pytz
import yaml
import pandas as pd
import numpy as np
import netCDF4 as nc
import xarray as xr
import warnings
warnings.filterwarnings("ignore")

# Import plotting libraries
import matplotlib.pyplot as plt
# %matplotlib inline

# Import the OOI M2M tool
sys.path.append("/home/andrew/Documents/OOI-CGSN/ooinet/ooinet/")
from m2m import M2M
from phsen import PHSEN

from m2m import M2M

# Import user info for connecting to OOINet via M2M
userinfo = yaml.load(open("/home/andrew/Documents/OOI-CGSN/QAQC_Sandbox/user_info.yaml"))
username = userinfo["apiname"]
token = userinfo["apikey"]

# Initialize the M2M tool
OOI = M2M(username, token)


# +
def process_dataset(ds):
    
    # Remove the *_qartod_executed variables
    qartod_pattern = re.compile(r"^.+_qartod_executed.+$")
    for v in ds.variables:
        if qartod_pattern.match(v):
            # the shape of the QARTOD executed should compare to the provenance variable
            if ds[v].shape[0] != ds["provenance"].shape[0]:
                ds = ds.drop_vars(v)
                
    # Reset the dimensions and coordinates
    ds = ds.swap_dims({"obs": "time"})
    ds = ds.reset_coords()
    keys = ["obs", "id", "provenance", "driver_timestamp", "ingestion_timestamp",
            'port_timestamp', 'preferred_timestamp']
    for key in keys:
        if key in ds.variables:
            ds = ds.drop_vars(key)
    ds = ds.sortby('time')

    # clear-up some global attributes we will no longer be using
    keys = ['DODS.strlen', 'DODS.dimName', 'DODS_EXTRA.Unlimited_Dimension', '_NCProperties', 'feature_Type']
    for key in keys:
        if key in ds.attrs:
            del(ds.attrs[key])
            
    # Fix the dimension encoding
    if ds.encoding['unlimited_dims']:
        del ds.encoding['unlimited_dims']
    
    # resetting cdm_data_type from Point to Station and the featureType from point to timeSeries
    ds.attrs['cdm_data_type'] = 'Station'
    ds.attrs['featureType'] = 'timeSeries'

    # update some of the global attributes
    ds.attrs['acknowledgement'] = 'National Science Foundation'
    ds.attrs['comment'] = 'Data collected from the OOI M2M API and reworked for use in locally stored NetCDF files.'

    return ds


def load_datasets(datasets, ds=None):
    
    while len(datasets) > 0:
        
        dset = datasets.pop()
        new_ds = xr.open_dataset(dset)
        new_ds = process_dataset(new_ds)
        
        if ds is None:
            ds = new_ds
        else:
            ds = xr.concat([new_ds, ds], dim="time")
            
        ds = load_datasets(datasets, ds)
        
    return ds


# -

# ---
# ### Deployment Info

# Import the deployment info for Pioneer PCO2W:

# +
# Central Surface Mooring
CNSM_deployments = pd.read_csv('/home/andrew/Documents/OOI-CGSN/asset-management/deployment/CP01CNSM_Deploy.csv')

# Inshore Surface Mooring
ISSM_deployments = pd.read_csv('/home/andrew/Documents/OOI-CGSN/asset-management/deployment/CP03ISSM_Deploy.csv')

# Offshore Surface Mooring
OSSM_deployments = pd.read_csv('/home/andrew/Documents/OOI-CGSN/asset-management/deployment/CP04OSSM_Deploy.csv')
# -

# Select the PCO2W instruments from the whole list of instruments on the moorings:

# +
# Central Surface Mooring
mask = CNSM_deployments['Reference Designator'].apply(lambda x: True if 'PCO2W' in x else False)
CNSM_deployments = CNSM_deployments[mask]

# Inshore Surface Mooring
mask = ISSM_deployments['Reference Designator'].apply(lambda x: True if 'PCO2W' in x else False)
ISSM_deployments = ISSM_deployments[mask]

# Offshore Surface Mooring
mask = OSSM_deployments['Reference Designator'].apply(lambda x: True if 'PCO2W' in x else False)
OSSM_deployments = OSSM_deployments[mask]
# -

# ------------------------------
# ### Water Sampling Data

# Import the discrete sample data for the Pioneer cruises:

# Import the discrete sample data for Pioneer cruises
basepath = '/home/andrew/Documents/OOI-CGSN/ooicgsn-water-sampling/Pioneer/'
for file in sorted(os.listdir('/home/andrew/Documents/OOI-CGSN/ooicgsn-water-sampling/Pioneer/')):
    if file.endswith('.txt'):
        pass
    else:
        try:
            Bottles = Bottles.append(pd.read_csv(basepath + '/' + file))
        except:
            Bottles = pd.read_csv(basepath + '/' + file)

# Replace all -9999999 hold values with nans
Bottles = Bottles.replace(-9999999, np.nan)
Bottles = Bottles.replace(str(-9999999), np.nan)

# Reformat the date columns
Bottles['Start Time [UTC]'] = Bottles['Start Time [UTC]'].apply(lambda x: pd.to_datetime(x))
Bottles['Bottle Closure Time [UTC]'] = Bottles['Bottle Closure Time [UTC]'].apply(lambda x: pd.to_datetime(x))


# +
# Clean up the data sets to convert sample values stored as strings to floats
# Convert all real measurements still as strings to floats
def clean_types(x):
    if type(x) is str:
        try:
            x = float(x)
            return x
        except:
            return x
    else:
        return x
    
for col in Bottles.columns:
    Bottles[col] = Bottles[col].apply(lambda x: clean_types(x))


# +
# Clean up the DIC, TA, and pH measurements to remove sample placeholders
def clean_dic_and_ta(x):
    if type(x) is str:
        x = np.nan
    elif x < 1900:
        x = np.nan
    else:
        pass
    return x

def clean_pH(x):
    if type(x) is str:
        x = np.nan
    elif x > 8:
        x = np.nan
    elif x < 0:
        x = np.nan
    else:
        pass
    return x

Bottles['Discrete DIC [µmol/kg]'] = Bottles['Discrete DIC [µmol/kg]'].apply(lambda x: clean_dic_and_ta(x))
Bottles['Discrete Alkalinity [µmol/kg]'] = Bottles['Discrete Alkalinity [µmol/kg]'].apply(lambda x: clean_dic_and_ta(x))


# +
# Clean the nutrients to remove sample placeholdes and values not sig. dif. from zero
def clean_nutrients(x):
    if type(x) is str:
        if '<' in x:
            x = 0
        else:
            x = np.nan
    else:
        pass
    return x

Bottles['Discrete Ammonium [uM]'] = Bottles['Discrete Ammonium [uM]'].apply(lambda x: clean_nutrients(x))
Bottles['Discrete Silicate [uM]'] = Bottles['Discrete Silicate [uM]'].apply(lambda x: clean_nutrients(x))
Bottles['Discrete Phosphate [uM]'] = Bottles['Discrete Phosphate [uM]'].apply(lambda x: clean_nutrients(x))
Bottles['Discrete Nitrate [uM]'] = Bottles['Discrete Nitrate [uM]'].apply(lambda x: clean_nutrients(x))
Bottles['Discrete Nitrite [uM]'] = Bottles['Discrete Nitrite [uM]'].apply(lambda x: clean_nutrients(x))
# -

# ---
# #### Calculate TEOS-10 Properties for Bottle Data 
# The TEOS-10 properties are considered to be derived from thermodynamic principles whereas the previous TEOS-80 was derived from empirical observations. Here, we'll add in the parameters for conservative temperature, absolute salinity, and neutral density.

import gsw

# +
# Calculate some key physical parameters to get density based on TEOS-10
SP = Bottles[["Salinity 1, uncorrected [psu]","Salinity 1, uncorrected [psu]"]].mean(axis=1)
T = Bottles[['Temperature 1 [deg C]','Temperature 2 [deg C]']].mean(axis=1)
P = Bottles["Pressure [db]"]
LAT = Bottles["Latitude [deg]"]
LON = Bottles["Longitude [deg]"]

# Absolute salinity
SA = gsw.conversions.SA_from_SP(SP, P, LON, LAT)
Bottles["Absolute Salinity [g/kg]"] = SA

# Conservative temperature
CT = gsw.conversions.CT_from_t(SA, T, P)
Bottles["Conservative Temperature"] = CT

# Density
RHO = gsw.density.rho(SA, CT, P)
Bottles["Density [kg/m^3]"] = RHO

# Calculate potential density
SIGMA0 = gsw.density.sigma0(SA, CT)
# -

# ---
# #### Calculate Carbon System Parameters
# The discrete water samples were tested for Total Alkalinity, Dissolved Inorganic Carbon, and pH [Total Scale]. I calculate the discrete water sample pCO<sub>2</sub> concentrations from the TA and DIC using the ```CO2SYS``` program. 

# Calculate the Carbon System Parameters
from PyCO2SYS import CO2SYS

PAR1 = Bottles['Discrete Alkalinity [µmol/kg]']
PAR2 = Bottles['Discrete DIC [µmol/kg]']
PAR1TYPE = 1
PAR2TYPE = 2
SAL = Bottles['Discrete Salinity [psu]']
TEMPIN = 25
TEMPOUT = Bottles['Temperature 1 [deg C]']
PRESIN = 0
PRESOUT = Bottles['Pressure [db]']
SI = Bottles['Discrete Silicate [uM]']
PO4 = Bottles['Discrete Phosphate [uM]']
PHSCALEIN = 1
K1K2CONSTANTS = 1
K2SO4CONSTANTS = 1

CO2dict = CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT, PRESIN, PRESOUT, SI, PO4, PHSCALEIN, K1K2CONSTANTS, K2SO4CONSTANTS)

# ---
# #### Check accuracy of CO2SYS
# In order to demonstrate that the ```CO2SYS``` software package accurately calculates the pCO<sub>2</sub>, we can compare the pH calculated by ```CO2SYS``` against the measured seawater pH. This serves as an independent check and bound on the error introduced by the carbonate system algorithms. 

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
# %matplotlib inline

# Find how well CO2SYS reproduces the measured pH values
mask = CO2dict['pHin'] == 8 # This is the "error" value returned from the CO2SYS program
CO2dict['pHin'][mask] = np.nan
pH = Bottles['Discrete pH [Total scale]'].apply(lambda x: clean_pH(x))
pHdf = pd.DataFrame(data=[pH.values, CO2dict['pHin']], index=['Measured','CO2sys']).T
pH_meas = pHdf.dropna()['Measured'].values.reshape(-1,1)
pH_calc = pHdf.dropna()['CO2sys'].values.reshape(-1,1)

# +
# Use sklearn linear regression model to determine the accuracy 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Fit a linear regression to the measured vs calculated pH values
regression = LinearRegression()
regression.fit(pH_meas, pH_calc)

# Get the regression values
pH_pred = regression.predict(pH_meas)
pH_mse = mean_squared_error(pH_pred, pH_calc)
pH_std = np.sqrt(pH_mse)
# -

# Look at how closely the pH measurements match eachother
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
ax.scatter(pH_meas, pH_calc)
ax.plot(pH_meas, pH_pred, c='r', label='Regression')
ax.plot(pH_meas, pH_pred-1.96*pH_std, ':', c='r', label='2 Stds')
ax.plot(pH_meas, pH_pred+1.96*pH_std, ':', c='r')
ax.set_xlabel('Discrete pH\n[Total Scale, T=25C]')
ax.set_ylabel('CO2SYS-calculated pH\n[Total Scale, T=25C]')
ax.text(0.8,0.6,f'Std: {pH_std.round(3)}', fontsize='16', transform=ax.transAxes)
ax.legend()
ax.grid()

# Next, filter the CO2SYS results to remove the results equal to "8" which is the dummy value returned by CO2SYS:

# Now, using the locations where the pH measurement came out as 8 (indicating error), replace those locations
# with NaNs to avoid using bad data
for key in CO2dict.keys():
    try:
        CO2dict[key][mask] = np.nan
    except ValueError:
        pass

# Now add the calculated carbon system parameters to the cruise info
Bottles['Calculated Alkalinity [µmol/kg]'] = CO2dict['TAlk']
Bottles['Calculated CO2aq [µmol/kg]'] = CO2dict['CO2out']
Bottles['Calculated CO3 [µmol/kg]'] = CO2dict['CO3out']
Bottles['Calculated DIC [µmol/kg]'] = CO2dict['TCO2']
Bottles['Calculated pCO2 [µatm]'] = CO2dict['pCO2out']
Bottles['Calculated pCO2in'] = CO2dict['pCO2in']
Bottles['Calculated pH'] = CO2dict['pHoutTOTAL']

# ### Compare with OSU data
# Next, to check the accuracy of the ```CO2SYS``` program, I'll check it against values calculated by Oregon State for one of their cruises. 

CE10 = pd.read_excel("/home/andrew/Documents/OOI-CGSN/QAQC_Sandbox/Data_Review/Instrument_Review/PCO2W/Data/Endurance-10_SR1813_Discrete_Summary.xlsx")

# Take a look at the measured data
CE10.replace(to_replace=-9999999, value=np.nan, inplace=True)
CE10.head()

# #### Calculate Physical Properties of Coastal Endurance
# ---------------------------------
# Calculate the TEOS-10 properties for the Coastal Endurance data following the same approach as above for the CGSN data using the ```gsw``` package.

# +
# Calculate some key physical parameters to get density based on TEOS-10
SP = CE10[["CTD Salinity 1 [psu]","CTD Salinity 2 [psu]"]].mean(axis=1)
T = CE10[['CTD Temperature 1 [deg C]','CTD Temperature 2 [deg C]']].mean(axis=1)
P = CE10["CTD Pressure [db]"]
LAT = CE10["CTD Latitude [deg]"]
LON = CE10["CTD Longitude [deg]"]

# Absolute salinity
SA = gsw.conversions.SA_from_SP(SP, P, LON, LAT)
CE10["CTD Absolute Salinity [g/kg]"] = SA

# Conservative temperature
CT = gsw.conversions.CT_from_t(SA, T, P)
CE10["CTD Conservative Temperature"] = CT

# Density
RHO = gsw.density.rho(SA, CT, P)
CE10["CTD Density [kg/m^3]"] = RHO

# Calculate potential density
SIGMA0 = gsw.density.sigma0(SA, CT)
# -

# Calculate the **Alkalinity** and **pH** from the **DIC** and **pCO2**:

PAR1 = CE10["Discrete DIC [umol/kg]"]
PAR2 = CE10["Discrete pCO2 [uatm]"]
PAR1TYPE = 2
PAR2TYPE = 4
SAL = CE10["CTD Salinity 1 [psu]"]
TEMPIN = CE10["CTD Temperature 1 [deg C]"]
TEMPOUT = CE10["CTD Temperature 1 [deg C]"]
PRESIN = CE10["CTD Pressure [db]"]
PRESOUT = CE10["CTD Pressure [db]"]
SI = CE10['Discrete Silicate [uM]']
SI = SI*1000/RHO
PO4 = CE10['Discrete Phosphate [uM]']
PO4 = PO4*1000/RHO
PHSCALEIN = 1
K1K2CONSTANTS = 1
K2SO4CONSTANTS = 1

CO2dict = CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT, PRESIN, PRESOUT, SI, PO4, PHSCALEIN, K1K2CONSTANTS, K2SO4CONSTANTS)

# #### Compare OSU's results vs. CO2SYS program's results
# I regress the Total Alkalinity and the pH calculated by the ```CO2SYS``` program against the Total Alkalinity and pH calculated by Oregon State for the Coastal Endurance array, respectively. This allows us to determine how reproducible our methods are and if we are in agreement across labs.

# +
# Calculate the regression of Total Alkalinity
TAlk = pd.DataFrame(data=[CE10["Calculated Alkalinity [umol/kg]"], CO2dict["TAlk"]], index=["CE","CO2SYS"]).T.dropna()
TAlk.sort_values(by="CE", inplace=True)
TAlk_CE = TAlk["CE"].values.reshape(-1,1)
TAlk_SYS = TAlk["CO2SYS"].values.reshape(-1,1)

# Fit a linear regression to the measured vs calculated Total Alkalinity values
CE_TAlk_reg = LinearRegression()
CE_TAlk_reg.fit(TAlk_CE, TAlk_SYS)

# Get the regression values
TAlk_pred = CE_TAlk_reg.predict(TAlk_CE)
TAlk_mse = mean_squared_error(TAlk_CE, TAlk_SYS)
TAlk_std = np.sqrt(TAlk_mse)

# Retrieve the intercept and regression coefficient 
b, m = CE_TAlk_reg.intercept_[0], CE_TAlk_reg.coef_[0][0]

# +
fig, ax = plt.subplots(figsize=(15,5))
ax.scatter(CE10["Calculated Alkalinity [umol/kg]"], CO2dict["TAlk"], s=64)
ax.plot(TAlk_CE, TAlk_pred, c='r', label='Regression')
ax.plot(TAlk_CE, TAlk_pred-1.96*TAlk_std, ':', c='r', label='2 Stds')
ax.plot(TAlk_CE, TAlk_pred+1.96*TAlk_std, ':', c='r')
ax.legend()
ax.set_xlabel("CE Calculated Alkalinity [$\mu$mol/kg]", fontsize=16)
ax.set_ylabel("CO2SYS Calculated\nAlkalinity [$\mu$mol/kg]", fontsize=16)
ax.text(0.5,0.3,f'y = {m.round(2)}*x + {b.round(2)}, Std: {TAlk_std.round(2)}', fontsize='16', transform=ax.transAxes)

ax.grid()

# +
# Calculate the regression of pH
pH = pd.DataFrame(data=[CE10["Calculated pH [Total scale]"], CO2dict["pHout"]], index=["CE","CO2SYS"]).T.dropna()
pH.sort_values(by="CE", inplace=True)
pH_CE = pH["CE"].values.reshape(-1,1)
pH_SYS = pH["CO2SYS"].values.reshape(-1,1)

# Fit a linear regression to the measured vs calculated pH values
CE_pH_reg = LinearRegression()
CE_pH_reg.fit(pH_CE, pH_SYS)

# Get the regression values
pH_pred = CE_pH_reg.predict(pH_CE)
pH_mse = mean_squared_error(pH_CE, pH_SYS)
pH_std = np.sqrt(pH_mse)

# Retrieve the intercept and regression coefficient 
b, m = CE_pH_reg.intercept_[0], CE_pH_reg.coef_[0][0]
# -

fig, ax = plt.subplots(figsize=(15,5))
ax.scatter(CE10["Calculated pH [Total scale]"], CO2dict["pHin"], s=64)
ax.plot(pH_CE, pH_pred, c='r', label='Regression')
ax.plot(pH_CE, pH_pred-1.96*pH_std, ':', c='r', label='2 Stds')
ax.plot(pH_CE, pH_pred+1.96*pH_std, ':', c='r')
ax.legend()
ax.set_xlabel("CE Calculated pH [Total Scale]", fontsize=16)
ax.set_ylabel("CO2SYS Calculated\npH [Total Scale]", fontsize=16)
ax.text(0.5,0.3,f'y = {m.round(4)}*x + {b.round(4)}, Std: {pH_std.round(4)}', fontsize='16', transform=ax.transAxes)
ax.grid()

# ---
# ## PCO2W Data
# The PCO2W data is downloaded from OOINet as netCDF files, indexed by the dimension "obs". This needs switched to the time dimension.

# Load all of the Pioneer PCO2W data streams
basepath = "/media/andrew/Files/Instrument_Data/PCO2W"

# ---
# #### CP01CNSM MFN

refdes = "CP01CNSM-MFD35-05-PCO2WB000"
method = "recovered_inst"
stream = "pco2w_abc_instrument"

OOINet.get

datasets = ['/'.join((basepath, refdes, method, stream, dset)) for dset in os.listdir('/'.join((basepath, refdes, method, stream)))]
datasets = sorted([dset for dset in datasets if "blank" not in dset])
CNSM_PCO2W = load_datasets(datasets)
# Select a single spectrum to reduce the size of the dataset
CNSM_PCO2W = CNSM_PCO2W.sel({"spectrum":0})
CNSM_PCO2W

# ---
# #### CP03ISSM MFN

refdes = "CP03ISSM-MFD35-05-PCO2WB000"
method = "recovered_inst"
stream = "pco2w_abc_instrument"

datasets = ['/'.join((basepath, refdes, method, stream, dset)) for dset in os.listdir('/'.join((basepath, refdes, method, stream)))]
datasets = sorted([dset for dset in datasets if "blank" not in dset])
ISSM_PCO2W = load_datasets(datasets)
# Select a single spectrum to reduce the size of the dataset
ISSM_PCO2W = ISSM_PCO2W.sel({"spectrum":0})
ISSM_PCO2W

# ---
# #### CP04OSSM MFN

refdes = "CP04OSSM-MFD35-05-PCO2WB000"
method = "recovered_inst"
stream = "pco2w_abc_instrument"

datasets = ['/'.join((basepath, refdes, method, stream, dset)) for dset in os.listdir('/'.join((basepath, refdes, method, stream)))]
datasets = sorted([dset for dset in datasets if "blank" not in dset])
OSSM_PCO2W = load_datasets(datasets)
# Select a single spectrum to reduce the size of the dataset
OSSM_PCO2W = OSSM_PCO2W.sel({"spectrum":0})
OSSM_PCO2W

# ---
# ## CTD Data
# Load the associated CTD data with the PCO2W data

# Load all of the Pioneer PCO2W data streams
basepath = "/media/andrew/Files/Instrument_Data/CTDBP/"

# #### CP01CNSM MFN

refdes = "CP01CNSM-MFD37-03-CTDBPD000"
method = "recovered_inst"
stream = "ctdbp_cdef_instrument_recovered"

datasets = ['/'.join((basepath, refdes, method, stream, dset)) for dset in os.listdir('/'.join((basepath, refdes, method, stream)))]
datasets = sorted([dset for dset in datasets if "blank" not in dset])
CNSM_CTDBP = load_datasets(datasets)
CNSM_CTDBP

# #### CP03ISSM MFN

refdes = "CP03ISSM-MFD37-03-CTDBPD000"
method = "recovered_inst"
stream = "ctdbp_cdef_instrument_recovered"

# Load all of the CTDBP datasets
datasets = ['/'.join((basepath, refdes, method, stream, dset)) for dset in os.listdir('/'.join((basepath, refdes, method, stream)))]
datasets = sorted([dset for dset in datasets if "blank" not in dset])
ISSM_CTDBP = load_datasets(datasets)
ISSM_CTDBP

# #### CP04OSSM MFN

refdes = "CP04OSSM-MFD37-03-CTDBPE000"
method = "recovered_inst"
stream = "ctdbp_cdef_instrument_recovered"

# Load all of the CTDBP datasets
datasets = ['/'.join((basepath, refdes, method, stream, dset)) for dset in os.listdir('/'.join((basepath, refdes, method, stream)))]
datasets = sorted([dset for dset in datasets if "blank" not in dset])
OSSM_CTDBP = load_datasets(datasets)
OSSM_CTDBP

# ---
# ## Data Exploration
# Going to look at some basic plots of the data to see how the instruments perform over time

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

xmin = min(CNSM_PCO2W.time.min().values, ISSM_PCO2W.time.min().values, OSSM_PCO2W.time.min().values)
xmax = max(CNSM_PCO2W.time.max().values, ISSM_PCO2W.time.max().values, OSSM_PCO2W.time.max().values)

# ---
# ### Cruise Data
# ---

Cruises = pd.read_csv('/home/andrew/Documents/OOI-CGSN/ooi-integration/asset-management/cruise/CruiseInformation.csv')
Cruises = Cruises.dropna()
mask = Cruises['notes'].apply(lambda x: True if 'Pioneer' in x else False)
Cruises[mask]

# Select just the relevant Pioneer cruises (ignoring glider deployments/etc)
indices = [3, 6, 11, 13, 21, 35, 46, 59, 79, 98, 110, 137, 156, 180]
PioneerCruises = Cruises.loc[indices]
PioneerCruises

startDates = list(PioneerCruises['cruiseStartDateTime'].apply(lambda x: pd.to_datetime(x)))
endDates = list(PioneerCruises['cruiseStopDateTime'].apply(lambda x: pd.to_datetime(x)))
labels = list(PioneerCruises["CUID"] + '\n' + PioneerCruises['notes'])

# +
# Create a timeline of pioneer cruises
levels = np.tile([-5, 5, -3, 3, -1, 1], int(np.ceil(len(PioneerCruises)/6)))[:len(PioneerCruises)]

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_title("Pioneer Cruise Dates", fontsize=16)
markerline, stemline, baseline = ax.stem(startDates, levels, linefmt="C3-", basefmt="k-", use_line_collection=True)
plt.setp(markerline, mec="k", mfc="w", zorder=3)

# Shift markers to the baseline by replacing the ydata with zeros
markerline.set_ydata(np.zeros(len(startDates)))

# Annotate lines 
vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
for d, l, r, va in zip(startDates, levels, labels, vert):
    ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3), textcoords="offset points", va=va, ha="center", fontsize=14)
    
# Format xaxis
ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=6))
ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
    
# Remove y-axis and spines
ax.get_yaxis().set_visible(False)
for spine in ["left", "top", "right"]:
    ax.spines[spine].set_visible(False)

ax.grid()
ax.margins(y=0.1)
fig.autofmt_xdate()
# -

# #### Deployment Periods
# Plot the deployments lengths of the PCO2Ws on each of the Pioneer Array Multi-Function Nodes (MFNs) on Gantt charts. The length of each deployment is based on the first-and-last timestamps for each deployment, along with the cruise names associated with the instrument deployment.
#

# +
# Plot the PCO2W Deployment periods and the associated cruises for the CNSM
mask = CNSM_deployments['CUID_Deploy'].apply(lambda x: False if x.startswith('#') else True)
deploymentNumber = [x for x in CNSM_deployments['deploymentNumber'][mask].values]
startDateTime = [pd.to_datetime(x) for x in CNSM_deployments['startDateTime'][mask].values]
stopDateTime = [pd.to_datetime(x) for x in CNSM_deployments['stopDateTime'][mask].values]
cuid_deploy = [x for x in CNSM_deployments['CUID_Deploy'][mask].values]
cuid_recover = [x for x in CNSM_deployments['CUID_Recover'][mask].values]
stopDateTime[-1] = pd.to_datetime(datetime.datetime.now())
ilen = len(deploymentNumber)
pos = np.arange(0.5, ilen*0.5+0.5, 0.5)

# Now try to plot the data
fig, ax = plt.subplots(figsize=(12,8))
for i in range(len(deploymentNumber)):
    start_date = mdates.date2num(startDateTime[i])
    end_date = mdates.date2num(stopDateTime[i])
    ax.barh((i*0.5)+0.5, end_date - start_date, left=start_date, height=0.3, align='center', edgecolor='black', alpha=0.75)
    try:
        ax.annotate(cuid_deploy[i], xy=(start_date+10, (i*0.5)+0.5), clip_on=True, va='center')
    except:
        pass
# Add in the cruise dates
ax.set_title('CP01CNSM MFN PCO2W', fontsize=16)
locsy, labelsy = plt.yticks(pos, deploymentNumber)
ax.set_ylabel('Deployments', fontsize=16)
ax.xaxis_date()
ax.grid()
fig.autofmt_xdate()
# -

# Get the associated bounds for the x-axis
xmin, xmax = ax.get_xbound()
xmin, xmax

# +
# Plot the PCO2W Deployment periods and the associated cruises for the ISSM
mask = ISSM_deployments['CUID_Deploy'].apply(lambda x: False if x.startswith('#') else True)
deploymentNumber = [x for x in ISSM_deployments['deploymentNumber'][mask].values]
startDateTime = [pd.to_datetime(x) for x in ISSM_deployments['startDateTime'][mask].values]
stopDateTime = [pd.to_datetime(x) for x in ISSM_deployments['stopDateTime'][mask].values]
stopDateTime[-1] = pd.to_datetime(datetime.datetime.now())
cuid_deploy = [x for x in ISSM_deployments['CUID_Deploy'][mask].values]
cuid_recover = [x for x in ISSM_deployments['CUID_Recover'][mask].values]

ilen = len(deploymentNumber)
pos = np.arange(0.5, ilen*0.5+0.5, 0.5)

# Now try to plot the data
fig, ax = plt.subplots(figsize=(12,8))
for i in range(len(deploymentNumber)):
    start_date = mdates.date2num(startDateTime[i])
    end_date = mdates.date2num(stopDateTime[i])
    ax.barh((i*0.5)+0.5, end_date - start_date, left=start_date, height=0.3, align='center', edgecolor='black', alpha=0.75)
    try:
        ax.annotate(cuid_deploy[i], xy=(start_date+10, (i*0.5)+0.5), clip_on=True, va='center')
    except:
        pass
# Add in the cruise dates
ax.set_title('CP03ISSM MFN PCO2W', fontsize=16)
locsy, labelsy = plt.yticks(pos, deploymentNumber)
ax.set_ylabel('Deployments', fontsize=16)
ax.set_xlim((xmin, xmax))
ax.xaxis_date()
ax.grid()
fig.autofmt_xdate()

# +
# Plot the PCO2W Deployment periods and the associated cruises
mask = OSSM_deployments['CUID_Deploy'].apply(lambda x: False if x.startswith('#') else True)
deploymentNumber = [x for x in OSSM_deployments['deploymentNumber'][mask].values]
startDateTime = [pd.to_datetime(x) for x in OSSM_deployments['startDateTime'][mask].values]
stopDateTime = [pd.to_datetime(x) for x in OSSM_deployments['stopDateTime'][mask].values]
stopDateTime[-1] = pd.to_datetime(datetime.datetime.now())
cuid_deploy = [x for x in OSSM_deployments['CUID_Deploy'][mask].values]
cuid_recover = [x for x in OSSM_deployments['CUID_Recover'][mask].values]

ilen = len(deploymentNumber)
pos = np.arange(0.5, ilen*0.5+0.5, 0.5)

# Now try to plot the data
fig, ax = plt.subplots(figsize=(12,8))
for i in range(len(deploymentNumber)):
    start_date = mdates.date2num(startDateTime[i])
    end_date = mdates.date2num(stopDateTime[i])
    ax.barh((i*0.5)+0.5, end_date - start_date, left=start_date, height=0.3, align='center', edgecolor='black', alpha=0.75)
    try:
        ax.annotate(cuid_deploy[i], xy=(start_date+10, (i*0.5)+0.5), clip_on=True, va='center')
    except:
        pass
# Add in the cruise dates
ax.set_title('CP04OSSM MFN PCO2W', fontsize=16)
locsy, labelsy = plt.yticks(pos, deploymentNumber)
ax.set_ylabel('Deployments', fontsize=16)
ax.set_xlim((xmin, xmax))
ax.xaxis_date()
ax.grid()
fig.autofmt_xdate()


# -

def adjust_pco2w_pressure(PCO2in, Pin, Pout, Tin):
    """Adjust pCO2 partial pressures from Pin to Pout."""
    V = 35 # molal volume of CO2 (35 ml/mol)
    R = 82.057 # Gas constant 82.057 (ml*atm)/(mol*K)
    T = Tin + 273.15 # Temperature (K)
    Pin = Pin/10.1325 # Convert from dbar to atm (1 atm = 10.1325 db)
    Pout = Pout/10.1325 # Convert from dbar to atm 
    
    # Change in pressure
    dP = Pin - Pout
    
    # Calculate the hydrostatic pressure effect
    coef = (V*dP)/(R*T)
    PCO2out = PCO2in/np.exp(coef)
    
    return PCO2out


# Plot the pCO2 data from the PCO2Ws with the cruise times:

# +
# Plot the PCO2W Deployment periods and the associated cruises
mask = CNSM_deployments['CUID_Deploy'].apply(lambda x: False if x.startswith('#') else True)
deploymentNumber = [x for x in CNSM_deployments['deploymentNumber'][mask].values]
startDateTime = [pd.to_datetime(x) for x in CNSM_deployments['startDateTime'][mask].values]
stopDateTime = [pd.to_datetime(x) for x in CNSM_deployments['stopDateTime'][mask].values]
cuid_deploy = [x for x in CNSM_deployments['CUID_Deploy'][mask].values]
cuid_recover = [x for x in CNSM_deployments['CUID_Recover'][mask].values]

# Plot the PCO2W data with each of the cruises as lines
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(CNSM_PCO2W.time, CNSM_PCO2W.pco2_seawater, s=1)
for i in range(len(startDateTime)):
    ax.plot([startDateTime[i], startDateTime[i]],[200, 1200], c='r')
    ax.annotate(cuid_deploy[i], xy=(startDateTime[i]+datetime.timedelta(days=5), 250))
ax.set_title('CP01CNSM MFN PCO2W', fontsize=16)
ax.set_xlim(xmin, xmax)
ax.set_ylim(200, 1200)
ax.set_ylabel('pCO$_{2}$ [$\mu$atm]', fontsize=16)
ax.grid()
fig.autofmt_xdate()

# +
# Plot the PCO2W Deployment periods and the associated cruises
mask = OSSM_deployments['CUID_Deploy'].apply(lambda x: False if x.startswith('#') else True)
deploymentNumber = [x for x in OSSM_deployments['deploymentNumber'][mask].values]
startDateTime = [pd.to_datetime(x) for x in OSSM_deployments['startDateTime'][mask].values]
stopDateTime = [pd.to_datetime(x) for x in OSSM_deployments['stopDateTime'][mask].values]
cuid_deploy = [x for x in OSSM_deployments['CUID_Deploy'][mask].values]
cuid_recover = [x for x in OSSM_deployments['CUID_Recover'][mask].values]

# Plot the PCO2W data with each of the cruises as lines
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(OSSM_PCO2W.time, OSSM_PCO2W.pco2_seawater, s=1)
for i in range(len(startDateTime)):
    ax.plot([startDateTime[i], startDateTime[i]],[200, 1200], c='r')
    ax.annotate(cuid_deploy[i], xy=(startDateTime[i]+datetime.timedelta(days=5), 250))
ax.set_title('CP04OSSM MFN PCO2W', fontsize=16)
ax.set_xlim(xmin, xmax)
ax.set_ylim(200, 1200)
ax.set_ylabel('pCO$_{2}$ [$\mu$atm]', fontsize=16)
ax.grid()
fig.autofmt_xdate()

# +
# Plot the PCO2W Deployment periods and the associated cruises
mask = ISSM_deployments['CUID_Deploy'].apply(lambda x: False if x.startswith('#') else True)
deploymentNumber = [x for x in ISSM_deployments['deploymentNumber'][mask].values]
startDateTime = [pd.to_datetime(x) for x in ISSM_deployments['startDateTime'][mask].values]
stopDateTime = [pd.to_datetime(x) for x in ISSM_deployments['stopDateTime'][mask].values]
cuid_deploy = [x for x in ISSM_deployments['CUID_Deploy'][mask].values]
cuid_recover = [x for x in ISSM_deployments['CUID_Recover'][mask].values]

# Plot the PCO2W data with each of the cruises as lines
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(ISSM_PCO2W.time, ISSM_PCO2W.pco2_seawater, s=1)
for i in range(len(startDateTime)):
    ax.plot([startDateTime[i], startDateTime[i]],[200, 1200], c='r')
    ax.annotate(cuid_deploy[i], xy=(startDateTime[i]+datetime.timedelta(days=5), 250))
ax.set_title('CP03ISSM MFN PCO2W', fontsize=16)
ax.set_xlim(xmin, xmax)
ax.set_ylim(200, 1200)
ax.set_ylabel('pCO$_{2}$ [$\mu$atm]', fontsize=16)
ax.grid()
fig.autofmt_xdate()


# -

# ---
# ## CP01CNSM PCO2W
# First, we'll focus on the CNSM PCO2W measurements. The 

# #### Select the nearest cruise the start and end of the time period

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


# ## Deployment 1
# Deployment: KN-214 (Pioneer 1) <br>
# Recovery: KN-217 (Pioneer 2)

# Get the deployments
deploy_cruise = 'KN-214'
recover_cruise = 'KN-217'

# Find the CNSM deployment
CNSM_deployments[CNSM_deployments['CUID_Deploy'] == deploy_cruise]

# Find the ISSM deployment
ISSM_deployments[ISSM_deployments['CUID_Deploy'] == deploy_cruise]

# Find the OSSM deployment
OSSM_deployments[OSSM_deployments['CUID_Deploy'] == deploy_cruise]

# Get the first deployment of the CNSM PCO2W instrument and interpolate the co-located CTD data to the PCO2W measurements:

# Interpolate the ctd data to the pco2w timestamps
cnsm_pco2w_dep1 = CNSM_PCO2W.where(CNSM_PCO2W.deployment == 1, drop=True)
cnsm_ctdbp_dep1 = CNSM_CTDBP.where(CNSM_CTDBP.deployment == 1, drop=True).interp_like(cnsm_pco2w_dep1)
# Add the ctd data to the pco2w dataset
cnsm_pco2w_dep1['pressure'] = cnsm_ctdbp_dep1.ctdbp_seawater_pressure
cnsm_pco2w_dep1['salinity'] = cnsm_ctdbp_dep1.practical_salinity
cnsm_pco2w_dep1['temperature'] = cnsm_ctdbp_dep1.ctdbp_seawater_temperature
cnsm_pco2w_dep1['density'] = cnsm_ctdbp_dep1.density

# Select the bottle data associated with the deployment cruise KN-214:

# +
# Get the data
KN214 = Bottles[Bottles['Cruise']=='KN-214']

# Select just the data with pCO2 data
pco2mask = KN214['Calculated pCO2 [µatm]'].notna()
KN214 = KN214[pco2mask]
KN214.head()
# -

# Plot the pCO2 profile from the Bottle Data
fig, ax = plt.subplots(figsize=(5,10))
ax.scatter(KN214["Calculated pCO2 [µatm]"], KN214["Pressure [db]"], c="tab:red")
ax.invert_yaxis()
ax.grid()
ax.set_xlabel("pCO$_{2}$ [$\mu$atm]", fontsize=16)
ax.set_ylabel("Pressure [dbar]", fontsize=16)
ax.set_title("CNSM Bottle Data", fontsize=16)

# Plot just the pCO2 for the given time period
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(KN214['Bottle Closure Time [UTC]'], KN214['Calculated pCO2 [µatm]'], c='r', label='Discrete')
ax.plot(cnsm_pco2w_dep1.time, cnsm_pco2w_dep1.pco2_seawater, label='PCO2W')
ax.set_ylabel("pCO2 [$\mu$atm]", fontsize=16)
ax.set_title("CP01CNSM MFN Deployment 1", fontsize=16)
ax.grid()
ax.legend()
fig.autofmt_xdate()

# #### Comparing Deployment Data
# Next, compare the Bottle measurements and PCO2W measurements of pCO2 from the deployment cruise. First, adjust the pCO2 measurments for the pressure effect on partial pressures. Then, identify the Bottle measurements which were sampled near to the CNSM MFN.

# Adjust the bottle pCO2 measurements for hydrostatic pressure 
KN214['pCO2 Pressure-corrected'] = KN214[['Pressure [db]','Temperature 1 [deg C]','Calculated pCO2 [µatm]']].apply(lambda x: adjust_pco2w_pressure(x[2],x[0],1,x[1]), axis=1)

KN214["Target Asset"].unique()

# Select the Bottle samples from the CNSM deployment
cnsm = KN214["Target Asset"] == "CS"

# Adjust the PCO2W pCO2 measurements for hydrostatic pressure (this assumes its measured at surface pressure)
cnsm_pco2w_dep1 = cnsm_pco2w_dep1.assign(pco2_pressure_corrected = lambda x: adjust_pco2w_pressure(x.pco2_seawater, 1, 135, x.temperature))

# Select the first three days of data for 
pco2w_kn214 = cnsm_pco2w_dep1.sel(time=slice('2013-11-21','2013-11-25'))
mfn = (KN214['Pressure [db]'][cnsm] > 100) & (KN214['Pressure [db]'][cnsm] < 150)

# Convert the time series to a stationary series to calculate the standard deviation correctly
pco2dif = pco2w_kn214.pco2_pressure_corrected.diff("time")

pco2dif.values.mean(), pco2dif.values.std()

fig, ax = plt.subplots(figsize=(15,5))
ax.plot(pco2dif.time, pco2dif)
ax.plot(pco2dif.time, np.full(pco2dif.time.shape, pco2dif.values.mean()), color='k', ls=':')
ax.fill_between(pco2dif.time, pco2dif.values.mean()-1.96*pco2dif.values.std(), pco2dif.values.mean()+1.96*pco2dif.values.std(), alpha=0.25)
ax.grid()
ax.set_ylabel('$\Delta$pCO$_{2}$', fontsize=16)
ax.set_title('CP01CNSM PCO2W First-Differenced Time Series', fontsize=16)
fig.autofmt_xdate()

# +
# Take the mean of the values
bottle_pco2_avg = KN214['Calculated pCO2 [µatm]'][cnsm][mfn].mean()
bottle_pco2_std = KN214['Calculated pCO2 [µatm]'][cnsm][mfn].std()

# Take the 
pco2w_pco2_avg = pco2w_kn214.pco2_pressure_corrected.values.mean()
pco2w_pco2_std = pco2w_kn214.pco2_pressure_corrected.diff(dim='time').values.std()

# +
# Plot the PCO2 time series vs time
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(15,5))
ax0.scatter(KN214['Bottle Closure Time [UTC]'][cnsm][mfn], KN214['Calculated pCO2 [µatm]'][cnsm][mfn], c='r', label='Bottles')
ax0.plot(pco2w_kn214.time, pco2w_kn214.pco2_pressure_corrected, label='PCO2W')
ax0.fill_between(pco2w_kn214.time, pco2w_kn214.pco2_pressure_corrected-pco2w_pco2_std*1.96, pco2w_kn214.pco2_pressure_corrected+pco2w_pco2_std*1.96, alpha=0.25)
ax0.set_ylabel('Pressure-corrected pCO$_{2}$ [µatm]', fontsize=16)
#ax0.set_xlabel('Datetime')
ax0.grid()
ax0.legend()
# Plot the PCO2 vs pressure
ax1.scatter(pco2w_kn214.pco2_pressure_corrected, pco2w_kn214.pressure)
ax1.scatter(KN214['Calculated pCO2 [µatm]'][cnsm][mfn], KN214['Pressure [db]'][cnsm][mfn], c='r')
ax1.set_xlabel('Pressure-corrected pCO$_{2}$ [µatm]', fontsize=16)
ax1.set_ylabel('Pressure [dbar]', fontsize=16)
ax1.set_ylim((150, 100))
ax1.grid()

fig.autofmt_xdate()
# -

# Plot the results
pco2w_pco2_avg, pco2w_pco2_std

bottle_pco2_avg, bottle_pco2_std

# ---
# ## Deployment 3
# Deploy Cruise: AT-27 <br>
# Recover Cruise: AT-31

# Take a look at the deployment three data
# Interpolate the ctd data to the pco2w timestamps
cnsm_pco2w_dep3 = CNSM_PCO2W.where(CNSM_PCO2W.deployment == 3, drop=True)
cnsm_ctdbp_dep3 = CNSM_CTDBP.where(CNSM_CTDBP.deployment == 3, drop=True).interp_like(cnsm_pco2w_dep3)
# Add the ctd data to the pco2w dataset
cnsm_pco2w_dep3['pressure'] = cnsm_ctdbp_dep3.ctdbp_seawater_pressure
cnsm_pco2w_dep3['practical_salinity'] = cnsm_ctdbp_dep3.practical_salinity
cnsm_pco2w_dep3['temperature'] = cnsm_ctdbp_dep3.ctdbp_seawater_temperature
cnsm_pco2w_dep3['density'] = cnsm_ctdbp_dep3.density


# Calculate new variables for the dataset
# Absolute salinity
absolute_salinity = gsw.SA_from_SP(cnsm_pco2w_dep3.practical_salinity, cnsm_pco2w_dep3.pressure, cnsm_pco2w_dep3.lon, cnsm_pco2w_dep3.lat)
absolute_salinity = pd.Series(absolute_salinity, index=cnsm_pco2w_dep3.time.values)
absolute_salinity.index.name = 'time'
cnsm_pco2w_dep3['absolute_salinity'] = absolute_salinity
# Conservative temperature
conservative_temp = gsw.CT_from_t(cnsm_pco2w_dep3.absolute_salinity, cnsm_pco2w_dep3.temperature, cnsm_pco2w_dep3.pressure)
conservative_temp = pd.Series(conservative_temp, index=cnsm_pco2w_dep3.time.values)
conservative_temp.index.name = 'time'
cnsm_pco2w_dep3['conservative_temp'] = conservative_temp
# Potential density
sigma = gsw.sigma0(cnsm_pco2w_dep3.absolute_salinity, cnsm_pco2w_dep3.conservative_temp)
sigma = pd.Series(sigma, index=cnsm_pco2w_dep3.time.values)
sigma.index.name = 'time'
cnsm_pco2w_dep3['sigma'] = sigma

# Get the associated discrete sample data
at27 = Bottles['Cruise'].apply(lambda x: True if '27' in x else False)
AT27 = Bottles[at27]
AT27.head()

at31 = Bottles['Cruise'].apply(lambda x: True if '31' in x else False)
AT31 = Bottles[at31]
AT31.head()

# +
# Plot the Bottle Data
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
ax0.scatter(AT27["Calculated pCO2 [µatm]"], AT27["Pressure [db]"], c="tab:red")
ax0.grid()
ax0.set_xlabel("pCO$_{2}$ [$\mu$atm]", fontsize=16)
ax0.set_ylabel("Pressure [dbar]", fontsize=16)
ax0.invert_yaxis()
ax0.set_title("CP01CNSM MFN Deployment 3", fontsize=16)

ax1.scatter(AT31["Calculated pCO2 [µatm]"], AT31["Pressure [db]"], c="tab:blue")
ax1.grid()
ax1.set_xlabel("pCO$_{2}$ [$\mu$atm]", fontsize=16)
ax1.invert_yaxis()
ax1.set_title("CP01CNSM MFN Recovery 3", fontsize=16)
# -

# Now, need to apply corrections for hydrostatic pressure
# Adjust the bottle pCO2 measurements for hydrostatic pressure 
AT27['pCO2 Pressure-corrected'] = AT27[['Pressure [db]','Temperature 1 [deg C]','Calculated pCO2 [µatm]']].apply(lambda x: adjust_pco2w_pressure(x[2],x[0],1,x[1]), axis=1)
AT31['pCO2 Pressure-corrected'] = AT31[['Pressure [db]','Temperature 1 [deg C]','Calculated pCO2 [µatm]']].apply(lambda x: adjust_pco2w_pressure(x[2],x[0],1,x[1]), axis=1)

# Adjust the PCO2W pCO2 measurements for hydrostatic pressure (this assumes its measured at surface pressure)
cnsm_pco2w_dep3 = cnsm_pco2w_dep3.assign(pco2_pressure_corrected = lambda x: adjust_pco2w_pressure(x.pco2_seawater, 1, 135, x.temperature))

# Just select the bottle samples associated with the central surface mooring
at27_cnsm = (AT27['Target Asset'] == 'CNSM') & (AT27['Pressure [db]'] >= 100)
at31_cnsm = (AT31['Target Asset'] == 'CNSM') & (AT31['Pressure [db]'] >= 100)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(cnsm_pco2w_dep3.time, cnsm_pco2w_dep3.pco2_pressure_corrected, label='PCO2W')
ax.scatter(AT27['Bottle Closure Time [UTC]'][at27_cnsm], AT27['Calculated pCO2 [µatm]'][at27_cnsm], c='r', label='AT27', s=84)
ax.scatter(AT31['Bottle Closure Time [UTC]'][at31_cnsm], AT31['Calculated pCO2 [µatm]'][at31_cnsm], c='y', label='AT31', s=84)
ax.grid()
ax.set_title('CP01CNSM MFN PCO2W: Deployment 3', fontsize=16)
ax.legend(loc='upper left')
ax.set_ylabel('pCO$_{2}$ [$\mu$atm]', fontsize=16)
fig.autofmt_xdate()

pco2w_var = cnsm_pco2w_dep3.pco2_pressure_corrected.diff(dim="time")
pco2w_var

3.72280475*5

np.mean(pco2w_var), np.std(pco2w_var)

# +
# Select start and end times for the deployment of the PCO2W
tmin = pd.to_datetime(cnsm_pco2w_dep3.time.values.min())
tmax = tmin + datetime.timedelta(days=3)
# Select the first three days of data for 
pco2w_at27 = cnsm_pco2w_dep3.sel(time=slice(tmin,tmax))

# For the recovery
tmax = pd.to_datetime(cnsm_pco2w_dep3.time.values.max())
tmin = tmax - datetime.timedelta(days=3)
#
pco2w_at31 = cnsm_pco2w_dep3.sel(time=slice(tmin,tmax))

# +
# Calculate the standard deviations of the start and end time 
pco2w_at27_dif = pco2w_at27.pco2_pressure_corrected.diff('time')
pco2w_at27_avg, pco2w_at27_std = pco2w_at27_dif.values.mean(), pco2w_at27_dif.values.std()
pco2w_at31_dif = pco2w_at31.pco2_pressure_corrected.diff('time')
pco2w_at31_avg, pco2w_at31_std = pco2w_at31_dif.values.mean(), pco2w_at31_dif.values.std()

# Plot the first difference
fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(12,8))
ax0.plot(pco2w_at27_dif)
ax0.fill_between(range(len(pco2w_at27_dif)), pco2w_at27_avg-1.96*pco2w_at27_std, pco2w_at27_avg+1.96*pco2w_at27_std, alpha=0.25) 
ax0.grid()
ax0.set_ylabel('$\Delta$pCO$_{2}$', fontsize=16)

ax1.plot(pco2w_at31_dif)
ax1.fill_between(range(len(pco2w_at31_dif)), pco2w_at31_avg-1.96*pco2w_at31_std, pco2w_at31_avg+1.96*pco2w_at31_std, alpha=0.25) 
ax1.grid()
ax1.set_ylabel('$\Delta$pCO$_{2}$', fontsize=16)

# +
# Okay, the first difference time series look good. Now lets plot the full time series
# Plot the 
ymin, ymax = cnsm_pco2w_dep3.pco2_pressure_corrected.values.min(), cnsm_pco2w_dep3.pco2_pressure_corrected.values.max()
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax0.plot(pco2w_at27.time, pco2w_at27.pco2_pressure_corrected, label='PCO2W')
ax0.fill_between(pco2w_at27.time, pco2w_at27.pco2_pressure_corrected+1.96*pco2w_at27_std, pco2w_at27.pco2_pressure_corrected-1.96*pco2w_at27_std, alpha=0.25)
ax0.scatter(AT27['Bottle Closure Time [UTC]'][at27_cnsm], AT27['pCO2 Pressure-corrected'][at27_cnsm], c='r', label='AT27')
ax0.grid()
ax0.legend()
ax0.set_title('Deployment', fontsize=16)
ax0.set_ylim(ymin, ymax)
ax0.set_ylabel('pCO$_{2}$ [$\mu$atm]', fontsize=16)

ax1.plot(pco2w_at31.time, pco2w_at31.pco2_pressure_corrected, label='PCO2W')
ax1.scatter(AT31['Bottle Closure Time [UTC]'][at31_cnsm], AT31['Calculated pCO2 [µatm]'][at31_cnsm], c='r', label='AT31')
ax1.fill_between(pco2w_at31.time, pco2w_at31.pco2_pressure_corrected+1.96*pco2w_at31_std, pco2w_at31.pco2_pressure_corrected-1.96*pco2w_at31_std, alpha=0.25)
ax1.grid()
ax1.legend(loc='lower right')
ax1.set_title('Recovery', fontsize=16)
ax1.set_ylim(ymin, ymax)

fig.autofmt_xdate()
# -

# Now calculate the direct comparisons
pco2w_at27.pco2_pressure_corrected.values.mean(), pco2w_at27_std

AT27['Calculated pCO2 [µatm]'][at27_cnsm].mean(), AT27['Calculated pCO2 [µatm]'][at27_cnsm].std()

pco2w_at31.pco2_pressure_corrected.values.mean(), pco2w_at31_std

AT31['Calculated pCO2 [µatm]'][at31_cnsm].mean(), AT31['Calculated pCO2 [µatm]'][at31_cnsm].std()

# ### Analysis of Covarying Properties
# Deployment 3 of the CNSM PCO2W is one of the few timeseries of pCO2 which lasted the entire deployment. However, most time series has significant data or sampling gaps, making it significantly more difficult to analyze instrument performance. One way of filling in the missing data gaps is to find other water properties which strongly correlate with pCO2 and may be used to predict and fill the missing gaps. 
#
# I decompose the pCO2 time series as a function of the principal water measurements of temperature, salinity, density, and oxygen using Principal-Component-Analysis (also known as Empirical Orthogonal Functions) to do so.

# +
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1, figsize=(20,20))

# Plot the salinity
ax0.scatter(cnsm_pco2w_dep3.time, cnsm_pco2w_dep3.absolute_salinity, c=cnsm_pco2w_dep3.pco2_pressure_corrected)
ax0.set_ylabel('Absolute\nsalinity [g/kg]', fontsize=12)
ax0.set_title('CNSM MFN Deployment 3')
ax0.grid()

ax1.scatter(cnsm_pco2w_dep3.time, cnsm_pco2w_dep3.conservative_temp, c=cnsm_pco2w_dep3.pco2_pressure_corrected)
ax1.set_ylabel('$\Theta$ [$^{\circ}$C]', fontsize=12)
ax1.grid()

ax2.scatter(cnsm_pco2w_dep3.time, cnsm_pco2w_dep3.sigma, c=cnsm_pco2w_dep3.pco2_pressure_corrected)
ax2.set_ylabel('Potential\nDensity (kg/m$^{3}$)', fontsize=12)
ax2.grid()

ax3.scatter(cnsm_pco2w_dep3.time, cnsm_pco2w_dep3.pco2_pressure_corrected, c=cnsm_pco2w_dep3.pco2_pressure_corrected)
ax3.set_ylabel('pCO$_{2}$ [$\mu$atm]')
ax3.grid()

fig.autofmt_xdate()

# +
# Plot a T-S Diagram of the CTD Data
# First, create a grid of the 
temp = cnsm_pco2w_dep3.conservative_temp.values
salt = cnsm_pco2w_dep3.absolute_salinity.values

# Get the min and max values
smin = salt.min() - (0.01 * salt.min())
smax = salt.max() + (0.01 * salt.max())
tmin = temp.min() - (0.1 * temp.min())
tmax = temp.max() + (0.1 * temp.max())

# Calculate how many gridcells are needed in the x and y dimensions
xdim = np.int(np.round((smax-smin)/0.1+1.0))
ydim = np.int(np.round((tmax-tmin)+1.0))

# Create an empty grid of zeros
dens = np.zeros((ydim, xdim))

# Create temp and salt vectors of appropriate dimensions
ti = np.linspace(1, ydim-1, ydim)+tmin
si = np.linspace(1, xdim-1, xdim)*0.1 + smin

# Loop to fill in grid with densities
for j in range(0, int(ydim)):
    for i in range(0, int(xdim)):
        dens[j,i] = gsw.rho(si[i],ti[j],0)

# Subtract 1000 to convert to sigma_t
dens = dens-1000

# =============================================
# Plot the data
fig, ax = plt.subplots(figsize=(10,8))
CS = ax.contour(si, ti, dens, linestyles='dashed', colors='k')
ax.clabel(CS, fontsize=12, inline=1, fmt='%1.2f')
ST = ax.scatter(salt, temp, c=cnsm_pco2w_dep3.pco2_pressure_corrected)
plt.colorbar(ST, label='pCO$_{2}$ [$\mu$atm]')
ax.set_xlabel('Absolute Salinity (g/kg)', fontsize=16)
ax.set_ylabel('Conservative Temp ($^{\circ}$C)', fontsize=16)
ax.set_title('CNSM MFN Deployment 3', fontsize=16)
ax.grid()

# +
# Plot some of the data as a scatter plot against each other to look for correlations with pCO2
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

ax0.scatter(cnsm_pco2w_dep3.absolute_salinity, cnsm_pco2w_dep3.pco2_seawater, c=cnsm_pco2w_dep3.time)
ax0.set_xlabel('Absolute Salinity [g/kg]', fontsize=14)
ax0.set_ylabel('pCO$_{2}$ [$\mu$atm]', fontsize=14)
ax0.grid()

ax1.scatter(cnsm_pco2w_dep3.conservative_temp, cnsm_pco2w_dep3.pco2_seawater, c=cnsm_pco2w_dep3.time)
ax1.set_xlabel('Conservative Temperature ($^{\circ}$C)', fontsize=14)
ax1.grid()

ax2.scatter(cnsm_pco2w_dep3.sigma, cnsm_pco2w_dep3.pco2_seawater, c=cnsm_pco2w_dep3.time)
st = ax2.set_xlabel('Potential Density (kg/m$^{3}$)', fontsize=14)
#fig.colorbar(st, ax=ax2)
ax2.grid()


# +
# Do time series plots of each of the variables with pCO2
fig, (ax00, ax10, ax20) = plt.subplots(nrows=3, ncols=1, figsize=(15,9))

# First plot is to compare pCO2 and temperature
ax00.scatter(cnsm_pco2w_dep3.time, cnsm_pco2w_dep3.pco2_pressure_corrected, c="tab:blue")
ax00.set_ylabel("pCO$_{2}$ [$\mu$atm]", fontsize=14)
ax00.grid()

ax01 = ax00.twinx()
ax01.scatter(cnsm_pco2w_dep3.time, cnsm_pco2w_dep3.conservative_temp, c="tab:red", alpha=0.5)
ax01.set_ylabel("Conservative T\n[$^{\circ}$C]", fontsize=14)

# =================
# Plot the pCO2 vs the salinity
ax10.scatter(cnsm_pco2w_dep3.time, cnsm_pco2w_dep3.pco2_pressure_corrected, c="tab:blue")
ax10.set_ylabel("pCO$_{2}$ [$\mu$atm]", fontsize=14)
ax10.grid()

ax11 = ax10.twinx()
ax11.scatter(cnsm_pco2w_dep3.time, cnsm_pco2w_dep3.absolute_salinity, c="tab:green", alpha=0.5)
ax11.set_ylabel("Absolute salinity\n[g/kg]", fontsize=14)

# =================
# Plot the pCO2 vs the density
ax20.scatter(cnsm_pco2w_dep3.time, cnsm_pco2w_dep3.pco2_pressure_corrected, c="tab:blue")
ax20.set_ylabel("pCO$_{2}$ [$\mu$atm]", fontsize=14)
ax20.grid()

ax21 = ax20.twinx()
ax21.scatter(cnsm_pco2w_dep3.time, cnsm_pco2w_dep3.sigma, c="tab:orange", alpha=0.5)
ax21.set_ylabel("Potential Density", fontsize=14)

fig.autofmt_xdate()
# -

# Calculate some Pearson Correlations
pco2 = cnsm_pco2w_dep3.pco2_pressure_corrected.to_dataframe()
sigma = cnsm_pco2w_dep3.sigma.to_dataframe()
temp = cnsm_pco2w_dep3.conservative_temp.to_dataframe()
sal = cnsm_pco2w_dep3.absolute_salinity.to_dataframe()

import xarray as xr

df

# Pearson Correlation
df = pco2.merge(sigma, left_index=True, right_index=True).merge(temp, left_index=True, right_index=True).merge(sal, left_index=True, right_index=True)
columns = [x for x in df.columns if 'obs' not in x and 'spectrum' not in x]
df = df[columns]

df.corr()

# Rolling window of Pearson's Correlations
# First, interpolate missing data
df_interpolated = df.interpolate()
# Set window size
r_window_size = 24 # Daily window
# Calculate the rolling window
rolling_sigma = df_interpolated["pco2_pressure_corrected"].rolling(window=r_window_size, center=True).corr(df_interpolated["sigma"])
rolling_temp = df_interpolated["pco2_pressure_corrected"].rolling(window=r_window_size, center=True).corr(df_interpolated["conservative_temp"])
rolling_sal = df_interpolated["pco2_pressure_corrected"].rolling(window=r_window_size, center=True).corr(df_interpolated["absolute_salinity"])

# +
# Plot some of the results
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(15,9))

ax0.plot(rolling_sigma, c="tab:orange")
ax0.grid()
ax0.set_ylabel("Pearson R-value")
ax0.set_title("Potential Density")

ax1.plot(rolling_temp, c="tab:blue")
ax1.grid()
ax1.set_ylabel("Pearson R-value")
ax1.set_title("Conservative Temperature")

ax2.plot(rolling_sal, c="tab:green")
ax2.grid()
ax2.set_ylabel("Pearson R-value")
ax2.set_title("Absolute Salinity")

fig.autofmt_xdate()


# -

# ### Time Lagged Cross Correlation
# Time-lagged cross correlation can identify directionality between two signals. Note that this does not imply causality, just time-lagged correlation.

def crosscorr(x, y, lag=0, wrap=False):
    """
    Lag-N cross correlation.
    Shifted data is filled with NaNs.
    
    Args:
        lag - int, default 0
        x, y - pandas.Series objects of equal length and timebase
        
    Returns:
        crosscorr: float
    """
    if wrap:
        shifted_y = y.shift(lag)
        shifted_y.iloc[:lag] = y.iloc[-lag:].values
        return x.corr(shifted_y)
    else:
        return x.corr(y.shift(lag))


24*7

d1 = df["pco2_pressure_corrected"]
d2 = df["absolute_salinity"]
rs = [crosscorr(d1, d2, lag) for lag in range(-24*7,24*7+1)]
offset = np.ceil(len(rs)/2)-np.argmax(rs)

fig, ax = plt.subplots(figsize=(15,5))
ax.plot([x for x in range(-24*7,24*7+1)], rs)
ax.axvline(np.ceil(len(rs)/2)-(24*7+1), color="k", linestyle="--", label="Center")
ax.axvline(np.argmax(np.abs(rs))-(24*7), color="r", linestyle="--", label="Peak")
ax.grid()

np.argmax(np.abs(rs))-(24), (np.argmax(np.abs(rs))-(24))/24

np.max(np.abs(rs))

-224/24

# +
# PCA Example
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
# -

dpCO2 = cnsm_pco2w_dep3.pco2_pressure_corrected.diff(dim='time')
dS = cnsm_pco2w_dep3.absolute_salinity.diff(dim='time')
dP = cnsm_pco2w_dep3.pressure.diff(dim='time')
dT = cnsm_pco2w_dep3.conservative_temp.diff(dim='time')
dRho = cnsm_pco2w_dep3.density.diff(dim='time')

fig, ax = plt.subplots(figsize=(15,5))
ax.plot(dpCO2.time, dpCO2.values)

fig, ax = plt.subplots(figsize=(15,5))
ax.plot(dS.time, dS.values)

fig, ax = plt.subplots(figsize=(15,5))
ax.plot(dT.time, dT.values)

fig, ax = plt.subplots(figsize=(15,5))
ax.plot(dRho.time, dRho.values)

# Try with the full data
pCO2 = cnsm_pco2w_dep3.pco2_pressure_corrected.values
S = cnsm_pco2w_dep3.absolute_salinity.values
T = cnsm_pco2w_dep3.conservative_temp.values
#P = cnsm_pco2w_dep3.pressure.values
rho = cnsm_pco2w_dep3.sigma.values

X = np.vstack((pCO2, S, T, rho)).T
X.shape

df = pd.DataFrame(data=X, index=cnsm_pco2w_dep3.time.values, columns=['CO2','S','T','rho'])

df

# Fill missing values with the median
df.fillna(method='bfill', inplace=True)

X = df.values
sc = StandardScaler()
X_std = sc.fit_transform(X)

pca = decomposition.PCA()
X_pca = pca.fit(X_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_),'.-')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.grid()

pca = decomposition.PCA(n_components = 0.99)
X_pca = pca.fit_transform(X_std)
print(pca.n_components_)

pd.DataFrame(pca.components_, columns=df.columns)

n_pcs = pca.n_components_
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
initial_feature_names = df.columns
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

most_important_names

# #### Oxygen
# Now, add in the oxygen data as a check on the pCO2 data

root = "/media/andrew/Files/Instrument_Data/DOSTA/CP01CNSM-MFD37-04-DOSTAD000/telemetered"
datasets = ['/'.join((root, dset)) for dset in os.listdir(root)]

CNSM_DOSTA = xr.open_mfdataset(datasets)
CNSM_DOSTA = CNSM_DOSTA.swap_dims({"obs":"time"})
CNSM_DOSTA = CNSM_DOSTA.sortby("time")

# Plot the DOSTA data
fig, ax = plt.subplots(figsize=(15,5))
ax.scatter(CNSM_DOSTA.time, CNSM_DOSTA.dissolved_oxygen)

cnsm_dosta_dep3 = CNSM_DOSTA.where(CNSM_DOSTA.deployment == 3, drop=True).interp_like(cnsm_pco2w_dep3)
cnsm_dosta_dep3

fig, ax = plt.subplots(figsize=(15,5))
ax.scatter(cnsm_dosta_dep3.time, cnsm_dosta_dep3.dissolved_oxygen)
ax.grid()
ax.set_ylabel('Oxygen concentration ($\mu$mol kg$^{-1}$)')

dosta3 = CNSM_DOSTA.where(CNSM_DOSTA.deployment == 3, drop=True)
dosta3

fig, ax = plt.subplots(figsize=(15,5))
ax.scatter(dosta3.time, dosta3.dissolved_oxygen)
ax.grid()
ax.set_ylabel('Oxygen concentration ($\mu$mol kg$^{-1}$)')



# ---
# ## Deployment 5
# Deploy Cruise: AR-04 <br>
# Recover Cruise - AR-08B

# Take a look at the deployment three data
# Interpolate the ctd data to the pco2w timestamps
cnsm_pco2w_dep5 = CNSM_PCO2W.where(CNSM_PCO2W.deployment == 5, drop=True)
cnsm_ctdbp_dep5 = CNSM_CTDBP.where(CNSM_CTDBP.deployment == 5, drop=True).interp_like(cnsm_pco2w_dep5)
# Add the ctd data to the pco2w dataset
cnsm_pco2w_dep5['pressure'] = cnsm_ctdbp_dep5.ctdbp_seawater_pressure
cnsm_pco2w_dep5['salinity'] = cnsm_ctdbp_dep5.practical_salinity
cnsm_pco2w_dep5['temperature'] = cnsm_ctdbp_dep5.ctdbp_seawater_temperature
cnsm_pco2w_dep5['density'] = cnsm_ctdbp_dep5.density

Bottles['Cruise'].unique()

# Get the associated discrete sample data
ar04 = Bottles['Cruise'].apply(lambda x: True if 'AR04' in x else False)
AR04 = Bottles[ar04]
AR04.head()

ar08 = Bottles['Cruise'].apply(lambda x: True if '08' in x else False)
AR08 = Bottles[ar08]
AR08.head()

# Now, need to apply corrections for hydrostatic pressure
# Adjust the bottle pCO2 measurements for hydrostatic pressure 
AR04['pCO2 Pressure-corrected'] = AR04[['Pressure [db]','Temperature 1 [deg C]','Calculated pCO2 [µatm]']].apply(lambda x: adjust_pco2w_pressure(x[2],x[0],1,x[1]), axis=1)
AR08['pCO2 Pressure-corrected'] = AR08[['Pressure [db]','Temperature 1 [deg C]','Calculated pCO2 [µatm]']].apply(lambda x: adjust_pco2w_pressure(x[2],x[0],1,x[1]), axis=1)

# Adjust the PCO2W pCO2 measurements for hydrostatic pressure (this assumes its measured at surface pressure)
cnsm_pco2w_dep5 = cnsm_pco2w_dep5.assign(pco2_pressure_corrected = lambda x: adjust_pco2w_pressure(x.pco2_seawater, 1, 135, x.temperature))

# Just select the bottle samples associated with the central surface mooring
ar04_cnsm = (AR04['Target Asset'] == 'CNSM') & (AR04['Pressure [db]'] >= 100)
ar08_cnsm = (AR08['Target Asset'] == 'CNSM') & (AR08['Pressure [db]'] >= 100)

AR08[ar08_cnsm]['Calculated pCO2 [µatm]']

AR08['Target Asset']

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(cnsm_pco2w_dep5.time, cnsm_pco2w_dep5.pco2_pressure_corrected, label='PCO2W')
ax.scatter(AR04['Bottle Closure Time [UTC]'][ar04_cnsm], AR04['Calculated pCO2 [µatm]'][ar04_cnsm], c='r', label='AR04', s=84)
#ax.scatter(AR08['Bottle Closure Time [UTC]'][ar08_cnsm], AR08['Calculated pCO2 [µatm]'][ar08_cnsm], c='y', label='AR08', s=84)
ax.grid()
ax.set_title('CP01CNSM MFN PCO2W: Deployment 5', fontsize=16)
ax.legend(loc='upper left')
ax.set_ylabel('pCO$_{2}$ [$\mu$atm]', fontsize=16)
fig.autofmt_xdate()

# +
# Select start and end times for the deployment of the PCO2W
tmin = pd.to_datetime(cnsm_pco2w_dep5.time.values.min())
tmax = tmin + datetime.timedelta(days=3)
# Select the first three days of data for 
pco2w_ar04 = cnsm_pco2w_dep5.sel(time=slice(tmin,tmax))

# For the recovery - this time the instrument failed before the recovery cruise

# +
# Calculate the standard deviations of the start and end time 
pco2w_ar04_dif = pco2w_ar04.pco2_pressure_corrected.diff('time')
pco2w_ar04_avg, pco2w_ar04_std = pco2w_ar04_dif.values.mean(), pco2w_ar04_dif.values.std()
# pco2w_ar08_dif = pco2w_ar08.pco2_pressure_corrected.diff('time')
# pco2w_ar08_avg, pco2w_ar08_std = pco2w_ar08_dif.values.mean(), pco2w_ar08_dif.values.std()

# Plot the first difference
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(pco2w_ar04_dif)
ax.fill_between(range(len(pco2w_ar04_dif)), pco2w_ar04_avg-1.96*pco2w_ar04_std, pco2w_ar04_avg+1.96*pco2w_ar04_std, alpha=0.25) 
ax.grid()
ax.set_ylabel('$\Delta$pCO$_{2}$', fontsize=16)
# -

# Okay, the first difference time series look good. Now lets plot the full time series
# Plot the 
ymin, ymax = cnsm_pco2w_dep5.pco2_pressure_corrected.values.min(), cnsm_pco2w_dep5.pco2_pressure_corrected.values.max()
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(pco2w_ar04.time, pco2w_ar04.pco2_pressure_corrected, label='PCO2W')
ax.fill_between(pco2w_ar04.time, pco2w_ar04.pco2_pressure_corrected+1.96*pco2w_ar04_std, pco2w_ar04.pco2_pressure_corrected-1.96*pco2w_at27_std, alpha=0.25)
ax.scatter(AR04['Bottle Closure Time [UTC]'][ar04_cnsm], AR04['pCO2 Pressure-corrected'][ar04_cnsm], c='r', label='AR-04')
ax.grid()
ax.legend()
ax.set_title('Deployment', fontsize=16)
ax.set_ylim(ymin, ymax)
ax.set_ylabel('pCO$_{2}$ [$\mu$atm]', fontsize=16)
fig.autofmt_xdate()

# Now calculate the direct comparisons
pco2w_ar04.pco2_pressure_corrected.values.mean(), pco2w_ar04_std

AR04['Calculated pCO2 [µatm]'][ar04_cnsm].mean(), AR04['Calculated pCO2 [µatm]'][ar04_cnsm].std()

# **==========================================================================================================**
# ## Deployment 6
# Deploy Cruise: AR-08B (Pioneer 6) <br>
# Recover Cruise: AR-18 (Pioneer 7)

# Take a look at the deployment three data
# Interpolate the ctd data to the pco2w timestamps
cnsm_pco2w_dep6 = CNSM_PCO2W.where(CNSM_PCO2W.deployment == 6, drop=True)
cnsm_ctdbp_dep6 = CNSM_CTDBP.where(CNSM_CTDBP.deployment == 6, drop=True).interp_like(cnsm_pco2w_dep6)
# Add the ctd data to the pco2w dataset
cnsm_pco2w_dep6['pressure'] = cnsm_ctdbp_dep6.ctdbp_seawater_pressure
cnsm_pco2w_dep6['salinity'] = cnsm_ctdbp_dep6.practical_salinity
cnsm_pco2w_dep6['temperature'] = cnsm_ctdbp_dep6.ctdbp_seawater_temperature
cnsm_pco2w_dep6['density'] = cnsm_ctdbp_dep6.density

Bottles["Cruise"].unique()

# Get the associated discrete sample data for the deployment cruise
ar08 = Bottles['Cruise'].apply(lambda x: True if 'AR-08' in x else False)
AR08 = Bottles[ar08]
AR08.head()

# Get the associated discrete sample data for the recovery cruise
ar18 = Bottles["Cruise"].apply(lambda x: True if 'AR-18' in x else False)
AR18 = Bottles[ar18]
AR18.head()

# Now, need to apply corrections for hydrostatic pressure
# Adjust the bottle pCO2 measurements for hydrostatic pressure 
AR08['pCO2 Pressure-corrected'] = AR08[['Pressure [db]','Temperature 1 [deg C]','Calculated pCO2 [µatm]']].apply(lambda x: adjust_pco2w_pressure(x[2],x[0],1,x[1]), axis=1)
AR18['pCO2 Pressure-corrected'] = AR18[['Pressure [db]','Temperature 1 [deg C]','Calculated pCO2 [µatm]']].apply(lambda x: adjust_pco2w_pressure(x[2],x[0],1,x[1]), axis=1)

# Adjust the PCO2W pCO2 measurements for hydrostatic pressure (this assumes its measured at surface pressure)
cnsm_pco2w_dep6 = cnsm_pco2w_dep6.assign(pco2_pressure_corrected = lambda x: adjust_pco2w_pressure(x.pco2_seawater, 1, 135, x.temperature))

# +
# Plot the Bottle sample data
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8,10))

ax0.scatter(AR08["pCO2 Pressure-corrected"][ar08_cnsm], AR08["Pressure [db]"][ar08_cnsm])
ax0.set_ylabel("Pressure [dbar]", fontsize=16)
ax0.set_xlabel("pCO$_{2}$ [$\mu$atm]", fontsize=16)
ax0.invert_yaxis()
ax0.grid()

ax1.scatter(AR18["pCO2 Pressure-corrected"][ar18_cnsm], AR18["Pressure [db]"][ar18_cnsm])
ax1.set_xlabel("pCO$_{2}$ [$\mu$atm]", fontsize=16)
ax1.invert_yaxis()
ax1.grid()
# -

#
ar08_cnsm = AR08["Target Asset"] == "CNSM"
ar18_cnsm = AR18["Target Asset"] == "CNSM"

AR08["Discrete Phosphate [uM]"]

AR08[AR08["Target Asset"] == "CNSM"]["Discrete Phosphate [uM]"]

# **==========================================================================================================**
# ## Deployment 11
# Deploy cruise: AR-34 (Pioneer 12) <br>
# Recover Cruise: AR-39 (Pioneer 13)

# Get the deployments
deploy_cruise = 'AR-34'
recover_cruise = 'AR-39'

ar34 = df['Cruise'].apply(lambda x: True if 'AR34' in x else False)
AR34 = df[ar34]
AR34.head()

ar39 = df['Cruise'].apply(lambda x: True if 'AR39' in x else False)
AR39 = df[ar39]
AR39.head()

# Get the samples associated with the CP01CNSM 
ar34_cnsm = AR34['Target Asset'] == 'CSNM'
ar39_cnsm = AR39['Target Asset'] == 'CNSM'

AR34['Calculated pCO2 [µatm]'][ar34_cnsm]











# **==============================================================================================================**

from statsmodels.tsa.stattools import adfuller

# Check if the time series is stationary
adfuller(PCO2subset.pco2_pressure_corrected.diff(dim='time'))

# +
# Okay, time series is not stationary (duh) Now to find the correct differencing and build an ARIMA model for the data
# -

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(PCO2subset.pco2_pressure_corrected.diff(dim='time'))

# +
# So the pco2w data reaches stationarity with a single-order of differencing. Now find the AR model size
# -

plot_pacf(PCO2subset.pco2_pressure_corrected.diff(dim='time').values)

# +
from statsmodels.tsa.arima_model import ARIMA

# 4,0,1 model
model = ARIMA(PCO2subset.pco2_pressure_corrected.values, order=(1,0,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# -

# Plot the residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
ax[0].grid()
residuals.plot(kind='kde', title='Density', ax=ax[1])
ax[1].grid()
plt.show()

# Actual vs. fitted
model_fit.plot_predict(dynamic=False)
plt.grid()



# **==============================================================================================================**
# ### Modeling the time series data

# ## White Noise
#
# The simplest stochastic process. We'll draw from a univariate gaussian:

plt.style.use('default')

n = 500
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,6), gridspec_kw={'width_ratios':[3, 1]})
eps = np.random.normal(size=n)
ax[0].plot(eps)
ax[0].grid()
sns.distplot(eps, ax=ax[1])


# ## Moving Average
# A moving average process is defined as a weighted average of some previous values:
# $$
# X_{t} = \mu + \epsilon_{t} + \sum^{q}_{i=1}\theta_{i}\epsilon_{t-i}
# $$
# where $\theta$ are the parameters of the process and _q_ is the order of the process. With order we mean how many time steps _q_ we should include in the weighted average.
#
# Let's simulate an MA process. For every step _t_ we take the $\epsilon$ values up to _q_ time steps back. First, we create a function that given an 1D array creates a 2D array with rows that look _q_ indices back

def lag_view(x, order):
    """
    For every value X_i create a row that lags k values: [X_i-1, X_i-2,...,X_i-k]
    """
    y = x.copy()
    # Create features by shigting the window of order size by one step.
    # The result is a 2D array [[t1, t2, t3], [t2, t3, t4],..., [t_k-2, t_k-1, t_k]]
    x = np.array([y[-(i + order):][:order] for i in range(y.shape[0])])
    
    # Reverse the array as we started at the end and remove duplicates.
    # We truncate the features [order -1:] and the labels [order]
    # This is the shifting of the features with one time step compared to the labels
    x = np.stack(x)[::-1][order-1: -1]
    y = y[order:]
    
    return x, y


# In the above function, we create a 2D matrix which we truncate the input and output array so that all rows have lagging values.

lag_view(np.arange(10), 3)[0]


# Now we are able to easily take a look at different lags back in time. Let's simulate three different MA processes with order $q=1, q=6$ and $q=11$.

def ma_process(eps, theta):
    """
    Creates an MA(q) process with a zero mean (mean not included in implementation).
    :param eps: (array) White noise signal.
    :param theta: (array/list) Parameters of the process.
    """
    # Reverse the order of theta as X_t, X_t-1, X_t-k in an array is X_t-k, 
    theta = np.array([1] + list(theta))[::-1][:, None]
    eps_q, _ = lag_view(eps, len(theta))
    return eps_q @ theta


fig = plt.figure(figsize=(18, 4 * 3))
a = 310
for i in range(0, 11, 5):
    a += 1
    theta = np.random.uniform(0, 1, size=i + 1)
    plt.subplot(a)
    plt.grid()
    plt.title(f'$\\theta$ = {theta.round(2)}')
    plt.plot(ma_process(eps, theta))


# #### MA Processes from different orders
# Note that above we've chosen positive values for $\theta$, which isn't required. An MA process can have both positive and negative values for $\theta$. In the plots above we can see that when the order of **MA(q)** increases, the values are longer correlated with previous values. Actually, because the process is a weighted average of the $\epsilon$ values until lag $q$, the correlation drops to zero after this lag. Based on this property we can make an educated guess on about the order of an **MA(q)** process. This is good because inferring the order from the time series along is difficult.

# ## Autocorrelation
# When a value $X_{t}$ is correlated witha  previous value $X_{t-k}$, this is called autocorrelation. The auotcorrelation function is defined as:
# $$
# ACF(X_{t}, X_{t-k}) = \frac{E[(X_{t} - \mu_{t})(X_{t-k} - \mu_{t-k})]}{\sigma_{t}\sigma_{t-k}}
# $$
# Numerically we can approximate it by determining the correlation between different arrays, namely $X_{t}$ and array $X_{t-k}$. By doing so, we do need to truncate both arrays by $k$ elements in order to maintain an equal length.

# +
def pearson_correlation(x, y):
    return np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std())

def acf(x, lag=40):
    """
    Determine autocorrelation factors.
    :param x: (array) Time series.
    : param lag: (int) Number of lags.
    """
    return np.array([1] + [pearson_correlation(x[:-i], x[i:]) for i in range(1, lag)])


# -

lag = 40
# Create an ma(1) and ma(2) process
ma_1 = ma_process(eps, [1])
ma_2 = ma_process(eps, [0.2, -0.3, 0.8])

# Above we created an **MA(1)** and **MA(2)** process with different weights $\theta$. The weights for the models are:
# * MA(1): [1]
# * MA(2): [0.2, -0.3, 0.8]
#
# Below we apply the ACF on both series and we plot the results of both applied functions. We've also defined a helper function **bartletts_formula** which we use as a null hypothesis to determine if the correlation coefficients we've found are significant and not a statistical fluke. With this functions, we determine a confidence incterval $CI$.
# $$
# CI = \pm z_{1-\alpha/2} \sqrt{\frac{1 + 2\sum^{h-1}_{1<i<h-1}r^{2}_{i}}{N}}
# $$
# where $z_{1-\alpha/2}$ is the quantile function from the normal distribution. Quantile functions are the inverse of the cumulative distribution function and can be called with **scipy.stats.norm.ppf**. Any values outside of this confidence interval (below plotted in orange) are statistically significant.

import scipy as sp


# +
def bartletts_formula(acf_array, n):
    """
    Computes the Standard Error of an acf with Bartlett's formula
    :param acf_array: (array) Contaiing autocorrelation factors
    :param n: (int) Length of original time series sequence.
    """
    # The first value has autocorrelation with itself. So that value is skiped
    se = np.zeros(len(acf_array)-1)
    se[0] = 1 / np.sqrt(n)
    se[1:] = np.sqrt((1 + 2 * np.cumsum(acf_array[1:-1]**2)) / n)
    return se

def plot_acf(x, alpha=0.05, lag=40):
    """
    :param x: (array)
    :param alpha: (flt) Statistical significance for confidence interval
    :param lag: (int)
    """
    acf_val = acf(x, lag)
    plt.figure(figsize=(16, 4))
    plt.vlines(np.arange(lag), 0, acf_val)
    plt.scatter(np.arange(lag), acf_val, marker='o', alpha=1)
    plt.xlabel('lag')
    plt.ylabel('autocorrelation')
    plt.grid()
    
    # Determine confidence interval
    ci = sp.stats.norm.ppf(1 - alpha / 2.) * bartletts_formula(acf_val, len(x))
    plt.fill_between(np.arange(1, ci.shape[0] + 1), -ci, ci, alpha=0.25)


# -

for array in [ma_1, ma_2]:
    plot_acf(array)


# ## AR process
# In the section above we have seen and simulated an MA process and described the definition of autocorrelation to infer the order of a purely MA process. Now we are going to simulate another series called the Auot Regressive (RA) process. Again we're going to infer the order of the process using the Partial AutoCorrelation Function (PACF). 
#
# An **AR(p)** process is defined as:
# $$
# X_{t} = c + \epsilon_{t} \sum^{p}_{i=1}\phi_{i}X_{t-i}
# $$
# Now $\phi$ are the parameters of the process and $p$ is the order of the process. Where **MA(q)** is a weighted average over the error terms (white noise), **AR(p)** is a weighted average over the previous values of the series $X_{t-p}$. Note that this process also has a white noise variable, which makes this a stochastic series.

def ar_process(eps, phi):
    """
    Creates an AR process with a zero mean (c)
    """
    # Reverse the order of phi and add a 1 for current eps_t
    phi = np.r_[1, phi][::-1]
    ar = eps.copy()
    offset = len(phi)
    for i in range(offset, ar.shape[0]):
        ar[i - 1] = ar[i - offset: i] @ phi
    return ar


fig = plt.figure(figsize=(16, 4 * 3))
a = 310
for i in range(0, 11, 5):
    a += 1
    phi = np.random.normal(0, 0.1, size=i + 1)
    plt.subplot(a)
    plt.title(f'$\\phi$ = {phi.round(2)}')
    plt.plot(ar_process(eps, phi))
    plt.grid()

# ## AR Processes from different orders
# Below we create three new **AR(p)** processes and plot the ACF of these series:

plot_acf(ar_process(eps, [0.3, -0.3, 0.5]))
plot_acf(ar_process(eps, [0.5, -0.1, 0.1]))
plot_acf(ar_process(eps, [0.2, 0.5, 0.1]))


# By analyzing these plots we can tell that the ACF plot of the **AR** processes don't necessarily cut off after lag $p$. For the **AR(p)** process, the ACF clearly isn't decisive for determining the order of the process. Actually, for **AR** processes we use the **Partial Autocorrelation Function**.
#
# ## Partial Autocorrelation
# The partial autocorrelation function shows the autocorrelation of value $X_{t}$ and $X_{t-k}$ after the correlation between $X_{t}$ with the intermediate values $X_{t-1},...,X_{t-k+1}$ explained. Below we'll go through the steps required to determine partial autocorrelation.
#
# The partial correlation between $X_{t}$ and $X_{t-k}$ can be determined by training two linear models. 
#
# Let $\hat{X}_{t}$ and $\hat{X}_{t-k}$ be determined by a Linear Model optimized on $X_{t-1}...X_{t-k-1}$ and paameterized by $\alpha$ and $\beta$:
#
# $$
# \hat{X}_{t} = \alpha_{1}X_{t-1} + \alpha_{2}X_{t-2}...\alpha_{k-1}X_{t-k-1}
# $$
# $$
# \hat{X}_{t-k} = \beta_{1}X_{t-1} + \beta_{2}X_{t-2}...\beta_{k-1}X_{t-k-1}
# $$
#
# The partial correlation is then defined by the Pearson's coefficient of the residuals of both predicted values $\hat{X}_{t}$ and $\hat{X}_{t-k}$:
# $$
# PACF(X_{t},X_{t-k}) = corr((X_{t}-\hat{X}_{t}),(X_{t-k}-\hat{X}_{t-k}))
# $$

# ### Intermezzo: Linear Models
# Above we use a linear model to define the PACF. Later, we are also going to train an ARIMA model, which is a type of linear model. So let's quickly define a linear regression model:
# $$
# y=\beta X+\epsilon
# $$
# Where the parameters $\beta$ can be found via Ordinary-Least-Squares:
# $$
# \beta = (X^{T}X)^{-1}X^{T}y
# $$
#

def least_squares(x, y):
    return np.linalg.inv((x.T @ x)) @ (x.T @ y)


class LinearModel:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.beta = None
        self.intercept_ = None
        self.coef_ = None
        
    def _prepare_features(self, x):
        if self.fit_intercept:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x
    
    def fit(self, x, y):
        x = self._prepare_features(x)
        self.beta = least_squares(x, y)
        if self.fit_intercept:
            self.intercept_ = self.beta[0]
            self.coef_ = self.beta[1:]
        else:
            self.coef_ = self.beta
            
    def predict(self, x):
        x = self._prepare_features(x)
        # Now calculate y from y = Beta*X + error
        return x @ self.beta
    
    def fit_predict(self, x, y):
        """Fit and predict the data"""
        self.fit(x, y)
        return self.predict(x)



# Above we have defined a scikit-learn style class that can perform linear regression. We train by applying the **fit** method. If we also want to train the model with an intercept, we add ones to the feature matrix. This will result in a constant shift when applying $\beta X$.
#
# With the short intermezzo in place, we can finaly define the partial autocorrelation function and plot the results.

x=ar_process(eps, [0.3, -0.3, 0.5])


# +
def pacf(x, lag=40):
    """
    Calculate the Partial Autocorrelation Function.
    
    The PACF returns results:
        [1, pacf_lag_1, pacf_lag_2, pacf_lag_3,...]
    :param x: (array)
    :param lag: (int)
    """
    y=[]
    
    # Partial auto-correlation needs intermediate terms.
    # Thus, we start at index 3
    for i in range(3, lag + 2):
        backshifted = lag_view(x, i)[0]

        xt = backshifted[:, 0]
        feat = backshifted[:, 1:-1]
        xt_hat = LinearModel(fit_intercept=False).fit_predict(feat, xt)

        xt_k = backshifted[:, -1]
        xt_k_hat = LinearModel(fit_intercept=False).fit_predict(feat, xt_k)

        y.append(pearson_correlation(xt - xt_hat, xt_k - xt_k_hat))
    return np.array([1, acf(x, 2)[1]] +  y)


def plot_pacf(x, alpha=0.05, lag=40, title=None):
    """
    :param x: (array)
    :param alpha: (flt) Statistical significance for confidence interval.
    :parm lag: (int)
    """
    pacf_val = pacf(x, lag)
    plt.figure(figsize=(16, 4))
    plt.vlines(np.arange(lag + 1), 0, pacf_val)
    plt.scatter(np.arange(lag + 1), pacf_val, marker='o')
    plt.grid()
    plt.xlabel('lag')
    plt.ylabel('autocorrelation')
    
    # Determine confidence interval
    ci = sp.stats.norm.ppf(1 - alpha / 2.) * bartletts_formula(pacf_val, len(x))
    plt.fill_between(np.arange(1, ci.shape[0] + 1), -ci, ci, alpha=0.25)


# -

plot_pacf(ar_process(eps, [0.3, -0.3, 0.5]))
plot_pacf(ar_process(eps, [0.5, -0.1, 0.1]))
plot_pacf(ar_process(eps, [0.2, 0.5, 0.1]))


# In the above figures, we now see a significant cut off at lag 3 for all 3 autoregressive processes. We thus are able to infer the order of the processes. The relationship between AR and MA processes and the ACF and PACF plots are one to keep in mind, as they help with inferring the order of a certain series.
#
# |   | **AR(p)** | **MA(q)** | **ARMA(p,q)** |
# | - | --------- | --------- | ------------- |
# | ACF | Tails off | Cuts off after lag _q_ | Tails off |
# | PACF | Cuts off after lag _q_ | Tails off | Tails off |
#
# In the table above, we show the relationship between processes and autocorrelations. The **ARMA(p,q)** process is also included in this table, which is just a combination of an **AR(p)** and **MA(q)** series.
#
# Before we go and combine AR and MA processes, first we'll discuss the concept of Stationarity.

# ## Stationarity
# An ARMA model requires the time series to be **stationary**, whereas an ARIMA model does not. A **stationary** time series has a constant mean and a constant variance over time. For the white noise AR and MA time series we've defined above, this requirement holds, but for a lot of real-world data this does not. ARIMA models can work with data that isn't stationary, but instead has a trend. For time series that have recurring patterns (e.g. seasonality), ARIMA models don't work.
#
# When data shows a trend, we can remove the trend by differencing time step $X_{t}$ with $X_{t-1}$. We can difference **n** times until the data is stationary. We can test stationarity with a Dicker-Fuller test. 
#
# We can difference a time series by:
# $$
# \nabla X_{t} = X_{t} - X_{t-1}
# $$
#
# And we can undo the difference by taking the sum:
# $$
# X_{t} = \nabla X_{t} + \nabla X_{t-1}
# $$
#
# We can implement this with two recursive functions:

# +
def difference(x, d=1):
    if d==0:
        return x
    else:
        x = np.r_[x[0], np.diff(x)]
        return difference(x, d-1)
    
def undo_difference(x, d=1):
    if d==1:
        return np.cumsum(x)
    else:
        x = np.cumsum(x)
        return undo_difference(x, d-1)


# -

# # ARIMA 
# Now we have discussed all the pieces we need in order to define an ARIMA model with hyperparameters p, q, and d:
# * p = order of the AR model
# * q = order of the MA model
# * d = differencing order (how often we difference the data)
#
# The ARMA and ARIMA combination is defined as:
# $$
# X_{t} = c + \epsilon_{t} + \sum^{p}_{i=1}\phi_{i}X_{t-i} + \sum^{q}_{i=1}\theta_{i}\epsilon_{t-i}
# $$
#
# We see that the model is dependent on a white noise term $\epsilon$, which we don't know a priori. Therefore, we can use a trick for retrieving a quasi-white noise term. First, we train an **AR(p)** model and take the residuals as the $\epsilon_{t}$ terms. This will lead to an estimation of an **ARIMA** model. We could estimate the error terms $\epsilon$ more accurately by iteratively training the **ARIMA** model by updating the redisuals every iteration. For now, we'll accept the quasi-white-noise method. With these white noise terms, we can start modelling the full **ARIMA(q, p, d)** model.
#
# Below we've defined the ARIMA class which inherits from LinearModel. Because of this inheritance, we can call the fit and predict methods from the parent with the super function:

class ARIMA(LinearModel):
    def __init__(self, q, d, p):
        """
        An ARIMA Model:
        :param q: (int) Order of the MA model
        :param p: (int) Order of the RA model
        :param d: (int) Number of times to difference
        """
        super().__init__(True)
        self.p = p
        self.q = q
        self.d = d
        self.ar = None
        self.resid = None
        
    def prepare_features(self, x):
        if self.d > 0:
            x = difference(x, self.d)
        
        ar_features = None
        ma_features = None
        
        # Determine the features and the epsilon terms for the MA process
        if self.q > 0:
            if self.ar is None:
                self.ar = ARIMA(0, 0, self.p)
                self.ar.fit_predict(x)
            eps = self.ar.resid
            eps[0] = 0
            
            # Prepend with zeros as there are no residuals_t-k in the first X_t
            ma_features, _ = lag_view(np.r_[np.zeros(self.q), eps], self.q)
            
        # Determine the features for the AR process
        if self.p > 0:
            # Prepend with zeros as there are no X_t-k in the first X_t
            ar_features = lag_view(np.r_[np.zeros(self.p), x], self.p)[0]
            
        if ar_features is not None and ma_features is not None:
            n = min(len(ar_features), len(ma_features)) 
            ar_features = ar_features[:n]
            ma_features = ma_features[:n]
            features = np.hstack((ar_features, ma_features))
        elif ma_features is not None: 
            n = len(ma_features)
            features = ma_features[:n]
        else:
            n = len(ar_features)
            features = ar_features[:n]
        
        return features, x[:n]
    
    def fit(self, x):
        features, x = self.prepare_features(x)
        super().fit(features, x)
        return features
            
    def fit_predict(self, x): 
        """
        Fit and transform input
        :param x: (array) with time series.
        """
        features = self.fit(x)
        return self.predict(x, prepared=(features))
    
    def predict(self, x, **kwargs):
        """
        :param x: (array)
        :kwargs:
            prepared: (tpl) containing the features, eps and x
        """
        features = kwargs.get('prepared', None)
        if features is None:
            features, x = self.prepare_features(x)
        
        y = super().predict(features)
        self.resid = x - y

        return self.return_output(y)
    
    def return_output(self, x):
        if self.d > 0:
            x = undo_difference(x, self.d) 
        return x
    
    def forecast(self, x, n):
        """
        Forecast the time series.
        
        :param x: (array) Current time steps.
        :param n: (int) Number of time steps in the future.
        """
        features, x = self.prepare_features(x)
        y = super().predict(features)
        
        # Append n time steps as zeros. Because the epsilon terms are unknown
        y = np.r_[y, np.zeros(n)]
        for i in range(n):
            feat = np.r_[y[-(self.p + n) + i: -n + i], np.zeros(self.q)]
            y[x.shape[0] + i] = super().predict(feat[None, :])
        return self.return_output(y)

# The above `ARIMA` class inherits from `LinearModel`. The first step is to initiate the parent and pass the boolean `True` so that we also fit an intercept for the model. In the `prepare_features` method (note that this one has no _ prefix and thus differs from the parent method), we create the features for the linear regression model. The features comprise of the lagging time steps $X_{t-k}$ with order **q**, which is the **AR** portion of the model, and of the lagging error terms $\epsilon_{t-k}$, which is the **MA** part of the model. In this method we also train an **AR** model first, so that we can use the residuals of that model as error terms. Note that we prepend the $\epsilon$ and the $X$ with _n_ zeros, where _n_ is equal to the order **q** and **p**, respectively. This is done because there are no values $\epsilon_{t-q}$ and $X_{t-p}$ at time $X_{0}$. Furthermore, we just implement some methods inspired by scikit-learning naming convention, e.g. the `fit_predict`, `fit`, and `predict` methods. 


