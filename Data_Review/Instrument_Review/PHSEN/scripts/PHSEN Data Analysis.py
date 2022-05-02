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

# # PHSEN Data Analysis
#
# ---
# ### Purpose
# The purpose of this notebook is to analyze the performance of the Sunburst Sensors, LLC. SAMI-pH (PHSEN) pH seawater measurements at the Pioneer Array. This is done on a deployment-by-deployment, site-by-site comparison with the pH measurements from discete water samples collected by Niskin Bottle casts during deployment and recovery of the instrumentation during mooring maintainence. 
#
# ---
# ### Datasets
# There are three main sources of data sources:
# * **Deployments**: These are the deployment master sheets from OOI Asset Management. They contain the deployment numbers, deployment start times and cruise, and recovery times and cruise, for all of the instrumentation deployed. 
# * **PHSEN**: This is the Sunburst Sensors, LLC. SAMI-pH sensor. It is calibrated for pH values from 70-9.0 pH units for salinities from 25-40 psu. Manufacturers stated accuracy of 0.003 pH units, precision < 0.001 pH units, and long-term drift of < 0.001 pH units / 6 months. The data is downloaded from the Ocean Observatories data portal (OOINet) as netCDF files.
# * **CTDBP**: This is the collocated SeaBird CTD with the PHSEN. The data is downloaded from the Ocean Observatories data portal (OOINet) as netCDF files. These data are needed since the PCO2W datasets do not contain either temperature (T), salinity (S), pressure (P), or density ($\rho$) data needed to compare with the discrete sampling.
# * **Discrete Water Samples**: These are discrete water samples collected via Niskin Bottle casts during deployment and recovery of the moored instrumentation. The data is downloaded from OOI Alfresco website as excel files. Parameters sampled include oxygen, salinity, nutrient concentrations (phosphate, nitrate, nitrite, ammonium, silicate), chlorophyll concentrations, and the carbon system. The carbon system parameters sampled are Total Alkalinity (TA), Dissolved Inorganic Carbon (DIC), and pH. 
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

import matplotlib.pyplot as plt
# %matplotlib inline

# Import the OOI M2M tool
sys.path.append("/home/andrew/Documents/OOI-CGSN/ooinet/ooinet/")
from m2m import M2M
from phsen import PHSEN

# #### Set OOINet API access
# In order access and download data from OOINet, need to have an OOINet api username and access token. Those can be found on your profile after logging in to OOINet. Your username and access token should NOT be stored in this notebook/python script (for security). It should be stored in a yaml file, kept in the same directory, named user_info.yaml.

from m2m import M2M

# Import user info for connecting to OOINet via M2M
userinfo = yaml.load(open("../../../../user_info.yaml"))
username = userinfo["apiname"]
token = userinfo["apikey"]

# #### Connect to OOINet

OOINet = M2M(username, token)


# ---
# ### Define Useful Functions

def check_thredds_table(thredds_table, refdes, method, stream, parameters):
    """Function which checks the thredds_table for if a request has been implemented before"""
    
    # Filter for the refdes-method-stream
    request_history = thredds_table[(thredds_table["refdes"] == refdes) &
                                    (thredds_table["method"] == method) & 
                                    (thredds_table["stream"] == stream) &
                                    (thredds_table["parameters"] == parameters)]
    
    if len(request_history) == 0:
        if parameters == "All":
            thredds_url = OOINet.get_thredds_url(refdes, method, stream)
        else:
            thredds_url = OOINet.get_thredds_url(refdes, method, stream, parameters=parameters)
    
        # Save the request to the table
        thredds_table = thredds_table.append({
            "refdes": refdes,
            "method": method,
            "stream": stream,
            "request_date": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "thredds_url": thredds_url,
            "parameters": parameters,
        }, ignore_index = True)
    else:
        thredds_url = request_history[request_history["request_date"] == np.max(request_history["request_date"])]["thredds_url"].iloc[0]
    
    return thredds_table, thredds_url


# ---
# ## Irminger Array

# ### Identify PHSEN instruments
# Next, we want to identify the deployed PHSEN instruments. Based on the results of the PHSEN tech refresh analysis, we have identified the following sensors as having relatively robust datasets: **GI03FLMA** and **GI03FLMB**. Additionally, we include the PHSEN on the surface mooring **GI01SUMO**.

# Start with the GI01SUMO PHSEN 
OOI.search_datasets(array="GI01SUMO", instrument="PHSEN", English_names=True)

# Flanking Mooring A
OOI.search_datasets(array="GI03FLMA", instrument="PHSEN", English_names=True)

# Flanking Mooring B
OOI.search_datasets(array="GI03FLMB", instrument="PHSEN", English_names=True)

# ## Load PHSEN Data
# With the identified data sets, want to download the data.
#
# #### GI01SUMO-RII11-02-PHSENE041

refdes = "GI01SUMO-RII11-02-PHSENE041"

vocab = OOI.get_vocab(refdes)
vocab

deployments = OOI.get_deployments(refdes)
deployments

datastreams = OOI.get_datastreams(refdes)
datastreams

# Select the relevant method and stream
method = "recovered_inst"
stream = "phsen_abcdef_instrument"

# If the data has been previously downloaded, can open the data from the local directory:

load_dir = f"/media/andrew/Files/Instrument_Data/PHSEN/{refdes}/{method}/{stream}"
datasets = ["/".join((load_dir, dset)) for dset in os.listdir(load_dir)]

gi01sumo41 = load_datasets(datasets)
gi01sumo41

gi01sumo41

# #### GI01SUMO-RII11-02-PHSENE042

refdes = "GI01SUMO-RII11-02-PHSENE042"

vocab = OOI.get_vocab(refdes)
vocab

deployments = OOI.get_deployments(refdes)
deployments

datastreams = OOI.get_datastreams(refdes)
datastreams

# Select the relevant method and stream
method = "recovered_inst"
stream = "phsen_abcdef_instrument"

# If the data has been previously downloaded, can open the data from the local directory:

load_dir = f"/media/andrew/Files/Instrument_Data/PHSEN/{refdes}/{method}/{stream}"
datasets = sorted(["/".join((load_dir, dset)) for dset in os.listdir(load_dir)])

gi01sumo42 = load_datasets(datasets, ds=None)
gi01sumo42;

# #### GI03FLMA-RIS01-04-PHSENF000

refdes = "GI03FLMA-RIS01-04-PHSENF000"

vocab = OOI.get_vocab(refdes)
vocab

deployments = OOI.get_deployments(refdes)
deployments

datastreams = OOI.get_datastreams(refdes)
datastreams

# Select the method and stream
method = "recovered_inst"
stream = "phsen_abcdef_instrument"

# If the data has already been downloaded, can read it in from the local directory:

load_dir = f"/media/andrew/Files/Instrument_Data/PHSEN/{refdes}/{method}/{stream}"
datasets = sorted(["/".join((load_dir, dset)) for dset in os.listdir(load_dir)])

gi03flma = load_datasets(datasets, ds=None)
gi03flma;

# #### GI03FLMB-RIS01-04-PHSENF000

refdes = "GI03FLMB-RIS01-04-PHSENF000"

vocab = OOI.get_vocab(refdes)
vocab

deployments = OOI.get_deployments(refdes)
deployments

datastreams = OOI.get_datastreams(refdes)
datastreams

# Select the method and stream
method = "recovered_inst"
stream = "phsen_abcdef_instrument"

# If the data has already been downloaded, can read it in from the local directory:

load_dir = f"/media/andrew/Files/Instrument_Data/PHSEN/{refdes}/{method}/{stream}"
datasets = ["/".join((load_dir, dset)) for dset in os.listdir(load_dir)]

gi03flmb = load_datasets(datasets, ds=None)
gi03flmb;

# ### Reformat the PHSEN dataset
# The PHSEN datasets need to be reprocessed to allow easy filtering and calculations on the instrument's blank measurements.

gi01sumo_rii11_02_phsene041 = xr.open_dataset("../data/GI01SUMO_RII11_02_PHSENE041.nc")
gi01sumo_rii11_02_phsene042 = xr.open_dataset("../data/GI01SUMO_RII11_02_PHSENE042.nc")
gi03flma_ris01_04_phsenf000 = xr.open_dataset("../data/GI03FLMA_RIS01_04_PHSENF000.nc")
gi03flmb_ris01_04_phsenf000 = xr.open_dataset("../data/GI03FLMB_RIS01_04_PHSENF000.nc")

gi01sumo_rii11_02_phsene041


# # Process the Data
# The next step is to process the data following
#
# #### Blanks
# The quality of the blanks for the two wavelengths at 434 and 578 nm directly influences the quality of the pH seawater measurements. Each time stamp of the blank consists of four blank samples. The vendor suggests that the intensity of the blanks should fall between 341 counts and 3891 counts. Consequently, the approach is to average the four blank measurements into a single blank average, run the gross range test for blank counts outside of the suggested range 341 - 3891 counts, and then combine the results of the blanks at 434 and 578 nm

def blanks_mask(ds):
    """Generate a mask based on the blanks values for the PHSEN data."""
    # Average the blanks
    blanks_434 = ds.blank_signal_434.mean(dim="blanks")
    blanks_578 = ds.blank_signal_578.mean(dim="blanks")
    
    # Filter the blanks
    mask434 = (blanks_434 > 341) & (blanks_434 < 3891)
    mask578 = (blanks_578 > 341) & (blanks_578 < 3891)
    
    # Merge the blanks
    blanks_mask = np.stack([mask434, mask578]).T
    blanks_mask = np.all(blanks_mask, axis=1)   
    
    return blanks_mask


gi01sumo_rii11_02_phsene041_blanks = blanks_mask(gi01sumo_rii11_02_phsene041)
gi01sumo_rii11_02_phsene042_blanks = blanks_mask(gi01sumo_rii11_02_phsene042)
gi03flma_ris01_04_phsenf000_blanks = blanks_mask(gi03flma_ris01_04_phsenf000)
gi03flmb_ris01_04_phsenf000_blanks = blanks_mask(gi03flmb_ris01_04_phsenf000)

# #### Gross Range
# The next filter is the gross range test on the pH values. Chris Wingard suggests that the pH values should fall between 7.4 and 8.6 pH units.

# +
gi01sumo_rii11_02_phsene041_gross_range = (gi01sumo_rii11_02_phsene041.seawater_ph > 7.4) & (gi01sumo_rii11_02_phsene041.seawater_ph < 8.6)
gi01sumo_rii11_02_phsene041_gross_range = gi01sumo_rii11_02_phsene041_gross_range.values

gi01sumo_rii11_02_phsene042_gross_range = (gi01sumo_rii11_02_phsene042.seawater_ph > 7.4) & (gi01sumo_rii11_02_phsene042.seawater_ph < 8.6)
gi01sumo_rii11_02_phsene042_gross_range = gi01sumo_rii11_02_phsene042_gross_range.values

gi03flma_ris01_04_phsenf000_gross_range = (gi03flma_ris01_04_phsenf000.seawater_ph > 7.4) & (gi03flma_ris01_04_phsenf000.seawater_ph < 8.6)
gi03flma_ris01_04_phsenf000_gross_range = gi03flma_ris01_04_phsenf000_gross_range.values

gi03flmb_ris01_04_phsenf000_gross_range = (gi03flmb_ris01_04_phsenf000.seawater_ph > 7.4) & (gi03flmb_ris01_04_phsenf000.seawater_ph < 8.6)
gi03flmb_ris01_04_phsenf000_gross_range = gi03flmb_ris01_04_phsenf000_gross_range.values


# -

# #### Noise Filter
# Lastly, I want to filter the pH sensor for when it is excessively noisy. This involves several steps. First, have to separate by deployment to avoid cross-deployment false error. Second, take the first-order difference of the pH values. Then, run the gross range test on the first-order difference with (min, max) values of (-0.04, 0.4) (suggested by Chris Wingard). 

def noise_filter(ds, noise_min, noise_max):
    """Filter for noise on the PHSEN values."""
    # First calculate the first-difference of the pH values
    # which is the "noise" in the measurment
    for depNum in np.unique(ds.deployment.values):
        depNoise = ds.seawater_ph.where(ds.deployment==depNum).diff("time")
        try:
            noise = noise.fillna(value=depNoise)
        except:
            noise = depNoise
    
    # Filter the noise values based on the min/max inputs
    mask = (noise > noise_min) & (noise < noise_max)
    mask = np.insert(mask.values, 0, mask.values[1])
    
    # Return the mask
    return mask


gi01sumo_rii11_02_phsene041_noise = noise_filter(gi01sumo_rii11_02_phsene041, -0.04, 0.04)
gi01sumo_rii11_02_phsene042_noise = noise_filter(gi01sumo_rii11_02_phsene042, -0.04, 0.04)
gi03flma_ris01_04_phsenf000_noise = noise_filter(gi03flma_ris01_04_phsenf000, -0.04, 0.04)
gi03flmb_ris01_04_phsenf000_noise = noise_filter(gi03flmb_ris01_04_phsenf000, -0.04, 0.04)

# #### All Filter
# Merge the different masks together into a single mask to eliminate all the bad or noisy data points.

# +
# GI01SUMO PHSENE041
gi01sumo_rii11_02_phsene041_mask = np.stack([gi01sumo_rii11_02_phsene041_blanks, gi01sumo_rii11_02_phsene041_gross_range, gi01sumo_rii11_02_phsene041_noise]).T
gi01sumo_rii11_02_phsene041_mask = np.all(gi01sumo_rii11_02_phsene041_mask, axis=1)
# Add the mask as a variable to the the DataSet
gi01sumo_rii11_02_phsene041["mask"] = (("time"), gi01sumo_rii11_02_phsene041_mask)
# Drop the bad data from the DataSet
gi01sumo_rii11_02_phsene041 = gi01sumo_rii11_02_phsene041.where(gi01sumo_rii11_02_phsene041.mask, drop=True)

# GI01SUMO PHSENE042
gi01sumo_rii11_02_phsene042_mask = np.stack([gi01sumo_rii11_02_phsene042_blanks, gi01sumo_rii11_02_phsene042_gross_range, gi01sumo_rii11_02_phsene042_noise]).T
gi01sumo_rii11_02_phsene042_mask = np.all(gi01sumo_rii11_02_phsene042_mask, axis=1)
# Add the mask as a variable to the DataSet
gi01sumo_rii11_02_phsene042["mask"] = (("time"), gi01sumo_rii11_02_phsene042_mask)
# Drop the bad data from the DataSet
gi01sumo_rii11_02_phsene042 = gi01sumo_rii11_02_phsene042.where(gi01sumo_rii11_02_phsene042.mask, drop=True)

# GI03FLMA
gi03flma_ris01_04_phsenf000_mask = np.stack([gi03flma_ris01_04_phsenf000_blanks, gi03flma_ris01_04_phsenf000_gross_range, gi03flma_ris01_04_phsenf000_noise]).T
gi03flma_ris01_04_phsenf000_mask = np.all(gi03flma_ris01_04_phsenf000_mask, axis=1)
# Add the mask as a variable to the DataSet
gi03flma_ris01_04_phsenf000["mask"] = (("time"), gi03flma_ris01_04_phsenf000_mask)
# Drop the bad data from the DataSet
gi03flma_ris01_04_phsenf000 = gi03flma_ris01_04_phsenf000.where(gi03flma_ris01_04_phsenf000.mask, drop=True)

# GI03FLMB
gi03flmb_ris01_04_phsenf000_mask = np.stack([gi03flmb_ris01_04_phsenf000_blanks, gi03flmb_ris01_04_phsenf000_gross_range, gi03flmb_ris01_04_phsenf000_noise]).T
gi03flmb_ris01_04_phsenf000_mask = np.all(gi03flmb_ris01_04_phsenf000_mask, axis=1)
# Add the mask as a variable to the DataSet
gi03flmb_ris01_04_phsenf000["mask"] = (("time", gi03flmb_ris01_04_phsenf000_mask))
# Drop the bad data from the DataSet
gi03flmb_ris01_04_phsenf000 = gi03flmb_ris01_04_phsenf000.where(gi03flmb_ris01_04_phsenf000.mask, drop=True)
# -

# ## CTD Data
# Next, identify the associated CTD data with the given PHSEN data. We want the CTDs which are colocated with the PHSEN data. 

# Start with the GI01SUMO CTDs 
OOI.search_datasets(array="GI01SUMO", instrument="CTD", English_names=True)

# Flanking Mooring A: GI03FLMA CTDs 
OOI.search_datasets(array="GI03FLMA", instrument="CTD", English_names=True)

# Flanking Mooring B: GI03FLMB
OOI.search_datasets(array="GI03FLMB", instrument="CTD", English_names=True)

# #### GI01SUMO-RII11-02-CTDMOQ011
# Want the CTD data from the same depth as the PHSEN **GI01SUMO-RII11-02-PHSENE041** located at 20 meters depth.

refdes = "GI01SUMO-RII11-02-CTDMOQ011"

vocab = OOI.get_vocab(refdes)
vocab

deployments = OOI.get_deployments(refdes)
deployments

datastreams = OOI.get_datastreams(refdes)
datastreams

# Select the method and stream
method = "recovered_inst"
stream = "ctdmo_ghqr_instrument_recovered"

# If the data has already been downloaded, can read it in from the local directory:

load_dir = f"/media/andrew/Files/Instrument_Data/CTDMO/{refdes}/{method}/{stream}"
datasets = ["/".join((load_dir, dset)) for dset in os.listdir(load_dir)]

gi01sumo41_ctd = load_datasets(datasets, ds=None)
gi01sumo41_ctd = gi01sumo41_ctd.sortby("time")
gi01sumo41_ctd;

# #### GI01SUMO-RII11-02-CTDMOQ013
# Next, want to get the CTD data associated with the PHSEN **GI01SUMO-RII11-02-PHSENE042** located at 100m depth

refdes = "GI01SUMO-RII11-02-CTDMOQ013"

vocab = OOI.get_vocab(refdes)
vocab

deployments = OOI.get_deployments(refdes)
deployments

datastreams = OOI.get_datastreams(refdes)
datastreams

# Select the method and stream
method = "recovered_inst"
stream = "ctdmo_ghqr_instrument_recovered"

# If the data has already been downloaded, can read it in from the local directory:

load_dir = f"/media/andrew/Files/Instrument_Data/CTDMO/{refdes}/{method}/{stream}"
datasets = ["/".join((load_dir, dset)) for dset in os.listdir(load_dir)]

gi01sumo42_ctd = load_datasets(datasets, ds=None)
gi01sumo42_ctd = gi01sumo42_ctd.sortby("time")
gi01sumo42_ctd;

# #### GI03FLMA-RIM01-02-CTDMOG040

refdes = "GI03FLMA-RIM01-02-CTDMOG040"

vocab = OOI.get_vocab(refdes)
vocab

deployments = OOI.get_deployments(refdes)
deployments

datastreams = OOI.get_datastreams(refdes)
datastreams

method = "recovered_inst"
stream = "ctdmo_ghqr_instrument_recovered"

# If the data has already been downloaded, can read it in from the local directory:

load_dir = f"/media/andrew/Files/Instrument_Data/CTDMO/{refdes}/{method}/{stream}"
datasets = ["/".join((load_dir, dset)) for dset in os.listdir(load_dir)]

gi03flma_ctd = load_datasets(datasets, ds=None)
gi03flma_ctd = gi03flma_ctd.sortby("time")
gi03flma_ctd;

# #### GI03FLMB-RIM01-02-CTDMOG060

refdes = "GI03FLMB-RIM01-02-CTDMOG060"

vocab = OOI.get_vocab(refdes)
vocab

deployments = OOI.get_deployments(refdes)
deployments

datastreams = OOI.get_datastreams(refdes)
datastreams

# Select method and stream
method = "recovered_inst"
stream = "ctdmo_ghqr_instrument_recovered"

# If the data has already been downloaded, can read it in from the local directory:

load_dir = f"/media/andrew/Files/Instrument_Data/CTDMO/{refdes}/{method}/{stream}"
datasets = ["/".join((load_dir, dset)) for dset in os.listdir(load_dir)]

gi03flmb_ctd = load_datasets(datasets, ds=None)
gi03flmb_ctd = gi03flmb_ctd.sortby("time")
gi03flmb_ctd;


# ---
# ## Interpolate CTD data to PHSEN data
# Next, we want to interpolate the CTD temperature, salinity, and pressure to the timestamps of the PHSEN measurements. Then, we add the interpolated temperature, salinity, and pressure to the PHSEN datasets.

def interp_ctd(deployments, ctd_ds, ph_ds, ds=None):
    
    while len(deployments) > 0:

        # Get the deployment number
        depNum, deployments = deployments[0], deployments[1:]

        # Get the relevant ctd and ph data
        ctd = ctd_ds.where(ctd_ds.deployment == depNum, drop=True)
        ph = ph_ds.where(ph_ds.deployment == depNum, drop=True)
        
        # Sort the data by time
        ctd = ctd.sortby("time")
        ph = ph.sortby("time")

        # Interpolate the data
        if len(ph) == 0:
            # This is the case where we can skip
            continue
        elif len(ctd) == 0:
            # This is the case where the ctd failed
            continue
            # nan_array = np.empty(len(ph))
            # nan_array[:] = np.nan
            # for v in ctd.var
        else:
            ph_ctd = ctd.interp_like(ph)
            ph_ctd = ph_ctd.sortby("time")

        # Now merge the datasets
        if ds is None:
            ds = ph_ctd
        else:
            ds = xr.concat([ds, ph_ctd], dim="time")

        ds = interp_ctd(datasets, ctd_ds, ph_ds, ds)
        ds = ds.sortby("time")

    return ds


# #### GI01SUMO-RII11-02-PHSENE041

deployments = sorted(np.unique(gi01sumo41.deployment))
deployments

# Interpolate the ctd data to the pH data:

tsp = interp_ctd(deployments, gi01sumo41_ctd, gi01sumo41, ds=None)
tsp

# Add the temperature/salinity/pressure data to the pH data:

gi01sumo41 = gi01sumo41.sortby("time")

gi01sumo41["temperature"] = ("time", tsp.ctdmo_seawater_temperature)
gi01sumo41["pressure"] = ("time", tsp.ctdmo_seawater_pressure)
gi01sumo41["practical_salinity"] = ("time", tsp.practical_salinity)

# +
fig, ax = plt.subplots(figsize=(12,8))

ax.plot(tsp.time, tsp.ctdmo_seawater_temperature, linestyle="", marker=".", color="tab:blue")
ax.plot(gi01sumo41.time, gi01sumo41.temperature, linestyle="", marker=".", color="tab:red")
# -

# #### GI01SUMO-RII11-02-PHSENE042

deployments = sorted(np.unique(list(gi01sumo42.deployment.values)))
deployments

# Interpolate the ctd data to the pH data:

tsp = interp_ctd(deployments, gi01sumo42_ctd, gi01sumo42, ds=None)
tsp

# Add the temperature/salinity/pressure data to the pH data:

gi01sumo42 = gi01sumo42.sortby("time")

gi01sumo42["temperature"] = ("time", tsp.ctdmo_seawater_temperature)
gi01sumo42["pressure"] = ("time", tsp.ctdmo_seawater_pressure)
gi01sumo42["practical_salinity"] = ("time", tsp.practical_salinity)

# +
fig, ax = plt.subplots(figsize=(12,8))

ax.plot(tsp.time, tsp.ctdmo_seawater_temperature, linestyle="", marker=".", color="tab:blue")
ax.plot(gi01sumo42.time, gi01sumo42.temperature, linestyle="", marker=".", color="tab:red")
ax.grid()
# -

# #### GI03FLMA-RIS01-04-PHSENF000

deployments = sorted(np.unique(gi03flma.deployment))
deployments

# Interpolate the ctd data to the pH data:

tsp = interp_ctd(deployments, gi03flma_ctd, gi03flma, ds=None)
tsp

# Add the temperature/salinity/pressure data to the pH data:

gi03flma = gi03flma.sortby("time")

gi03flma["temperature"] = ("time", tsp.ctdmo_seawater_temperature)
gi03flma["pressure"] = ("time", tsp.ctdmo_seawater_pressure)
gi03flma["practical_salinity"] = ("time", tsp.practical_salinity)

# +
fig, ax = plt.subplots(figsize=(12,8))

ax.plot(tsp.time, tsp.ctdmo_seawater_temperature, linestyle="", marker=".", color="tab:blue")
ax.plot(gi03flma.time, gi03flma.temperature, linestyle="", marker=".", color="tab:red")
ax.grid()
# -

# #### GI03FLMB-RIS01-04-PHSENF000

deployments = sorted(np.unique(gi03flmb.deployment))
deployments

# Interpolate the ctd data to the pH data:

tsp = interp_ctd(deployments, gi03flmb_ctd, gi03flmb, ds=None)
tsp

# Add the temperature/salinity/pressure data to the pH data:

gi03flmb = gi03flmb.sortby("time")

gi03flmb["temperature"] = ("time", tsp.ctdmo_seawater_temperature)
gi03flmb["pressure"] = ("time", tsp.ctdmo_seawater_pressure)
gi03flmb["practical_salinity"] = ("time", tsp.practical_salinity)

# +
fig, ax = plt.subplots(figsize=(12,8))

ax.plot(gi03flmb.time, gi03flmb.temperature, linestyle="", marker=".", color="tab:blue")
ax.plot(tsp.time, tsp.ctdmo_seawater_temperature, linestyle="", marker=".", color="tab:red")
ax.grid()
# -

# ------------------------------
# ## Water Sampling Data
#
# Multiple samples on a Niskin: In this case, I'm going to calculate the mean value from the two samples for the Niskin.

# ### Irminger Bottle Data

irminger_bottle_data = pd.read_csv("../data/Irminger_Sea_Discrete_Sampling_Summary.csv")
irminger_bottle_data

# Replace -9999999 with NaNs
irminger_bottle_data = irminger_bottle_data.replace(to_replace="-9999999", value=np.nan)
irminger_bottle_data = irminger_bottle_data.replace(to_replace=-9999999, value=np.nan)
irminger_bottle_data


# +
# Convert times from strings to 
def convert_times(x):
    if type(x) is str:
        x = x.replace(" ","").replace("Z","")
        x = pd.to_datetime(x)
    else:
        pass
    return x

irminger_bottle_data["Start Time [UTC]"] = irminger_bottle_data["Start Time [UTC]"].apply(lambda x: convert_times(x))
irminger_bottle_data["CTD Bottle Closure Time [UTC]"] = irminger_bottle_data["CTD Bottle Closure Time [UTC]"].apply(lambda x: convert_times(x))
# -

# Get the carbon data
subset=["Discrete Alkalinity [umol/kg]","Discrete DIC [umol/kg]","Discrete pH [Total scale]"]
irminger_carbon_data = irminger_bottle_data.dropna(how="all", subset=subset)
irminger_carbon_data

# ---
# #### Calculate TEOS-10 Properties for Bottle Data 
# The TEOS-10 properties are considered to be derived from thermodynamic principles whereas the previous TEOS-80 was derived from empirical observations. Here, we'll add in the parameters for conservative temperature, absolute salinity, and neutral density.

import gsw

# +
# Calculate some key physical parameters to get density based on TEOS-10
SP = irminger_carbon_data[["CTD Salinity 1 [psu]", "CTD Salinity 2 [psu]"]].mean(axis=1)
T = irminger_carbon_data[["CTD Temperature 1 [deg C]", "CTD Temperature 2 [deg C]"]].mean(axis=1)
P = irminger_carbon_data["CTD Pressure [db]"]
LAT = irminger_carbon_data["CTD Latitude [deg]"]
LON = irminger_carbon_data["CTD Longitude [deg]"]

# Absolute salinity
SA = gsw.conversions.SA_from_SP(SP, P, LON, LAT)
irminger_carbon_data["CTD Absolute Salinity [g/kg]"] = SA

# Conservative temperature
CT = gsw.conversions.CT_from_t(SA, T, P)
irminger_carbon_data["CTD Conservative Temperature"] = CT

# Density
RHO = gsw.density.rho(SA, CT, P)
irminger_carbon_data["CTD Density [kg/m^3]"] = RHO

# Calculate potential density
SIGMA0 = gsw.density.sigma0(SA, CT)
irminger_carbon_data["CTD Sigma [kg/m^3]"] = RHO
# -

# ---
# #### Calculate Carbon System Parameters
# The discrete water samples were tested for Total Alkalinity, Dissolved Inorganic Carbon, and pH [Total Scale]. I calculate the discrete water sample pCO<sub>2</sub> concentrations from the TA and DIC using the ```CO2SYS``` program. 
#
# Use of ```CO2SYS``` requires, at a minimum, the following inputs:
# * ```PAR1```: First measured carbon system measurement
# * ```PAR2```: Second measured carbon system measurement
# * ```PAR1_type```: The type of PAR1 
#         * 1 = Total Alkalinity umol/kg
#         * 2 = DIC umol/kg
#         * 3 = pH Total Scale
#         * 4 = pCO2
#         * 5 = fCO2
# * ```PAR2_type```: The type of PAR2
#
# The following are optional hydrographic inputs:
# * ```salinity```: practical salinity
# * ```temperature```: the temperature at which PAR1 and PAR2 are provided (in C)
# * ```pressure```: the water pressure at which ```PAR1``` and ```PAR2``` are measured
#

# Calculate the Carbon System Parameters
import PyCO2SYS as pyco2

# +
# Get the key parameters
DIC = irminger_carbon_data["Discrete DIC [umol/kg]"]
TA = irminger_carbon_data["Discrete Alkalinity [umol/kg]"]
PH = irminger_carbon_data["Discrete pH [Total scale]"]
SAL = irminger_carbon_data["Discrete Salinity [psu]"]

# Set the input hydrographic parameters at which the DIC/TA/pH lab measurements were performed
TEMP_IN = 25
PRES_IN = 0

# Get the hydrographic parameters at which the samples were taken
TEMP_OUT = irminger_carbon_data[["CTD Temperature 1 [deg C]","CTD Temperature 1 [deg C]"]].mean(axis=1, skipna=True)
PRES_OUT = irminger_carbon_data["CTD Pressure [db]"]

# Nutrient inputs = need to fill NaNs with zeros otherwise will return NaNs
SIO4 = irminger_carbon_data["Discrete Silicate [uM]"].fillna(value=0)
PO4 = irminger_carbon_data["Discrete Phosphate [uM]"].fillna(value=0)
NH4 = irminger_carbon_data["Discrete Ammonium [uM]"].fillna(value=0)
# -

CO2dict = pyco2.CO2SYS_nd(TA, DIC, 1, 2, SAL, total_ammonia=NH4, total_phosphate=PO4, total_silicate=SIO4, 
                          temperature_out=TEMP_OUT, pressure_out=PRES_OUT)
CO2dict.keys()

# ---
# #### Check accuracy of CO2SYS
# In order to demonstrate that the ```CO2SYS``` software package accurately calculates the pCO<sub>2</sub>, we can compare the pH calculated by ```CO2SYS``` against the measured seawater pH. This serves as an independent check and bound on the error introduced by the carbonate system algorithms. 

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
# %matplotlib inline

PHout = CO2dict["pH"]

# Next, check the pH results to see how the calculated pH compares with the measured pH
df = pd.DataFrame(data=[PH.values, PHout], index=["Measured", "CO2sys"]).T.dropna()
df = df[df["CO2sys"] > 7]
df = df.sort_values(by="Measured")
PHmeas = df["Measured"].values.reshape(-1,1)
PHcalc = df["CO2sys"].values.reshape(-1,1)

# +
# Use sklearn linear regression model to determine the accuracy 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Fit a linear regression to the measured vs calculated pH values
regression = LinearRegression()
regression.fit(PHmeas, PHcalc)

# Get the regression values
PHpred = regression.predict(PHmeas)
PHmse = mean_squared_error(PHpred, PHcalc)
PHstd = np.sqrt(PHmse)
# -

PHpred.reshape(-1)

# Look at how closely the pH measurements match eachother
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
ax.scatter(PHmeas, PHcalc)
ax.plot(PHmeas, PHpred, c='r', label='Regression')
#ax.fill_between(PHmeas, PHpred-1.96*PHstd, PHpred+1.96*PHstd, color="tab:red")
ax.plot(PHmeas, PHpred-1.96*PHstd, ':', c='r', label='2 Stds')
ax.plot(PHmeas, PHpred+1.96*PHstd, ':', c='r')
ax.set_xlabel('Discrete pH\n[Total Scale, T=25C]')
ax.set_ylabel('CO2SYS-calculated pH\n[Total Scale, T=25C]')
ax.text(0.8,0.6,f'Std: {PHstd.round(3)}', fontsize='16', transform=ax.transAxes)
ax.legend()
ax.grid()

# **Results:** The CO2sys calculates the pH to within +/- 0.019 std pH units of the measured pH. 
#

# Now add the calculated carbon system parameters to the cruise info
irminger_carbon_data["Calculated Alkalinity [umol/kg]"] = CO2dict["alkalinity"]
irminger_carbon_data["Calculated CO2aq [umol/kg]"] = CO2dict["aqueous_CO2_out"]
irminger_carbon_data["Calculated CO3 [umol/kg]"] = CO2dict["carbonate_out"]
irminger_carbon_data["Calculated DIC [umol/kg]"] = CO2dict["dic"]
irminger_carbon_data["Calculated pCO2 [uatm]"] = CO2dict["pCO2_out"]
# Bottles["Calculated pH"] = CO2dict["pH_out"]

# **Adjusted pH:** Since we now know that CO2SYS can reproduce the pH measurements from the DIC/TA values, we can input the measured pH in order to adjust the pH for pressure and temperature.

CO2dict = pyco2.CO2SYS_nd(DIC, PH, 2, 3, SAL, total_ammonia=NH4, total_phosphate=PO4, total_silicate=SIO4, 
                          temperature_out=TEMP_OUT, pressure_out=PRES_OUT)

irminger_carbon_data["Calculated pH"] = CO2dict["pH_total_out"]

# ---
# ## Data Comparison
# Finally, we can go ahead and begin comparing the Bottle Data with the PHSEN data.

# ### GI03FLMA PHSEN

gi03flma_ris01_04_phsenf000

# Annotations
annotations = OOINet.get_annotations("GI03FLMA-RIS01-04-PHSENF000")
gi03flma_ris01_04_phsenf000 = OOINet.add_annotation_qc_flag(gi03flma_ris01_04_phsenf000, annotations)

# Drop the bad data
gi03flma_ris01_04_phsenf000 = gi03flma_ris01_04_phsenf000.where(gi03flma_ris01_04_phsenf000.rollup_annotations_qc_results != 9, drop=True)


# Identify the associated discrete sample values

def filter_target_asset(x, target):
    if type(x) is str:
        if target.lower() in x.lower():
            return True
        else:
            return False
    else:
        return False


gi03flma_ris01_04_phsenf000.pressure.mean()

irminger_carbon_data["Target Asset"].unique()

mask = irminger_carbon_data["Target Asset"].apply(lambda x: filter_target_asset(x, "flma"))
flma = irminger_carbon_data[mask]
surface = (flma["CTD Pressure [db]"] >= 15) & (flma["CTD Pressure [db]"] <= 45)

# +
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the data with colors based on deployment
discrete = ax.plot(flma[surface]["CTD Bottle Closure Time [UTC]"], 
                   flma[surface]["Calculated pH"],
                   marker="*", linestyle="", c="tab:red", markersize=12, label="Discrete Samples")
scatter = ax.scatter(gi03flma_ris01_04_phsenf000.time, 
                     gi03flma_ris01_04_phsenf000.seawater_ph,
                     c=gi03flma_ris01_04_phsenf000.deployment)
ax.grid()
ax.set_title("-".join((gi03flma_ris01_04_phsenf000.attrs["id"].split("-")[0:4])), fontsize=16)
ax.set_ylabel(gi03flma_ris01_04_phsenf000.seawater_ph.attrs["long_name"], fontsize=16)
#ax.set_ylim((200, 500))

# Generate the figure legend
marker = scatter.legend_elements()[0]
marker.append(discrete[0])
label = ["Deployment " + x for x in scatter.legend_elements()[1]]
label.append("Discrete Water Sample")
legend =(marker, label)
legend1 = ax.legend(*legend, fontsize=16, edgecolor="black", loc='center left', bbox_to_anchor=(1, 0.5))

fig.autofmt_xdate()


# -

# Compare a particular deployment to a given deployment

def plot_timeseries_with_errorbars(ds, param, bottles, bottle_param):
    """Make plotting of time series quick"""
    
    # First, calculate the average and standard deviation of the time series
    ds_dif = ds[param].diff("time")
    ds_avg, ds_std = (ds[param].mean().values, ds[param].std().values)
    
    # Next, find the bottle samples within the relevant time frame, plus/minus three days
    tmin = ds.time.min().values - pd.Timedelta(days=7)
    tmax = ds.time.max().values + pd.Timedelta(days=7)
    bottles = bottles[(bottles["CTD Bottle Closure Time [UTC]"] > tmin) & 
                      (bottles["CTD Bottle Closure Time [UTC]"] < tmax)]
    
    # Plot the figure
    fig, ax = plt.subplots(figsize=(12,8))
    
    scatter = ax.plot(ds.time, ds[param], marker=".", c="tab:blue")
    fill = ax.fill_between(ds.time, ds[param]-ds_std*1.96, ds[param]+ds_std*1.96, alpha=0.5)
    
    discrete = ax.plot(bottles["Start Time [UTC]"], bottles[bottle_param],
                       marker=".", linestyle="", markersize=24, color="tab:red")
    
    ax.grid()
    ax.legend(["Timeseries", "Discrete Samples"], fontsize=12, loc="best", edgecolor="black")
    ax.set_ylabel(ds[param].attrs["long_name"], fontsize=16)
    title = "-".join((ds.id.split("-")[0:4])) + " Deployment " + str(int(np.unique(ds.deployment)[0]))
    ax.set_title(title, fontsize=16)
    ax.set_ylim((ds[param].min().values-3*ds_std, ds[param].max().values+3*ds_std))
    fig.autofmt_xdate()
    
    return fig, ax, ds_avg, ds_std


deployment_data = gi03flma_ris01_04_phsenf000.where(gi03flma_ris01_04_phsenf000.deployment == 4, drop=True)

fig, ax, ds_avg, ds_std = plot_timeseries_with_errorbars(deployment_data, "seawater_ph",
                                                         flma[surface], "Calculated pH")

# ### GI03FLMB PHSEN

gi03flmb_ris01_04_phsenf000

# Annotations
annotations = OOINet.get_annotations("GI03FLMB-RIS01-04-PHSENF000")
gi03flmb_ris01_04_phsenf000 = OOINet.add_annotation_qc_flag(gi03flmb_ris01_04_phsenf000, annotations)

# Drop the bad data
gi03flmb_ris01_04_phsenf000 = gi03flmb_ris01_04_phsenf000.where(gi03flmb_ris01_04_phsenf000.rollup_annotations_qc_results != 9, drop=True)

# Identify the associated discrete sample values

gi03flmb_ris01_04_phsenf000.pressure.mean()

mask = irminger_carbon_data["Target Asset"].apply(lambda x: filter_target_asset(x, "flmb"))
flmb = irminger_carbon_data[mask]
surface = (flmb["CTD Pressure [db]"] >= 15) & (flmb["CTD Pressure [db]"] <= 45)

# +
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the data with colors based on deployment
discrete = ax.plot(flmb[surface]["CTD Bottle Closure Time [UTC]"], 
                   flmb[surface]["Calculated pH"],
                   marker="*", linestyle="", c="tab:red", markersize=12, label="Discrete Samples")
scatter = ax.scatter(gi03flmb_ris01_04_phsenf000.time, 
                     gi03flmb_ris01_04_phsenf000.seawater_ph,
                     c=gi03flmb_ris01_04_phsenf000.deployment)
ax.grid()
ax.set_title("-".join((gi03flmb_ris01_04_phsenf000.attrs["id"].split("-")[0:4])), fontsize=16)
ax.set_ylabel(gi03flmb_ris01_04_phsenf000.seawater_ph.attrs["long_name"], fontsize=16)
#ax.set_ylim((200, 500))

# Generate the figure legend
marker = scatter.legend_elements()[0]
marker.append(discrete[0])
label = ["Deployment " + x for x in scatter.legend_elements()[1]]
label.append("Discrete Water Sample")
legend =(marker, label)
legend1 = ax.legend(*legend, fontsize=16, edgecolor="black", loc='center left', bbox_to_anchor=(1, 0.5))

fig.autofmt_xdate()
# -

deployment_data = gi03flmb_ris01_04_phsenf000.where(gi03flmb_ris01_04_phsenf000.deployment == 2, drop=True)

fig, ax, ds_avg, ds_std = plot_timeseries_with_errorbars(deployment_data, "seawater_ph",
                                                         flmb[surface], "Calculated pH")

# ### GI01SUMO PHSENE041

gi01sumo_rii11_02_phsene041

# Annotations
annotations = OOINet.get_annotations("GI01SUMO-RII11-02-PHSENE041")
gi01sumo_rii11_02_phsene041 = OOINet.add_annotation_qc_flag(gi01sumo_rii11_02_phsene041, annotations)

# Drop the bad data
gi01sumo_rii11_02_phsene041 = gi01sumo_rii11_02_phsene041.where(gi01sumo_rii11_02_phsene041.rollup_annotations_qc_results != 9, drop=True)

# Identify the associated discrete sample values

gi01sumo_rii11_02_phsene041.pressure.mean().values, gi01sumo_rii11_02_phsene041.pressure.std().values

mask = irminger_carbon_data["Target Asset"].apply(lambda x: filter_target_asset(x, "sumo"))
sumo = irminger_carbon_data[mask]
surface = (sumo["CTD Pressure [db]"] >= 15) & (sumo["CTD Pressure [db]"] <= 45)

# +
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the data with colors based on deployment
discrete = ax.plot(sumo[surface]["CTD Bottle Closure Time [UTC]"], 
                   sumo[surface]["Calculated pH"],
                   marker="*", linestyle="", c="tab:red", markersize=12, label="Discrete Samples")
scatter = ax.scatter(gi01sumo_rii11_02_phsene041.time, 
                     gi01sumo_rii11_02_phsene041.seawater_ph,
                     c=gi01sumo_rii11_02_phsene041.deployment)
ax.grid()
ax.set_title("-".join((gi01sumo_rii11_02_phsene041.attrs["id"].split("-")[0:4])), fontsize=16)
ax.set_ylabel(gi01sumo_rii11_02_phsene041.seawater_ph.attrs["long_name"], fontsize=16)
#ax.set_ylim((200, 500))

# Generate the figure legend
marker = scatter.legend_elements()[0]
marker.append(discrete[0])
label = ["Deployment " + x for x in scatter.legend_elements()[1]]
label.append("Discrete Water Sample")
legend =(marker, label)
legend1 = ax.legend(*legend, fontsize=16, edgecolor="black", loc='center left', bbox_to_anchor=(1, 0.5))

fig.autofmt_xdate()
# -

deployment_data = gi01sumo_rii11_02_phsene041.where(gi01sumo_rii11_02_phsene041.deployment == 2, drop=True)

fig, ax, ds_avg, ds_std = plot_timeseries_with_errorbars(deployment_data, "seawater_ph",
                                                         sumo[surface], "Calculated pH")

# ### GI01SUMO PHSENE042

gi01sumo_rii11_02_phsene042

# Annotations
annotations = OOINet.get_annotations("GI01SUMO-RII11-02-PHSENE042")
gi01sumo_rii11_02_phsene042 = OOINet.add_annotation_qc_flag(gi01sumo_rii11_02_phsene042, annotations)

# Drop the bad values
gi01sumo_rii11_02_phsene042 = gi01sumo_rii11_02_phsene042.where(gi01sumo_rii11_02_phsene042.rollup_annotations_qc_results != 9, drop=True)

# Identify the associated discrete sample values

gi01sumo_rii11_02_phsene042.pressure.mean().values, gi01sumo_rii11_02_phsene042.pressure.std().values

mask = irminger_carbon_data["Target Asset"].apply(lambda x: filter_target_asset(x, "sumo"))
sumo = irminger_carbon_data[mask]
surface = (sumo["CTD Pressure [db]"] >= 80) & (sumo["CTD Pressure [db]"] <= 120)

# +
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the data with colors based on deployment
discrete = ax.plot(sumo[surface]["CTD Bottle Closure Time [UTC]"], 
                   sumo[surface]["Calculated pH"],
                   marker="*", linestyle="", c="tab:red", markersize=12, label="Discrete Samples")
scatter = ax.scatter(gi01sumo_rii11_02_phsene042.time, 
                     gi01sumo_rii11_02_phsene042.seawater_ph,
                     c=gi01sumo_rii11_02_phsene042.deployment)
ax.grid()
ax.set_title("-".join((gi01sumo_rii11_02_phsene042.attrs["id"].split("-")[0:4])), fontsize=16)
ax.set_ylabel(gi01sumo_rii11_02_phsene042.seawater_ph.attrs["long_name"], fontsize=16)
#ax.set_ylim((200, 500))

# Generate the figure legend
marker = scatter.legend_elements()[0]
marker.append(discrete[0])
label = ["Deployment " + x for x in scatter.legend_elements()[1]]
label.append("Discrete Water Sample")
legend =(marker, label)
legend1 = ax.legend(*legend, fontsize=16, edgecolor="black", loc='center left', bbox_to_anchor=(1, 0.5))

fig.autofmt_xdate()
# -

deployment_data = gi01sumo_rii11_02_phsene042.where(gi01sumo_rii11_02_phsene042.deployment == 3, drop=True)

fig, ax, ds_avg, ds_std = plot_timeseries_with_errorbars(deployment_data, "seawater_ph",
                                                         sumo[surface], "Calculated pH")

# ---
# ## Pioneer Array

# ### Identify PHSEN instruments
# Next, we want to identify the deployed PHSEN instruments. Based on the results of the PHSEN tech refresh analysis, we have identified the following sensors as having relatively robust datasets: **CP01CNSM** and **CP03ISSM**. Additionally, we include the PHSEN on the surface mooring **CP04OSSM**.

OOINet.search_datasets(array="CP01CNSM", instrument="PHSEN")

OOINet.search_datasets(array="CP03ISSM", instrument="PHSEN")

OOINet.search_datasets(array="CP04OSSM", instrument="PHSEN")

# ### Identify associated CTDs
# Now, we want to identify the associated CTD datasets to the PHSEN datasets

OOINet.search_datasets(array="CP01CNSM", instrument="CTDBP")

OOINet.search_datasets(array="CP03ISSM", instrument="CTDBP")

OOINet.search_datasets(array="CP04OSSM", instrument="CTDBP")

# ### Download Datasets

refdes = "CP04OSSM-MFD35-06-PHSEND000"

metadata = OOINet.get_metadata(refdes)
metadata = metadata.groupby(by=["refdes","method","stream"]).agg(lambda x: pd.unique(x.values.ravel()).tolist())
metadata = metadata.reset_index()
metadata = metadata.applymap(lambda x: x[0] if len(x) == 1 else x)
metadata

method = "recovered_inst"
stream = "phsen_abcdef_instrument"
parameters = "All"

# +
# Load the thredds table
thredds_table = pd.read_csv("../data/thredds_table.csv")

# Check the thredds_table for if the data has already been loaded
thredds_table, thredds_url = check_thredds_table(thredds_table, refdes, method, stream, parameters)
print(thredds_url)

# Save the thredds_table
thredds_table.to_csv("../data/thredds_table.csv", index=False)
# -

# Get the THREDDS catalog for the reference designator
thredds_catalog = OOINet.get_thredds_catalog(thredds_url)

# Parse the datasets
netCDF_files = sorted(OOINet.parse_catalog(thredds_catalog, exclude=["gps","blank","ENG","CTDBP"]))
netCDF_files

# Load the netCDF datasets
phsen = OOINet.load_netCDF_datasets(netCDF_files)
phsen = phsen.sortby("time")
phsen

# Reformat the PHSEN datasets
# PHSEN = PHSEN()
phsen = PHSEN.phsen_instrument(phsen)
phsen = phsen.sortby("time")
phsen

# #### Download Associated Dataset

refdes = "CP04OSSM-MFD37-03-CTDBPE000"

metadata = OOINet.get_metadata(refdes)
metadata = metadata.groupby(by=["refdes","method","stream"]).agg(lambda x: pd.unique(x.values.ravel()).tolist())
metadata = metadata.reset_index()
metadata = metadata.applymap(lambda x: x[0] if len(x) == 1 else x)
metadata

method = "recovered_inst"
stream = "ctdbp_cdef_instrument_recovered"
parameters = "All"

# +
# Load the thredds table
thredds_table = pd.read_csv("../data/thredds_table.csv")

# Check the thredds_table for if the data has already been loaded
thredds_table, thredds_url = check_thredds_table(thredds_table, refdes, method, stream, parameters)
print(thredds_url)

# Save the thredds_table
thredds_table.to_csv("../data/thredds_table.csv")

# +
# Get the catalog
thredds_catalog = OOINet.get_thredds_catalog(thredds_url)

# Parse the datasets
netCDF_files = sorted(OOINet.parse_catalog(thredds_catalog, exclude=["gps","blank","ENG"]))
netCDF_files
# -

# Load the netCDF datasets
ctdbp = OOINet.load_netCDF_datasets(netCDF_files)
ctdbp = ctdbp.sortby("time")
ctdbp


# #### Interpolate the CTD parameters to the PHSEN dataset

def interpolate_ctd(phsen, ctd):
    
    deployments = np.unique(phsen.deployment.values)
    
    ds = None
    
    for depNum in deployments:
        
        # Get the relevant ctd and pco2 data
        ctd_dep = ctd.where(ctd.deployment == depNum, drop=True)
        phsen_dep = phsen.where(phsen.deployment == depNum, drop=True)  

        # Sort by time
        ctd_dep = ctd_dep.sortby("time")
        phsen_dep = phsen_dep.sortby("time")

        # Drop duplicated indexes
        ctd_dep = ctd_dep.sel(time=~ctd_dep.get_index("time").duplicated())

        # Interpolate the data
        if len(ctd_dep.time) <= 1:
            # Need to create a dummy dataset of all NaNs to append
            data_vars = {}
            for var in ctd_dep.variables:
                if "time" in var:
                    continue
                array = np.empty(phsen.time.size)
                array[:] = np.nan
                data_vars.update({var: (["time"], array)})
            coords = phsen_dep.coords
            attrs = phsen_dep.attrs
            phsen_ctd = xr.Dataset(
                data_vars = data_vars,
                coords = coords,
                attrs = attrs)
        elif len(phsen_dep.time) <= 1:
            continue
        else:
            phsen_ctd = ctd_dep.interp_like(phsen)
            phsen_ctd = phsen_ctd.sortby("time")

        if ds is None:
            ds = phsen_ctd
        else:
            ds = xr.concat([ds, phsen_ctd], dim="time")


# +
ds = None

deployments = np.unique(phsen.deployment.values)

for depNum in deployments:
    
    # Get the relevant ctd and pco2 data
    ctd = ctdbp.where(ctdbp.deployment == depNum, drop=True)
    co2 = phsen.where(phsen.deployment == depNum, drop=True)  
    
    # Sort by time
    ctd = ctd.sortby("time")
    co2 = co2.sortby("time")
    
    # Drop duplicated indexes
    ctd = ctd.sel(time=~ctd.get_index("time").duplicated())
    
    # Interpolate the data
    if len(ctd.time) <= 1:
        # Need to create a dummy dataset of all NaNs to append
        data_vars = {}
        for var in ctd.variables:
            if "time" in var:
                continue
            array = np.empty(co2.time.size)
            array[:] = np.nan
            data_vars.update({var: (["time"], array)})
        coords = co2.coords
        attrs = co2.attrs
        co2_ctd = xr.Dataset(
            data_vars = data_vars,
            coords = coords,
            attrs = attrs)
    elif len(co2.time) <= 1:
        continue
    else:
        co2_ctd = ctd.interp_like(co2)
        co2_ctd = co2_ctd.sortby("time")
        
    if ds is None:
        ds = co2_ctd
    else:
        ds = xr.concat([ds, co2_ctd], dim="time")
# -

ds

phsen["temperature"] = ("time", ds.ctdbp_seawater_temperature)
phsen["practical_salinity"] = ("time", ds.practical_salinity)
phsen["pressure"] = ("time", ds.ctdbp_seawater_pressure)
phsen["density"] = ("time", ds.density)

# #### Save the Data

filename = "_".join((phsen.attrs["id"].split("-")[0:4]))
filename

phsen.to_netcdf(f"../data/{filename}.nc")

# ---
# ## Load the Pioneer PHSEN Datasets

os.listdir("../data/")

# +
# Central Surface Mooring
cp01cnsm_rid26_06_phsend000 = xr.open_dataset("../data/CP01CNSM_RID26_06_PHSEND000.nc")
cp01cnsm_mfd35_06_phsend000 = xr.open_dataset("../data/CP01CNSM_MFD35_06_PHSEND000.nc")

# Inshore Surface Mooring
cp03issm_rid26_06_phsend000 = xr.open_dataset("../data/CP03ISSM_RID26_06_PHSEND000.nc")
cp03issm_mfd35_06_phsend000 = xr.open_dataset("../data/CP03ISSM_MFD35_06_PHSEND000.nc")

# Offshore Surface Mooring
cp04ossm_rid26_06_phsend000 = xr.open_dataset("../data/CP04OSSM_RID26_06_PHSEND000.nc")
cp04ossm_mfd35_06_phsend000 = xr.open_dataset("../data/CP04OSSM_MFD35_06_PHSEND000.nc")
# -

# ---
# ## Process the Data
# The next step is to process the data following
#
# #### Blanks
# The quality of the blanks for the two wavelengths at 434 and 578 nm directly influences the quality of the pH seawater measurements. Each time stamp of the blank consists of four blank samples. The vendor suggests that the intensity of the blanks should fall between 341 counts and 3891 counts. Consequently, the approach is to average the four blank measurements into a single blank average, run the gross range test for blank counts outside of the suggested range 341 - 3891 counts, and then combine the results of the blanks at 434 and 578 nm

# +
cp01cnsm_rid26_06_phsend000_blanks = blanks_mask(cp01cnsm_rid26_06_phsend000)
cp01cnsm_mfd35_06_phsend000_blanks = blanks_mask(cp01cnsm_mfd35_06_phsend000)

cp03issm_rid26_06_phsend000_blanks = blanks_mask(cp03issm_rid26_06_phsend000)
cp03issm_mfd35_06_phsend000_blanks = blanks_mask(cp03issm_mfd35_06_phsend000)

cp04ossm_rid26_06_phsend000_blanks = blanks_mask(cp04ossm_rid26_06_phsend000)
cp04ossm_mfd35_06_phsend000_blanks = blanks_mask(cp04ossm_mfd35_06_phsend000)
# -

# #### Gross Range
# The next filter is the gross range test on the pH values. Chris Wingard suggests that the pH values should fall between 7.4 and 8.6 pH units.

# +
# Central Surface Mooring
cp01cnsm_rid26_06_phsend000_gross_range = (cp01cnsm_rid26_06_phsend000.seawater_ph > 7.4) & (cp01cnsm_rid26_06_phsend000.seawater_ph < 8.6)
cp01cnsm_rid26_06_phsend000_gross_range = cp01cnsm_rid26_06_phsend000_gross_range.values

cp01cnsm_mfd35_06_phsend000_gross_range = (cp01cnsm_mfd35_06_phsend000.seawater_ph > 7.4) & (cp01cnsm_mfd35_06_phsend000.seawater_ph < 8.6)
cp01cnsm_mfd35_06_phsend000_gross_range = cp01cnsm_mfd35_06_phsend000_gross_range.values

# Inshore Surface Mooring
cp03issm_rid26_06_phsend000_gross_range = (cp03issm_rid26_06_phsend000.seawater_ph > 7.4) & (cp03issm_rid26_06_phsend000.seawater_ph < 8.6)
cp03issm_rid26_06_phsend000_gross_range = cp03issm_rid26_06_phsend000_gross_range.values

cp03issm_mfd35_06_phsend000_gross_range = (cp03issm_mfd35_06_phsend000.seawater_ph > 7.4) & (cp03issm_mfd35_06_phsend000.seawater_ph < 8.6)
cp03issm_mfd35_06_phsend000_gross_range = cp03issm_mfd35_06_phsend000_gross_range.values

# Offshore Surface Mooring
cp04ossm_rid26_06_phsend000_gross_range = (cp04ossm_rid26_06_phsend000.seawater_ph > 7.4) & (cp04ossm_rid26_06_phsend000.seawater_ph < 8.6)
cp04ossm_rid26_06_phsend000_gross_range = cp04ossm_rid26_06_phsend000_gross_range.values

cp04ossm_mfd35_06_phsend000_gross_range = (cp04ossm_mfd35_06_phsend000.seawater_ph > 7.4) & (cp04ossm_mfd35_06_phsend000.seawater_ph < 8.6)
cp04ossm_mfd35_06_phsend000_gross_range = cp04ossm_mfd35_06_phsend000_gross_range.values
# -

# #### Noise Filter
# Lastly, I want to filter the pH sensor for when it is excessively noisy. This involves several steps. First, have to separate by deployment to avoid cross-deployment false error. Second, take the first-order difference of the pH values. Then, run the gross range test on the first-order difference with (min, max) values of (-0.04, 0.4) (suggested by Chris Wingard). 

# +
# Central Surface Mooring
cp01cnsm_rid26_06_phsend000_noise = noise_filter(cp01cnsm_rid26_06_phsend000, -0.04, 0.04)
cp01cnsm_mfd35_06_phsend000_noise = noise_filter(cp01cnsm_mfd35_06_phsend000, -0.04, 0.04)

# Inshore Surface Mooring
cp03issm_rid26_06_phsend000_noise = noise_filter(cp03issm_rid26_06_phsend000, -0.04, 0.04)
cp03issm_mfd35_06_phsend000_noise = noise_filter(cp03issm_mfd35_06_phsend000, -0.04, 0.04)

# Offshore Surface Mooring
cp04ossm_rid26_06_phsend000_noise = noise_filter(cp04ossm_rid26_06_phsend000, -0.04, 0.04)
cp04ossm_mfd35_06_phsend000_noise = noise_filter(cp04ossm_mfd35_06_phsend000, -0.04, 0.04)
# -

# #### All Filter
# Merge the different masks together into a single mask to eliminate all the bad or noisy data points.

# +
# Central Surface Mooring
# NSIF
cp01cnsm_rid26_06_phsend000_mask = np.stack([cp01cnsm_rid26_06_phsend000_blanks, cp01cnsm_rid26_06_phsend000_gross_range, cp01cnsm_rid26_06_phsend000_noise]).T
cp01cnsm_rid26_06_phsend000_mask = np.all(cp01cnsm_rid26_06_phsend000_mask, axis=1)
# Add the mask as a variable to the the DataSet
cp01cnsm_rid26_06_phsend000["mask"] = (("time"), cp01cnsm_rid26_06_phsend000_mask)
# Drop the bad data from the DataSet
cp01cnsm_rid26_06_phsend000 = cp01cnsm_rid26_06_phsend000.where(cp01cnsm_rid26_06_phsend000.mask, drop=True)

# MFN
cp01cnsm_mfd35_06_phsend000_mask = np.stack([cp01cnsm_mfd35_06_phsend000_blanks, cp01cnsm_mfd35_06_phsend000_gross_range, cp01cnsm_mfd35_06_phsend000_noise]).T
cp01cnsm_mfd35_06_phsend000_mask = np.all(cp01cnsm_mfd35_06_phsend000_mask, axis=1)
# Add the mask as a variable to the the DataSet
cp01cnsm_mfd35_06_phsend000["mask"] = (("time"), cp01cnsm_mfd35_06_phsend000_mask)
# Drop the bad data from the DataSet
cp01cnsm_mfd35_06_phsend000 = cp01cnsm_mfd35_06_phsend000.where(cp01cnsm_mfd35_06_phsend000.mask, drop=True)


# Inshore Surface Mooring
# NSIF
cp03issm_rid26_06_phsend000_mask = np.stack([cp03issm_rid26_06_phsend000_blanks, cp03issm_rid26_06_phsend000_gross_range, cp03issm_rid26_06_phsend000_noise]).T
cp03issm_rid26_06_phsend000_mask = np.all(cp03issm_rid26_06_phsend000_mask, axis=1)
# Add the mask as a variable to the the DataSet
cp03issm_rid26_06_phsend000["mask"] = (("time"), cp03issm_rid26_06_phsend000_mask)
# Drop the bad data from the DataSet
cp03issm_rid26_06_phsend000 = cp03issm_rid26_06_phsend000.where(cp03issm_rid26_06_phsend000.mask, drop=True)

# MFN
cp03issm_mfd35_06_phsend000_mask = np.stack([cp03issm_mfd35_06_phsend000_blanks, cp03issm_mfd35_06_phsend000_gross_range, cp03issm_mfd35_06_phsend000_noise]).T
cp03issm_mfd35_06_phsend000_mask = np.all(cp03issm_mfd35_06_phsend000_mask, axis=1)
# Add the mask as a variable to the the DataSet
cp03issm_mfd35_06_phsend000["mask"] = (("time"), cp03issm_mfd35_06_phsend000_mask)
# Drop the bad data from the DataSet
cp03issm_mfd35_06_phsend000 = cp03issm_mfd35_06_phsend000.where(cp03issm_mfd35_06_phsend000.mask, drop=True)

# Offshore Surface Mooring
# NSIF
cp04ossm_rid26_06_phsend000_mask = np.stack([cp04ossm_rid26_06_phsend000_blanks, cp04ossm_rid26_06_phsend000_gross_range, cp04ossm_rid26_06_phsend000_noise]).T
cp04ossm_rid26_06_phsend000_mask = np.all(cp04ossm_rid26_06_phsend000_mask, axis=1)
# Add the mask as a variable to the the DataSet
cp04ossm_rid26_06_phsend000["mask"] = (("time"), cp04ossm_rid26_06_phsend000_mask)
# Drop the bad data from the DataSet
cp04ossm_rid26_06_phsend000 = cp04ossm_rid26_06_phsend000.where(cp04ossm_rid26_06_phsend000.mask, drop=True)

# MFN
cp04ossm_mfd35_06_phsend000_mask = np.stack([cp04ossm_mfd35_06_phsend000_blanks, cp04ossm_mfd35_06_phsend000_gross_range, cp04ossm_mfd35_06_phsend000_noise]).T
cp04ossm_mfd35_06_phsend000_mask = np.all(cp04ossm_mfd35_06_phsend000_mask, axis=1)
# Add the mask as a variable to the the DataSet
cp04ossm_mfd35_06_phsend000["mask"] = (("time"), cp04ossm_mfd35_06_phsend000_mask)
# Drop the bad data from the DataSet
cp04ossm_mfd35_06_phsend000 = cp04ossm_mfd35_06_phsend000.where(cp04ossm_mfd35_06_phsend000.mask, drop=True)
# -

# ---
# ### Pioneer Discrete Water

pioneer_bottle_data = pd.read_csv("../data/Pioneer_Discrete_Summary_Data2.csv")
pioneer_bottle_data

# Replace -9999999 with NaNs
pioneer_bottle_data = pioneer_bottle_data.replace(to_replace="-9999999", value=np.nan)
pioneer_bottle_data = pioneer_bottle_data.replace(to_replace=-9999999, value=np.nan)
pioneer_bottle_data

# Convert the dates and times
pioneer_bottle_data["Start Time [UTC]"] = pioneer_bottle_data["Start Time [UTC]"].apply(lambda x: convert_times(x))
pioneer_bottle_data["CTD Bottle Closure Time [UTC]"] = pioneer_bottle_data["CTD Bottle Closure Time [UTC]"].apply(lambda x: convert_times(x))





# ---
# ## Data Analysis


