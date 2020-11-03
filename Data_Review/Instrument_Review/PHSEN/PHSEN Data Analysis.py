# -*- coding: utf-8 -*-
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

# Import user info for connecting to OOINet via M2M
userinfo = yaml.load(open("../../../user_info.yaml"))
username = userinfo["apiname"]
token = userinfo["apikey"]

# Initialize the M2M tool
OOI = M2M(username, token)


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

# If the data has not been previously downloaded, can either download or load directly from the THREDDS server:

catalog = OOI.get_thredds_catalog(thredds_url)

datasets = OOI.parse_catalog(catalog, exclude=["gps", "CTD"])

save_dir = f"/media/andrew/Files/Instrument_Data/PHSEN/{refdes}/{method}/{stream}/"
OOI.download_netCDF_files(datasets, save_dir)

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
gi01sumo42

# If the data has not been previously downloaded, can either download or load directly from the THREDDS server:

thredds_url = OOI.get_thredds_url(refdes, method, stream)
thredds_url

# Get the catalog
catalog = OOI.get_thredds_catalog(thredds_url)

datasets = OOI.parse_catalog(catalog, exclude=["gps", "CTD"])

save_dir = f"/media/andrew/Files/Instrument_Data/PHSEN/{refdes}/{method}/{stream}/"

OOI.download_netCDF_files(datasets, save_dir)

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
gi03flma

# If the data has not been previously downloaded, can either download or load directly from the THREDDS server:

thredds_url = OOI.get_thredds_url(refdes, method, stream)
thredds_url

# Get the catalog
catalog = OOI.get_thredds_catalog(thredds_url)
datasets = OOI.parse_catalog(catalog, exclude=["gps", "CTD"])

# Download the data
save_dir = f"/media/andrew/Files/Instrument_Data/PHSEN/{refdes}/{method}/{stream}/"
OOI.download_netCDF_files(datasets, save_dir)

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

# If the data has not been previously downloaded, can either download or load directly from the THREDDS server:

thredds_url = OOI.get_thredds_url(refdes, method, stream)
thredds_url

# Get the catalog
catalog = OOI.get_thredds_catalog(thredds_url)
datasets = OOI.parse_catalog(catalog, exclude=["gps", "CTD"])

# Can download the data
save_dir = f"/media/andrew/Files/Instrument_Data/PHSEN/{refdes}/{method}/{stream}/"
OOI.download_netCDF_files(datasets, save_dir)

# ### Reformat the PHSEN dataset
# The PHSEN datasets need to be reprocessed to allow easy filtering and calculations on the instrument's blank measurements.

PHSEN = PHSEN()

gi01sumo41 = PHSEN.phsen_instrument(gi01sumo41)
gi01sumo41

gi01sumo42 = PHSEN.phsen_instrument(gi01sumo42)
gi01sumo42

gi03flma = PHSEN.phsen_instrument(gi03flma)
gi03flma

gi03flmb = PHSEN.phsen_instrument(gi03flmb)
gi03flmb


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


gi01sumo41_blanks = blanks_mask(gi01sumo41)
gi01sumo42_blanks = blanks_mask(gi01sumo42)
gi03flma_blanks = blanks_mask(gi03flma)
gi03flmb_blanks = blanks_mask(gi03flmb)

# #### Gross Range
# The next filter is the gross range test on the pH values. Chris Wingard suggests that the pH values should fall between 7.4 and 8.6 pH units.

# +
gi01sumo41_gross_range = (gi01sumo41.seawater_ph > 7.4) & (gi01sumo41.seawater_ph < 8.6)
gi01sumo41_gross_range = gi01sumo41_gross_range.values

gi01sumo42_gross_range = (gi01sumo42.seawater_ph > 7.4) & (gi01sumo42.seawater_ph < 8.6)
gi01sumo42_gross_range = gi01sumo42_gross_range.values

gi03flma_gross_range = (gi03flma.seawater_ph > 7.4) & (gi03flma.seawater_ph < 8.6)
gi03flma_gross_range = gi03flma_gross_range.values

gi03flmb_gross_range = (gi03flmb.seawater_ph > 7.4) & (gi03flmb.seawater_ph < 8.6)
gi03flmb_gross_range = gi03flmb_gross_range.values


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


gi01sumo41_noise = noise_filter(gi01sumo41, -0.04, 0.04)
gi01sumo42_noise = noise_filter(gi01sumo42, -0.04, 0.04)
gi03flma_noise = noise_filter(gi03flma, -0.04, 0.04)
gi03flmb_noise = noise_filter(gi03flmb, -0.04, 0.04)

# #### All Filter
# Merge the different masks together into a single mask to eliminate all the bad or noisy data points.

# +
gi01sumo41_mask = np.stack([gi01sumo41_blanks, gi01sumo41_gross_range, gi01sumo41_noise]).T
gi01sumo41_mask = np.all(gi01sumo41_mask, axis=1)

gi01sumo42_mask = np.stack([gi01sumo42_blanks, gi01sumo42_gross_range, gi01sumo42_noise]).T
gi01sumo42_mask = np.all(gi01sumo42_mask, axis=1)

gi03flma_mask = np.stack([gi03flma_blanks, gi03flma_gross_range, gi03flma_noise]).T
gi03flma_mask = np.all(gi03flma_mask, axis=1)

gi03flmb_mask = np.stack([gi03flmb_blanks, gi03flmb_gross_range, gi03flmb_noise]).T
gi03flmb_mask = np.all(gi03flmb_mask, axis=1)

# +
fig, ax = plt.subplots(figsize=(12,8))

# Plot the pH data, and color the 
ax.scatter(gi01sumo41.time.values[gi01sumo41_mask], gi01sumo41.seawater_ph.values[gi01sumo41_mask], c=gi01sumo41.deployment.values[gi01sumo41_mask])
ax.set_ylabel(gi01sumo41.seawater_ph.attrs["long_name"], fontsize=12)
ax.set_xlabel(gi01sumo41.time.attrs["long_name"], fontsize=12)
ax.set_title(gi01sumo41.attrs["id"])
ax.grid()

fig.autofmt_xdate()

# +
fig, ax = plt.subplots(figsize=(12,8))

# Plot the pH data, and color the 
ax.scatter(gi01sumo42.time.values[gi01sumo42_mask], gi01sumo42.seawater_ph.values[gi01sumo42_mask], c=gi01sumo42.deployment.values[gi01sumo42_mask])
ax.set_ylabel(gi01sumo42.seawater_ph.attrs["long_name"], fontsize=12)
ax.set_xlabel(gi01sumo42.time.attrs["long_name"], fontsize=12)
ax.set_title(gi01sumo42.attrs["id"])
ax.grid()

fig.autofmt_xdate()

# +
fig, ax = plt.subplots(figsize=(12,8))

# Plot the pH data, and color the 
ax.scatter(gi03flma.time.values[gi03flma_mask], gi03flma.seawater_ph.values[gi03flma_mask], c=gi03flma.deployment.values[gi03flma_mask])
ax.set_ylabel(gi03flma.seawater_ph.attrs["long_name"], fontsize=12)
ax.set_xlabel(gi03flma.time.attrs["long_name"], fontsize=12)
ax.set_title(gi03flma.attrs["id"])
ax.grid()

fig.autofmt_xdate()

# +
fig, ax = plt.subplots(figsize=(12,8))

# Plot the pH data, and color the 
ax.scatter(gi03flmb.time.values[gi03flmb_mask], gi03flmb.seawater_ph.values[gi03flmb_mask], c=gi03flmb.deployment.values[gi03flmb_mask])
ax.set_ylabel(gi03flmb.seawater_ph.attrs["long_name"], fontsize=12)
ax.set_xlabel(gi03flmb.time.attrs["long_name"], fontsize=12)
ax.set_title(gi03flmb.attrs["id"])
ax.grid()

fig.autofmt_xdate()
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
gi01sumo41_ctd

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
gi01sumo42_ctd

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
gi03flma_ctd

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
gi03flmb_ctd


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
# -

# ------------------------------
# ## Water Sampling Data

del Bottles

# Import the discrete sample data for Pioneer cruises
basepath = '/home/andrew/Documents/OOI-CGSN/ooicgsn-water-sampling/Irminger/'
for file in sorted(os.listdir('/home/andrew/Documents/OOI-CGSN/ooicgsn-water-sampling/Irminger/')):
    if file.endswith('.txt'):
        pass
    else:
        try:
            Bottles = Bottles.append(pd.read_csv(basepath + '/' + file))
        except:
            Bottles = pd.read_csv(basepath + '/' + file)

Bottles

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

# ## Data Comparison

# ### GI03FLMA PHSEN vs Bottle Data

deployments = sorted(np.unique(gi03flma.deployment))
deployments

# Identify the bottle associated with GI03FLMA
Bottles["Target Asset"].unique()

gi03flma_bot = Bottles[Bottles["Target Asset"] == "GI03FLMA"]

gi03flma_bot["Discrete pH [Total scale]"]


