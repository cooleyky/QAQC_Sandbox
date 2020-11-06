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

from m2m import M2M

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
gi01sumo41;

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

PHSEN = PHSEN()

gi01sumo41 = PHSEN.phsen_instrument(gi01sumo41)
gi01sumo41 = gi01sumo41.sortby("time")
gi01sumo41;

gi01sumo42 = PHSEN.phsen_instrument(gi01sumo42)
gi01sumo42 = gi01sumo42.sortby("time")
gi01sumo42;

gi03flma = PHSEN.phsen_instrument(gi03flma)
gi03flma = gi03flma.sortby("time")
gi03flma;

gi03flmb = PHSEN.phsen_instrument(gi03flmb)
gi03flmb = gi03flmb.sortby("time")
gi03flmb;


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
# GI01SUMO PHSENE041
gi01sumo41_mask = np.stack([gi01sumo41_blanks, gi01sumo41_gross_range, gi01sumo41_noise]).T
gi01sumo41_mask = np.all(gi01sumo41_mask, axis=1)
# Add the mask as a variable to the the DataSet
gi01sumo41["mask"] = (("time"), gi01sumo41_mask)
# Drop the bad data from the DataSet
gi01sumo41 = gi01sumo41.where(gi01sumo41.mask, drop=True)

# GI01SUMO PHSENE042
gi01sumo42_mask = np.stack([gi01sumo42_blanks, gi01sumo42_gross_range, gi01sumo42_noise]).T
gi01sumo42_mask = np.all(gi01sumo42_mask, axis=1)
# Add the mask as a variable to the DataSet
gi01sumo42["mask"] = (("time"), gi01sumo42_mask)
# Drop the bad data from the DataSet
gi01sumo42 = gi01sumo42.where(gi01sumo42.mask, drop=True)

# GI03FLMA
gi03flma_mask = np.stack([gi03flma_blanks, gi03flma_gross_range, gi03flma_noise]).T
gi03flma_mask = np.all(gi03flma_mask, axis=1)
# Add the mask as a variable to the DataSet
gi03flma["mask"] = (("time"), gi03flma_mask)
# Drop the bad data from the DataSet
gi03flma = gi03flma.where(gi03flma.mask, drop=True)

# GI03FLMB
gi03flmb_mask = np.stack([gi03flmb_blanks, gi03flmb_gross_range, gi03flmb_noise]).T
gi03flmb_mask = np.all(gi03flmb_mask, axis=1)
# Add the mask as a variable to the DataSet
gi03flmb["mask"] = (("time", gi03flmb_mask))
# Drop the bad data from the DataSet
gi03flmb = gi03flmb.where(gi03flmb.mask, drop=True)

# +
fig, ax = plt.subplots(figsize=(12,8))

# Plot the pH data, and color the 
scatter = ax.scatter(gi01sumo41.time, gi01sumo41.seawater_ph, c=gi01sumo41.deployment)
legend1 = ax.legend(*scatter.legend_elements(), title="Deployment #", fontsize=16)
ax.set_ylabel(gi01sumo41.seawater_ph.attrs["long_name"], fontsize=16)
ax.set_title(gi01sumo41.attrs["id"], fontsize=16)
ax.grid()

fig.autofmt_xdate()
# -

gi01sumo41.attrs["id"]

# +
fig, ax = plt.subplots(figsize=(12,8))

# Plot the pH data, and color the 
scatter = ax.scatter(gi01sumo42.time, gi01sumo42.seawater_ph, c=gi01sumo42.deployment)
legend1 = ax.legend(*scatter.legend_elements(), title="Deployment #", fontsize=16)
ax.set_ylabel(gi01sumo42.seawater_ph.attrs["long_name"], fontsize=16)
ax.set_title(gi01sumo42.attrs["id"], fontsize=16)
ax.grid()

fig.autofmt_xdate()
# -

gi01sumo42.attrs["id"]

# +
fig, ax = plt.subplots(figsize=(12,8))

# Plot the pH data, and color the 
scatter = ax.scatter(gi03flma.time, gi03flma.seawater_ph, c=gi03flma.deployment)
legend1 = ax.legend(*scatter.legend_elements(), title="Deployment #", fontsize=16)
ax.set_ylabel(gi03flma.seawater_ph.attrs["long_name"], fontsize=16)
ax.set_title(gi03flma.attrs["id"], fontsize=16)
ax.grid()

fig.autofmt_xdate()
# -

gi03flma.attrs["id"]

# +
fig, ax = plt.subplots(figsize=(12,8))

# Plot the pH data, and color the 
scatter = ax.scatter(gi03flmb.time, gi03flmb.seawater_ph, c=gi03flmb.deployment)
legend1 = ax.legend(*scatter.legend_elements(), title="Deployment #", fontsize=16)
ax.set_ylabel(gi03flmb.seawater_ph.attrs["long_name"], fontsize=16)
ax.set_title(gi03flmb.attrs["id"], fontsize=16)
ax.grid()

fig.autofmt_xdate()
# -

gi03flmb.attrs["id"]

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

import json

# Need to append and clean up the bottle datasets:

# +
Bottles = pd.DataFrame()

ARRAY = "/media/andrew/Files/Water_Sampling/Global_Irminger_Sea_Array"
for CRUISE in sorted(os.listdir(ARRAY)):
    PATH = "/".join((ARRAY, CRUISE, "Ship_data", "Water_Sampling"))
    if os.path.isdir(PATH):
        for file in sorted(os.listdir(PATH)):
            if "discrete" in file.lower() and file.endswith(".xlsx"):
                BOTTLE = "/".join((PATH, file))
                Bottles = Bottles.append(pd.read_excel(BOTTLE), ignore_index=True)
            else:
                pass
            
Bottles.head()
# -

Bottles.columns


# If the the bottle data from all of the cruises has already been appended - then can just load it:

# +
#Bottles = pd.read_excel("Irminger_Summary.xlsx")
#Bottles.head()
# -

# #### Split cells with multiple values into lists

# +
# First, want to split entries with a "," into a list
def split_entry(x):
    if type(x) == str and "," in x:
        x = x.split(",")
    return x

for col in Bottles:
    if "CTD" in col or "Discrete" in col:
        Bottles[col] = Bottles[col].apply(lambda x: split_entry(x))
    else:
        pass


# -

# #### Clean up the entries and convert multiple measurements from the same Niskin into a single measurement

# +
def convert_entry(x):
    if type(x) is str:
        try:
            x = float(x)
        except:
            pass
    elif type(x) is list:
        try:
            x = [float(i) for i in x]
            x = float(np.nanmean(x))
        except:
            pass
    return x

for col in Bottles:
    if "CTD" in col or "Discrete" in col:
        Bottles[col] = Bottles[col].apply(lambda x: convert_entry(x))
    else:
        pass
# -

# #### Replace placeholders with NaNs

Bottles = Bottles.replace(to_replace="-9999999", value=np.nan)
Bottles = Bottles.replace(to_replace=-9999999, value=np.nan)

# #### Convert times to pandas datetimes

Bottles["Start Time [UTC]"] = Bottles["Start Time [UTC]"].apply(lambda x: pd.to_datetime(x))
Bottles["CTD Bottle Closure Time [UTC]"] = Bottles["CTD Bottle Closure Time [UTC]"].apply(lambda x: pd.to_datetime(x))


# #### Replace Sample IDs with fill values

# +
def clean_carbon(x):
    if type(x) is float:
        if (x > 10) and (x < 2000):
            return np.nan
        else:
            return x
        
Bottles["Discrete Alkalinity [umol/kg]"] = Bottles["Discrete Alkalinity [umol/kg]"].apply(lambda x: clean_carbon(x))
Bottles["Discrete DIC [umol/kg]"] = Bottles["Discrete DIC [umol/kg]"].apply(lambda x: clean_carbon(x))
Bottles["Discrete pH [Total scale]"] = Bottles["Discrete pH [Total scale]"].apply(lambda x: clean_carbon(x))


# +
def clean_nutrients(x):
    if type(x) is float: # or type(x) is np.float64:
        return x
    else:
        if type(x) is list:
            return np.nan
        elif "-" in x:
            return np.nan
        elif "<" in x:
            return 0
        else:
            return x
        
Bottles["Discrete Nitrate [uM]"] = Bottles["Discrete Nitrate [uM]"].apply(lambda x: clean_nutrients(x))
Bottles["Discrete Nitrite [uM]"] = Bottles["Discrete Nitrite [uM]"].apply(lambda x: clean_nutrients(x))
Bottles["Discrete Phosphate [uM]"] = Bottles["Discrete Phosphate [uM]"].apply(lambda x: clean_nutrients(x))
Bottles["Discrete Silicate [uM]"] = Bottles["Discrete Silicate [uM]"].apply(lambda x: clean_nutrients(x))
Bottles["Discrete Ammonium [uM]"] = Bottles["Discrete Ammonium [uM]"].apply(lambda x: clean_nutrients(x))


# +
def clean_chlorophyll(x):
    if type(x) is float:
        return x
    else:
        if type(x) is list:
            return np.nan
        elif "/" in x:
            return np.nan
        else:
            return x
        
Bottles["Discrete Chlorophyll [ug/L]"] = Bottles["Discrete Chlorophyll [ug/L]"].apply(lambda x: clean_chlorophyll(x))
Bottles["Discrete Phaeopigment [ug/L]"] = Bottles["Discrete Phaeopigment [ug/L]"].apply(lambda x: clean_chlorophyll(x))
# -

# ---
# #### Calculate TEOS-10 Properties for Bottle Data 
# The TEOS-10 properties are considered to be derived from thermodynamic principles whereas the previous TEOS-80 was derived from empirical observations. Here, we'll add in the parameters for conservative temperature, absolute salinity, and neutral density.

import gsw

# +
# Calculate some key physical parameters to get density based on TEOS-10
SP = Bottles[["CTD Salinity 1 [psu]", "CTD Salinity 2 [psu]"]].mean(axis=1)
T = Bottles[["CTD Temperature 1 [deg C]", "CTD Temperature 2 [deg C]"]].mean(axis=1)
P = Bottles["CTD Pressure [db]"]
LAT = Bottles["CTD Latitude [deg]"]
LON = Bottles["CTD Longitude [deg]"]

# Absolute salinity
SA = gsw.conversions.SA_from_SP(SP, P, LON, LAT)
Bottles["CTD Absolute Salinity [g/kg]"] = SA

# Conservative temperature
CT = gsw.conversions.CT_from_t(SA, T, P)
Bottles["CTD Conservative Temperature"] = CT

# Density
RHO = gsw.density.rho(SA, CT, P)
Bottles["CTD Density [kg/m^3]"] = RHO

# Calculate potential density
SIGMA0 = gsw.density.sigma0(SA, CT)
Bottles["CTD Sigma [kg/m^3]"] = RHO
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
DIC = Bottles["Discrete DIC [umol/kg]"]
TA = Bottles["Discrete Alkalinity [umol/kg]"]
PH = Bottles["Discrete pH [Total scale]"]
SAL = Bottles["Discrete Salinity [psu]"]

# Set the input hydrographic parameters at which the DIC/TA/pH lab measurements were performed
TEMP_IN = 25
PRES_IN = 0

# Get the hydrographic parameters at which the samples were taken
TEMP_OUT = Bottles[["CTD Temperature 1 [deg C]","CTD Temperature 1 [deg C]"]].mean(axis=1, skipna=True)
PRES_OUT = Bottles["CTD Pressure [db]"]

# Nutrient inputs = need to fill NaNs with zeros otherwise will return NaNs
SIO4 = Bottles["Discrete Silicate [uM]"].fillna(value=0)
PO4 = Bottles["Discrete Phosphate [uM]"].fillna(value=0)
NH4 = Bottles["Discrete Ammonium [uM]"].fillna(value=0)
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
Bottles["Calculated Alkalinity [umol/kg]"] = CO2dict["alkalinity"]
Bottles["Calculated CO2aq [umol/kg]"] = CO2dict["aqueous_CO2_out"]
Bottles["Calculated CO3 [umol/kg]"] = CO2dict["carbonate_out"]
Bottles["Calculated DIC [umol/kg]"] = CO2dict["dic"]
Bottles["Calculated pCO2 [uatm]"] = CO2dict["pCO2_out"]
# Bottles["Calculated pH"] = CO2dict["pH_out"]

# **Adjusted pH:** Since we now know that CO2SYS can reproduce the pH measurements from the DIC/TA values, we can input the measured pH in order to adjust the pH for pressure and temperature.

CO2dict = pyco2.CO2SYS_nd(DIC, PH, 2, 3, SAL, total_ammonia=NH4, total_phosphate=PO4, total_silicate=SIO4, 
                          temperature_out=TEMP_OUT, pressure_out=PRES_OUT)

Bottles["Calculated pH"] = CO2dict["pH_total_out"]

# ---
# ## Data Comparison
# Finally, we can go ahead and begin comparing the Bottle Data with the PHSEN data.

# ### GI03FLMA PHSEN vs Bottle Data

deployments = sorted(np.unique(gi03flma.deployment))
deployments

# Identify the bottle associated with GI03FLMA
Bottles["Target Asset"].unique()

# Filter for the associated bottles collected a GI03FLMA
flma_mask = Bottles["Target Asset"].apply(lambda x: True if "flma" in x.lower() else False)
gi03flma_bot = Bottles[flma_mask]
gi03flma_bot.head()

# +
# Plot the data
fig, ax = plt.subplots(figsize=(15,10))

scatter = ax.scatter(gi03flma.time, gi03flma.seawater_ph, c=gi03flma.deployment)
discrete = ax.plot(gi03flma_bot["Start Time [UTC]"], gi03flma_bot["Calculated pH"],
             marker=".", linestyle="", markersize=20, color="tab:red")
ax.set_ylabel("In-situ pH [Total Scale]", fontsize=16)
ax.set_ylim((7.4, 8.4))
ax.set_title(gi03flma.id, fontsize=16)
ax.set_xlabel("Date", fontsize=16)
ax.grid()

# Generate the figure legend
marker = scatter.legend_elements()[0]
marker.append(discrete[0])
label = ["Deployment " + x for x in scatter.legend_elements()[1]]
label.append("Discrete Water Sample")
legend =(marker, label)
legend1 = ax.legend(*legend, fontsize=12, edgecolor="black")

fig.autofmt_xdate()
# -

gi03flma.lat, gi03flma.lon

gi03flma_bot[gi03flma_bot["Calculated pH"].notna()][["Cruise", "Start Time [UTC]"]]

# There are three apparent cruises on which we have discrete water sample data for comparison with the in-situ PHSEN data: **AT30-01**, **AR7-01**, **AR21**.
#
# Next, we want to look at the individual deployments, calculate the standard deviations, and compare with the discrete water bottle samples.
#
# **Deployment 2**

flma_bot2 = gi03flma_bot[gi03flma_bot["Calculated pH"].notna()].loc[[229, 324]]
flma_dep2 = gi03flma.where(gi03flma.deployment == 2, drop=True)
flma_dep2_dif = flma_dep2.where(flma_dep2.mask.astype(bool), drop=True).diff("time")
flma_dep2_avg, flma_dep2_std = (flma_dep2_dif.seawater_ph.values.mean(), flma_dep2_dif.seawater_ph.values.std())

flma_dep2_avg, flma_dep2_std

# +
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.plot(flma_dep2.time, flma_dep2.seawater_ph, marker=".", c="tab:blue")
fill = ax.fill_between(flma_dep2.time, flma_dep2.seawater_ph-flma_dep2_std*1.96, 
                      flma_dep2.seawater_ph+flma_dep2_std*1.96, alpha=0.5)
discrete = ax.plot(flma_bot2["Start Time [UTC]"], flma_bot2["Calculated pH"],
             marker=".", linestyle="", markersize=20, color="tab:red")
ax.grid()
ax.legend(["PHSEN","Discrete Samples"], fontsize=12)
ax.set_ylabel("pH [Total Scale]", fontsize=16)
ax.set_title(f"{flma_dep2_dif.id} Deployment 2", fontsize=16)
fig.autofmt_xdate()
# -

np.min(flma_dep2.time), np.max(flma_dep2.time)

# +
# Do the deployment
deploy_bot = flma_bot2.loc[229]
recover_bot = flma_bot2.loc[324]

flma_dep2_deploy = flma_dep2.sel(time=slice("2015-08-19","2015-08-22"))
flma_dep2_recover = flma_dep2.sel(time=slice("2016-07-09","2016-07-13"))

# +
fig, (ax0, ax1) = plt.subplots(ncols=2, nrows=1, figsize=(12,8))

ax0.plot(flma_dep2_deploy.time, flma_dep2_deploy.seawater_ph, marker=".", c="tab:blue")
ax0.fill_between(flma_dep2_deploy.time, flma_dep2_deploy.seawater_ph-flma_dep2_std*1.96, 
                 flma_dep2_deploy.seawater_ph+flma_dep2_std*1.96, alpha=0.5)
ax0.plot(deploy_bot["Start Time [UTC]"], deploy_bot["Calculated pH"],
         marker=".", linestyle="", markersize=20, color="tab:red")
ax0.grid()
ax0.set_ylabel("pH [Total Scale]", fontsize=16)
ax0.legend(["PHSEN", "Discrete Samples"], loc="lower right", fontsize=12)
ax0.set_title("Deployment")


ax1.plot(flma_dep2_recover.time, flma_dep2_recover.seawater_ph, marker=".", c="tab:blue")
ax1.fill_between(flma_dep2_recover.time, flma_dep2_recover.seawater_ph-flma_dep2_std*1.96, 
                 flma_dep2_recover.seawater_ph+flma_dep2_std*1.96, alpha=0.5)
ax1.plot(recover_bot["Start Time [UTC]"], recover_bot["Calculated pH"],
         marker=".", linestyle="", markersize=20, color="tab:red")
ax1.grid()
ax1.set_title("Recovery")

fig.autofmt_xdate()
# -

# **Deployment 3**

flma_bot3 = gi03flma_bot[gi03flma_bot["Calculated pH"].notna()].loc[[324, 428]]
flma_dep3 = gi03flma.where(gi03flma.deployment == 3, drop=True)
flma_dep3_dif = flma_dep3.where(flma_dep3.mask.astype(bool), drop=True).diff("time")
flma_dep3_avg, flma_dep3_std = (flma_dep3_dif.seawater_ph.values.mean(), flma_dep3_dif.seawater_ph.values.std())

flma_dep3_avg, flma_dep3_std

# +
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.plot(flma_dep3.time, flma_dep3.seawater_ph, marker=".", c="tab:blue")
fill = ax.fill_between(flma_dep3.time, flma_dep3.seawater_ph-flma_dep2_std*1.96, 
                      flma_dep3.seawater_ph+flma_dep3_std*1.96, alpha=0.5)
discrete = ax.plot(flma_bot3["Start Time [UTC]"], flma_bot3["Calculated pH"],
             marker=".", linestyle="", markersize=20, color="tab:red")
ax.grid()
ax.legend(["PHSEN","Discrete Samples"], fontsize=12)
ax.set_ylabel("pH [Total Scale]", fontsize=16)
ax.set_title(f"{flma_dep3_dif.id} Deployment 3", fontsize=16)
fig.autofmt_xdate()
# -

np.min(flma_dep3.time), np.max(flma_dep3.time)

# +
# Do the deployment
deploy_bot = flma_bot3.loc[324]
recover_bot = flma_bot3.loc[428]

flma_dep3_deploy = flma_dep3.sel(time=slice("2016-07-17","2016-07-20"))
flma_dep3_recover = flma_dep3.sel(time=slice("2017-08-05","2017-08-09"))

# +
fig, (ax0, ax1) = plt.subplots(ncols=2, nrows=1, figsize=(12,8))

ax0.plot(flma_dep3_deploy.time, flma_dep3_deploy.seawater_ph, marker=".", c="tab:blue")
ax0.fill_between(flma_dep3_deploy.time, flma_dep3_deploy.seawater_ph-flma_dep3_std*1.96, 
                 flma_dep3_deploy.seawater_ph+flma_dep3_std*1.96, alpha=0.5)
ax0.plot(deploy_bot["Start Time [UTC]"], deploy_bot["Calculated pH"],
         marker=".", linestyle="", markersize=20, color="tab:red")
ax0.grid()
ax0.set_ylabel("pH [Total Scale]", fontsize=16)
ax0.legend(["PHSEN", "Discrete Samples"], loc="lower right", fontsize=12)
ax0.set_title("Deployment")


ax1.plot(flma_dep3_recover.time, flma_dep3_recover.seawater_ph, marker=".", c="tab:blue")
ax1.fill_between(flma_dep3_recover.time, flma_dep3_recover.seawater_ph-flma_dep3_std*1.96, 
                 flma_dep3_recover.seawater_ph+flma_dep3_std*1.96, alpha=0.5)
ax1.plot(recover_bot["Start Time [UTC]"], recover_bot["Calculated pH"],
         marker=".", linestyle="", markersize=20, color="tab:red")
ax1.grid()
ax1.set_title("Recovery")

fig.autofmt_xdate()
# -

# **Deployment 4**

flma_bot4 = gi03flma_bot[gi03flma_bot["Calculated pH"].notna()].loc[[428]]
flma_dep4 = gi03flma.where(gi03flma.deployment == 4, drop=True)
flma_dep4_dif = flma_dep4.where(flma_dep4.mask.astype(bool), drop=True).diff("time")
flma_dep4_avg, flma_dep4_std = (flma_dep4_dif.seawater_ph.values.mean(), flma_dep4_dif.seawater_ph.values.std())

flma_dep4_avg, flma_dep4_std

# +
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.plot(flma_dep4.time, flma_dep4.seawater_ph, marker=".", c="tab:blue")
fill = ax.fill_between(flma_dep4.time, flma_dep4.seawater_ph-flma_dep2_std*1.96, 
                      flma_dep4.seawater_ph+flma_dep4_std*1.96, alpha=0.5)
discrete = ax.plot(flma_bot4["Start Time [UTC]"], flma_bot4["Calculated pH"],
             marker=".", linestyle="", markersize=20, color="tab:red")
ax.grid()
ax.legend(["PHSEN","Discrete Samples"], fontsize=12)
ax.set_ylabel("pH [Total Scale]", fontsize=16)
ax.set_title(f"{flma_dep4_dif.id} Deployment 4", fontsize=16)
fig.autofmt_xdate()
# -

np.min(flma_dep4.time), np.max(flma_dep4.time)

# +
# Do the deployment
deploy_bot = flma_bot4.loc[428]
#recover_bot = flma_bot4.loc[428]

flma_dep4_deploy = flma_dep4.sel(time=slice("2017-08-12","2017-08-16"))
flma_dep4_recover = flma_dep4.sel(time=slice("2018-06-11","2018-06-15"))

# +
fig, ax = plt.subplots(figsize=(12,8))

ax.plot(flma_dep4_deploy.time, flma_dep4_deploy.seawater_ph, marker=".", c="tab:blue")
ax.fill_between(flma_dep4_deploy.time, flma_dep4_deploy.seawater_ph-flma_dep4_std*1.96, 
                 flma_dep4_deploy.seawater_ph+flma_dep4_std*1.96, alpha=0.5)
ax.plot(deploy_bot["Start Time [UTC]"], deploy_bot["Calculated pH"],
         marker=".", linestyle="", markersize=20, color="tab:red")
ax.grid()
ax.set_ylabel("pH [Total Scale]", fontsize=16)
ax.legend(["PHSEN", "Discrete Samples"], loc="lower right", fontsize=12)
ax.set_title("Deployment")

fig.autofmt_xdate()

# +
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(12, 24))

scatter = ax0.plot(flma_dep2.time.values[flma_dep2.mask.astype(bool)], flma_dep2.seawater_ph.values[flma_dep2.mask.astype(bool)],
                  marker=".", c="tab:blue")
fill = ax0.fill_between(flma_dep2.time[flma_dep2.mask.astype(bool)], flma_dep2.seawater_ph[flma_dep2.mask.astype(bool)]-flma_dep2_std*1.96, 
                      flma_dep2.seawater_ph[flma_dep2.mask.astype(bool)]+flma_dep2_std*1.96, alpha=0.5)
discrete = ax0.plot(flma_bot2["Start Time [UTC]"], flma_bot2["Calculated pH"],
             marker=".", linestyle="", markersize=20, color="tab:red")
ax0.grid()
ax0.legend(["PHSEN","Discrete Samples"], fontsize=12)
ax0.set_ylabel("pH [Total Scale]", fontsize=16)
ax0.set_title(f"{flma_dep2_dif.id}\nDeployment 2", fontsize=16)

scatter = ax1.plot(flma_dep3.time.values[flma_dep3.mask.astype(bool)], flma_dep3.seawater_ph.values[flma_dep3.mask.astype(bool)],
                  marker=".", c="tab:blue")
fill = ax1.fill_between(flma_dep3.time[flma_dep3.mask.astype(bool)], flma_dep3.seawater_ph[flma_dep3.mask.astype(bool)]-flma_dep2_std*1.96, 
                      flma_dep3.seawater_ph[flma_dep3.mask.astype(bool)]+flma_dep3_std*1.96, alpha=0.5)
discrete = ax1.plot(flma_bot3["Start Time [UTC]"], flma_bot3["Calculated pH"],
             marker=".", linestyle="", markersize=20, color="tab:red")
ax1.grid()
ax1.legend(["PHSEN","Discrete Samples"], fontsize=12)
#ax1.set_ylabel("pH [Total Scale]", fontsize=16)
ax1.set_title(f"Deployment 3", fontsize=16)

scatter = ax2.plot(flma_dep4.time.values[flma_dep4.mask.astype(bool)], flma_dep4.seawater_ph.values[flma_dep4.mask.astype(bool)],
                  marker=".", c="tab:blue")
fill = ax2.fill_between(flma_dep4.time[flma_dep4.mask.astype(bool)], flma_dep4.seawater_ph[flma_dep4.mask.astype(bool)]-flma_dep2_std*1.96, 
                      flma_dep4.seawater_ph[flma_dep4.mask.astype(bool)]+flma_dep4_std*1.96, alpha=0.5)
discrete = ax2.plot(flma_bot4["Start Time [UTC]"], flma_bot4["Calculated pH"],
             marker=".", linestyle="", markersize=20, color="tab:red")
ax2.grid()
#ax2.legend(["PHSEN","Discrete Samples"], fontsize=12)
ax2.set_ylabel("pH [Total Scale]", fontsize=16)
ax2.set_title(f"Deployment 4", fontsize=16)
# -

# ### GI03FLMB-RIS01-04-PHSENF000 vs. Bottle Data

deployments = sorted(np.unique(gi03flma.deployment))
deployments

# Identify the bottle associated with GI03FLMA
Bottles["Target Asset"].unique()

flmb_mask = Bottles["Target Asset"].apply(lambda x: True if "flmb" in x.lower() else False)
gi03flmb_bot = Bottles[flmb_mask]
gi03flmb_bot.head()

# +
# Plot the data
fig, ax = plt.subplots(figsize=(12,8))

scatter = ax.scatter(gi03flmb.time, gi03flmb.seawater_ph, c=gi03flmb.deployment)
discrete = ax.plot(gi03flmb_bot["Start Time [UTC]"], gi03flmb_bot["Calculated pH"],
             marker=".", linestyle="", markersize=20, color="tab:red")
ax.set_ylabel("In-situ pH [Total Scale]", fontsize=16)
ax.set_ylim((7.4, 8.4))
ax.set_title(gi03flmb.id, fontsize=16)
#ax.set_xlabel("Date", fontsize=16)
ax.grid()

# Generate the figure legend
marker = scatter.legend_elements()[0]
marker.append(discrete[0])
label = ["Deployment " + x for x in scatter.legend_elements()[1]]
label.append("Discrete Water Sample")
legend =(marker, label)
legend1 = ax.legend(*legend, fontsize=12, edgecolor="black")

fig.autofmt_xdate()
# -

gi03flmb.lat, gi03flmb.lon

gi03flmb_bot[gi03flmb_bot["Calculated pH"].notna()][["Cruise", "Start Time [UTC]"]]

# **Deployment 1**

flmb_bot1 = gi03flmb_bot[gi03flmb_bot["Calculated pH"].notna()].loc[[217, 241]]
flmb_dep1 = gi03flmb.where(gi03flmb.deployment == 1, drop=True)
flmb_dep1_dif = flmb_dep1.where(flmb_dep1.mask.astype(bool), drop=True).diff("time")
flmb_dep1_avg, flmb_dep1_std = (flmb_dep1_dif.seawater_ph.values.mean(), flmb_dep1_dif.seawater_ph.values.std())

flmb_dep1_avg, flmb_dep1_std

# +
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.plot(flmb_dep1.time, flmb_dep1.seawater_ph, marker=".", c="tab:blue")
fill = ax.fill_between(flmb_dep1.time, flmb_dep1.seawater_ph-flmb_dep1_std*1.96, 
                       flmb_dep1.seawater_ph+flmb_dep1_std*1.96, alpha=0.5)
discrete = ax.plot(flmb_bot1["Start Time [UTC]"], flmb_bot1["Calculated pH"],
             marker=".", linestyle="", markersize=20, color="tab:red")
ax.grid()
ax.legend(["PHSEN","Discrete Samples"], fontsize=12)
ax.set_ylabel("pH [Total Scale]", fontsize=16)
ax.set_title(f"{flmb_dep1_dif.id} Deployment 1", fontsize=16)
fig.autofmt_xdate()
# -

np.min(flmb_dep1.time), np.max(flmb_dep1.time)

flmb_bot1

# +
# Do the deployment
#deploy_bot = flma_bot4.loc[428]
recover_bot = flmb_bot1

flmb_dep1_deploy = flmb_dep1.sel(time=slice("2014-09-16","2014-09-20"))
flmb_dep1_recover = flmb_dep1.sel(time=slice("2015-08-10","2015-08-21"))

# +
fig, ax = plt.subplots(figsize=(12,8))

ax.plot(flmb_dep1_recover.time, flmb_dep1_recover.seawater_ph, marker=".", c="tab:blue")
ax.fill_between(flmb_dep1_recover.time, flmb_dep1_recover.seawater_ph-flmb_dep1_std*1.96, 
                 flmb_dep1_recover.seawater_ph+flmb_dep1_std*1.96, alpha=0.5)
ax.plot(recover_bot["Start Time [UTC]"], recover_bot["Calculated pH"],
         marker=".", linestyle="", markersize=20, color="tab:red")
ax.grid()
ax.set_ylabel("pH [Total Scale]", fontsize=16)
ax.legend(["PHSEN", "Discrete Samples"], loc="lower right", fontsize=12)
ax.set_title("Recovery")

fig.autofmt_xdate()
# -

# **Deployment 2**

flmb_bot2 = gi03flmb_bot[gi03flmb_bot["Calculated pH"].notna()].loc[[217, 241, 336]]
flmb_dep2 = gi03flmb.where(gi03flmb.deployment == 2, drop=True)
flmb_dep2_dif = flmb_dep2.where(flmb_dep2.mask.astype(bool), drop=True).diff("time")
flmb_dep2_avg, flmb_dep2_std = (flmb_dep2_dif.seawater_ph.values.mean(), flmb_dep2_dif.seawater_ph.values.std())

flmb_dep2_avg, flmb_dep2_std

# +
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.plot(flmb_dep2.time, flmb_dep2.seawater_ph, marker=".", c="tab:blue")
fill = ax.fill_between(flmb_dep2.time, flmb_dep2.seawater_ph-flmb_dep2_std*1.96, 
                       flmb_dep2.seawater_ph+flmb_dep2_std*1.96, alpha=0.5)
discrete = ax.plot(flmb_bot2["Start Time [UTC]"], flmb_bot2["Calculated pH"],
             marker=".", linestyle="", markersize=20, color="tab:red")
ax.grid()
ax.legend(["PHSEN","Discrete Samples"], fontsize=12)
ax.set_ylabel("pH [Total Scale]", fontsize=16)
ax.set_title(f"{flmb_dep2_dif.id} Deployment 2", fontsize=16)
ax.set_ylim((8.0, 8.2))
fig.autofmt_xdate()
# -

np.min(flmb_dep2.time), np.max(flmb_dep2.time)

flmb_bot2

# +
# Do the deployment
deploy_bot = flmb_bot2.iloc[0:2]
recover_bot = flmb_bot2.iloc[2]

flmb_dep2_deploy = flmb_dep2.sel(time=slice("2015-08-21","2015-08-25"))
flmb_dep2_recover = flmb_dep2.sel(time=slice("2016-07-07","2016-07-14"))
# -

deploy_bot

# +
fig, (ax0, ax1) = plt.subplots(ncols=2, nrows=1, figsize=(12,8))

ax0.plot(flmb_dep2_deploy.time, flmb_dep2_deploy.seawater_ph, marker=".", c="tab:blue")
ax0.fill_between(flmb_dep2_deploy.time, flmb_dep2_deploy.seawater_ph-flmb_dep2_std*1.96, 
                 flmb_dep2_deploy.seawater_ph+flmb_dep2_std*1.96, alpha=0.5)
ax0.plot(deploy_bot["Start Time [UTC]"], deploy_bot["Calculated pH"],
         marker=".", linestyle="", markersize=20, color="tab:red")
ax0.grid()
ax0.set_ylabel("pH [Total Scale]", fontsize=16)
ax0.legend(["PHSEN", "Discrete Samples"], loc="lower right", fontsize=12)
ax0.set_title("Deployment")


ax1.plot(flmb_dep2_recover.time, flmb_dep2_recover.seawater_ph, marker=".", c="tab:blue")
ax1.fill_between(flmb_dep2_recover.time, flmb_dep2_recover.seawater_ph-flmb_dep2_std*1.96, 
                 flmb_dep2_recover.seawater_ph+flmb_dep2_std*1.96, alpha=0.5)
ax1.plot(recover_bot["Start Time [UTC]"], recover_bot["Calculated pH"],
         marker=".", linestyle="", markersize=20, color="tab:red")
ax1.grid()
ax1.set_title("Recovery")

fig.autofmt_xdate()
# -

# **Deployment 3**

flmb_bot3 = gi03flmb_bot[gi03flmb_bot["Calculated pH"].notna()].loc[[336, 417]]
flmb_dep3 = gi03flmb.where(gi03flmb.deployment == 3, drop=True)
flmb_dep3_dif = flmb_dep3.where(flmb_dep3.mask.astype(bool), drop=True).diff("time")
flmb_dep3_avg, flmb_dep3_std = (np.nanmean(flmb_dep3_dif.seawater_ph.values), np.nanstd(flmb_dep3_dif.seawater_ph.values))

flmb_dep3_avg, flmb_dep3_std

# +
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.plot(flmb_dep3.time, flmb_dep3.seawater_ph, marker=".", c="tab:blue")
fill = ax.fill_between(flmb_dep3.time, flmb_dep3.seawater_ph-flmb_dep3_std*1.96, 
                       flmb_dep3.seawater_ph+flmb_dep3_std*1.96, alpha=0.5)
discrete = ax.plot(flmb_bot3["Start Time [UTC]"], flmb_bot3["Calculated pH"],
             marker=".", linestyle="", markersize=20, color="tab:red")
ax.grid()
ax.legend(["PHSEN","Discrete Samples"], fontsize=12)
ax.set_ylabel("pH [Total Scale]", fontsize=16)
ax.set_title(f"{flmb_dep3_dif.id} Deployment 3", fontsize=16)
fig.autofmt_xdate()
# -

np.min(flmb_dep3.time), np.max(flmb_dep3.time)

flmb_bot3

# +
# Do the deployment
deploy_bot = flmb_bot3.iloc[0]
recover_bot = flmb_bot3.iloc[1]

flmb_dep3_deploy = flmb_dep3.sel(time=slice("2016-07-14","2016-07-18"))
flmb_dep3_recover = flmb_dep3.sel(time=slice("2017-08-03","2017-08-08"))

# +
fig, (ax0, ax1) = plt.subplots(ncols=2, nrows=1, figsize=(12,8))

ax0.plot(flmb_dep3_deploy.time, flmb_dep3_deploy.seawater_ph, marker=".", c="tab:blue")
ax0.fill_between(flmb_dep3_deploy.time, flmb_dep3_deploy.seawater_ph-flmb_dep3_std*1.96, 
                 flmb_dep3_deploy.seawater_ph+flmb_dep3_std*1.96, alpha=0.5)
ax0.plot(deploy_bot["Start Time [UTC]"], deploy_bot["Calculated pH"],
         marker=".", linestyle="", markersize=20, color="tab:red")
ax0.grid()
ax0.set_ylabel("pH [Total Scale]", fontsize=16)
ax0.legend(["PHSEN", "Discrete Samples"], loc="lower right", fontsize=12)
ax0.set_title("Deployment")


ax1.plot(flmb_dep3_recover.time, flmb_dep3_recover.seawater_ph, marker=".", c="tab:blue")
ax1.fill_between(flmb_dep3_recover.time, flmb_dep3_recover.seawater_ph-flmb_dep3_std*1.96, 
                 flmb_dep3_recover.seawater_ph+flmb_dep3_std*1.96, alpha=0.5)
ax1.plot(recover_bot["Start Time [UTC]"], recover_bot["Calculated pH"],
         marker=".", linestyle="", markersize=20, color="tab:red")
ax1.grid()
ax1.set_title("Recovery")

fig.autofmt_xdate()
# -

# **Deployment 4**

flmb_bot4 = gi03flmb_bot[gi03flmb_bot["Calculated pH"].notna()].loc[[417]]
flmb_dep4 = gi03flmb.where(gi03flmb.deployment == 4, drop=True)
flmb_dep4_dif = flmb_dep4.where(flmb_dep4.mask.astype(bool), drop=True).diff("time")
flmb_dep4_avg, flmb_dep4_std = (flmb_dep4_dif.seawater_ph.values.mean(), flmb_dep4_dif.seawater_ph.values.std())

flmb_dep4_avg, flmb_dep4_std

# +
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.plot(flmb_dep4.time, flmb_dep4.seawater_ph, marker=".", c="tab:blue")
fill = ax.fill_between(flmb_dep4.time, flmb_dep4.seawater_ph-flmb_dep4_std*1.96, 
                       flmb_dep4.seawater_ph+flmb_dep4_std*1.96, alpha=0.5)
discrete = ax.plot(flmb_bot4["Start Time [UTC]"], flmb_bot4["Calculated pH"],
             marker=".", linestyle="", markersize=20, color="tab:red")
ax.grid()
ax.legend(["PHSEN","Discrete Samples"], fontsize=12)
ax.set_ylabel("pH [Total Scale]", fontsize=16)
ax.set_title(f"{flmb_dep4_dif.id} Deployment 4", fontsize=16)
fig.autofmt_xdate()
# -

np.min(flmb_dep4.time), np.max(flmb_dep4.time)

flmb_bot4

# +
# Do the deployment
deploy_bot = flmb_bot4.iloc[0]
#recover_bot = flmb_bot3.iloc[1]

flmb_dep4_deploy = flmb_dep4.sel(time=slice("2017-08-12","2017-08-17"))
#flmb_dep3_recover = flmb_dep3.sel(time=slice("2017-08-03","2017-08-08"))

# +
fig, ax = plt.subplots(figsize=(12,8))

ax.plot(flmb_dep4_deploy.time, flmb_dep4_deploy.seawater_ph, marker=".", c="tab:blue")
ax.fill_between(flmb_dep4_deploy.time, flmb_dep4_deploy.seawater_ph-flmb_dep4_std*1.96, 
                 flmb_dep4_deploy.seawater_ph+flmb_dep4_std*1.96, alpha=0.5)
ax.plot(deploy_bot["Start Time [UTC]"], deploy_bot["Calculated pH"],
         marker=".", linestyle="", markersize=20, color="tab:red")
ax.grid()
ax.set_ylabel("pH [Total Scale]", fontsize=16)
ax.legend(["PHSEN", "Discrete Samples"], loc="lower right", fontsize=12)
ax.set_title("Deployment")

fig.autofmt_xdate()
# -



# **Deployment 5**

flmb_bot5 = gi03flmb_bot[gi03flmb_bot["Calculated pH"].notna()].loc[[417]]
flmb_dep5 = gi03flmb.where(gi03flmb.deployment == 5, drop=True)
flmb_dep5_dif = flmb_dep5.where(flmb_dep5.mask.astype(bool), drop=True).diff("time")
flmb_dep5_avg, flmb_dep5_std = (flmb_dep5_dif.seawater_ph.values.mean(), flmb_dep5_dif.seawater_ph.values.std())

flmb_dep5_avg, flmb_dep5_std

# +
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.plot(flmb_dep5.time, flmb_dep5.seawater_ph, marker=".", c="tab:blue")
fill = ax.fill_between(flmb_dep5.time, flmb_dep5.seawater_ph-flmb_dep5_std*1.96, 
                       flmb_dep5.seawater_ph+flmb_dep5_std*1.96, alpha=0.5)
discrete = ax.plot(flmb_bot5["Start Time [UTC]"], flmb_bot5["Calculated pH"],
             marker=".", linestyle="", markersize=20, color="tab:red")
ax.grid()
ax.legend(["PHSEN","Discrete Samples"], fontsize=12)
ax.set_ylabel("pH [Total Scale]", fontsize=16)
ax.set_title(f"{flmb_dep5_dif.id} Deployment 5", fontsize=16)
fig.autofmt_xdate()
# -

# ### GI01SUMO-RII11-02-PHSENE041

deployments = sorted(np.unique(gi03flma.deployment))
deployments

# Identify the bottle associated with GI03FLMA
Bottles["Target Asset"].unique()

gi01sumo_mask = Bottles["Target Asset"].apply(lambda x: True if "sumo" in x.lower() else False)
gi01sumo_bot = Bottles[gi01sumo_mask]
gi01sumo_bot.head()

# The GI01SUMO PHSENE041 is at 20 meters depth

mask41 = (gi01sumo_bot["CTD Pressure [db]"] > 10) & (gi01sumo_bot["CTD Pressure [db]"] < 50)
mask42 = gi01sumo_bot["CTD Pressure [db]"] > 60

gi01sumo41_bot = gi01sumo_bot[mask41]
gi01sumo42_bot = gi01sumo_bot[mask42]

# +
# Plot the data
fig, ax = plt.subplots(figsize=(12,8))

scatter = ax.scatter(gi01sumo41.time, gi01sumo41.seawater_ph, c=gi01sumo41.deployment)
discrete = ax.plot(gi01sumo41_bot["Start Time [UTC]"], gi01sumo41_bot["Calculated pH"],
             marker=".", linestyle="", markersize=20, color="tab:red")
ax.set_ylabel("In-situ pH [Total Scale]", fontsize=16)
ax.set_ylim((7.4, 8.4))
ax.set_title(gi01sumo41.id, fontsize=16)
#ax.set_xlabel("Date", fontsize=16)
ax.grid()

# Generate the figure legend
marker = scatter.legend_elements()[0]
marker.append(discrete[0])
label = ["Deployment " + x for x in scatter.legend_elements()[1]]
label.append("Discrete Water Sample")
legend =(marker, label)
legend1 = ax.legend(*legend, fontsize=12, edgecolor="black")

fig.autofmt_xdate()
# -

gi01sumo41.id

# +
# Plot the data
fig, ax = plt.subplots(figsize=(12,8))

scatter = ax.scatter(gi01sumo42.time, gi01sumo42.seawater_ph, c=gi01sumo42.deployment)
discrete = ax.plot(gi01sumo42_bot["Start Time [UTC]"], gi01sumo42_bot["Calculated pH"],
             marker=".", linestyle="", markersize=20, color="tab:red")
ax.set_ylabel("In-situ pH [Total Scale]", fontsize=16)
ax.set_ylim((7.4, 8.4))
ax.set_title(gi01sumo42.id, fontsize=16)
#ax.set_xlabel("Date", fontsize=16)
ax.grid()

# Generate the figure legend
marker = scatter.legend_elements()[0]
marker.append(discrete[0])
label = ["Deployment " + x for x in scatter.legend_elements()[1]]
label.append("Discrete Water Sample")
legend =(marker, label)
legend1 = ax.legend(*legend, fontsize=12, edgecolor="black")

fig.autofmt_xdate()
# -

gi01sumo42.lat, gi01sumo42.lon


