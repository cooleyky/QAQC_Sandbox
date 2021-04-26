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

# # CTD: Calculate QARTOD Tables
#
# **Author: Andrew Reed**
#
# ### Purpose
# The purpose of this notebook is to calculate the QARTOD test data tables for the CGSN-maintained CTD instruments, including both fixed instruments and those on vehicles. First, all of the relevant CGSN CTD instruments and associated data streams should be identified using the Download Data Notebook. With the data downloaded locally, we can 
#
# ### Method
# The approach outlined here attempts to conform to the approach to outlined in the QARTOD Manual for In-situ Temperature and Salinity Observations [1]. The 
#
# ### References
# 1. U.S. Integrated Ocean Observing System, 2015. Manual for Real-Time Quality Control of In-situ Temperature and Salinity Data Version 2.0: A Guide to Quality Control and Quality Assurance of In-situ Temperature and Salinity Observations. 56 pp.

# Import libraries
import os, shutil, sys, time, re, requests, csv, datetime, pytz
import time
import yaml
import pandas as pd
import numpy as np
import netCDF4 as nc
import xarray as xr
from fuzzywuzzy import process
import warnings
warnings.filterwarnings("ignore")

# Import the OOINet M2M Tool
sys.path.append("/home/andrew/Documents/OOI-CGSN/ooinet/ooinet/")
from m2m import M2M

# Import QARTOD utils
sys.path.append("../../")
from utils import *

import matplotlib.pyplot as plt
# %matplotlib inline

# #### Set OOINet API access
# In order access and download data from OOINet, need to have an OOINet api username and access token. Those can be found on your profile after logging in to OOINet. Your username and access token should NOT be stored in this notebook/python script (for security). It should be stored in a yaml file, kept in the same directory, named user_info.yaml.

userinfo = yaml.load(open("../../../user_info.yaml"))
username = userinfo["apiname"]
token = userinfo["apikey"]

# #### Connect to OOINet

OOINet = M2M(username, token)

# ---
# ## Datasets
# First, the ```Download_Data``` notebook should be run first. Then, if all the datasets for a given instrument have already been identified, then want to simply load the identified data streams from local memory:

# Try to load the CGSN DOSTA datasets
try:
    cgsn_datasets = pd.read_csv("../data/cgsn_datasets.csv")
# If they haven't been downloaded yet
except:
    print("Downloading Dataset list...")
    datasets = OOINet.search_datasets(instrument="CTD", English_names=True)
    # Save the dataset results
    cgsn = datasets["array"].apply(lambda x: True if x.startswith(("CP","GA","GI","GP","GS")) else False)
    cgsn_datasets = datasets[cgsn]
    cgsn_datasets.to_csv("../data/cgsn_datasets.csv", index=False)

# Look at the cgsn_datasets
cgsn_datasets.head()

# #### Reference Designators
# From all the identified datasets, we can get all the reference_designators. We also want to split out the fixed platforms from the vehicles (mobile assets) and wire-following-profilers

reference_designators = sorted(cgsn_datasets["refdes"])
fixed_platforms = [refdes for refdes in reference_designators if "MOAS" not in refdes and "CTDPF" not in refdes]
fixed_platforms

# #### Select a Reference Designator

refdes = fixed_platforms[0]
refdes

# ---
# ## Metadata 
# The metadata contains the following important key pieces of data for each reference designator: **method**, **stream**, **particleKey**, and **count**. The method and stream are necessary for identifying and loading the relevant dataset. The particleKey tells us which data variables in the dataset we should be calculating the QARTOD parameters for. The count lets us know which dataset (the recovered instrument, recovered host, or telemetered) contains the most data and likely has the best record to use to calculate the QARTOD tables. 

metadata = OOINet.get_metadata(refdes)
metadata

# Groupby based on the reference designator - method - stream to get the unique values for each data stream

metadata = metadata.groupby(by=["refdes","method","stream"]).agg(lambda x: pd.unique(x.values.ravel()).tolist())
metadata = metadata.reset_index()
metadata.head()

# Convert single item lists back to entries

metadata = metadata.applymap(lambda x: x[0] if len(x) == 1 else x)
metadata.head()

# #### Select a referenc

method, stream, particleKey, pdId = metadata[metadata["count"] == np.max(metadata["count"])][["method","stream","particleKey","pdId"]].iloc[0]
method, stream, particleKey, pdId

# ---
# ## Load Data
# When calculating the QARTOD data tables, we only want to utilize the most complete data record available for a given reference designator. We can identify this by filtering for the largest value under ```count``` which indicates the number of particles in the system for a given dataset. The more particles, the more availabe data. While most of the time this will be the recovered_inst stream, in cases of instrument loss or failure, it may be the record recovered from the mooring host computer (recovered_host) or even data which was telemetered back to shore.

# #### Request Data
# Next, we need to query OOINet to get the netCDF files to load. To speed up the request, we can do several things. First, we utilize the ```pdId``` field from the metadata to limit our request to only the relevant parameters which we will be calculating the QARTOD Tables for. Second, we can load and store our requests to OOINet. The constructed netCDF files and THREDDS catalog remain active for up to 6 months; this can significantly shorten the download time by not needing OOINet to generate new netCDF files.
#
# To build the request, we need the ```method```, ```stream```, ```particleKey```, and ```pdId``` for the stream with the most available data for the given reference designator.

# +
# First, want to load my record of requests and check if I've made this specific request before
thredds_table = pd.read_csv("../data/thredds_table.csv")

# Filter for the refdes-method-stream
request_history = thredds_table[(thredds_table["refdes"] == refdes) & (thredds_table["method"] == method) & (thredds_table["stream"] == stream)]

# If the length of the request history is 0, request hasn't been previously fufilled
if len(request_history) == 0:
    thredds_url = OOINet.get_thredds_url(refdes, method, stream)#, parameters=parameters)
    # Save the request to the table
    thredds_table = thredds_table.append({
        "refdes": refdes,
        "method": method,
        "stream": stream,
        "request_date": datetime.datetime.now().strftime("YYYY-mm-ddTHH:MM:SSZ"),
        "thredds_url": thredds_url,
        "parameters": None
    }, ignore_index=True)
    
    # Save the updates to the thredds_table
    thredds_table.to_csv("../data/thredds_table.csv", index=False)
else:
    thredds_url = request_history["thredds_url"].iloc[0]
# -

thredds_url

thredds_table

# Test a storage table to save the QARTOD tables
columns = ["refdes", "method", "stream", "request_date", "thredds_url", "parameters"]
thredds_table = 







# ---
# ## Fixed Platforms
# The CTDs located on fixed depth platforms are the simplest to calculate the QARTOD tables for, since they do not require data binning based on depth or pressure. Fixed platforms for CGSN include those on the surface buoys, Near-Surface Instrument Frames (NSIFs), and Multi-Function Nodes (MFNs) of surface moorings as well as the instrument risers on Flanking Moorings. The (mostly) fixed-depth nature of these instruments allows us to simply pass in the datasets and which data variable to calculate the QARTOD tables for.
#
# The QARTOD manual on In-situ Temperature and Salinity recommend testing the temperature and the salinity values. For CTDs, this typically i

data_vars = [""]















# #### Reference Designators
# Identify all of the reference designators associated with the fixed platforms. Any reference designator with ```MOAS``` indicates mobile asset, while ```WFP``` indicates wire-following profiler, and ```HYPM``` indicates hybrid-profiler mooring.

for refdes in fixed_platforms[0:5]:
    print(refdes)

refdes = fixed_platforms[0]
refdes = "GI03FLMA-RIM01-02-CTDMOG035"

# #### Metadata
# Next, identify the metadata associated with a specific reference designator

refdes_metadata = metadata[metadata["refdes"] == refdes]
refdes_metadata

# For fixed platforms, the recovered_inst method frequently has _different_ particleKey names (the data variables in the dataset) than the recovered_host or telemetered data streams!

# #### Load Data
# When calculating the QARTOD data tables, we only want to utilize the most complete data record available for a given reference designator. We can identify this by filtering for the largest value under ```count``` which indicates the number of particles in the system for a given dataset. The more particles, the more availabe data. While most of the time this will be the recovered_inst stream, in cases of instrument loss or failure, it may be the record recovered from the mooring host computer (recovered_host) or even data which was telemetered back to shore.

# Get the directory where the data is stored

base_path = "/media/andrew/Files/Instrument_Data"
sensor = refdes.split("-")[-1][0:6]
sensor_path = "/".join((base_path, sensor))

# Load the data for recoverd instrument

method, stream, params = refdes_metadata[refdes_metadata["count"] == np.max(refdes_metadata["count"])][["method","stream","particleKey"]].iloc[0]
method, stream, params

# Generate the path to the directory with the data
data_path = "/".join((sensor_path, refdes, method, stream))
netCDF_datasets = ["/".join((data_path, x)) for x in sorted(os.listdir(data_path))]
# Filter the datasets to eliminate "blank" datasets, which sometimes get downloaded
netCDF_datasets = [dset for dset in netCDF_datasets if "blank" not in dset]
# Load the data
data = load_datasets(netCDF_datasets)
# Resort the data by time
data = data.sortby("time")

# #### Annotations
# The annotations associated with a specific reference designator may contain relevant information on the performance or reliability of the data for a given dataset. The annotations are downloaded from OOINet as a json and processed into a pandas dataframe. Each annotation may apply to the entire dataset, to a specific stream, or to a specific variable. With the downloaed annotations, we can use the information contained in the ```qcFlag``` column to translate the annotations into QC flags, which can then be used to filter out bad data. 

annotations = OOINet.get_annotations(refdes)
annotations

# Pass in the annotations and the dataset to add the annotation ```qcFlag``` values to the dataset

data = OOINet.add_annotation_qc_flag(data, annotations)
data

# Use the added ```qcFlag``` values to filter out bad (```qcFlag``` value of 9) data from the dataset

data = data.where(data.rollup_annotations_qc_results != 9, drop=True)
data

# ### Plot the data
# The code below allows for quickly plotting the key CTD data parameters. The data range plotted can also be adjusted using tstart/tend.

print(params)

pres = process.extractOne("pressure", params)[0]
temp = process.extractOne("temperature", params)[0]
cond = process.extractOne("conductivity", params)[0]
pres, temp, cond

# +
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(15,12))

tstart, tend = "2010-01-01", "2021-01-01"
#tstart, tend = "2015-05-01", "2016-01-01"

ax0.plot(data.time.loc[dict(time=slice(tstart, tend))],
         data[temp].loc[dict(time=slice(tstart, tend))],
         marker=".", linestyle="", color="tab:red")
ax0.set_ylabel(data[temp].attrs["long_name"])
ax0.set_title(data.attrs["id"])
ax0.grid()

ax1.plot(data.time.loc[dict(time=slice(tstart, tend))],
         data[cond].loc[dict(time=slice(tstart, tend))],
         marker=".", linestyle="", color="tab:blue")
ax1.set_ylabel(data[cond].attrs["long_name"])
ax1.grid()

ax2.plot(data.time.loc[dict(time=slice(tstart, tend))],
         data[pres].loc[dict(time=slice(tstart, tend))],
         marker=".", linestyle="", color="black")
ax2.set_ylabel(data[pres].attrs["long_name"])
ax2.grid()
ax2.invert_yaxis()

fig.autofmt_xdate()
# -

# ## Gross Range
# The Gross Range QARTOD test consists of two parameters: a fail range which indicates when the data is bad, and a suspect range which indicates when data is either questionable or interesting. The fail range values are set based upon the instrument/measurement and associated calibration. For example, the conductivity sensors are calibration for measurements between 0 (freshwater) and 9 (highly-saline waters). The suspect range values are calculated based on the mean of the available data $\pm$3$\sigma$.
#
# The complicating factor of not having matching data variable names (```particleKey```) for the same data depending on the stream requires us to match the data variable being used to calculate the suspect range. We can use "fuzzy-logic string matching" from the ```fuzzywuzzy``` library to calculate the percent-similarity between particleKeys to correctly match up the results with the data variables.
#
# The results for the gross range should be saved as a table as follows:
#
# | subsite | node | sensor | stream | parameter | qcConfig |
# | ------- | ---- | ------ | ------ | --------- | -------- |
# | CP01CNSM | RID27 | 03-CTDBPC000 | ctdbp_cdef_dcl_instrument_recovered| conductivity | {'qartod' {'gross_range_test': {'suspect_span': [1.21, 6.58], 'fail_span': [0, 9]}}} |
# | CP01CNSM | RID27 | 03-CTDBPC000 |	ctdbp_cdef_instrument_recovered | ctdbp_seawater_conductivity | {'qartod': {'gross_range_test': {'suspect_span': [1.21, 6.58], 'fail_span': [0, 9]}}} |
# | CP01CNSM | RID27 | 03-CTDBPC000 | ctdbp_cdef_dcl_instrument | conductivity | {'qartod': {'gross_range_test': {'suspect_span': [1.21, 6.58], 'fail_span': [0, 9]}}} |
#

# #### Single Reference Designator
# First, step through the process for a single reference designator. Start by initializing the gross range table to save the results:

gross_range_table = pd.DataFrame(columns=["subsite", "node", "sensor", "stream", "parameter", "qcConfig"])

# Build the gross range table for each of the streams and data variables (particleKeys) for the given reference designator

# +
# Split the reference designator into the relevant 
subsite, node, sensor = refdes.split("-",2)

for ind in refdes_metadata.index:
    stream, particleKeys = refdes_metadata[["stream", "particleKey"]].loc[ind]

    for pKey in particleKeys:
        if pKey == "depth":
            continue

        # Get the appropriate parameter
        param = process.extractOne(pKey, params)[0]

        # Get the correct fail ranges based on the values
        if "temp" in pKey:
            fail_min, fail_max = (-5, 35)
        elif "cond" in pKey:
            if "MO" in sensor:
                fail_min, fail_max = (0, 6)
            else:
                fail_min, fail_max = (0, 9)
        else:
            fail_min, fail_max = (0, 5000)

        # Calculate the gross_range
        gross_range = Gross_Range(fail_min, fail_max)
        gross_range.fit(data, param, sigma=3)
        gross_range.make_qcConfig()

        # Append the results
        gross_range_table = gross_range_table.append({
            "subsite": subsite,
            "node": node,
            "sensor": sensor,
            "stream": stream,
            "parameter": pKey,
            "qcConfig": gross_range.qcConfig
        }, ignore_index=True)
# -

# Results

gross_range_table.loc[0]["qcConfig"], gross_range_table.loc[1]["qcConfig"], gross_range_table.loc[2]["qcConfig"], 

# Save the results

gross_range_table.to_csv(f"../results/gross_range/{refdes}.csv", index=False)

# ---
# ## Climatology
# For the climatology QARTOD test, First, we bin the data by month and take the mean. The binned-montly means are then fit with a 1-or-2 cycle harmonic via Ordinary-Least-Squares (OLS) regression. Ranges are calculated based on the 3$\sigma$ calculated from the OLS-fitting.  
#
# The results for the climatology table should be saved as a table as follows:
#
# | subsite | node | sensor | stream | parameter | qcConfig |
# | ------- | ---- | ------ | ------ | --------- | -------- |
# | CP01CNSM | RID27 | 03-CTDBPC000 | ctdbp_cdef_dcl_instrument_recovered| {'inp': 'conductivity', 'tinp': 'time', 'zinp': None} | {'qartod': {'climatology': {'config': [{'tspan': [0, 1], 'vspan': [2.97, 4.11], 'period': 'month'}, {'tspan': [1, 2], 'vspan': [2.75, 3.89], 'period': 'month'}, {'tspan': [2, 3], 'vspan': [2.72, 3.86], 'period': 'month'}, {'tspan': [3, 4], 'vspan': [2.89, 4.03], 'period': 'month'}, {'tspan': [4, 5], 'vspan': [3.2, 4.34], 'period': 'month'}, {'tspan': [5, 6], 'vspan': [3.59, 4.73], 'period': 'month'}, {'tspan': [6, 7], 'vspan': [3.94, 5.08], 'period': 'month'}, {'tspan': [7, 8], 'vspan': [4.16, 5.3], 'period': 'month'}, {'tspan': [8, 9], 'vspan': [4.19, 5.33], 'period': 'month'}, {'tspan': [9, 10], 'vspan': [4.02, 5.16], 'period': 'month'}, {'tspan': [10, 11], 'vspan': [3.7, 4.84], 'period': 'month'}, {'tspan': [11, 12], 'vspan': [3.32, 4.46], 'period': 'month'}]}}} |
# | CP01CNSM | RID27 | 03-CTDBPC000 |	ctdbp_cdef_instrument_recovered | {'inp': 'ctdbp_seawater_conductivity', 'tinp': 'time', 'zinp': None} | {'qartod': {'climatology': {'config': [{'tspan': [0, 1],'vspan': [2.97, 4.11],'period': 'month'},{'tspan': [1, 2], 'vspan': [2.75, 3.89], 'period': 'month'},{'tspan': [2, 3], 'vspan': [2.72, 3.86], 'period': 'month'},{'tspan': [3, 4], 'vspan': [2.89, 4.03], 'period': 'month'},{'tspan': [4, 5], 'vspan': [3.2, 4.34], 'period': 'month'},{'tspan': [5, 6], 'vspan': [3.59, 4.73], 'period': 'month'},{'tspan': [6, 7], 'vspan': [3.94, 5.08], 'period': 'month'},{'tspan': [7, 8], 'vspan': [4.16, 5.3], 'period': 'month'},{'tspan': [8, 9], 'vspan': [4.19, 5.33], 'period': 'month'},{'tspan': [9, 10], 'vspan': [4.02, 5.16], 'period': 'month'},{'tspan': [10, 11], 'vspan': [3.7, 4.84], 'period': 'month'},{'tspan': [11, 12], 'vspan': [3.32, 4.46], 'period': 'month'}]}}} |
# | CP01CNSM | RID27 | 03-CTDBPC000 | ctdbp_cdef_dcl_instrument | {'inp': 'conductivity', 'tinp': 'time', 'zinp': None} | {'qartod': {'climatology': {'config': [{'tspan': [0, 1],'vspan': [2.97, 4.11],'period': 'month'},{'tspan': [1, 2], 'vspan': [2.75, 3.89], 'period': 'month'},{'tspan': [2, 3], 'vspan': [2.72, 3.86], 'period': 'month'},{'tspan': [3, 4], 'vspan': [2.89, 4.03], 'period': 'month'},{'tspan': [4, 5], 'vspan': [3.2, 4.34], 'period': 'month'},{'tspan': [5, 6], 'vspan': [3.59, 4.73], 'period': 'month'},{'tspan': [6, 7], 'vspan': [3.94, 5.08], 'period': 'month'},{'tspan': [7, 8], 'vspan': [4.16, 5.3], 'period': 'month'},{'tspan': [8, 9], 'vspan': [4.19, 5.33], 'period': 'month'},{'tspan': [9, 10], 'vspan': [4.02, 5.16], 'period': 'month'},{'tspan': [10, 11], 'vspan': [3.7, 4.84], 'period': 'month'},{'tspan': [11, 12], 'vspan': [3.32, 4.46], 'period': 'month'}]}}} |
#

# Initialize the climatology table to save the results

climatology_table = pd.DataFrame(columns=["subsite", "node", "sensor", "stream", "parameters", "qcConfig"])

# Build the gross range table for each of the streams and data variables (particleKeys) for the given reference designator

# +
# Split the reference designator into the relevant 
subsite, node, sensor = refdes.split("-",2)

for ind in refdes_metadata.index:
    stream, particleKeys = refdes_metadata[["stream", "particleKey"]].loc[ind]

    for pKey in particleKeys:
        if pKey == "depth":
            continue

        # Get the appropriate parameter
        param = process.extractOne(pKey, params)[0]

        # Calculate the climatology
        climatology = Climatology()
        climatology.fit(data, param)
        climatology.make_qcConfig()
        
        # Plot the results for checking
        fig, ax = plt.subplots(figsize=(12, 8))
        # Plot the instrument_recovered data
        s = ax.scatter(data.time, data[param], c="tab:blue")
        ax.plot(climatology.fitted_data, c="tab:red")
        ax.fill_between(climatology.fitted_data.index, climatology.fitted_data + 3*climatology.sigma,
                        climatology.fitted_data - 3*climatology.sigma, color="tab:red", alpha=0.3)
        ax.set_ylabel(data[param].attrs['long_name'])
        ax.set_title(data.attrs["id"])
        ax.grid()
        fig.autofmt_xdate()

        # Append the results
        climatology_table = climatology_table.append({
            "subsite": subsite,
            "node": node,
            "sensor": sensor,
            "stream": stream,
            "parameters": {"inp":pKey, "tinp":"time", "zinp": None},
            "qcConfig": climatology.qcConfig
        }, ignore_index=True)
# -

# Check the results

climatology_table

# Save the results of the climatology fitting:

climatology_table.to_csv(f"../results/climatology/{refdes}.csv", index=False)

# #### Script all the reference designators
# With the gross range and climatology methods developed, we can script it so that we calculate the gross range and climatology for all of the individual results

# +
base_path = "/media/andrew/Files/Instrument_Data"

for refdes in sorted(fixed_platforms):
    
    # ------------------------------------------------
    # Identify metadata associated w/selected metadata
    refdes_metadata = metadata[metadata["refdes"] == refdes]
    
    # Remove all "bad" data sources
    mask = refdes_metadata["method"].apply(lambda x: True if "bad" not in x else False)
    refdes_metadata = refdes_metadata[mask]
    if len(refdes_metadata) == 0:
        print(f"No data for {refdes}")
        continue
    else:
        print(refdes)
    
    # -------------
    # Load the data
    # First, identify which data source has the most data
    method, stream, params = refdes_metadata[refdes_metadata["count"] == np.max(refdes_metadata["count"])][["method","stream","particleKey"]].iloc[0]

    # Second, load the appropriate datasource
    sensor = refdes.split("-")[-1][0:6]
    # Generate the path to the directory with the data
    data_path = "/".join((base_path, sensor, refdes, method, stream))
    netCDF_datasets = ["/".join((data_path, x)) for x in sorted(os.listdir(data_path))]
    # Filter the datasets to eliminate "blank" datasets, which sometimes get downloaded
    netCDF_datasets = [dset for dset in netCDF_datasets if "blank" not in dset]
    # Load the data
    data = load_datasets(netCDF_datasets)
    
    # ----------------------------------------
    # Download and add the annotation qc flags
    try:
        annotations = OOINet.get_annotations(refdes)
        data = OOINet.add_annotation_qc_flag(data, annotations)
        # Drop the bad qc flags
        data = data.where(data.rollup_annotations_qc_results != 9, drop=True)
    except:
        pass
    
    # ------------------------------------------------
    # Calculate the Gross Range and Climatology values
    gross_range_table = pd.DataFrame(columns=["subsite", "node", "sensor", "stream", "parameter", "qcConfig"])
    climatology_table = pd.DataFrame(columns=["subsite", "node", "sensor", "stream", "parameters", "qcConfig"])
    
    subsite, node, sensor = refdes.split("-",2)

    for ind in refdes_metadata.index:
        stream, particleKeys = refdes_metadata[["stream", "particleKey"]].loc[ind]

        for pKey in particleKeys:
            if pKey == "depth":
                continue

            # Get the appropriate parameter
            param = process.extractOne(pKey, params)[0]

            # Get the correct fail ranges based on the values
            if "temp" in pKey:
                fail_min, fail_max = (-5, 35)
            elif "cond" in pKey:
                fail_min, fail_max = (0, 9)
            else:
                fail_min, fail_max = (0, 5000)

            # Calculate the gross_range
            gross_range = Gross_Range(fail_min, fail_max)
            gross_range.fit(data, param, sigma=3)
            gross_range.make_qcConfig()

            # Append the results
            gross_range_table = gross_range_table.append({
                "subsite": subsite,
                "node": node,
                "sensor": sensor,
                "stream": stream,
                "parameter": pKey,
                "qcConfig": gross_range.qcConfig
            }, ignore_index=True)
            
            # Filter out the "suspect" values from the data
            data = data.where((data[param] <= gross_range.suspect_max) &
                              (data[param] >= gross_range.suspect_min), 
                               drop=True)
            
            # Calculate the climatology
            climatology = Climatology()
            climatology.fit(data, param)
            climatology.make_qcConfig()

            # Append the results
            climatology_table = climatology_table.append({
                "subsite": subsite,
                "node": node,
                "sensor": sensor,
                "stream": stream,
                "parameters": {"inp":pKey, "tinp":"time", "zinp": None},
                "qcConfig": climatology.qcConfig
            }, ignore_index=True)
        
    # -------------------------------------------
    # Save the gross range and climatology tables
    gross_range_table.to_csv(f"../results/gross_range/{refdes}.csv", index=False)
    climatology_table.to_csv(f"../results/climatology/{refdes}.csv", index=False)
# -







# ---
# ## Check Results
# As a check on the results, compare the QARTOD Tables on an array or two as a check on the results

# +
import json

def parse_qcConfig(x):
    x = x.replace("None",'"None"')
    x = x.replace("'",'"')
    x = json.loads(x)
    return x

def create_qc_timeseries(ds, qcConfig):
    
    # Parse the qcConfig object
    qcdf = pd.DataFrame(qcConfig["qartod"]["climatology"]["config"])
    qcdf = qcdf.drop(columns="period")
    qcdf["tstart"] = qcdf["tspan"].apply(lambda x: int(x[0]))
    qcdf["tend"] = qcdf["tspan"].apply(lambda x: int(x[1]))
    qcdf["vmin"] = qcdf["vspan"].apply(lambda x: float(x[0]))
    qcdf["vmax"] = qcdf["vspan"].apply(lambda x: float(x[1]))
    
    tmin = pd.to_datetime(np.min(ds.time).values)
    tmax = pd.to_datetime(np.max(ds.time).values)
    
    dates = pd.date_range(start=tmin, end=tmax, freq="M", normalize=True)
    vmin = []
    vmax = []
    for t in dates:
        # Get the month value
        vmin.append(qcdf[qcdf["tend"] == t.month]["vmin"].iloc[0])
        vmax.append(qcdf[qcdf["tend"] == t.month]["vmax"].iloc[0]) 
        
    df = pd.DataFrame(data=np.array([vmin, vmax]).T, index=dates, columns=["vmin","vmax"])
    
    return df
# -



# ### Argentine Basin Array

argentine_refdes = [refdes for refdes in reference_designators if refdes.startswith("GA")]
argentine_refdes = [refdes for refdes in argentine_refdes if "MOAS" not in refdes]
argentine_refdes[0:5]

# #### GA01SUMO Surface Mooring CTD at 250 m

# +
argentine_vocab = pd.DataFrame()

for refdes in argentine_refdes:
    argentine_vocab = argentine_vocab.append(OOINet.get_vocab(refdes), ignore_index=True)
# -

argentine_vocab

# Surface Mooring CTD at 250 m
refdes = "GA01SUMO-RII11-02-CTDMOQ015"

refdes_metadata = metadata[metadata["refdes"] == refdes]
method, stream, params = refdes_metadata[refdes_metadata["count"] == np.max(refdes_metadata["count"])][["method","stream","particleKey"]].iloc[0]
method, stream, params

# +
# First, want to load my record of requests and check if I've made this specific request before
thredds_table = pd.read_csv("../data/thredds_table.csv")

# Filter for the refdes-method-stream
request_history = thredds_table[(thredds_table["refdes"] == refdes) & (thredds_table["method"] == method) & (thredds_table["stream"] == stream)]

# If the length of the request history is 0, request hasn't been previously fufilled
if len(request_history) == 0:
    thredds_url = OOINet.get_thredds_url(refdes, method, stream)
    # Save the request to the table
    thredds_table = thredds_table.append({
        "refdes": refdes,
        "method": method,
        "stream": stream,
        "request_date": datetime.datetime.now().strftime("YYYY-mm-ddTHH:MM:SSZ"),
        "thredds_url": thredds_url,
        "parameters": parameters
    }, ignore_index=True)
    
    # Save the updates to the thredds_table
    thredds_table.to_csv("../data/thredds_table.csv", index=False)
else:
    thredds_url = request_history["thredds_url"].iloc[0]
# -

thredds_table[thredds_table["refdes"] == refdes]

catalog = OOINet.get_thredds_catalog(thredds_url)
catalog = OOINet.parse_catalog(catalog, exclude=["gps"])
catalog

ga01sumo_data = OOINet.load_netCDF_datasets(sorted(catalog))
ga01sumo_data

annotations = OOINet.get_annotations(refdes)
ga01sumo_data = OOINet.add_annotation_qc_flag(ga01sumo_data, annotations)
ga01sumo_data = ga01sumo_data.where(ga01sumo_data.rollup_annotations_qc_results != 9, drop=True)

# +
# Load the gross_range and climatology
refdes = "GA01SUMO-RII11-02-CTDMOQ015"
for file in os.listdir("../results/gross_range/"):
    if refdes in file:
        ga01sumo_gross_range = pd.read_csv("../results/gross_range/" + file)

for file in os.listdir("../results/climatology/"):
    if refdes in file:
        ga01sumo_climatology = pd.read_csv("../results/climatology/" + file)
        
# Clean up the qartod table to be parseable
ga01sumo_gross_range["qcConfig"] = ga01sumo_gross_range["qcConfig"].apply(lambda x: parse_qcConfig(x))
ga01sumo_climatology["parameters"] = ga01sumo_climatology["parameters"].apply(lambda x: parse_qcConfig(x))
ga01sumo_climatology["qcConfig"] = ga01sumo_climatology["qcConfig"].apply(lambda x: parse_qcConfig(x))
# -

ga01sumo_climatology_ts = {}
for ind in ga01sumo_climatology.index:
    # Get the data
    clim = ga01sumo_climatology.loc[ind]
    # Get the qcConfig
    param = clim["parameters"]["inp"]
    qcConfig = clim["qcConfig"]
    # Generate the time series
    ts = create_qc_timeseries(ga01sumo_data, qcConfig)
    # Save the results
    ga01sumo_climatology_ts.update({
        param: ts
    })

# ### Flanking mooring A

refdes = "GA03FLMA-RIM01-02-CTDMOG046"

refdes_metadata = metadata[metadata["refdes"] == refdes]
method, stream, params = refdes_metadata[refdes_metadata["count"] == np.max(refdes_metadata["count"])][["method","stream","particleKey"]].iloc[0]
method, stream, params

# +
# First, want to load my record of requests and check if I've made this specific request before
thredds_table = pd.read_csv("../data/thredds_table.csv")

# Filter for the refdes-method-stream
request_history = thredds_table[(thredds_table["refdes"] == refdes) & (thredds_table["method"] == method) & (thredds_table["stream"] == stream)]

# If the length of the request history is 0, request hasn't been previously fufilled
if len(request_history) == 0:
    thredds_url = OOINet.get_thredds_url(refdes, method, stream)
    # Save the request to the table
    thredds_table = thredds_table.append({
        "refdes": refdes,
        "method": method,
        "stream": stream,
        "request_date": datetime.datetime.now().strftime("YYYY-mm-ddTHH:MM:SSZ"),
        "thredds_url": thredds_url,
        "parameters": None
    }, ignore_index=True)
    
    # Save the updates to the thredds_table
    thredds_table.to_csv("../data/thredds_table.csv", index=False)
else:
    thredds_url = request_history["thredds_url"].iloc[0]
# -

thredds_table[thredds_table["refdes"] == refdes]

catalog = OOINet.get_thredds_catalog(thredds_url)
catalog = OOINet.parse_catalog(catalog, exclude=["gps"])
catalog

ga03flma_data = OOINet.load_netCDF_datasets(sorted(catalog))
ga03flma_data

annotations = OOINet.get_annotations(refdes)
ga03flma_data = OOINet.add_annotation_qc_flag(ga03flma_data, annotations)
ga03flma_data = ga03flma_data.where(ga03flma_data.rollup_annotations_qc_results != 9, drop=True)

# +
# Load the gross_range and climatology
refdes = "GA03FLMA-RIM01-02-CTDMOG046"
for file in os.listdir("../results/gross_range/"):
    if refdes in file:
        ga03flma_gross_range = pd.read_csv("../results/gross_range/" + file)
        
for file in os.listdir("../results/climatology/"):
    if refdes in file:
        ga03flma_climatology = pd.read_csv("../results/climatology/" + file)
        
# Clean up the qartod table to be parseable
ga03flma_gross_range["qcConfig"] = ga03flma_gross_range["qcConfig"].apply(lambda x: parse_qcConfig(x))
ga03flma_climatology["parameters"] = ga03flma_climatology["parameters"].apply(lambda x: parse_qcConfig(x))
ga03flma_climatology["qcConfig"] = ga03flma_climatology["qcConfig"].apply(lambda x: parse_qcConfig(x))
# -

ga03flma_climatology_ts = {}
for ind in ga03flma_climatology.index:
    # Get the data
    clim = ga03flma_climatology.loc[ind]
    # Get the qcConfig
    param = clim["parameters"]["inp"]
    qcConfig = clim["qcConfig"]
    # Generate the time series
    ts = create_qc_timeseries(ga03flma_data, qcConfig)
    # Save the results
    ga03flma_climatology_ts.update({
        param: ts
    })

# ### Flanking Mooring B

refdes = "GA03FLMB-RIM01-02-CTDMOG066"

refdes_metadata = metadata[metadata["refdes"] == refdes]
method, stream, params = refdes_metadata[refdes_metadata["count"] == np.max(refdes_metadata["count"])][["method","stream","particleKey"]].iloc[0]

# +
# First, want to load my record of requests and check if I've made this specific request before
thredds_table = pd.read_csv("../data/thredds_table.csv")

# Filter for the refdes-method-stream
request_history = thredds_table[(thredds_table["refdes"] == refdes) & (thredds_table["method"] == method) & (thredds_table["stream"] == stream)]

# If the length of the request history is 0, request hasn't been previously fufilled
if len(request_history) == 0:
    thredds_url = OOINet.get_thredds_url(refdes, method, stream)
    # Save the request to the table
    thredds_table = thredds_table.append({
        "refdes": refdes,
        "method": method,
        "stream": stream,
        "request_date": datetime.datetime.now().strftime("YYYY-mm-ddTHH:MM:SSZ"),
        "thredds_url": thredds_url,
        "parameters": None
    }, ignore_index=True)
    
    # Save the updates to the thredds_table
    thredds_table.to_csv("../data/thredds_table.csv", index=False)
else:
    thredds_url = request_history["thredds_url"].iloc[0]
# -

catalog = OOINet.get_thredds_catalog(thredds_url)
catalog = OOINet.parse_catalog(catalog, exclude=["gps"])
catalog

ga03flmb_data = OOINet.load_netCDF_datasets(sorted(catalog))
ga03flmb_data

# Get annotations and filter
annotations = OOINet.get_annotations(refdes)
ga03flmb_data = OOINet.add_annotation_qc_flag(ga03flmb_data, annotations)
ga03flmb_data = ga03flmb_data.where(ga03flmb_data.rollup_annotations_qc_results != 9, drop=True)

# +
# Load the gross_range and climatology
refdes = "GA03FLMB-RIM01-02-CTDMOG066"
for file in os.listdir("../results/gross_range/"):
    if refdes in file:
        ga03flmb_gross_range = pd.read_csv("../results/gross_range/" + file)
        
for file in os.listdir("../results/climatology/"):
    if refdes in file:
        ga03flmb_climatology = pd.read_csv("../results/climatology/" + file)
        
# Clean up the qartod table to be parseable
ga03flmb_gross_range["qcConfig"] = ga03flmb_gross_range["qcConfig"].apply(lambda x: parse_qcConfig(x))
ga03flmb_climatology["parameters"] = ga03flmb_climatology["parameters"].apply(lambda x: parse_qcConfig(x))
ga03flmb_climatology["qcConfig"] = ga03flmb_climatology["qcConfig"].apply(lambda x: parse_qcConfig(x))
# -

ga03flmb_climatology_ts = {}
for ind in ga03flmb_climatology.index:
    # Get the data
    clim = ga03flmb_climatology.loc[ind]
    # Get the qcConfig
    param = clim["parameters"]["inp"]
    qcConfig = clim["qcConfig"]
    # Generate the time series
    ts = create_qc_timeseries(ga03flmb_data, qcConfig)
    # Save the results
    ga03flmb_climatology_ts.update({
        param: ts
    })

# #### Comparison Plots

os.listdir("../results/")

# +
# Plot the temperature data
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharex=True, sharey=True)

ax1.plot(ga01sumo_data.time, ga01sumo_data.ctdmo_seawater_temperature, linestyle="", marker=".", color="tab:red")
ax1.fill_between(ga01sumo_climatology_ts["ctdmo_seawater_temperature"].index,
                 ga01sumo_climatology_ts["ctdmo_seawater_temperature"]["vmin"],
                 ga01sumo_climatology_ts["ctdmo_seawater_temperature"]["vmax"],
                 alpha=0.3, color="tab:red")
ax1.set_ylabel(ga01sumo_data.ctdmo_seawater_temperature.attrs["long_name"], fontsize=14)
ax1.grid()
ax1.set_title("-".join(ga01sumo_data.attrs["id"].split("-")[0:4]), fontsize=14)

ax2.plot(ga03flma_data.time, ga03flma_data.ctdmo_seawater_temperature, linestyle="", marker=".", color="tab:red")
ax2.fill_between(ga03flma_climatology_ts["ctdmo_seawater_temperature"].index,
                 ga03flma_climatology_ts["ctdmo_seawater_temperature"]["vmin"],
                 ga03flma_climatology_ts["ctdmo_seawater_temperature"]["vmax"],
                 alpha=0.3, color="tab:red")
ax2.set_ylabel(ga03flma_data.ctdmo_seawater_temperature.attrs["long_name"], fontsize=14)
ax2.grid()
ax2.set_title("-".join(ga03flma_data.attrs["id"].split("-")[0:4]), fontsize=14)

ax3.plot(ga03flmb_data.time, ga03flmb_data.ctdmo_seawater_temperature, linestyle="", marker=".", color="tab:red")
# ax3.scatter(ga03flmb_data.time, ga03flmb_data.ctdmo_seawater_temperature, c=ga03flmb_data.deployment)
ax3.fill_between(ga03flmb_climatology_ts["ctdmo_seawater_temperature"].index,
                 ga03flmb_climatology_ts["ctdmo_seawater_temperature"]["vmin"],
                 ga03flmb_climatology_ts["ctdmo_seawater_temperature"]["vmax"],
                 alpha=0.3, color="tab:red")
ax3.set_ylabel(ga03flmb_data.ctdmo_seawater_temperature.attrs["long_name"], fontsize=14)
ax3.grid()
ax3.set_title("-".join(ga03flmb_data.attrs["id"].split("-")[0:4]), fontsize=14)

fig.autofmt_xdate()

# -

ga01sumo_monthly = ga01sumo_data.sortby("time").resample(time="M").mean()
ga03flma_monthly = ga03flma_data.sortby("time").resample(time="M").mean()
ga03flmb_monthly = ga03flmb_data.sortby("time").resample(time="M").mean()

# +
# Plot the temperature data
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharex=True, sharey=True)

ax1.plot(ga01sumo_monthly.time, ga01sumo_monthly.ctdmo_seawater_temperature, linestyle="-", marker=".", color="tab:red")
ax1.fill_between(ga01sumo_climatology_ts["ctdmo_seawater_temperature"].index,
                 ga01sumo_climatology_ts["ctdmo_seawater_temperature"]["vmin"],
                 ga01sumo_climatology_ts["ctdmo_seawater_temperature"]["vmax"],
                 alpha=0.3, color="tab:red")
ax1.set_ylabel(ga01sumo_data.ctdmo_seawater_temperature.attrs["long_name"], fontsize=14)
ax1.grid()
ax1.set_title("-".join(ga01sumo_data.attrs["id"].split("-")[0:4]), fontsize=14)

ax2.plot(ga03flma_monthly.time, ga03flma_monthly.ctdmo_seawater_temperature, linestyle="-", marker=".", color="tab:red")
ax2.fill_between(ga03flma_climatology_ts["ctdmo_seawater_temperature"].index,
                 ga03flma_climatology_ts["ctdmo_seawater_temperature"]["vmin"],
                 ga03flma_climatology_ts["ctdmo_seawater_temperature"]["vmax"],
                 alpha=0.3, color="tab:red")
ax2.set_ylabel(ga03flma_data.ctdmo_seawater_temperature.attrs["long_name"], fontsize=14)
ax2.grid()
ax2.set_title("-".join(ga03flma_data.attrs["id"].split("-")[0:4]), fontsize=14)

ax3.plot(ga03flmb_monthly.time, ga03flmb_monthly.ctdmo_seawater_temperature, linestyle="-", marker=".", color="tab:red")
ax3.fill_between(ga03flmb_climatology_ts["ctdmo_seawater_temperature"].index,
                 ga03flmb_climatology_ts["ctdmo_seawater_temperature"]["vmin"],
                 ga03flmb_climatology_ts["ctdmo_seawater_temperature"]["vmax"],
                 alpha=0.3, color="tab:red")
ax3.set_ylabel(ga03flmb_data.ctdmo_seawater_temperature.attrs["long_name"], fontsize=14)
ax3.grid()
ax3.set_title("-".join(ga03flmb_data.attrs["id"].split("-")[0:4]), fontsize=14)

fig.autofmt_xdate()

# +
# Plot the temperature data
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharex=True, sharey=True)

ax1.plot(ga01sumo_data.time, ga01sumo_data.ctdmo_seawater_conductivity, linestyle="", marker=".", color="tab:blue")
ax1.fill_between(ga01sumo_climatology_ts["ctdmo_seawater_conductivity"].index,
                 ga01sumo_climatology_ts["ctdmo_seawater_conductivity"]["vmin"],
                 ga01sumo_climatology_ts["ctdmo_seawater_conductivity"]["vmax"],
                 alpha=0.3, color="tab:blue")
ax1.set_ylabel(ga01sumo_data.ctdmo_seawater_conductivity.attrs["long_name"], fontsize=14)
ax1.grid()
ax1.set_title("-".join(ga01sumo_data.attrs["id"].split("-")[0:4]), fontsize=14)

ax2.plot(ga03flma_data.time, ga03flma_data.ctdmo_seawater_conductivity, linestyle="", marker=".", color="tab:blue")
ax2.fill_between(ga03flma_climatology_ts["ctdmo_seawater_conductivity"].index,
                 ga03flma_climatology_ts["ctdmo_seawater_conductivity"]["vmin"],
                 ga03flma_climatology_ts["ctdmo_seawater_conductivity"]["vmax"],
                 alpha=0.3, color="tab:blue")
ax2.set_ylabel(ga03flma_data.ctdmo_seawater_conductivity.attrs["long_name"], fontsize=14)
ax2.grid()
ax2.set_title("-".join(ga03flma_data.attrs["id"].split("-")[0:4]), fontsize=14)

ax3.plot(ga03flmb_data.time, ga03flmb_data.ctdmo_seawater_conductivity, linestyle="", marker=".", color="tab:blue")
ax3.fill_between(ga03flmb_climatology_ts["ctdmo_seawater_conductivity"].index,
                 ga03flmb_climatology_ts["ctdmo_seawater_conductivity"]["vmin"],
                 ga03flmb_climatology_ts["ctdmo_seawater_conductivity"]["vmax"],
                 alpha=0.3, color="tab:blue")
ax3.set_ylabel(ga03flmb_data.ctdmo_seawater_conductivity.attrs["long_name"], fontsize=14)
ax3.grid()
ax3.set_title("-".join(ga03flmb_data.attrs["id"].split("-")[0:4]), fontsize=14)
              
fig.autofmt_xdate()
# +
# Plot the temperature data
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharex=True, sharey=True)

ax1.plot(ga01sumo_monthly.time, ga01sumo_monthly.ctdmo_seawater_conductivity, linestyle="-", marker=".", color="tab:blue")
ax1.fill_between(ga01sumo_climatology_ts["ctdmo_seawater_conductivity"].index,
                 ga01sumo_climatology_ts["ctdmo_seawater_conductivity"]["vmin"],
                 ga01sumo_climatology_ts["ctdmo_seawater_conductivity"]["vmax"],
                 alpha=0.3, color="tab:blue")
ax1.set_ylabel(ga01sumo_data.ctdmo_seawater_conductivity.attrs["long_name"], fontsize=14)
ax1.grid()
ax1.set_title("-".join(ga01sumo_data.attrs["id"].split("-")[0:4]), fontsize=14)

ax2.plot(ga03flma_monthly.time, ga03flma_monthly.ctdmo_seawater_conductivity, linestyle="-", marker=".", color="tab:blue")
ax2.fill_between(ga03flma_climatology_ts["ctdmo_seawater_conductivity"].index,
                 ga03flma_climatology_ts["ctdmo_seawater_conductivity"]["vmin"],
                 ga03flma_climatology_ts["ctdmo_seawater_conductivity"]["vmax"],
                 alpha=0.3, color="tab:blue")
ax2.set_ylabel(ga03flma_data.ctdmo_seawater_conductivity.attrs["long_name"], fontsize=14)
ax2.grid()
ax2.set_title("-".join(ga03flma_data.attrs["id"].split("-")[0:4]), fontsize=14)

ax3.plot(ga03flmb_monthly.time, ga03flmb_monthly.ctdmo_seawater_conductivity, linestyle="-", marker=".", color="tab:blue")
ax3.fill_between(ga03flmb_climatology_ts["ctdmo_seawater_conductivity"].index,
                 ga03flmb_climatology_ts["ctdmo_seawater_conductivity"]["vmin"],
                 ga03flmb_climatology_ts["ctdmo_seawater_conductivity"]["vmax"],
                 alpha=0.3, color="tab:blue")
ax3.set_ylabel(ga03flmb_data.ctdmo_seawater_conductivity.attrs["long_name"], fontsize=14)
ax3.grid()
ax3.set_title("-".join(ga03flmb_data.attrs["id"].split("-")[0:4]), fontsize=14)
              
fig.autofmt_xdate()
# -

# ### Calculate correlations between the time series


ga01sumo_df = ga01sumo_data["ctdmo_seawater_conductivity"].to_dataframe().resample("H").mean()
ga03flma_df = ga03flma_data["ctdmo_seawater_conductivity"].to_dataframe().resample("H").mean()
ga03flmb_df = ga03flmb_data["ctdmo_seawater_conductivity"].to_dataframe().resample("H").mean()

ga01sumo_df.rename(columns={"ctdmo_seawater_conductivity":"ga01sumo"}, inplace=True)
ga03flma_df.rename(columns={"ctdmo_seawater_conductivity":"ga03flma"}, inplace=True)
ga03flmb_df.rename(columns={"ctdmo_seawater_conductivity":"ga03flmb"}, inplace=True)

# Calculate the Pearson correlation coefficient between the thre
df = ga01sumo_df.merge(ga03flma_df, left_index=True, right_index=True).merge(ga03flmb_df, left_index=True, right_index=True)
df

import scipy.stats as stats
import seaborn as sns

# Calculate the correlation between the different time series
df.corr()

r, p = stats.pearsonr(df.dropna()["ga01sumo"], df.dropna()["ga03flma"])
print(f"Scipy computed Pearson r: {r} and p-value: {p}")

f, ax = plt.subplots(figsize=(16, 8))
df[["ga03flma","ga03flmb"]].rolling(window=24,center=True).median().plot(ax=ax)
ax.set(xlabel="time", ylabel="Pearson r")
ax.grid()

# Plot the moving window synchrony (using a window of 1-day)
window_size=12 # This is one day
# Interpolate missing data
df_interpolated = df.interpolate()

rolling_r = df_interpolated["ga03flma"].rolling(window=window_size, center=True).corr(df_interpolated["ga03flmb"])

f, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 12), sharex=True)
df[["ga03flma","ga03flmb"]].rolling(window=window_size, center=True).median().plot(ax=ax[0])
ax[0].set(xlabel="Time", ylabel="Temperature")
rolling_r.plot(ax=ax[1])
ax[1].set(xlabel="Time", ylabel="Pearson r")


# +
# Use the event in July 2016 to test lag correlations at ARgentine basin.
# -

df_2016 = df[slice("2016-06-15","2016-09-01")]
df_2016.plot()


def crosscorr(datax, datay, lag=0, wrap=False):
    """Lag-N corss correlation.
    Shifted data filled with NaNs
    
    Parameters
    ----------
    lag: (int)
        default 0
    datax, datay: (pandas.Series)
    """
    
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))


# +
datax = df["ga03flma"]
datay = df["ga03flmb"]
rs = [crosscorr(datax, datay, lag) for lag in range(-24*30,24*30+1)]
xvals = np.linspace(-(len(rs)-1)/2, (len(rs)-1)/2, len(rs))
offset = np.ceil(len(rs)/2)-np.argmax(rs)

f, ax = plt.subplots(figsize=(12, 8))
ax.plot(xvals, rs)
ax.axvline(xvals[int(np.ceil(len(rs)/2)-1)], color="k", linestyle="--", label="Center")
ax.axvline(xvals[np.argmax(rs)-1], color="r", linestyle="--", label="Peak Synchrony")
ax.set_title(f"Offset = {offset} hours\nga03flma leads <> ga03flmb leads", fontsize=16)
ax.set_ylabel("Pearson R-coefficient", fontsize=12)
ax.set_xlabel("Lag (Hours)", fontsize=12)
ax.grid()
ax.legend(fontsize=12)

# +
# Look at the climatology and test the lags
# -

dfsumo = ga01sumo_climatology_ts["ctdmo_seawater_temperature"].resample("H").interpolate()
dfflma = ga03flma_climatology_ts["ctdmo_seawater_temperature"].resample("H").interpolate()
dfflmb = ga03flmb_climatology_ts["ctdmo_seawater_temperature"].resample("H").interpolate()

dfsumo.rename(columns={"vmin":"ga01sumo_min", "vmax":"ga01sumo_max"}, inplace=True)
dfflma.rename(columns={"vmin":"ga03flma_min", "vmax":"ga03flma_max"}, inplace=True)
dfflmb.rename(columns={"vmin":"ga03flmb_min", "vmax":"ga03flmb_max"}, inplace=True)

# Calculate the Pearson correlation coefficient between the thre
df_clim = dfsumo.merge(dfflma, left_index=True, right_index=True).merge(dfflmb, left_index=True, right_index=True)
df_clim = df_clim.drop(columns=[x for x in df_clim.columns if "max" in x])
df_clim

df_clim.corr()

df.corr()

# +
# Ca
datax = df_clim["ga03flma_min"]
datay = df_clim["ga03flmb_min"]
rs = [crosscorr(datax, datay, lag) for lag in range(-24*30*3,24*30*3+1)]
xvals = np.linspace(-(len(rs)-1)/2, (len(rs)-1)/2, len(rs))
offset = np.ceil(len(rs)/2)-np.argmax(rs)

f, ax = plt.subplots(figsize=(12, 8))
ax.plot(xvals, rs)
ax.axvline(xvals[int(np.ceil(len(rs)/2)-1)], color="k", linestyle="--", label="Center")
ax.axvline(xvals[np.argmax(rs)-1], color="r", linestyle="--", label="Peak Synchrony")
ax.set_title(f"Offset = {offset} hours\nga03flma leads <> ga03flmb leads", fontsize=16)
ax.set_ylabel("Pearson R-coefficient", fontsize=12)
ax.set_xlabel("Lag (Hours)", fontsize=12)
ax.grid()
ax.legend(fontsize=12)
# -

# ### Annotations

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

# +
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12,8), sharex=True, sharey=True)

sns.distplot(ga01sumo_data.ctdmo_seawater_temperature, hist=True,
             hist_kws={"edgecolor": "k", "linewidth": 1,  "color": "tab:red"}, kde_kws={"linewidth": 3},
             ax=ax1)
ax1.grid()
ax1.set_title("-".join(ga01sumo_data.attrs["id"].split("-")[0:4]))
#ax1.set_axisbelow(True)

sns.distplot(ga03flma_data.ctdmo_seawater_temperature, hist=True,
            hist_kws={"edgecolor": "k", "linewidth": 1, "color": "tab:red"}, kde_kws={"linewidth": 3},
            ax=ax2)
ax2.grid()
ax2.set_title("-".join(ga03flma_data.attrs["id"].split("-")[0:4]))

sns.distplot(ga03flmb_data.ctdmo_seawater_temperature, hist=True,
            hist_kws={"edgecolor": "k", "linewidth": 1, "color": "tab:red"}, kde_kws={"linewidth": 3},
            ax=ax3)
ax3.set_title("-".join(ga03flmb_data.attrs["id"].split("-")[0:4]))
ax3.grid()



# -

# Calculate the monthly correlations
dfsumo = ga01sumo_monthly["ctdmo_seawater_temperature"].to_dataframe().rename(columns={"ctdmo_seawater_temperature":"ga01sumo"})
dfflma = ga03flma_monthly["ctdmo_seawater_temperature"].to_dataframe().rename(columns={"ctdmo_seawater_temperature":"ga03flma"})
dfflmb = ga03flmb_monthly["ctdmo_seawater_temperature"].to_dataframe().rename(columns={"ctdmo_seawater_temperature":"ga03flmb"})

df_monthly_temp = dfsumo.merge(dfflma, left_index=True, right_index=True).merge(dfflmb, left_index=True, right_index=True)
df_monthly_temp.corr()

# +
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12,8), sharex=True, sharey=True)

sns.distplot(ga01sumo_data.ctdmo_seawater_conductivity, hist=True,
             hist_kws={"edgecolor": "k", "linewidth": 1,  "color": "tab:blue"}, kde_kws={"linewidth": 3},
             ax=ax1)
ax1.grid()
ax1.set_title("-".join(ga01sumo_data.attrs["id"].split("-")[0:4]))
#ax1.set_axisbelow(True)

sns.distplot(ga03flma_data.ctdmo_seawater_conductivity, hist=True,
            hist_kws={"edgecolor": "k", "linewidth": 1, "color": "tab:blue"}, kde_kws={"linewidth": 3},
            ax=ax2)
ax2.grid()
ax2.set_title("-".join(ga03flma_data.attrs["id"].split("-")[0:4]))

sns.distplot(ga03flmb_data.ctdmo_seawater_conductivity, hist=True,
            hist_kws={"edgecolor": "k", "linewidth": 1, "color": "tab:blue"}, kde_kws={"linewidth": 3},
            ax=ax3)
ax3.set_title("-".join(ga03flmb_data.attrs["id"].split("-")[0:4]))
ax3.grid()
# -

dfsumo = ga01sumo_monthly["ctdmo_seawater_conductivity"].to_dataframe().rename(columns={"ctdmo_seawater_conductivity":"ga01sumo"})
dfflma = ga03flma_monthly["ctdmo_seawater_conductivity"].to_dataframe().rename(columns={"ctdmo_seawater_conductivity":"ga03flma"})
dfflmb = ga03flmb_monthly["ctdmo_seawater_conductivity"].to_dataframe().rename(columns={"ctdmo_seawater_conductivity":"ga03flmb"})

df_monthly_cond = dfsumo.merge(dfflma, left_index=True, right_index=True).merge(dfflmb, left_index=True, right_index=True)
df_monthly_cond.corr()

# +
fig, ax = plt.subplots(figsize=(12, 6))

ax = sns.boxplot(ga01sumo_data.ctdmo_seawater_temperature)
# -

ga01sumo_gross_range.loc[4]["qcConfig"]


# ### Alternative Climatology Fit 1: Remove Outliers using IQR
#
# Next, I'm going to test the effect on the climatological fits and the time-series correlations based on removing data using the range test of (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR). This should be less susceptible to outlier events, and should hopefully help synchronize the seasonal fits.

def calc_interquartile_stats(ds, param):
    
    Q1 = np.percentile(ds[param], 25)
    Q3 = np.percentile(ds[param], 75)
    IQR = Q3 - Q1
    Qmin = Q1 - 1.5*IQR
    Qmax = Q3 + 1.5*IQR
    
    return {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Qmin": Qmin,
        "Qmax": Qmax
    }


ga01sumo_iqr = {
    "ctdmo_seawater_temperature": calc_interquartile_stats(ga01sumo_data, "ctdmo_seawater_temperature"),
    "ctdmo_seawater_conductivity": calc_interquartile_stats(ga01sumo_data, "ctdmo_seawater_conductivity")
}
ga03flma_iqr = {
    "ctdmo_seawater_temperature": calc_interquartile_stats(ga03flma_data, "ctdmo_seawater_temperature"),
    "ctdmo_seawater_conductivity": calc_interquartile_stats(ga03flma_data, "ctdmo_seawater_conductivity")
}
ga03flmb_iqr = {
    "ctdmo_seawater_temperature": calc_interquartile_stats(ga03flmb_data, "ctdmo_seawater_temperature"),
    "ctdmo_seawater_conductivity": calc_interquartile_stats(ga03flmb_data, "ctdmo_seawater_conductivity")
}

# Filter the data
# Temperature
ga01sumo_temp = ga01sumo_data
ga01sumo_temp = ga01sumo_temp.where((ga01sumo_temp.ctdmo_seawater_temperature >= ga01sumo_iqr["ctdmo_seawater_temperature"]["Qmin"])
                                    & (ga01sumo_temp.ctdmo_seawater_temperature <= ga01sumo_iqr["ctdmo_seawater_temperature"]["Qmax"]),
                                    drop=True)
# Conductivity
ga01sumo_cond = ga01sumo_data
ga01sumo_cond = ga01sumo_cond.where((ga01sumo_cond.ctdmo_seawater_conductivity >= ga01sumo_iqr["ctdmo_seawater_conductivity"]["Qmin"])
                                    & (ga01sumo_cond.ctdmo_seawater_conductivity <= ga01sumo_iqr["ctdmo_seawater_conductivity"]["Qmax"]),
                                    drop=True)

# +
# Calculate the climatologies for the filtered data
# Temperature
climatology = Climatology()
climatology.fit(ga01sumo_temp, "ctdmo_seawater_temperature")
ga01sumo_temp_clim = pd.DataFrame(climatology.fitted_data, columns=["mean"])
ga01sumo_temp_clim["sigma"] = climatology.sigma
ga01sumo_temp_clim["vmin"] = np.floor((ga01sumo_temp_clim["mean"] - 3*ga01sumo_temp_clim["sigma"])*100)/100
ga01sumo_temp_clim["vmax"] = np.ceil((ga01sumo_temp_clim["mean"] + 3*ga01sumo_temp_clim["sigma"])*100)/100

# Conductivity
climatology = Climatology()
climatology.fit(ga01sumo_cond, "ctdmo_seawater_conductivity")
ga01sumo_cond_clim = pd.DataFrame(climatology.fitted_data, columns=["mean"])
ga01sumo_cond_clim["sigma"] = climatology.sigma
ga01sumo_cond_clim["vmin"] = np.floor((ga01sumo_cond_clim["mean"] - 3*ga01sumo_cond_clim["sigma"])*100)/100
ga01sumo_cond_clim["vmax"] = np.ceil((ga01sumo_cond_clim["mean"] + 3*ga01sumo_cond_clim["sigma"])*100)/100
# -

# Filter the data
# Temperature
ga03flma_temp = ga03flma_data
ga03flma_temp = ga03flma_temp.where((ga03flma_temp.ctdmo_seawater_temperature >= ga03flma_iqr["ctdmo_seawater_temperature"]["Qmin"])
                                    & (ga03flma_temp.ctdmo_seawater_temperature <= ga03flma_iqr["ctdmo_seawater_temperature"]["Qmax"]),
                                    drop=True)
# Conductivity
ga03flma_cond = ga03flma_data
ga03flma_cond = ga03flma_cond.where((ga03flma_cond.ctdmo_seawater_conductivity >= ga03flma_iqr["ctdmo_seawater_conductivity"]["Qmin"])
                                    & (ga03flma_cond.ctdmo_seawater_conductivity <= ga03flma_iqr["ctdmo_seawater_conductivity"]["Qmax"]),
                                    drop=True)

# +
# Calculate the climatologies for the filtered data
# Temperature
climatology = Climatology()
climatology.fit(ga03flma_temp, "ctdmo_seawater_temperature")
ga03flma_temp_clim = pd.DataFrame(climatology.fitted_data, columns=["mean"])
ga03flma_temp_clim["sigma"] = climatology.sigma
ga03flma_temp_clim["vmin"] = np.floor((ga03flma_temp_clim["mean"] - 3*ga03flma_temp_clim["sigma"])*100)/100
ga03flma_temp_clim["vmax"] = np.ceil((ga03flma_temp_clim["mean"] + 3*ga03flma_temp_clim["sigma"])*100)/100

# Conductivity
climatology = Climatology()
climatology.fit(ga03flma_cond, "ctdmo_seawater_conductivity")
ga03flma_cond_clim = pd.DataFrame(climatology.fitted_data, columns=["mean"])
ga03flma_cond_clim["sigma"] = climatology.sigma
ga03flma_cond_clim["vmin"] = np.floor((ga03flma_cond_clim["mean"] - 3*ga03flma_cond_clim["sigma"])*100)/100
ga03flma_cond_clim["vmax"] = np.ceil((ga03flma_cond_clim["mean"] + 3*ga03flma_cond_clim["sigma"])*100)/100
# -

# Filter the data
# Temperature
ga03flmb_temp = ga03flmb_data
ga03flmb_temp = ga03flmb_temp.where((ga03flmb_temp.ctdmo_seawater_temperature >= ga03flmb_iqr["ctdmo_seawater_temperature"]["Qmin"])
                                    & (ga03flmb_temp.ctdmo_seawater_temperature <= ga03flmb_iqr["ctdmo_seawater_temperature"]["Qmax"]),
                                    drop=True)
# Conductivity
ga03flmb_cond = ga03flmb_data
ga03flmb_cond = ga03flmb_cond.where((ga03flmb_cond.ctdmo_seawater_conductivity >= ga03flmb_iqr["ctdmo_seawater_conductivity"]["Qmin"])
                                    & (ga03flmb_cond.ctdmo_seawater_conductivity <= ga03flmb_iqr["ctdmo_seawater_conductivity"]["Qmax"]),
                                    drop=True)

# +
# Calculate the climatologies for the filtered data
# Temperature
climatology = Climatology()
climatology.fit(ga03flmb_temp, "ctdmo_seawater_temperature")
ga03flmb_temp_clim = pd.DataFrame(climatology.fitted_data, columns=["mean"])
ga03flmb_temp_clim["sigma"] = climatology.sigma
ga03flmb_temp_clim["vmin"] = np.floor((ga03flmb_temp_clim["mean"] - 3*ga03flmb_temp_clim["sigma"])*100)/100
ga03flmb_temp_clim["vmax"] = np.ceil((ga03flmb_temp_clim["mean"] + 3*ga03flmb_temp_clim["sigma"])*100)/100

# Conductivity
climatology = Climatology()
climatology.fit(ga03flmb_cond, "ctdmo_seawater_conductivity")
ga03flmb_cond_clim = pd.DataFrame(climatology.fitted_data, columns=["mean"])
ga03flmb_cond_clim["sigma"] = climatology.sigma
ga03flmb_cond_clim["vmin"] = np.floor((ga03flmb_cond_clim["mean"] - 3*ga03flmb_cond_clim["sigma"])*100)/100
ga03flmb_cond_clim["vmax"] = np.ceil((ga03flmb_cond_clim["mean"] + 3*ga03flmb_cond_clim["sigma"])*100)/100
# -

# #### Plot the results

# +
# Plot the temperature data
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharex=True, sharey=True)

ax1.plot(ga01sumo_cond.time, ga01sumo_cond.ctdmo_seawater_conductivity, linestyle="", marker=".", color="tab:blue")
ax1.fill_between(ga01sumo_cond_clim.index,
                 ga01sumo_cond_clim["vmin"],
                 ga01sumo_cond_clim["vmax"],
                 alpha=0.3, color="tab:blue")
ax1.set_ylabel(ga01sumo_cond.ctdmo_seawater_conductivity.attrs["long_name"], fontsize=14)
ax1.grid()
ax1.set_title("-".join(ga01sumo_data.attrs["id"].split("-")[0:4]), fontsize=14)

ax2.plot(ga03flma_cond.time, ga03flma_cond.ctdmo_seawater_conductivity, linestyle="", marker=".", color="tab:blue")
ax2.fill_between(ga03flma_cond_clim.index,
                 ga03flma_cond_clim["vmin"],
                 ga03flma_cond_clim["vmax"],
                 alpha=0.3, color="tab:blue")
ax2.set_ylabel(ga03flma_cond.ctdmo_seawater_conductivity.attrs["long_name"], fontsize=14)
ax2.grid()
ax2.set_title("-".join(ga03flma_cond.attrs["id"].split("-")[0:4]), fontsize=14)

ax3.plot(ga03flmb_cond.time, ga03flmb_cond.ctdmo_seawater_conductivity, linestyle="", marker=".", color="tab:blue")
ax3.fill_between(ga03flmb_cond_clim.index,
                 ga03flmb_cond_clim["vmin"],
                 ga03flmb_cond_clim["vmax"],
                 alpha=0.3, color="tab:blue")
ax3.set_ylabel(ga03flmb_cond.ctdmo_seawater_conductivity.attrs["long_name"], fontsize=14)
ax3.grid()
ax3.set_title("-".join(ga03flmb_cond.attrs["id"].split("-")[0:4]), fontsize=14)
              
fig.autofmt_xdate()

# +
# Plot the temperature data
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharex=True, sharey=True)

ax1.plot(ga01sumo_temp.time, ga01sumo_temp.ctdmo_seawater_temperature, linestyle="", marker=".", color="tab:red")
ax1.fill_between(ga01sumo_temp_clim.index,
                 ga01sumo_temp_clim["vmin"],
                 ga01sumo_temp_clim["vmax"],
                 alpha=0.3, color="tab:red")
ax1.set_ylabel(ga01sumo_temp.ctdmo_seawater_temperature.attrs["long_name"], fontsize=14)
ax1.grid()
ax1.set_title("-".join(ga01sumo_temp.attrs["id"].split("-")[0:4]), fontsize=14)

ax2.plot(ga03flma_temp.time, ga03flma_temp.ctdmo_seawater_temperature, linestyle="", marker=".", color="tab:red")
ax2.fill_between(ga03flma_temp_clim.index,
                 ga03flma_temp_clim["vmin"],
                 ga03flma_temp_clim["vmax"],
                 alpha=0.3, color="tab:red")
ax2.set_ylabel(ga03flma_cond.ctdmo_seawater_conductivity.attrs["long_name"], fontsize=14)
ax2.grid()
ax2.set_title("-".join(ga03flma_temp.attrs["id"].split("-")[0:4]), fontsize=14)

ax3.plot(ga03flmb_temp.time, ga03flmb_temp.ctdmo_seawater_temperature, linestyle="", marker=".", color="tab:red")
ax3.fill_between(ga03flmb_temp_clim.index,
                 ga03flmb_temp_clim["vmin"],
                 ga03flmb_temp_clim["vmax"],
                 alpha=0.3, color="tab:red")
ax3.set_ylabel(ga03flmb_cond.ctdmo_seawater_conductivity.attrs["long_name"], fontsize=14)
ax3.grid()
ax3.set_title("-".join(ga03flmb_temp.attrs["id"].split("-")[0:4]), fontsize=14)
              
fig.autofmt_xdate()

# +
# Plot the temperature data
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharex=True, sharey=True)

ax1.plot(ga01sumo_cond.time, ga01sumo_cond.ctdmo_seawater_conductivity, linestyle="", marker=".", color="tab:blue")
ax1.fill_between(ga01sumo_cond_clim.index,
                 ga01sumo_cond_clim["vmin"],
                 ga01sumo_cond_clim["vmax"],
                 alpha=0.3, color="tab:blue")
ax1.set_ylabel(ga01sumo_cond.ctdmo_seawater_conductivity.attrs["long_name"], fontsize=14)
ax1.grid()
ax1.set_title("-".join(ga01sumo_data.attrs["id"].split("-")[0:4]), fontsize=14)

ax2.plot(ga03flma_cond.time, ga03flma_cond.ctdmo_seawater_conductivity, linestyle="", marker=".", color="tab:blue")
ax2.fill_between(ga03flma_cond_clim.index,
                 ga03flma_cond_clim["vmin"],
                 ga03flma_cond_clim["vmax"],
                 alpha=0.3, color="tab:blue")
ax2.set_ylabel(ga03flma_cond.ctdmo_seawater_conductivity.attrs["long_name"], fontsize=14)
ax2.grid()
ax2.set_title("-".join(ga03flma_cond.attrs["id"].split("-")[0:4]), fontsize=14)

ax3.plot(ga03flmb_cond.time, ga03flmb_cond.ctdmo_seawater_conductivity, linestyle="", marker=".", color="tab:blue")
ax3.fill_between(ga03flmb_cond_clim.index,
                 ga03flmb_cond_clim["vmin"],
                 ga03flmb_cond_clim["vmax"],
                 alpha=0.3, color="tab:blue")
ax3.set_ylabel(ga03flmb_cond.ctdmo_seawater_conductivity.attrs["long_name"], fontsize=14)
ax3.grid()
ax3.set_title("-".join(ga03flmb_cond.attrs["id"].split("-")[0:4]), fontsize=14)
              
fig.autofmt_xdate()
# -

# ### Alternative 2: Using three-sigma filter

gross_range_temp

# Filter the data
# Temperature
ga01sumo_temp = ga01sumo_data
ga01sumo_temp = ga01sumo_temp.where((ga01sumo_temp.ctdmo_seawater_temperature >= gross_range_temp["ga01sumo"]["3-sigma"][0])
                                    & (ga01sumo_temp.ctdmo_seawater_temperature <= gross_range_temp["ga01sumo"]["3-sigma"][1]),
                                    drop=True)
# Conductivity
ga01sumo_cond = ga01sumo_data
ga01sumo_cond = ga01sumo_cond.where((ga01sumo_cond.ctdmo_seawater_conductivity >= gross_range_cond["ga01sumo"]["3-sigma"][0])
                                    & (ga01sumo_cond.ctdmo_seawater_conductivity <= gross_range_cond["ga01sumo"]["3-sigma"][1]),
                                    drop=True)

# +
# Calculate the climatologies for the filtered data
# Temperature
climatology = Climatology()
climatology.fit(ga01sumo_temp, "ctdmo_seawater_temperature")
ga01sumo_temp_clim = pd.DataFrame(climatology.fitted_data, columns=["mean"])
ga01sumo_temp_clim["sigma"] = climatology.sigma
ga01sumo_temp_clim["vmin"] = np.floor((ga01sumo_temp_clim["mean"] - 3*ga01sumo_temp_clim["sigma"])*100)/100
ga01sumo_temp_clim["vmax"] = np.ceil((ga01sumo_temp_clim["mean"] + 3*ga01sumo_temp_clim["sigma"])*100)/100

# Conductivity
climatology = Climatology()
climatology.fit(ga01sumo_cond, "ctdmo_seawater_conductivity")
ga01sumo_cond_clim = pd.DataFrame(climatology.fitted_data, columns=["mean"])
ga01sumo_cond_clim["sigma"] = climatology.sigma
ga01sumo_cond_clim["vmin"] = np.floor((ga01sumo_cond_clim["mean"] - 3*ga01sumo_cond_clim["sigma"])*100)/100
ga01sumo_cond_clim["vmax"] = np.ceil((ga01sumo_cond_clim["mean"] + 3*ga01sumo_cond_clim["sigma"])*100)/100
# -

# Filter the data
# Temperature
ga03flma_temp = ga03flma_data
ga03flma_temp = ga03flma_temp.where((ga03flma_temp.ctdmo_seawater_temperature >= gross_range_temp["ga03flma"]["3-sigma"][0])
                                    & (ga03flma_temp.ctdmo_seawater_temperature <= gross_range_temp["ga03flma"]["3-sigma"][1]),
                                    drop=True)
# Conductivity
ga03flma_cond = ga03flma_data
ga03flma_cond = ga03flma_cond.where((ga03flma_cond.ctdmo_seawater_conductivity >= gross_range_cond["ga03flma"]["3-sigma"][0])
                                    & (ga03flma_cond.ctdmo_seawater_conductivity <= gross_range_cond["ga03flma"]["3-sigma"][1]),
                                    drop=True)

# +
# Calculate the climatologies for the filtered data
# Temperature
climatology = Climatology()
climatology.fit(ga03flma_temp, "ctdmo_seawater_temperature")
ga03flma_temp_clim = pd.DataFrame(climatology.fitted_data, columns=["mean"])
ga03flma_temp_clim["sigma"] = climatology.sigma
ga03flma_temp_clim["vmin"] = np.floor((ga03flma_temp_clim["mean"] - 3*ga03flma_temp_clim["sigma"])*100)/100
ga03flma_temp_clim["vmax"] = np.ceil((ga03flma_temp_clim["mean"] + 3*ga03flma_temp_clim["sigma"])*100)/100

# Conductivity
climatology = Climatology()
climatology.fit(ga03flma_cond, "ctdmo_seawater_conductivity")
ga03flma_cond_clim = pd.DataFrame(climatology.fitted_data, columns=["mean"])
ga03flma_cond_clim["sigma"] = climatology.sigma
ga03flma_cond_clim["vmin"] = np.floor((ga03flma_cond_clim["mean"] - 3*ga03flma_cond_clim["sigma"])*100)/100
ga03flma_cond_clim["vmax"] = np.ceil((ga03flma_cond_clim["mean"] + 3*ga03flma_cond_clim["sigma"])*100)/100
# -

# Filter the data
# Temperature
ga03flmb_temp = ga03flmb_data
ga03flmb_temp = ga03flmb_temp.where((ga03flmb_temp.ctdmo_seawater_temperature >= gross_range_temp["ga03flmb"]["3-sigma"][0])
                                    & (ga03flmb_temp.ctdmo_seawater_temperature <= gross_range_temp["ga03flmb"]["3-sigma"][1]),
                                    drop=True)
# Conductivity
ga03flmb_cond = ga03flmb_data
ga03flmb_cond = ga03flmb_cond.where((ga03flmb_cond.ctdmo_seawater_conductivity >= gross_range_cond["ga03flmb"]["3-sigma"][0])
                                    & (ga03flmb_cond.ctdmo_seawater_conductivity <= gross_range_cond["ga03flmb"]["3-sigma"][1]),
                                    drop=True)

# +
# Calculate the climatologies for the filtered data
# Temperature
climatology = Climatology()
climatology.fit(ga03flmb_temp, "ctdmo_seawater_temperature")
ga03flmb_temp_clim = pd.DataFrame(climatology.fitted_data, columns=["mean"])
ga03flmb_temp_clim["sigma"] = climatology.sigma
ga03flmb_temp_clim["vmin"] = np.floor((ga03flmb_temp_clim["mean"] - 3*ga03flmb_temp_clim["sigma"])*100)/100
ga03flmb_temp_clim["vmax"] = np.ceil((ga03flmb_temp_clim["mean"] + 3*ga03flmb_temp_clim["sigma"])*100)/100

# Conductivity
climatology = Climatology()
climatology.fit(ga03flmb_cond, "ctdmo_seawater_conductivity")
ga03flmb_cond_clim = pd.DataFrame(climatology.fitted_data, columns=["mean"])
ga03flmb_cond_clim["sigma"] = climatology.sigma
ga03flmb_cond_clim["vmin"] = np.floor((ga03flmb_cond_clim["mean"] - 3*ga03flmb_cond_clim["sigma"])*100)/100
ga03flmb_cond_clim["vmax"] = np.ceil((ga03flmb_cond_clim["mean"] + 3*ga03flmb_cond_clim["sigma"])*100)/100

# +
# Plot the temperature data
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharex=True, sharey=True)

ax1.plot(ga01sumo_temp.time, ga01sumo_temp.ctdmo_seawater_temperature, linestyle="", marker=".", color="tab:red")
ax1.fill_between(ga01sumo_temp_clim.index,
                 ga01sumo_temp_clim["vmin"],
                 ga01sumo_temp_clim["vmax"],
                 alpha=0.3, color="tab:red")
ax1.set_ylabel(ga01sumo_temp.ctdmo_seawater_temperature.attrs["long_name"], fontsize=14)
ax1.grid()
ax1.set_title("-".join(ga01sumo_temp.attrs["id"].split("-")[0:4]), fontsize=14)

ax2.plot(ga03flma_temp.time, ga03flma_temp.ctdmo_seawater_temperature, linestyle="", marker=".", color="tab:red")
ax2.fill_between(ga03flma_temp_clim.index,
                 ga03flma_temp_clim["vmin"],
                 ga03flma_temp_clim["vmax"],
                 alpha=0.3, color="tab:red")
ax2.set_ylabel(ga03flma_cond.ctdmo_seawater_conductivity.attrs["long_name"], fontsize=14)
ax2.grid()
ax2.set_title("-".join(ga03flma_temp.attrs["id"].split("-")[0:4]), fontsize=14)

ax3.plot(ga03flmb_temp.time, ga03flmb_temp.ctdmo_seawater_temperature, linestyle="", marker=".", color="tab:red")
ax3.fill_between(ga03flmb_temp_clim.index,
                 ga03flmb_temp_clim["vmin"],
                 ga03flmb_temp_clim["vmax"],
                 alpha=0.3, color="tab:red")
ax3.set_ylabel(ga03flmb_cond.ctdmo_seawater_conductivity.attrs["long_name"], fontsize=14)
ax3.grid()
ax3.set_title("-".join(ga03flmb_temp.attrs["id"].split("-")[0:4]), fontsize=14)
              
fig.autofmt_xdate()
# -

# #### Monthly Statistics
# Want to count the number of days per month with data, then groupby the month

ga01sumo_data.ctdmo_seawater_temperature

ga01sumo_month = ga01sumo_data["ctdmo_seawater_temperature"].to_dataframe()
ga01sumo_month

daycount = ga01sumo_data["ctdmo_seawater_temperature"].to_dataframe().resample("D").count()
daycount

# Convert to a binary count of yes/no data
daycount["ctdmo_seawater_temperature"] = daycount["ctdmo_seawater_temperature"].apply(lambda x: 1 if x > 0 else 0)
daycount

monthcount = daycount.resample("M").count()
monthcount

bins = np.arange(0, 410, 10)
bins

# +
# Convert the count of data to a binary yes/no based on if data/no data
fig, ax = plt.subplots(figsize=(12,8))

sns.distplot(daycount, bins=bins, hist=True,
             hist_kws={"edgecolor": "k", "linewidth": 1}, kde_kws={"linewidth": 3},
             ax=ax)
ax.grid(alpha=0.3)
ax.set_axisbelow(True)

# +
fig, ax = plt.subplots(figsize=(12, 6))

ax = sns.boxplot(daycount["ctdmo_seawater_temperature"])
# -

daycount[daycount["ctdmo_seawater_temperature"] == np.min(daycount["ctdmo_seawater_temperature"])]

# ### Gross Range Comparison
#
# Need to look at a comparison of the gross range data

gross_range_temp = {
    "ga01sumo": {
        "5-sigma": None,
        "3-sigma": None,
        "1.5*IQR": None
    },
    "ga03flma": {
        "5-sigma": None,
        "3-sigma": None,
        "1.5*IQR": None,
    },
    "ga03flmb": {
        "5-sigma": None,
        "3-sigma": None,
        "1.5*IQR": None
    }
}

gross_range_cond = {
    "ga01sumo": {
        "5-sigma": None,
        "3-sigma": None,
        "1.5*IQR": None
    },
    "ga03flma": {
        "5-sigma": None,
        "3-sigma": None,
        "1.5*IQR": None,
    },
    "ga03flmb": {
        "5-sigma": None,
        "3-sigma": None,
        "1.5*IQR": None
    }
}

# #### GA01SUMO
#
# **Calculate the gross range values**

ga01sumo_temp_gross_range = Gross_Range(-5, 35)
# 5-sigma range
ga01sumo_temp_gross_range.fit(ga01sumo_data, "ctdmo_seawater_temperature")
gross_range_temp["ga01sumo"]["5-sigma"] = ga01sumo_temp_gross_range.suspect_min, ga01sumo_temp_gross_range.suspect_max
# 3-sigma range
ga01sumo_temp_gross_range.fit(ga01sumo_data, "ctdmo_seawater_temperature", sigma=3)
gross_range_temp["ga01sumo"]["3-sigma"] = ga01sumo_temp_gross_range.suspect_min, ga01sumo_temp_gross_range.suspect_max
# Interquartile Range
IQR = calc_interquartile_stats(ga01sumo_data, "ctdmo_seawater_temperature")
gross_range_temp["ga01sumo"]["1.5*IQR"] = (np.round(IQR["Qmin"], 2), np.round(IQR["Qmax"], 2))

ga03flma_temp_gross_range = Gross_Range(-5, 35)
# 5-sigma range
ga03flma_temp_gross_range.fit(ga03flma_data, "ctdmo_seawater_temperature")
gross_range_temp["ga03flma"]["5-sigma"] = ga03flma_temp_gross_range.suspect_min, ga03flma_temp_gross_range.suspect_max
# 3-sigma range
ga03flma_temp_gross_range.fit(ga03flma_data, "ctdmo_seawater_temperature", sigma=3)
gross_range_temp["ga03flma"]["3-sigma"] = ga03flma_temp_gross_range.suspect_min, ga03flma_temp_gross_range.suspect_max
# Interquartile Range
IQR = calc_interquartile_stats(ga03flma_data, "ctdmo_seawater_temperature")
gross_range_temp["ga03flma"]["1.5*IQR"] = (np.round(IQR["Qmin"], 2), np.round(IQR["Qmax"], 2))

ga03flmb_temp_gross_range = Gross_Range(-5, 35)
# 5-sigma range
ga03flmb_temp_gross_range.fit(ga03flmb_data, "ctdmo_seawater_temperature")
gross_range_temp["ga03flmb"]["5-sigma"] = ga03flmb_temp_gross_range.suspect_min, ga03flmb_temp_gross_range.suspect_max
# 3-sigma range
ga03flmb_temp_gross_range.fit(ga03flmb_data, "ctdmo_seawater_temperature", sigma=3)
gross_range_temp["ga03flmb"]["3-sigma"] = ga03flmb_temp_gross_range.suspect_min, ga03flmb_temp_gross_range.suspect_max
# Interquartile Range
IQR = calc_interquartile_stats(ga03flmb_data, "ctdmo_seawater_temperature")
gross_range_temp["ga03flmb"]["1.5*IQR"] = (np.round(IQR["Qmin"], 2), np.round(IQR["Qmax"], 2))

gross_range_temp

ga01sumo_cond_gross_range = Gross_Range(0, 9)
# 5-sigma range
ga01sumo_cond_gross_range.fit(ga01sumo_data, "ctdmo_seawater_conductivity")
gross_range_cond["ga01sumo"]["5-sigma"] = ga01sumo_cond_gross_range.suspect_min, ga01sumo_cond_gross_range.suspect_max
# 3-sigma range
ga01sumo_cond_gross_range.fit(ga01sumo_data, "ctdmo_seawater_conductivity", sigma=3)
gross_range_cond["ga01sumo"]["3-sigma"] = ga01sumo_cond_gross_range.suspect_min, ga01sumo_cond_gross_range.suspect_max
# Interquartile Range
IQR = calc_interquartile_stats(ga01sumo_data, "ctdmo_seawater_conductivity")
gross_range_cond["ga01sumo"]["1.5*IQR"] = (np.round(IQR["Qmin"], 2), np.round(IQR["Qmax"], 2))

ga03flma_cond_gross_range = Gross_Range(0, 9)
# 5-sigma range
ga03flma_cond_gross_range.fit(ga03flma_data, "ctdmo_seawater_conductivity")
gross_range_cond["ga03flma"]["5-sigma"] = ga03flma_cond_gross_range.suspect_min, ga03flma_cond_gross_range.suspect_max
# 3-sigma range
ga03flma_cond_gross_range.fit(ga03flma_data, "ctdmo_seawater_conductivity", sigma=3)
gross_range_cond["ga03flma"]["3-sigma"] = ga03flma_cond_gross_range.suspect_min, ga03flma_cond_gross_range.suspect_max
# Interquartile Range
IQR = calc_interquartile_stats(ga03flma_data, "ctdmo_seawater_conductivity")
gross_range_cond["ga03flma"]["1.5*IQR"] = (np.round(IQR["Qmin"], 2), np.round(IQR["Qmax"], 2))

ga03flmb_cond_gross_range = Gross_Range(0, 9)
# 5-sigma range
ga03flmb_cond_gross_range.fit(ga03flmb_data, "ctdmo_seawater_conductivity")
gross_range_cond["ga03flmb"]["5-sigma"] = ga03flmb_cond_gross_range.suspect_min, ga03flmb_cond_gross_range.suspect_max
# 3-sigma range
ga03flmb_cond_gross_range.fit(ga03flmb_data, "ctdmo_seawater_conductivity", sigma=3)
gross_range_cond["ga03flmb"]["3-sigma"] = ga03flmb_cond_gross_range.suspect_min, ga03flmb_cond_gross_range.suspect_max
# Interquartile Range
IQR = calc_interquartile_stats(ga03flmb_data, "ctdmo_seawater_conductivity")
gross_range_cond["ga03flmb"]["1.5*IQR"] = (np.round(IQR["Qmin"], 2), np.round(IQR["Qmax"], 2))

gross_range_cond

# **Plot the comparison of different comparison**

# +
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharex=True, sharey=True)

# GA01SUMO
ax1.plot(ga01sumo_data.time, ga01sumo_data.ctdmo_seawater_temperature, linestyle="", marker=".", color="tab:red", label="Data")
ax1.fill_between(ga01sumo_data.time, gross_range_temp["ga01sumo"]["5-sigma"][0], gross_range_temp["ga01sumo"]["5-sigma"][1], color="tab:gray", alpha=0.3, label="5-sigma")
ax1.fill_between(ga01sumo_data.time, gross_range_temp["ga01sumo"]["3-sigma"][0], gross_range_temp["ga01sumo"]["3-sigma"][1], color="tab:blue", alpha=0.3, label="3-sigma")
ax1.fill_between(ga01sumo_data.time, gross_range_temp["ga01sumo"]["1.5*IQR"][0], gross_range_temp["ga01sumo"]["1.5*IQR"][1], color="tab:red", alpha=0.3, label="Q1-1.5*IQR :: Q3+1.5*IQR")
ax1.legend()
ax1.grid()
ax1.set_ylabel(ga01sumo_data.ctdmo_seawater_temperature.attrs["long_name"])
ax1.set_title("-".join(ga01sumo_data.attrs["id"].split("-")[0:4]))

# GA03FLMA
ax2.plot(ga03flma_data.time, ga03flma_data.ctdmo_seawater_temperature, linestyle="", marker=".", color="tab:red", label="Data")
ax2.fill_between(ga03flma_data.time, gross_range_temp["ga03flma"]["5-sigma"][0], gross_range_temp["ga03flma"]["5-sigma"][1], color="tab:gray", alpha=0.3, label="5-sigma")
ax2.fill_between(ga03flma_data.time, gross_range_temp["ga03flma"]["3-sigma"][0], gross_range_temp["ga03flma"]["3-sigma"][1], color="tab:blue", alpha=0.3, label="3-sigma")
ax2.fill_between(ga03flma_data.time, gross_range_temp["ga03flma"]["1.5*IQR"][0], gross_range_temp["ga03flma"]["1.5*IQR"][1], color="tab:red", alpha=0.3, label="Q1-1.5*IQR :: Q3+1.5*IQR")
ax2.legend()
ax2.grid()
ax2.set_ylabel(ga03flma_data.ctdmo_seawater_temperature.attrs["long_name"])
ax2.set_title("-".join(ga03flma_data.attrs["id"].split("-")[0:4]))


# GA03FLMB
ax3.plot(ga03flmb_data.time, ga03flmb_data.ctdmo_seawater_temperature, linestyle="", marker=".", color="tab:red", label="Data")
ax3.fill_between(ga03flmb_data.time, gross_range_temp["ga03flmb"]["5-sigma"][0], gross_range_temp["ga03flmb"]["5-sigma"][1], color="tab:gray", alpha=0.3, label="5-sigma")
ax3.fill_between(ga03flmb_data.time, gross_range_temp["ga03flmb"]["3-sigma"][0], gross_range_temp["ga03flmb"]["3-sigma"][1], color="tab:blue", alpha=0.3, label="3-sigma")
ax3.fill_between(ga03flmb_data.time, gross_range_temp["ga03flmb"]["1.5*IQR"][0], gross_range_temp["ga03flmb"]["1.5*IQR"][1], color="tab:red", alpha=0.3, label="Q1-1.5*IQR :: Q3+1.5*IQR")
ax3.legend()
ax3.grid()
ax3.set_ylabel(ga03flmb_data.ctdmo_seawater_temperature.attrs["long_name"])
ax3.set_title("-".join(ga03flmb_data.attrs["id"].split("-")[0:4]))

fig.autofmt_xdate()

# +
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharex=True, sharey=True)

# GA01SUMO
ax1.plot(ga01sumo_data.time, ga01sumo_data.ctdmo_seawater_conductivity, linestyle="", marker=".", color="tab:blue", label="Data")
ax1.fill_between(ga01sumo_data.time, gross_range_cond["ga01sumo"]["5-sigma"][0], gross_range_cond["ga01sumo"]["5-sigma"][1], color="tab:gray", alpha=0.3, label="5-sigma")
ax1.fill_between(ga01sumo_data.time, gross_range_cond["ga01sumo"]["3-sigma"][0], gross_range_cond["ga01sumo"]["3-sigma"][1], color="tab:blue", alpha=0.3, label="3-sigma")
ax1.fill_between(ga01sumo_data.time, gross_range_cond["ga01sumo"]["1.5*IQR"][0], gross_range_cond["ga01sumo"]["1.5*IQR"][1], color="tab:red", alpha=0.3, label="Q1-1.5*IQR :: Q3+1.5*IQR")
ax1.legend()
ax1.grid()
ax1.set_ylabel(ga01sumo_data.ctdmo_seawater_conductivity.attrs["long_name"])
ax1.set_title("-".join(ga01sumo_data.attrs["id"].split("-")[0:4]))

# GA03FLMA
ax2.plot(ga03flma_data.time, ga03flma_data.ctdmo_seawater_conductivity, linestyle="", marker=".", color="tab:blue", label="Data")
ax2.fill_between(ga03flma_data.time, gross_range_cond["ga03flma"]["5-sigma"][0], gross_range_cond["ga03flma"]["5-sigma"][1], color="tab:gray", alpha=0.3, label="5-sigma")
ax2.fill_between(ga03flma_data.time, gross_range_cond["ga03flma"]["3-sigma"][0], gross_range_cond["ga03flma"]["3-sigma"][1], color="tab:blue", alpha=0.3, label="3-sigma")
ax2.fill_between(ga03flma_data.time, gross_range_cond["ga03flma"]["1.5*IQR"][0], gross_range_cond["ga03flma"]["1.5*IQR"][1], color="tab:red", alpha=0.3, label="Q1-1.5*IQR :: Q3+1.5*IQR")
ax2.legend()
ax2.grid()
ax2.set_ylabel(ga03flma_data.ctdmo_seawater_conductivity.attrs["long_name"])
ax2.set_title("-".join(ga03flma_data.attrs["id"].split("-")[0:4]))


# GA03FLMB
ax3.plot(ga03flmb_data.time, ga03flmb_data.ctdmo_seawater_conductivity, linestyle="", marker=".", color="tab:blue", label="Data")
ax3.fill_between(ga03flmb_data.time, gross_range_cond["ga03flmb"]["5-sigma"][0], gross_range_cond["ga03flmb"]["5-sigma"][1], color="tab:gray", alpha=0.3, label="5-sigma")
ax3.fill_between(ga03flmb_data.time, gross_range_cond["ga03flmb"]["3-sigma"][0], gross_range_cond["ga03flmb"]["3-sigma"][1], color="tab:blue", alpha=0.3, label="3-sigma")
ax3.fill_between(ga03flmb_data.time, gross_range_cond["ga03flmb"]["1.5*IQR"][0], gross_range_cond["ga03flmb"]["1.5*IQR"][1], color="tab:red", alpha=0.3, label="Q1-1.5*IQR :: Q3+1.5*IQR")
ax3.legend()
ax3.grid()
ax3.set_ylabel(ga03flmb_data.ctdmo_seawater_conductivity.attrs["long_name"])
ax3.set_title("-".join(ga03flmb_data.attrs["id"].split("-")[0:4]))

fig.autofmt_xdate()
# -

ga01sumo_data.ctdmo_seawater_conductivity


