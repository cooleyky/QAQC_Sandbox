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

# # CTD Download Data
#
# **Author: Andrew Reed**
#
# ### Purpose
# The purpose of this notebook is to calculate the QARTOD test data tables for the CGSN-maintained PCO2W instruments. This requires the following steps:
# 1. Identify all CGSN PCO2W instruments and associated data streams in OOINet
# 2. Download all the relevant data sets
# 3. Load the 
# 3. For each data stream, identify the primary data variables (parameters) for each data stream
#  

# Import libraries
import os, shutil, sys, time, re, requests, csv, datetime, pytz
import time
import yaml
import pandas as pd
import numpy as np
import netCDF4 as nc
import xarray as xr
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
# ## Identify Data Streams
#
# The first step is to identify all of the data streams associated with the ```CTDBP```, ```CTDMO```, and ```CTDPF``` instruments. This involves retrieving all combinations of reference designator - method - stream. This can be done by querying UFrame and iteratively walking through all of the API endpoints. The results are saved into a csv file so this step doesn't have to be repeated each time.
#
# First, set the instrument to search for using OOI terminology:

instrument = "CTD"

# ### Query OOINet for Datasets <br>
# If the data streams for a given instrument have not yet been identified from OOINet, then want to query OOINet for the data sets and save them to the local memory:

datasets = OOINet.search_datasets(instrument=instrument)
datasets.head()

# Check the reference designators that we got only CTD values

datasets["refdes"].unique()

# Filter for the instruments on the **Coastal Pioneer (CP)**, **Global Irminger (GI)**, **Global Argentine (GA)**, **Global Station Papa (GP)**, and **Global Southern (GS)** arrays:

cgsn_mask = datasets["array"].apply(lambda x: True if x.startswith(("CP","GI","GA","GP","GS")) else False)
cgsn_datasets = datasets[cgsn_mask]
cgsn_datasets.head()

# Save the data streams:

cgsn_datasets.to_csv("../data/cgsn_datasets.csv", index=False)

# ### Load Datasets
# If all the datasets for a given instrument have already been identified, then want to simply load the identified data streams from local memory:

cgsn_datasets = pd.read_csv("../data/cgsn_datasets.csv")
cgsn_datasets.head()

# #### Reference Designators
# From all the identified datasets, we can get all the reference_designators for the ```PCO2W```

reference_designators = sorted(cgsn_datasets["refdes"])
reference_designators

# Select a single reference designator (for development)

refdes = reference_designators[0]
refdes

# **Data Streams** <br>
# We can directly query OOINet for all the available data streams for a given reference designator.

datastreams = OOINet.get_datastreams(refdes)
datastreams

# **Science-relevant Data Streams** <br>
# In order to identify the data streams which contain the relevant _science_ parameters, we can query for the metadata for a given reference designator as well as the parameter data levels, then use those parameter data levels to filter the metadata to only the relevant data streams remain.

metadata = OOINet.get_metadata(refdes)
metadata

# #### Sensor Parameters
# Each instrument returns multiple parameters containing a variety of low-level instrument output and metadata. However, we are interested in science-relevant parameters for calculating the relevant QARTOD test limits. We can identify the science parameters based on the preload database, which designates the science parameters with a "data level" of L1 or L2. 
#
# Consequently, we through several steps to identify the relevant parameters. First, we query the preload database with the relevant metadata for a reference designator. Then, we filter the metadata for the science-relevant data streams. 

data_levels = OOINet.get_parameter_data_levels(metadata)

mask = metadata["pdId"].apply(lambda x: OOINet.filter_parameter_ids(x, data_levels))
metadata = metadata[mask]
metadata

# Iterate through all of the reference designators, download the relevant metadata, and save the output

# +
metadata = pd.DataFrame()

# Script to get all of the metadata for the CTD reference designators
for refdes in sorted(reference_designators):
    # Get the specific metadata for the reference designator
    refdes_metadata = OOINet.get_metadata(refdes)
    
    # Get the data levels for the metadata reference designator
    data_levels = OOINet.get_parameter_data_levels(refdes_metadata)
    
    # Select the relevant sensor parameters
    mask = refdes_metadata["pdId"].apply(lambda x: OOINet.filter_parameter_ids(x, data_levels))
    refdes_metadata = refdes_metadata[mask]
    
    # Save the reference designator specific metadata into a single dataframe
    metadata = metadata.append(refdes_metadata)
# -

# ### Load Metadata

metadata = pd.read_csv("../data/metadata.csv")
metadata.head()

# Groupby based on the reference designator - method - stream to get the unique values for each data stream:

metadata = metadata.groupby(by=["refdes","method","stream"]).agg(lambda x: pd.unique(x.values.ravel()).tolist())

metadata = metadata.reset_index()
metadata

# ## Download Data

base_dir = "/media/andrew/Files/Instrument_Data"

# thredds_table = pd.DataFrame(columns=["refdes", "method", "stream", "request_date", "thredds_url"])
thredds_table = pd.read_csv("../data/thredds_table.csv")

# First, select the metadata associated with the reference designator
refdes = sorted(reference_designators)[4]
refdes

# Get the associated metadata with the reference_designator
refdes_metadata = metadata[metadata["refdes"] == refdes]
refdes_metadata

for ind in refdes_metadata.index:
    
    # Get the key values to check for download
    # ----------------------------------------
    refdes, method, stream, pdId, pKeys = refdes_metadata[["refdes", "method", "stream", "pdId", "particleKey"]].loc[ind]
    
    
    # Check if the data has already been downloaded
    # ---------------------------------------------
    inst_class = refdes.split("-")[-1][0:6]
    data_path = f"{base_dir}/{inst_class}/{refdes}/{method}/{stream}"
    beginDT = None
    if not os.path.exists(data_path):
        # Need to create a directory in order to download the data
        os.makedirs(data_path)
    else:
        if not os.listdir(data_path):
            # Directory is empty. Take not actions
            pass
        else:
            # Get the last timestamp of the files
            files = sorted(os.listdir(data_path))
            dates = [pd.to_datetime(f.split("-")[-1].split(".")[0]) for f in files]
            beginDT = max(dates).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    
    # Download data
    # -----------------------------------------------
    # First, generate the parameters to pass to the data request
    parameters = ",".join(pdId)

    # Check if the THREDDS table already has a request checked for the data
    
    # Get the request url
    if beginDT is None:
        thredds_url = OOINet.get_thredds_url(refdes, method, stream)
        print(thredds_url)
    else:
        thredds_url = OOINet.get_thredds_url(refdes, method, stream, beginDT=beginDT)
        print(thredds_url)
        # If the thredds_url is "Not Found", need to save the result anyways to check later
        if thredds_url is None:
            thredds_table = thredds_table.append({
                "refdes": refdes,
                "method": method,
                "stream": stream,
                "request_date": request_date,
                "thredds_url": "NOT FOUND"
            }, ignore_index=True)
            # Continue on to the next request
            continue

    # Save the thredds_url for later storage
    request_date = datetime.datetime.now(tz=pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    thredds_table = thredds_table.append({
        "refdes": refdes,
        "method": method,
        "stream": stream,
        "request_date": request_date,
        "thredds_url": thredds_url
    }, ignore_index=True)

    # Get the catalog from the THREDDS server
    catalog = OOINet.get_thredds_catalog(thredds_url)

    # Parse the catalog, dropping extraneous datasets
    netCDF_files = OOINet.parse_catalog(catalog, exclude=["ENG", "gps"])
    netCDF_files = sorted(netCDF_files)

    # Download the data
    OOINet.download_netCDF_files(netCDF_files, save_dir=data_path)

thredds_table

# Save the THREDDS data table to speed up future requests
thredds_table.to_csv("../data/thredds_table.csv", index=False)
os.listdir("../data/")


