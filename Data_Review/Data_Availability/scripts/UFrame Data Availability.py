# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # UFrame Data Availability
#
# Author: Andrew Reed<br>
# Version: 1.0<br>
# Date: 2019-11-21<br>
#
# ---
# ### Purpose
# This notebook seeks to answer the following questions:
# 1. For each unique data stream returned by deployed CGSN assets - what is the data availability on a per day and per deployment basis?
# 2. How does the data availability vary for each mooring deployed and operated by CGSN?
# 3. How does the data availability in UFrame compare with OMS++?
#
# ---
# ### Updates
# Version: 1.1<br>
# Date: 2019-11-26
#
# The netCDF requests on a deployment-by-deployment basis causes a real issue for UFrame, with the end-result being the requests either getting throttled or killed. Consequently, the major change will be to download all of the data for a given stream, and then calculate on a deployment and per-day basis. 
#
# <br>
#
# Version 1.2<br>
# Date: 2020-01-21
#
# Functionality to calculate the statistics for all deployments for a particular reference designator's data streams works as intended. Currently it operates using HITL execution.
#
# Loading the netCDF requests in a scripted loop is skipping some deployments or throwing Exceptions when loading into xarray. Not sure why, but I think its linked to the time it takes to create the datasets by OOINet. The automated piece needs further development.
#
# <br>
#
# Version 1.3<br>
# Date: 2020-6-7
#
# Changed the deployment info to request it from OOINet instead of relying on the deploy sheets from GitHub. This makes the data internally consistent.
#
#
#

import os, time, re, requests, csv, pytz, sys
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import warnings
warnings.filterwarnings("ignore")

sys.path.append("/home/andrew/Documents/OOI-CGSN/ooinet/ooinet/")
sys.path.append("/home/andrew/Documents/OOI-CGSN/oceanobservatories/ooi-data-explorations/python")

from m2m import M2M
from ooi_data_explorations.uncabled import process_pco2w
from ooi_data_explorations.common import add_annotation_qc_flags

# Import user info for connecting to OOINet via M2M
userinfo = yaml.load(open("../../../user_info.yaml"))
username = userinfo["apiname"]
token = userinfo["apikey"]

from utils import *

process_pco2w.pco2w_instrument()

# ---
# ### Deployment Information
# To properly identify and calculate the data availability on a deployment level, need to pull in the deployment information for each array/platform. This can be ingested from the deployment sheets on an array-by-array basis. Once the deployment csvs are loaded, they are parsed to retain only the deployment number, startDateTime, and stopDateTime.
#
# Since we are concerned with what is available in OOINet. So, instead of pulling from the deploy sheets, I want to request the data from OOINet based upon the reference designator.

# Load all of the deployment sheets except for the gliders/auvs (i.e. mobile assets = MOAS)
deployment_csvs = pd.DataFrame()
for sheet in os.listdir("/home/andrew/Documents/OOI-CGSN/asset-management/deployment"):
    if "MOAS" not in sheet:
        deployment_csvs = deployment_csvs.append(pd.read_csv("/".join(("/home/andrew/Documents/OOI-CGSN/asset-management/deployment",sheet))))

# Select the instrument you are interested in
mask = deployment_csvs["Reference Designator"].apply(lambda x: True if "PCO2W" in x else False)
deployment_csvs = deployment_csvs[mask]
deployment_csvs.head()

reference_designators = sorted(deployment_csvs["Reference Designator"].unique())
reference_designators

# Initialize the OOINet object to connect of OOINet
OOINet = M2M(username, token)

# refdes = sorted(reference_designators)[0]
refdes = "CP01CNSM-MFD35-05-PCO2WB000"

refdes_deployments = OOINet.get_deployments(refdes)
refdes_deployments

# ---
# ## Metadata 
# The metadata contains the following important key pieces of data for each reference designator: **```method```**, **```stream```**, **```particleKey```**, and **```count```**. The method and stream are necessary for identifying and loading the relevant dataset. The particleKey tells us which data variables are in the dataset. 

# Specify which instrument we want and see the method options
metadata = OOINet.get_metadata(refdes)
metadata

# Groupby based on the reference designator - method - stream to get the unique values for each data stream

data_levels = OOINet.get_parameter_data_levels(metadata)
data_levels


# +
def filter_parameter_data_level(pdId, pid_dict):
    """Filter for processed data products."""
    data_level = pid_dict.get(pdId)
    if data_level is not None:
        if data_level > 0:
            return True
        else:
            return False
    else:
        return False
    
# Get the science-level data streams
mask = metadata["pdId"].apply(lambda x: filter_parameter_data_level(x, data_levels))
metadata = metadata[mask]
metadata = metadata.groupby(by=["refdes","stream"]).agg(lambda x: pd.unique(x.values.ravel()).tolist())
metadata = metadata.reset_index()
metadata = metadata.applymap(lambda x: x[0] if len(x) == 1 else x)
metadata.head()
# -

# Filter out any datastreams with ```blank``` in the stream

mask = metadata["stream"].apply(lambda x: True if "blank" not in x else False)
metadata = metadata[mask]
metadata

# The result is a dataframe with the datastreams which contain the science-relevant data. It also contains the ```method``` and ```stream``` needed to request and downloaded the data

# ---
# ## Request Data
# With the array, node, sensor, method, and stream identified after querying the m2m api, we can build the request for actual data. Below we follow these steps:
# 1. Request the thredds server url from the OOI API using the appropriate data request url for all time
# 2. Request all of the available datasets from the Thredds server
#     * This may take awhile for OOI to assemble all of the datasets. You can rerun the get_netcdf_datasets as many times as you want until the list of datasets stops growing.
# 3. Filter out datasets for associated but unwanted datasets such as engineering and CTD datasets.

# First, select the method and stream

method = "telemetered"
stream = "pco2w_abc_dcl_instrument"

# Next, get the first deployment to load

beginDT = refdes_deployments[refdes_deployments["deploymentNumber"] == 11]["deployStart"].values[0]
endDT = refdes_deployments[refdes_deployments["deploymentNumber"] == 11]["deployEnd"].values[0]

# +
# Try a new thredds request only for specific parameter ids
kwargs = {
    "beginDT": beginDT,
    "endDT": endDT
}

new_thredds_url = OOINet.get_thredds_url(refdes, method, stream, **kwargs)
# -

new_catalog = OOINet.get_thredds_catalog(new_thredds_url)

new_catalog = OOINet.parse_catalog(new_catalog, exclude=["ENG","CTD","blank"])
new_catalog

data = OOINet.load_netCDF_datasets(new_catalog)
data = process_pco2w.pco2w_datalogger(data)

# +
days = time_periods(beginDT, endDT)
days = [day.strip('Z') for day in days]

results = pd.DataFrame(columns=["day", "total", "missing", "present", "percent_present"])

for k in np.arange(0, len(days)-1, 1):
    # Get a subset of the data
    subset_data = data.sel(time=slice(days[k], days[k+1]))
    
    # Calculate the number of "present" and "missing" data values
    total = len(subset_data.pco2_seawater)
    missing = subset_data.pco2_seawater.isnull().sum().values
    present = subset_data.pco2_seawater.count().values
    percent_present = (present-missing)/total*100
    
    # Save the results in a dataframe
    results = results.append({
        "day": days[k],
        "total": total,
        "missing": missing,
        "present": present,
        "percent_present": percent_present
    }, ignore_index=True)
    
# -

results

# Create summary statistics
summary = pd.DataFrame("deploymentNumber", "First 30 days", "Total Good", "Last 30 days")

days[k], days[k+1]

subset_data = data.sel(time=slice(days[0],days[1]))

len(subset_data.pco2_seawater)

total = len(subset_data.pco2_seawater)
nans = subset_data.pco2_seawater.isnull().sum().values
good = (total - nans)
percent_good = good/total*100

subset_data.pco2_seawater[1] = np.nan

subset_data.pco2_seawater

# ---
# Build some functions to calculate the daily bin values

days = time_periods(beginDT, endDT)
days = [day.strip('Z') for day in days]
days = pd.to_datetime(days)
days

# **====================================================================================================================**
# ### Sensor Metadata Information & \_FillValues
# In order to properly calculate the daily data availability statistics, we need to know the \_FillValue property for each variable in the dataset. Unfortunately, current OOI netCDF files are not fully CF-compliant, so the \_FillValue property is ignored when loading the netCDF file. We can get around this by querying for the sensor metadata information, which contains the \_FillValues used by Stream Engine to fill out the datasets. 
#
# Once the sensor metadata is loaded, we can parse the metadata to identify the \_FillValue for each variable in the dataset.

# Get the metadata associated with the specific reference designator
refdes_metadata = OOINet.get_metadata(refdes)
refdes_metadata

# **====================================================================================================================**
# ### Request Data
# With the array, node, sensor, method, and stream identified after querying the m2m api, we can build the request for actual data. Below we follow these steps:
# 1. Request the thredds server url from the OOI API using the appropriate data request url for all time
# 2. Request all of the available datasets from the Thredds server
#     * This may take awhile for OOI to assemble all of the datasets. You can rerun the get_netcdf_datasets as many times as you want until the list of datasets stops growing.
# 3. Filter out datasets for associated but unwanted datasets such as engineering and CTD datasets.

# Get the thredds url
thredds_url = OOINet.get_thredds_url(refdes, method, stream)
thredds_url

# Find and return the netCDF datasets from the thredds url
catalog = OOINet.get_thredds_catalog(thredds_url)
catalog

# Filter the datasets
datasets = [d for d in datasets if 'ENG' not in d]
if 'CTD' not in sensor:
    datasets = [d for d in datasets if 'CTD' not in d]
datasets = sorted(datasets)
datasets

for deploy_num, start_time, stop_time in deploy_df.values:
    dset_num = 'deployment' + str(deploy_num).zfill(4)
    dset = [d for d in datasets if dset_num in d]
    if len(dset) == 0:
        continue
    print(dset_num + ' ' + dset[0])

# **====================================================================================================================** 
# ### Calculate Data Availability 
# With the downloaded data and associated sensor metadata, we can go ahead and calculate the percent of data available per deployment calculated on a per-day business. The method used for calculating the  is, as follows:
# 1. Bin the data into day-long periods spanning midnight-to-midnight, with the first and last days shortened by the respective deployment and recovery time.
# 2. For each variable in the dataset, make the following calculate on a per-day basis:
#     1. Calculate the number of NaNs in the dataset
#     2. Calculate the number of \_FillValues in the dataset
#     3. Subtract the number of NaNs and FillValues from the total number of data points
#     4. Divide by the total number of data points to get the percent data available
# 3. Save the percent data available in each day
# 4. Concatenate all of the daily stats into a pandas dataframe, which is saved as individual deployments
#
# With the daily statistics for each deployment, we can then calculate the statistics for the entire deployment. Then, we concatenate all of the deployments into a separate results dataframe, which is also saved as a summary of all of the deployment-level statistics.

save_dir = '/home/andrew/Documents/OOI-CGSN/QAQC_Sandbox/Data_Review/Data Availability/Results'

# Now, calculate the data availability
for deploy_num, start_time, stop_time in deploy_df.values:
    
    # Get the appropriate dataset to load the data
    dset_num = 'deployment' + str(deploy_num).zfill(4)
    dset = [d for d in datasets if dset_num in d]
    if len(dset) == 0:
        print('No data for deployment ' + str(deploy_num))
        continue
    
    # Load the netcdf file 
    ds = load_netcdf_datasets(dset)
    
    # Convert to a dataframe
    data = ds.to_dataframe()
    data = data.swaplevel()
    data = data.droplevel(level=1)
    
    # Strip the non-relevant columns and variables from the dataframe
    data.drop(columns=[x for x in data.columns if 'qc' in x], inplace=True)
    
    # Get the fillValues for the data
    fillValues = get_UFrame_fillValues(data, metadata, stream)
    
    # Create an array of days from midnight-to-midngith, with shortened first and last days
    days = time_periods(start_time, stop_time)
    days = [day.strip('Z') for day in days]
    
    # Slice the dataframe based on the start and end dates
    daily_stats = pd.DataFrame(columns=data.columns)
    for i in range(len(days)-1):
        
        # Slice the dataframe for only the data wihtin the time range
        subset_data = data.loc[days[i]:days[i+1]]
        
        # Remove any burn-in data that may have been captured
        subset_data = subset_data[subset_data['deployment'] == deploy_num]
        
        # Calculate the data availability for a particular day
        data_availability = calc_UFrame_data_availability(subset_data, fillValues)
        
        # Put the data availability into a dataframe
        daily_stats = daily_stats.append(data_availability, ignore_index = True)
        
    # Need to save the daily stats for each deployment
    filename = '_'.join(('Deployment', str(deploy_num), pd.datetime.now().strftime('%Y-%m-%d')+'.csv'))
    save_path = '/'.join((save_dir, array, node, sensor, method, stream))
    make_dirs(save_path)
    daily_stats.to_csv('/'.join((save_path, filename)))
    
    # Now, calculate the overall deployment statistics of the daily data availability
    deployment_stats = daily_stats.describe()
    
    # Add in the deployment number and reindex
    deployment_stats['deployment'] = deploy_num
    deployment_stats.set_index('deployment', append=True, inplace=True)
    deployment_stats = deployment_stats.reorder_levels(['deployment',None])
    
    # Finally, put everything into a results dataframe
    try:
        results = results.append(deployment_stats)
    except:
        results = deployment_stats


# Save the summary results as well:

results.to_csv('/'.join((save_path,'Data_Availability.csv')))















# This below is a scripted loop which, for whatever reason, breaks when loading the datasets. Needs further development. Potentially useful in the future but basic functionality to get the stats exists above.

# +
# Get the sensor metadata
metadata = get_sensor_metadata('/'.join((url, array, node, sensor, 'metadata')), username, token)

# Iterate through each individual deployment 
for deploy_num, start_time, end_time in deploy_df.values:
    
    # create an array of days from midnight-to-midnight, with shortened first and last days
    days = time_periods(start_time, end_time)
    
    # Get the start (min_time) and stop (max_time) time range in which to request data and format properly
    min_time = pd.to_datetime(days[0]).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    max_time = pd.to_datetime(days[-1]).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    
    # Request the data for the deployment time frame
    data = request_UFrame_data(url, array, node, sensor, method, stream, min_time, max_time, username, token)
    if data is None:
        continue
    
    # Strip the non-relevant columns and variables from the dataframe
    data.drop(columns=[x for x in data.columns if 'qc' in x], inplace=True)
    
    # Get the fillValues for the data
    fillValues = get_UFrame_fillValues(data, metadata, stream)
    
    # Slice the dataframe based on the start and end dates
    daily_stats = pd.DataFrame(columns=data.columns)
    for i in range(len(days)-1):
    
        # Get the start and stop dates, remove Z to make timezone agnostic
        startDate = days[i].strip('Z')
        stopDate = days[i+1].strip('Z')
    
        # Slice the dataframe for only data within the time range
        subset_data = data.loc[startDate:stopDate]
    
        # Calculate the data availability for a particular day
        data_availability = calc_UFrame_data_availability(subset_data, fillValues)
    
        # Put the data availability into a dataframe
        daily_stats = daily_stats.append(data_availability, ignore_index=True)
      
    # Need to save the daily stats for each deployment
    filename = '_'.join(('Deployment', str(deploy_num), pd.datetime.now().strftime('%Y-%m-%d')+'.csv'))
    save_path = '/'.join((save_dir, array, node, sensor, method, stream))
    make_dirs(save_path)
    daily_stats.to_csv('/'.join((save_path, filename)))
    
    # Now, calculate the overall deployment statistics of the daily data availability
    deployment_stats = daily_stats.describe()
    
    # Add in the deployment number and reindex
    deployment_stats['deployment'] = deploy_num
    deployment_stats.set_index('deployment', append=True, inplace=True)
    deployment_stats = deployment_stats.reorder_levels(['deployment',None])
    
    # Finally, put everything into a results dataframe
    try:
        results = results.append(deployment_stats)
    except:
        results = deployment_stats
