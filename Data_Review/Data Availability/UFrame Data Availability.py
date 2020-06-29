# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # UFrame Data Availability
#
# Author: Andrew Reed<br>
# Version: 1.0<br>
# Date: 2019-11-21<br>
#
# ### Purpose
# This notebook seeks to answer the following questions:
# 1. For each unique data stream returned by deployed CGSN assets - what is the data availability on a per day and per deployment basis?
# 2. How does the data availability vary for each mooring deployed and operated by CGSN?
# 3. How does the data availability in UFrame compare with OMS++?
#
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
#

import os, time, re, requests, csv, pytz
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import warnings
warnings.filterwarnings("ignore")

from utils import *

# **====================================================================================================================**
# ### Deployment Information
# To properly identify and calculate the data availability on a deployment level, need to pull in the deployment information for each array/platform. This can be ingested from the deployment sheets on an array-by-array basis. Once the deployment csvs are loaded, they are parsed to retain only the deployment number, startDateTime, and stopDateTime.

for sheet in os.listdir('/home/andrew/Documents/OOI-CGSN/asset-management/deployment/'):
    if sheet.startswith(('CP','G')) and 'MOAS' not in sheet:
        print(sheet)

# Import the respective mooring deploy sheet:

deploy_info = pd.read_csv('/home/andrew/Documents/OOI-CGSN/asset-management/deployment/CP01CNSM_Deploy.csv')
deploy_info.head()

# +
# This chunck of code identifies the start and stop times of the deployments from the associated
# deploy csv from ooi asset management. 
deploy_num = ()
startDateTime = ()
stopDateTime = ()
for i in np.unique(deploy_info['deploymentNumber']):
    subset = deploy_info[deploy_info['deploymentNumber'] == i]
    t0 = np.unique(subset['startDateTime'])[0]
    t1 = np.unique(subset['stopDateTime'])[0]
    # Put the data into tuples
    deploy_num = deploy_num + (i,)
    startDateTime = startDateTime + (t0,)
    stopDateTime = stopDateTime + (t1,)
    
deploy_df = pd.DataFrame(data=zip(deploy_num, startDateTime, stopDateTime),
                         columns=['deploymentNumber','startDateTime','stopDateTime'])
# -

deploy_df

# **====================================================================================================================**
# ### UFrame Asset Identification
# UFrame offers the ability to query different api end-points and get the available assets. This allows for the dynamic building of a query to programmatically request and load data from UFrame for each sensor on a particular array. Our steps below are to:
# 1. Select a reference designator from the deployment info sheet
# 2. Identify which methods and streams to download by querying the UFrame API

# First, load the user info so we can query the OOI m2m api
user_info = yaml.load(open('../../user_info.yaml'))
username = user_info['apiname']
token = user_info['apikey']

# Next, declare the url of ooi-net api
url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv'

# List the available reference designators for the particular mooring deployment:
print(deploy_info['Reference Designator'].unique())

refdes = 'CP01CNSM-MFD35-01-ADCPTF000'
array, node, sensor = refdes.split('-',2)
array, node, sensor

# Specify which instrument we want and see the method options
methods = get_and_print_api('/'.join((url, array, node, sensor)), username, token)

# Specify which method and return the available streams
method = 'telemetered'
streams = get_and_print_api('/'.join((url, array, node, sensor, method)), username, token)

# Specify the stream name
stream = 'adcp_velocity_earth'

# **====================================================================================================================**
# ### Sensor Metadata Information & \_FillValues
# In order to properly calculate the daily data availability statistics, we need to know the \_FillValue property for each variable in the dataset. Unfortunately, current OOI netCDF files are not fully CF-compliant, so the \_FillValue property is ignored when loading the netCDF file. We can get around this by querying for the sensor metadata information, which contains the \_FillValues used by Stream Engine to fill out the datasets. 
#
# Once the sensor metadata is loaded, we can parse the metadata to identify the \_FillValue for each variable in the dataset.

# Get the sensor metadata
metadata = get_sensor_metadata('/'.join((url, array, node, sensor, 'metadata')), username, token)
metadata.head()

# **====================================================================================================================**
# ### Request Data
# With the array, node, sensor, method, and stream identified after querying the m2m api, we can build the request for actual data. Below we follow these steps:
# 1. Request the thredds server url from the OOI API using the appropriate data request url for all time
# 2. Request all of the available datasets from the Thredds server
#     * This may take awhile for OOI to assemble all of the datasets. You can rerun the get_netcdf_datasets as many times as you want until the list of datasets stops growing.
# 3. Filter out datasets for associated but unwanted datasets such as engineering and CTD datasets.

data_request_url = '/'.join((url, array, node, sensor, method, stream))
data_request_url

thredds_url = get_thredds_url(data_request_url, None, None, username, token)
thredds_url

# Find and return the netCDF datasets from the thredds url
datasets = get_netcdf_datasets(thredds_url)
datasets

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
