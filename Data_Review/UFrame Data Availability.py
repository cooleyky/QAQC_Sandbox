# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
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
# 2. How does the data availability vary for each 
# 3. How does the data availability in UFrame compare with OMS++?
#
# ### Updates
# Version: 1.1<br>
# Date: 2019-11-26
#
# The netCDF requests on a deployment-by-deployment basis causes a real issue for UFrame, with the end-result being the requests either getting throttled or killed. Consequently, the major change will be to download all of the data for a given stream, and then calculate on a deployment and per-day basis. 

import os, time, re, requests, csv, pytz
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import warnings
warnings.filterwarnings("ignore")

# **====================================================================================================================**
# ### Deployment Information
# To properly identify and calculate the data availability on a deployment level, need to pull in the deployment information for each array/platform. This can be ingested from the deployment sheets on an array-by-array basis. Once the deployment csvs are loaded, they are parsed to retain only the deployment number, startDateTime, and stopDateTime.

deploy_info = pd.read_csv('/home/andrew/Documents/OOI-CGSN/asset-management/deployment/CP01CNSM_Deploy.csv')

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
# UFrame offers the ability to query different api end-points and get the available assets. This allows for the dynamic building of a query to programmatically request and load data from UFrame for each sensor on a particular array.

# +
# Specify some functions to convert timestamps
ntp_epoch = datetime.datetime(1900, 1, 1)
unix_epoch = datetime.datetime(1970, 1, 1)
ntp_delta = (unix_epoch - ntp_epoch).total_seconds()

def ntp_seconds_to_datetime(ntp_seconds):
    return datetime.datetime.utcfromtimestamp(ntp_seconds - ntp_delta).replace(microsecond=0)
  
def convert_time(ms):
    if ms is None:
        return None
    else:
        return datetime.datetime.utcfromtimestamp(ms/1000)


# -

# Function to make an API request and both print and return the results
def get_and_print_api(url):
    r = requests.get(url, auth=(username, token))
    data = r.json()
    for d in data:
        print(d)
    
    # Return the data
    return data


# First, load the user info so we can query the OOI m2m api
user_info = yaml.load(open('../user_info.yaml'))
username = user_info['apiname']
token = user_info['apikey']

# Next, declare the url of ooi-net api
url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv'

# Specify which array we want and see the node options
array = 'CP01CNSM'
nodes = get_and_print_api('/'.join((url, array)))

# Specify which node we want and see the inst options
node = 'RID27'
sensors = get_and_print_api('/'.join((url, array, node)))

# Specify which instrument we want and see the method options
sensor = '04-DOSTAD000'
methods = get_and_print_api('/'.join((url, array, node, sensor)))

# Specify which method and return the available streams
method = 'recovered_host'
streams = get_and_print_api('/'.join((url, array, node, sensor, method)))

# Specify the stream name
stream = 'dosta_abcdjm_dcl_instrument_recovered'

# **====================================================================================================================**
# ### Request Data
# With the array, node, sensor, method, and stream identified after querying the m2m api, we can build the request for actual data. Instead of requesting _all_ of the data available for the array-node-sensor-method-stream, I'll request the data for a particular deployment using the startDateTime and stopDateTime parsed from the deployment csvs. This will make the query for data significantly faster.

data_request_url = '/'.join((url, array, node, sensor, method, stream))

thredds_url = get_thredds_url(data_request_url, None, None, username, token)
thredds_url

datasets = requests.get(thredds_url).text
datasets

datasets = get_netcdf_datasets(thredds_url)
datasets

ds = xr.open

df2 = pd.DataFrame(columns=dfs[0].columns)
for key in dfs.keys():
    df2 = pd.concat([df2, dfs.get(key)])

dfs

ds = ds.swap_dims({'obs':'time'})

ds = ds.sortby('time')

df = ds.to_dataframe()

ds = xr.open_dataset(datasets[0])
ds = ds.swap_dims({'obs':'time'})

# Strip the data
data.drop(columns=[x for x in data.columns if 'qc' in x], inplace=True)

deployment = data[data['deployment'] == 1]
deployment.head()


# First, define necessary functions to make the requests:

def get_thredds_url(data_request_url, min_time, max_time, username, token):
    """
    Returns the associated thredds url for the desired dataset(s)
    """
    
    # Ensure proper datetime format for the request
    if min_time is not None:
        min_time = pd.to_datetime(min_time).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        max_time = pd.to_datetime(max_time).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    
    # Build the query
    params = {
        'beginDT':min_time,
        'endDT':max_time,
    }
    
    # Request the data
    r = requests.get(data_request_url, params=params, auth=(username, token))
    if r.status_code == 200:
        data_urls = r.json()
    else:
        print(r.reason)
        
    # The asynchronous data request is contained in the 'allURLs' key,
    # in which we want to find the url to the thredds server
    for d in data_urls['allURLs']:
        if 'thredds' in d:
            thredds_url = d
    
    return thredds_url


# Next, we want to download the relevant datasets from the thredds server url
def get_netcdf_datasets(thredds_url):
    import time
    datasets = []
    counter = 0
    tds_url = 'https://opendap.oceanobservatories.org/thredds/dodsC/'
    while not datasets:
        datasets = requests.get(thredds_url).text
        urls = re.findall(r'href=[\'"]?([^\'" >]+)', datasets)
        x = re.findall(r'(ooi/.*?.nc)', datasets)
        for i in x:
            if i.endswith('.nc') == False:
                x.remove(i)
        for i in x:
            try:
                float(i[-4])
            except:
                x.remove(i)
        datasets = [os.path.join(tds_url, i) for i in x]
        if not datasets: 
            print(f'Re-requesting data: {counter}')
            counter = counter + 1
            time.sleep(30)
    return datasets


def load_netcdf_datasets(datasets):
    # Determine number of datasets that need to be opened
    # If the time range spans more than one deployment, need xarray.open_mfdatasets
    if len(datasets) > 1:
        try:
            ds = xr.open_mfdataset(datasets)
        except Exception as exc:
            if '_FillValue type mismatch' in exc.args[1]:
                # Retry the request with #fillmismatch argument
                ds = xr.open_mfdataset([x+'#fillmismatch' for x in datasets])
                
        # Need to switch dimensions and sort by the time dimension
        ds = ds.swap_dims({'obs':'time'})
        ds = ds.sortby('time')
        
        return ds
    
    # If a single deployment, will be a single datasets returned
    elif len(datasets) == 1:
        try:
            ds = xr.open_dataset(datasets[0])
        except Exception as exc:
            if '_FillValue type mismatch' in exc.args[1]:
                # Retry the request with #fillmismatch argument
                ds = xr.open_dataset(datasets[0]+'#fillmismatch')
                
        # Need to switch dimensions and sort by the time dimension
        ds = ds.swap_dims({'obs':'time'})
        ds = ds.sortby('time')
                
        return ds
    
    # If there are no available datasets on the thredds server, let user know
    else:
        print('No datasets to open')
        return None


def request_UFrame_data(url, array, node, sensor, method, stream, min_time, max_time, username, token):
    """
    Function which requests a netCDF dataset from OOI UFrame
    and returns a pandas dataframe with the requested dataset
    """
    
    # Build the request url
    data_request_url = '/'.join((url, array, node, sensor, method, stream))
    
    # Query the thredds url
    thredds_url = get_thredds_url(data_request_url, min_time, max_time, username, token)

    # Find and return the netCDF datasets from the thredds url
    datasets = get_netcdf_datasets(thredds_url)
    
    # Load the netCDF files from the datasets
    ds = load_netcdf_datasets(datasets)
    
    # Convert the xarray dataset to a pandas dataframe for ease of use
    df = ds.to_dataframe()
    
    # Return the dataframe
    return df


# **====================================================================================================================**
# ### Sensor Metadata Information & \_FillValues
# In order to properly calculate the daily data availability statistics, we need to know the \_FillValue property for each variable in the dataset. Unfortunately, current OOI netCDF files are not fully CF-compliant, so the \_FillValue property is ignored when loading the netCDF file. We can get around this by querying for the sensor metadata information, which contains the \_FillValues used by Stream Engine to fill out the datasets. 
#
# Once the sensor metadata is loaded, we can parse the metadata to identify the \_FillValue for each variable in the dataset.

# Define a function to return the sensor metadata
def get_sensor_metadata(metadata_url, username=username, token=token):
    
    # Request and download the relevant metadata
    r = requests.get(metadata_url, auth=(username, token))
    if r.status_code == 200:
        metadata = r.json()
    else:
        print(r.reason)
        
    # Put the metadata into a dataframe
    df = pd.DataFrame(columns=metadata.get('parameters')[0].keys())
    for mdata in metadata.get('parameters'):
        df = df.append(mdata, ignore_index=True)
    
    return df


# Define a function to filter the metadata for the relevant variable info
def get_UFrame_fillValues(data, metadata, stream):
    
    # Filter the metadata down to a particular sensor stream
    mdf = metadata[metadata['stream']==stream]
    
    # Based on the variables in data, identify the associated fill values as key:value pair
    fillValues = {}
    for varname in data.columns:
        fv = mdf[mdf['particleKey'] == varname]['fillValue']
        # If nothing, default is NaN
        if len(fv) == 0:
            fv = np.nan
        else:
            fv = list(fv)[0]
            # Empty values in numpy default to NaNs
            if fv == 'empty':
                fv = np.nan
            else:
                fv = float(fv)
        # Update the dictionary
        fillValues.update({
            varname: fv
        })
        
    return fillValues


# Get the sensor metadata
metadata = get_sensor_metadata('/'.join((url, array, node, sensor, 'metadata')), username, token)
metadata.head()

# Get the fill values
fillValues = get_UFrame_fillValues(data, metadata, stream)
fillValues


# **====================================================================================================================** 
# ### Calculate Data Availability 
# With the downloaded data and associated sensor metadata, we can go ahead and calculate the percent of data available per deployment calculated on a per-day business. The method used is, as follows:
# 1. Bin the data into day-long periods spanning midnight-to-midnight, with the first and last days shortened by the respective deployment and recovery time.
# 2. For each variable in the dataset, make the following calculate on a per-day basis:
#     1. Calculate the number of NaNs in the dataset
#     2. Calculate the number of \_FillValues in the dataset
#     3. Subtract the number of NaNs and FillValues from the total number of data points
#     4. Divide by the total number of data points to get the percent data available
# 3. Save the percent data available in each day

# Define a function to calculate the data availability for a day
def calc_UFrame_data_availability(subset_data, fillValues):
    
    # Initialize a dictionary to store results
    data_availability = {}
    
    for col in subset_data.columns:
        
        # Check for NaNs in each col
        nans = len(subset_data[subset_data[col].isnull()][col])
    
        # Check for values with fill values
        fv = fillValues.get(col)
        if fv is not None:
            if np.isnan(fv):
                fills = 0
            else:
                fills = len(subset_data[subset_data[col] == fv][col])
        else:
            fills = 0
        
        # Get the length of the whole dataframe
        num_data = len(subset_data[col])
        
        # If there is no data in the time period, 
        if num_data == 0:
            data_availability.update({
            col: 0,
            })
        else:
            # Calculate the statistics for the nans, fills, and length
            num_bad = nans + fills
            num_good = num_data - num_bad
            per_good = (num_good/num_data)*100
    
            # Update the dictionary with the stats for a particular variable
            data_availability.update({
                col: per_good
            })
        
    return data_availability


# Define a function to bin the time period into midnight-to-midnight days
def time_periods(startDateTime, stopDateTime):
    """
    Generates an array of dates with midnight-to-midnight
    day timespans. The startDateTime and stopDateTime are
    then append to the first and last dates.
    """
    startTime = pd.to_datetime(startDateTime)
    if type(stopDateTime) == float:
        stopDateTime = pd.datetime.now()        
    stopTime = pd.to_datetime(stopDateTime)
    days = pd.date_range(start=startTime.ceil('D'), end=stopTime.floor('D'), freq='D')
    
    # Generate a list of times
    days = [x.strftime('%Y-%m-%dT%H:%M:%SZ') for x in days]
    days.insert(0, startTime.strftime('%Y-%m-%dT%H:%M:%SZ'))
    days.append(stopTime.strftime('%Y-%m-%dT%H:%M:%SZ'))
    
    return days


def make_dirs(path):
    """
    Function which checks if a path exists,
    and if it doesn't, makes the directory.
    """
    check = os.path.exists(path)
    if not check:
        os.makedirs(path)
    else:
        print(f'"{path}" already exists' )


# **====================================================================================================================**
# # Download all data sets
#
# Can't successfully download data because fuck UFrame. I can't do anything programmatically.

array, node, sensor, method, stream

save_dir = '/home/andrew/Documents/OOI-CGSN/QAQC_Sandbox/Data_Review/Data Availability/Results'

deploy_df

data.head()

metadata.head()

# Now, calculate the data availability
for deploy_num, start_time, stop_time in deploy_df.values:
    
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


results

results.to_csv('/'.join((save_path,'Data_Availability.csv')))



# +
# Get the sensor metadata
metadata = get_sensor_metadata('/'.join((url, array, node, sensor, 'metadata')), username, token)

# Iterate through the 
for deploy_num, start_time, end_time in deploy_df.values:
    
    # create an array of days from midnight-to-midnight, with shortened first and last days
    days = time_periods(start_time, end_time)
    
    # Get the start (min_time) and stop (max_time) time range in which to request data and format properly
    min_time = pd.to_datetime(days[0]).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    max_time = pd.to_datetime(days[-1]).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    
    # Request the data
    try:
        data = request_UFrame_data(url, array, node, sensor, method, stream, min_time, max_time, username, token)
    except:
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
