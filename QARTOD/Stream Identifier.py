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

# # Stream Identifier for QARTOD Parameters
#
# ### Purpose
# The purpose of this notebook is to identify the necessary UFrame data streams and data parameters from CGSN-controlled instruments for quality control by QARTOD algorithms. 

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

# #### Set OOINet API access
# In order access and download data from OOINet, need to have an OOINet api username and access token. Those can be found on your profile after logging in to OOINet. Your username and access token should NOT be stored in this notebook/python script (for security). It should be stored in a yaml file, kept in the same directory, named user_info.yaml.

# Import user info for accessing UFrame
userinfo = yaml.load(open('../user_info.yaml'))
username = userinfo['apiname']
token = userinfo['apikey']

# #### Define relevant UFrame api urls paths

data_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv'
anno_url = 'https://ooinet.oceanobservatories.org/api/m2m/12580/anno/find'
vocab_url = 'https://ooinet.oceanobservatories.org/api/m2m/12586/vocab/inv'
asset_url = 'https://ooinet.oceanobservatories.org/api/m2m/12587'
deploy_url = asset_url + '/events/deployment/query'
preload_url = 'https://ooinet.oceanobservatories.org/api/m2m/12575/parameter'
cal_url = asset_url + '/asset/cal'


# +
# Define some useful functions for working with the UFrame api
def get_api(url):
    r = requests.get(url, auth=(username, token))
    data = r.json()
    return data

# Function to make an API request and print the results
def get_and_print_api(url):
    r = requests.get(url, auth=(username, token))
    data = r.json()
    #for d in data:
     #   print(d)
    
    # Return the data
    return data
        
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

# **==================================================================================================================**
# ### Data Streams
# First, we want to identify all of the instruments and their associated data streams located on the CGSN arrays. This involves querying UFrame and walking-through all the reference designators, then saving the **array, node, sensor, method, and stream** information in a table.

def get_cgsn_data_streams(data_url, include_eng=True):
    """Query and download all CGSN data streams into a table."""
    
    # Initialize the table to store the results
    data_streams = pd.DataFrame(columns=['array', 'node', 'sensor', 'method', 'stream'])
    
    # Walk through all OOINet CGSN data arrays
    # Get the Pioneer and Global arrays
    arrays = [arr for arr in get_and_print_api(data_url) if arr.startswith(('CP','G','CE'))]
    for array in arrays:
        # Iterate through the nodes of a particular array
        nodes = get_and_print_api('/'.join((data_url, array)))
        for node in nodes:
            # Determine if to ignore engineering streams (default true)
            if include_eng:
                sensors = get_and_print_api('/'.join((data_url, array, node)))
            else:
                sensors = [sen for sen in get_and_print_api('/'.join((data_url, array, node))) if 'ENG' not in sen]
            # Iterate through the sensors for an array-node
            for sensor in sensors:
                # Iterate through all the methods for each array-node-sensor
                methods = get_and_print_api('/'.join((data_url, array, node, sensor)))
                for method in methods:
                    # Iterate through all the streams for each array-node-sensor-method
                    streams = get_and_print_api('/'.join((data_url, array, node, sensor, method)))
                    for stream in streams:
                        # Save into the data table
                        data_streams = data_streams.append({
                            'array':array,
                            'node':node,
                            'sensor':sensor,
                            'method':method,
                            'stream':stream
                        }, ignore_index=True)
                        
    # Return the resulting dataframe
    return data_streams            


# If you don't have the table of data streams or want to download the table again, run the following code:

# +
# # Download the data streams
# data_streams = get_cgsn_data_streams(data_url)
# data_streams.head()

# +
# # Save the data streams
# data_streams.to_csv('/media/andrew/Files/OOINet/data_streams.csv', index=False)
# -

# If you already have the data streams, load them into the notebook:

data_streams = pd.read_csv('/media/andrew/Files/OOINet/data_streams.csv')
data_streams.head()


# #### Filter Data Streams
# Next, we can filter the data streams down for only Pioneer, Endurance, Global, etc.

# +
def cgsn_mask(x):
    if x.startswith(('CG','CP','GI','GA','GS')):
        if 'MOAS' not in x:
            return True
        else:
            return False
    else:
        return False
    
def ce_mask(x):
    if x.startswith(('CE')):
        if 'MOAS' not in x:
            return True
        else:
            return False
    else:
        return False


# -

moas = data_streams['array'].apply(lambda x: True if 'MOAS' in x else False)
cgsn = data_streams['array'].apply(lambda x: cgsn_mask(x) )
ce = data_streams['array'].apply(lambda x: ce_mask(x))


# data_streams = data_streams[cgsn]

# **==================================================================================================================**
# ### Data Parameters
# With the data streams identified, the next step is to get the parameters to be tested associated with each data stream. 

# +
# Define functions for getting and filtering sensor metadata
def get_sensor_metadata(metadata_url, username=username, token=token):
    """Download the metadata for a specific sensor/instrument."""
   
    # Request and download the relevant metadata
    r = requests.get(metadata_url, auth=(username, token))
    if r.status_code == 200:
        mdata = r.json()
    else:
        print(r.reason)
        
    # Put the metadata into a dataframe
    metadata = pd.DataFrame.from_dict(mdata.get('parameters'))
    
    return metadata


def get_parameter_data_levels(metadata):
    """Return a dictionary of the data level for each parameter id."""
    pdIds = np.unique(metadata['pdId'])
    pid_dict = {}
    for pid in pdIds:
        # Query the preload information
        preload_info = get_and_print_api('/'.join((preload_url, pid.strip('PD'))))
        data_level = preload_info.get('data_level')
        # Update the dictionary
        pid_dict.update({pid: data_level})
    
    return pid_dict


def filter_parameter_ids(pdId, pid_dict):
    """Filter for only science parameters (data level = 1 or 2)."""
    data_level = pid_dict.get(pdId)
    if data_level == 1:
        return True
    elif data_level == 2:
        return True
    else:
        return False


# -

# #### Sensor Data Streams
# Next, select the specific data streams for a given sensor, e.g. all the data streams which relate to CTDs, or DOSTAs, etc.

sensor = 'PCO2A'
mask = data_streams['sensor'].apply(lambda x: True if sensor in x else False)
sensor_streams = data_streams[mask]
sensor_streams.head()

# Build the reference designator for the given data streams from the array-node-sensor:

sensor_streams['refdes'] = sensor_streams['array'] + '-' + sensor_streams['node'] + '-' + sensor_streams['sensor']

# #### Sensor Metadata
# Next, we want to download the metadata associated with the specific sensor we selected above. We do this by iterating through the available data streams for the sensor. First, check if we have the sensor metadata already.

# Load the sensor metadata
sensor_metadata = pd.read_csv('/media/andrew/Files/Instrument_Data/PCO2A/PCO2A_metadata.csv')
sensor_metadata.head()

# If we haven't already downloaded the sensor metadata, we can use the code block below to download that associated metadata for the given data streams:

# Now get the metadata
for refdes in np.unique(sensor_streams['refdes']):
    
    # Query the metadata for a particular refdes
    array, node, sensor = refdes.split('-', 2)
    metadata = get_sensor_metadata('/'.join((data_url, array, node, sensor, 'metadata')))
    metadata['refdes'] = refdes
    
    # Get the data levels of the metadata
    data_levels = get_parameter_data_levels(metadata)
    
    # Filter the metadata based on the data_levels to retain only those we want tested
    mask = metadata['pdId'].apply(lambda x: filter_parameter_ids(x, data_levels))
    metadata = metadata[mask]
    
    # Now, dynamically build a metadata dataframe for all the reference designatores
    try:
        sensor_metadata = sensor_metadata.append(metadata)
    except:
        sensor_metadata = metadata

# Check the resulting table of metadata
sensor_metadata.head()

# Save the metadata
sensor_metadata.to_csv('/media/andrew/Files/Instrument_Data/PCO2A/PCO2A_metadata.csv', index=False)

# #### Sensor Parameters
# Next, we merge the sensor metadata table with the sensor data streams table to get a table with the parameters for each sensor data stream:

# Merge on the reference designator and stream keys
sensor_parameters = sensor_streams.merge(sensor_metadata, left_on=['refdes','stream'], right_on=['refdes','stream'])
sensor_parameters.head()

# Save the sensor parameters
sensor_parameters.to_csv('/media/andrew/Files/Instrument_Data/PCO2A/PCO2A_parameters.csv', index=False)


# **====================================================================================================================**
# ## User range
# Next, for each array, need to calculate the mean and standard deviations for the given parameters

def get_async_url(data_request_url, username, token, min_time=None, max_time=None):
    """
    Return the associated async url for the desired data streams.
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
    
    # Request the download urls
    urls = requests.get(data_request_url, auth=(username, token)).json()

    # Get the async url
    async_url = urls['allURLs'][1]
    
    return async_url    


def get_netCDF_datasets(async_url):
    """
    Query the asynch url and return the netCDF datasets.
    """
    # This block of code works times the request until fufilled or the request times out (limit=10 minutes)
    start_time = time.time()
    check_complete = async_url + '/status.txt'
    r = requests.get(check_complete)
    while r.status_code != requests.codes.ok:
        check_complete = async_url + '/status.txt'
        r = requests.get(check_complete)
        elapsed_time = time.time() - start_time
        if elapsed_time > 10*60:
            print('Request time out')
        time.sleep(5)

    # Identify the netCDF urls
    datasets = requests.get(async_url).text
    x = re.findall(r'href=["](.*?.nc)', datasets)
    for i in x:
        if i.endswith('.nc') == False:
            x.remove(i)
        for i in x:
            try:
                float(i[-4])
            except:
                x.remove(i)
        datasets = [os.path.join(async_url, i) for i in x]
        
    return datasets


# #### Single-data stream
# If we want to download the datasets for a single datastream, we can use the following approach:

# List the available reference designators for the sensor
sensor_parameters['refdes'].unique()

# Select a single reference designator
refdes = 'CP01CNSM-SBD12-04-PCO2AA000'

# Get the associated data streams for the given reference designator
refdes_streams = sensor_parameters[sensor_parameters['refdes'] == refdes]
refdes_streams

# +
# Data request url
data_request_url = '/'.join((data_url, array, node, sensor, method, stream))

# Get the asynch url for the data request url
asynch_url = get_async_url(data_request_url, username, token)

# Get the netCDF datasets from the asynch url
datasets = get_netCDF_datasets(asynch_url)

# Download the datasets and save locally
datasets = [dset for dset in datasets if 'CTD' not in dset]
save_dir = '/'.join((os.getcwd(), array, node, sensor, method))
download_netCDF_datasets(datasets, save_dir)


# -

# #### Need to develop an improved method for getting the netCDF files (can't do json because only for synchronous requests)

def download_netCDF_datasets(datasets, save_dir):
    """Download the netCDF datasets for asynch"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    for dset in datasets:
        filename = wget.download(dset, out=save_dir)


array = 'CP01CNSM'
node = 'MFD37'
sensor = '04-DOSTAD000'
method = 'recovered_host'
stream = 'dosta_abcdjm_dcl_instrument_recovered'

data_request_url = '/'.join((data_url, array, node, sensor, method, stream))
data_request_url

# Next, load the netCDF files
ds = xr.open_mfdataset(['/'.join((save_dir,))])


# Now, want to request all of the data for a particular datastream
def get_thredds_url(data_request_url, username, token, min_time=None, max_time=None):
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


ds = xr.open_mfdataset(['/'.join((save_dir, x)) for x in os.listdir(save_dir)])

ds.to_netcdf('dosta_practice.nc')


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



# Load in the relevant data parameters
qartod = pd.read_csv('Results/CTD_parameters.csv')
qartod.drop(columns='Unnamed: 0', inplace=True)
qartod

refdes = np.unique(qartod['refdes'])
refdes

refdes = 'CP01CNSM-RID27-03-CTDBPC000'
subset = qartod[qartod['refdes'] == refdes]
subset

# Problem, the different data streams have different parameter keys. I don't want to accidentally ignore/miss some
# parameter keys based on the different methods/streams.
subset.drop_duplicates(subset=['refdes','stream','pdId'])

# Remove the bad data streams in the subset
mask = subset['method'].apply(lambda x: False if 'bad' in x else True)
subset = subset[mask]
subset

subset2 = subset.drop_duplicates(subset=['method','stream','pdId'])

pdid = 

# Iterate over the methods in a heirarchy
methods = np.unique(subset['method'])
for m in ('inst','host','recov','tele'):
    method = [x for x in methods if m in x]
    if len(method) > 0:
        break
method = method[0]
method

stream = subset[subset['method'] == method]['stream'].unique()[0]
stream

parameters = subset[subset['stream'] == stream]['pdId'].unique()

data_streams = pd.DataFrame(columns=['refdes','method','stream'])
refdes = np.unique(qartod['refdes'])
for ref in refdes:
    # Get a subset of the data for a given reference designator
    subset = qartod[qartod['refdes'] == ref]
    
    # Drop duplicate refdes, methods, and streams
    subset.drop_duplicates(subset=['refdes','method','stream'], inplace=True)
    
    # Next, remove the "bad" data streams in the subset
    mask = subset['method'].apply(lambda x: False if 'bad' in x else True)
    subset = subset[mask]
    # If all of the data streams are "bad", then what to do? I'm leaving blank
    if len(subset) == 0:
        continue
    
    # Iterate over the methods in a heirarchy
    methods = subset['method']
    for m in ('inst','host','recov','tele'):
        method = [x for x in methods if m in x]
        if len(method) > 0:
            break
    method = method[0]
    
    # Now get the stream associated with the method
    stream = subset[subset['method'] == method]['stream']
    stream = list(stream)[0]
    
    # Put the data into a new dataframe
    data_streams = data_streams.append({
        'refdes':ref,
        'method':method,
        'stream':stream
    }, ignore_index=True)


data_streams

# If any results already exist, load them
result = pd.read_csv("Results/Global_user_range.csv")
result.drop(columns='Unnamed: 0', inplace=True)
result


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


data_streams

# **==================================================================================================================**

data_streams.iloc[i]

# Go with the first entry to develop the code to do this automatically
i = 210 # 160, 206 not being returned
array, node, sensor = data_streams['refdes'].iloc[i].split('-',2)
method = data_streams['method'].iloc[i]
stream = data_streams['stream'].iloc[i]
url = '/'.join((data_url, array, node, sensor, method, stream))
url


# Request the data for the given refdes
thredds_url = get_thredds_url('/'.join((data_url, array, node, sensor, method, stream)), username, token)
thredds_url

# Get the datasets for the reference designator
datasets = get_netcdf_datasets(thredds_url)

datasets

datasets = [d for d in datasets if 'ENG' not in d]
datasets

ds = load_netcdf_datasets(datasets)

# Now, we want to get the important variables for the given reference designator
refdes = data_streams['refdes'].iloc[i]
variables = np.unique(qartod[qartod['refdes'] == refdes]['particleKey'])
variables

# +
#variables[-1] = 'temperature'
# -

# Take the variables and stick into a pandas dataframe
df = ds[variables].to_dataframe()
df.head()

# Calculate the statistics for each variable
# result = pd.DataFrame(columns=['refdes','particleKey','user_range'])
for var in variables:
    mean = df[var].mean()
    std = df[var].std()
#    if var == 'temperature':
 #       var = 'temp'
    result = result.append({
        'refdes':refdes,
        'particleKey':var,
        'user_range':(mean-3*std, mean+3*std)
    }, ignore_index=True)
result.tail()   

# +
# If the data streams didn't return anything
# -

result.to_csv('Results/Global_user_range.csv')





# Drop any duplicates from the results dataframe
dupes = result[result.duplicated()]
dupes

qartod.columns

# +
# Need to merge with the data_stream dataframe on the RefDes and particleKey 
# -

qartod = pd.read_csv('Results/Global_CTDs.csv')
qartod.drop(columns='Unnamed: 0', inplace=True)
qartod

qartod = qartod.merge(result, left_on=['refdes','particleKey'], right_on=['refdes','particleKey'], how='outer')

qartod



qartod = pd.read_csv('Results/Pioneer_CTD_user_ranges.csv')

# Add in the gross range based on the 
pressure_range = (0, 6000)
temp_range = (-5, 35)
ctdbp_cond_range = (0, 9)
ctdmo_cond_range = (0, 6)
ctdpf_cond_range = (0, 9)


def add_gross_range(pkey):
    if 'press' in pkey:
        gross_range = (0, 6000)
    elif 'temp' in pkey:
        gross_range = (-5, 35)
    elif 'cond' in pkey:
        if 'ctdbp' in pkey:
            gross_range = (0, 9)
        elif 'ctdmo' in pkey:
            gross_range = (0, 6)
        else:
            gross_range = (0, 9)
    else:
        gross_range = None
    
    return gross_range


qartod['gross_range'] = qartod['particleKey'].apply(lambda pkey: add_gross_range(pkey))
qartod

qartod.drop(columns='Unnamed: 0', inplace=True)
qartod

qartod.to_csv('Results/Pioneer_CTD_gross_range.csv')

for k in np.unique(qartod['particleKey']):
    print(k)

# +
# With the variables calculated, can 
# -

array = 'CP01CNSM'
node = 'MFD37'
sensor = '03-CTDBPD000'
method = 'recovered_host'
stream = 'ctdbp_cdef_dcl_instrument_recovered'

url = '/'.join((data_url, array, node, sensor, method, stream))
url

thredds_url = get_thredds_url(url, username, token)

datasets = get_netcdf_datasets(thredds_url)

datasets

ds = load_netcdf_datasets(datasets)
ds

df = ds.to_dataframe()

df['pressure'].mean()

np.unique(qartod[qartod['refdes'] == 'CP01CNSM-MFD37-03-CTDBPD000']['particleKey'])


