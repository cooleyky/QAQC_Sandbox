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

# # Data Stream Identifier for QARTOD Parameters
#
# #### Author: Andrew Reed
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

# Import the OOINet M2M tool
sys.path.append("/home/andrew/Documents/OOI-CGSN/ooinet/ooinet/")
from m2m import M2M

import matplotlib.pyplot as plt
# %matplotlib inline

# #### Set OOINet API access
# In order access and download data from OOINet, need to have an OOINet api username and access token. Those can be found on your profile after logging in to OOINet. Your username and access token should NOT be stored in this notebook/python script (for security). It should be stored in a yaml file, kept in the same directory, named user_info.yaml.

# Import user info for accessing UFrame
userinfo = yaml.load(open('../user_info.yaml'))
username = userinfo['apiname']
token = userinfo['apikey']

# #### Initialize the connection to OOINet

OOI = M2M(username, token)

# ---
# ## Identify Data Streams
# This section is necessary to identify all of the data stream associated with a specific instrument. This can be done by querying UFrame and iteratively walking through all of the API endpoints. The results are saved into a csv file so this step doesn't have to be repeated each time.
#
# First, check if the datasets have already been downloaded and saved locally:

os.listdir("/media/andrew/Files/Instrument_Data/DOSTA/")

datasets = OOI.search_datasets(instrument="DOSTA")
datasets = datasets.sort_values(by="array")
datasets

# Save the datasets locally so this process doesn't have to be re-run
datasets.to_csv("/media/andrew/Files/Instrument_Data/DOSTA/DOSTA_datasets.csv", index=False)

# #### CGSN Datasets
# With all of the datasets for the given instrument identified, want to filter for just the data from CGSN-relevant instruments on the **Coastal Pioneer (CP)**, **Global Irminger (GI)**, **Global Argentine (GA)**, **Global Station Papa (GP)**, and **Global Southern (GS)** arrays.

cgsn_mask = datasets["array"].apply(lambda x: True if x.startswith(("CP", "GA", "GI", "GP", "GS")) else False)
cgsn_datasets = datasets[cgsn_mask]
cgsn_datasets.head()

# ---
# ## Single Reference Designator
# The reference designator acts as a key for an instrument located at a specific location. First, select a reference designator (refdes) to request data from OOINet.

reference_designators = sorted(cgsn_datasets["refdes"])
refdes = reference_designators[0]
refdes

# #### Sensor Vocab
# The vocab provides information about the instrument model and type, its location (with descriptive names), depth, and manufacturer. Get the vocab for the given reference designator.

vocab = OOI.get_vocab(refdes)
vocab

# #### Sensor Deployments
# Download the deployment information for the selected reference designator:

deployments = OOI.get_deployments(refdes)
deployments

# #### Sensor Data Streams
# Next, select the specific data streams for the given reference designator

datastreams = OOI.get_datastreams(refdes)
datastreams

# #### Sensor Metadata
#
# Next, we want to download the metadata associated with the specific reference we selected above:

metadata = OOI.get_metadata(refdes)
metadata

# #### Sensor Parameters
# Each instrument returns multiple parameters containing a variety of low-level instrument output and metadata. However, we are interested in science-relevant parameters for calculating the relevant QARTOD test limits. We can identify the science parameters based on the preload database, which designates the science parameters with a "data level" of L1 or L2. 
#
# Consequently, we through several steps to identify the relevant parameters. First, we query the preload database with the relevant metadata for a reference designator. Then, we filter the metadata for the science-relevant data streams. 

data_levels = OOI.get_parameter_data_levels(metadata)

mask = metadata["pdId"].apply(lambda x: OOI.filter_parameter_ids(x, data_levels))
metadata = metadata[mask]
metadata

# #### Download Data
# To access data, there are two applicable methods. The first is to download the data and save the netCDF files locally. The second is to access and process the files remotely on the THREDDS server, without having to download the data. 

# Select a method and stream to download
method = "recovered_host"
stream = "dosta_abcdjm_dcl_instrument_recovered"

# Get the THREDDS url for the reference designator/method/stream
thredds_url = OOI.get_thredds_url(refdes, method, stream)
thredds_url

# Get the catalog
catalog = OOI.get_thredds_catalog(thredds_url)
catalog

# Identify the netCDF files from the THREDDS catalog
netCDF_files = OOI.parse_catalog(catalog, exclude=["gps", "CTD"])
for file in netCDF_files:
    print(file)
    print("\n")

# ##### Download datasets
# The first option is to download the relevant netCDF files to a local directory. This approach allows for the data to be access more quickly in the future, without having to go through the steps of requesting and waiting for the netCDF files to be generated on the THREDDS server. The downside is that there are terabytes of data on OOINet and downloading and saving all of the data, particularly when more is being generated every day, is not a particularly efficient approach.

# Specify the local directory to save the data to
save_dir = f"/media/andrew/Files/Instrument_Data/PCO2W/{refdes}/{method}/{stream}"

# ##### Remote access
# The second option is to remotely access the netCDF files directly on the THREDDS server. This approach avoids having to download all of the data to a local directory. However, it is not persistent across time.

pco2w = OOI.(a)





# ## Process Data
# With the 

blank_files = os.listdir(save_dir)
blank_files = ["/".join((save_dir, f)) for f in os.listdir(save_dir)]

with xr.open_mfdataset(blank_files, concat_dim="obs") as blanks:
    blanks = blanks.swap_dims({"obs":"time"})
    blanks = blanks.sortby("time")

blanks

# +
fig, ax = plt.subplots(figsize=(12,8))

for spectrum in np.unique(blanks.spectrum):
    ax.plot(blanks.time, blanks.where(blanks.spectrum==spectrum, drop=True).blank_light_measurements,
            linestyle="", marker=".", label=str(spectrum))
ax.grid()
ax.legend()
ax.set_ylabel("Counts")

fig.autofmt_xdate()


# +
fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(12,8))

ax0.plot(ds.time, ds.absorbance_blank_434/16384, linestyle="", marker=".", label="Blank 434")
ax0.plot(ds.time, ds.absorbance_blank_620/16384, linestyle="", marker=".", label="Blank 620")
ax0.legend()
ax0.set_ylabel("Blank Counts")
ax0.grid()
ax0.set_title(ds.attrs["id"])

ax1.plot(ds.time, ds.absorbance_ratio_434, linestyle="", marker=".", label="Blank Ratio 434")
ax1.plot(ds.time, ds.absorbance_ratio_620, linestyle="", marker=".", label="Blank Ratio 620")
ax1.legend()
ax1.grid()

# +
# Plot the data to take a look at it
fig, ax = plt.subplots(figsize=(12,8))

for depNum in np.unique(ds.deployment):
    ax.plot(ds.where(ds.deployment==depNum).time, ds.where(ds.deployment==depNum).pco2_seawater,
           linestyle="", marker=".", label=str(depNum))
ax.set_ylabel(ds.pco2_seawater.attrs["long_name"], fontsize=12)
ax.set_xlabel(ds.time.attrs["long_name"], fontsize=12)
ax.set_ylim(0, 2500)
ax.grid()
ax.set_title(ds.attrs["id"])
ax.legend()

fig.autofmt_xdate()
# -

blanks.blank_light_measurements

# Drop the extra spectrums because they aren't needed
ds = ds.where(ds.spectrum==0, drop=True)
ds

pco2w

parameters = metadata["particleKey"].unique()
parameters

gross_range_mask

# Pass a gross_range mask of (200, 2000) to it first
gross_range_mask = (ds.pco2_seawater.values >= 200) & (ds.pco2_seawater.values <= 2000)
gross_range_mask = gross_range_mask.reshape(-1)

# Next, calculate the user range based on the gross range filtered data
np.nanmean(ds.pco2w_thermistor_temperature.values), np.nanstd(ds.pco2w_thermistor_temperature.values)

ds.pco2_seawater[gross_range_mask].values.std()

# ---
# ## Process the Data
# This is for processing the data

ds

fig, ax = plt.subplots(figsize=(12,8))

# +
fig, ax = plt.subplots(figsize=(12,8))

for spectrum in np.unique(ds.spectrum):
    ax.plot(ds.time, ds.where(ds.spectrum==spectrum, drop=True).light_measurements, linestyle="", marker=".", label=str(spectrum))
ax.legend()
ax.grid()

# +
fig, ax = plt.subplots(figsize=(12,8))

ax.plot(ds.time, ds.where(ds.spectrum==0, drop=True).light_measurements, linestyle="", marker=".")
ax.plot(ds.time, ds.where(ds.spectrum==8, drop=True).light_measurements, linestyle="", marker=".")
ax.grid()
# -

ds.where(ds.spectrum==0, drop=True).light_measurements.shape

ds.light_measurements.shape



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


