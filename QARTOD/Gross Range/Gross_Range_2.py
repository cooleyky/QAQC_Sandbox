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

# Import libraries that will be used
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
userinfo = yaml.load(open('../../user_info.yaml'))
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
# First, we want to identify all of the instruments and their associated data streams located on the CGSN arrays. This involves querying UFrame and walking-through all the reference designators, then saving the **array, node, sensor, method, and stream** information in a table which is saved locally.

def get_cgsn_data_streams(data_url, include_eng=False):
    """Query and download all CGSN data streams into a table."""
    
    # Initialize the table to store the results
    data_streams = pd.DataFrame(columns=['array', 'node', 'sensor', 'method', 'stream'])
    
    # Walk through all OOINet CGSN data arrays
    # Get the Pioneer and Global arrays
    arrays = [arr for arr in get_and_print_api(data_url)]
    for arr in arrays:
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

# # Download the data streams
data_streams = get_cgsn_data_streams(data_url)
data_streams.head()

# +
# # Save the data streams
# data_streams.to_csv('/media/andrew/Files/OOINet/data_streams.csv', index=False)
# -

# If you already have the data streams, load them into the notebook:

data_streams = pd.read_csv('/media/andrew/Files/OOINet/data_streams.csv')
data_streams.head()

data_streams["array"].unique()


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

# Filter the data streams for only the relevant cgsn data streams:

data_streams = data_streams[cgsn]


# **==================================================================================================================**
# ### Data Parameters
# With the data streams identified, the next step is to get the parameters to be tested associated with each data stream. 

# +
# Define functions for getting and filtering sensor metadata
def get_metadata(metadata_url, username=username, token=token):
       
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
    # Check if pdId should be kept
    data_level = pid_dict.get(pdId)
    if data_level == 1:
        return True
    elif data_level == 2:
        return True
    else:
        return False


def get_sensor_metadata(sensor_streams):
    """Download the metadata for the given sensor."""
    
    # Now get the metadata
    for refdes in np.unique(sensor_streams['refdes']):

        # Query the metadata for a particular refdes
        array, node, sensor = refdes.split('-', 2)
        metadata = get_metadata('/'.join((data_url, array, node, sensor, 'metadata')))
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
            
    # Return the results
    return sensor_metadata


# -

# #### Sensor Data Streams
# Next, select the specific data streams for a given instrument type, e.g. all the data streams which relate to CTDs, or DOSTAs, etc:

inst = ''
mask = data_streams['sensor'].apply(lambda x: True if inst in x else False)
sensor_streams = data_streams[mask]
sensor_streams.head()

# Build the reference designator for the given data streams from the array-node-sensor:

sensor_streams['refdes'] = sensor_streams['array'] + '-' + sensor_streams['node'] + '-' + sensor_streams['sensor']

# Exclude any engineering streams in the system:

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
    metadata = get_metadata('/'.join((data_url, array, node, sensor, 'metadata')))
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

sensor_metadata.head()

# Check the resulting table of metadata
sensor_metadata.head()

# Save the metadata
sensor_metadata.to_csv('/media/andrew/Files/OOINet/metadata.csv', index=False)

# #### Sensor Parameters
# Next, we merge the sensor metadata table with the sensor data streams table to get a table with the parameters for each sensor data stream:

# Merge on the reference designator and stream keys
sensor_parameters = sensor_streams.merge(sensor_metadata, left_on=['refdes','stream'], right_on=['refdes','stream'])
sensor_parameters.head()

# # Save the sensor parameters
sensor_parameters.to_csv('/media/andrew/Files/OOINet/parameters.csv', index=False)

# **====================================================================================================================**
# ## Download Data
# Next, we need to download the data for each sensor in order to process the data and get the desired gross range values for the given sensor. To start with, we'll step through downloading and then processing a single sensor. We are going to request all the netCDF datasets for a given sensor reference-designator, i.e. all of the available data for a single location in the ocean. 
#
# There are three approaches to requesting the data: (1) Synchronous requests which return data in json format; (2) Asynchronous requests from the THREDDS data server which returns netCDF datasets; and (3) Asynchronous requests from the web archive which returns netCDF datasets. Option (1) is limited to 1000 data points, which is not suitable for our purposes. Option (2) allows for programmatically and directly opening datasets from the THREDDS server. However, the THREDDS server access is slow and generates netCDF files which are non-standard, which creates problems when working with xarray and does not lend itself to automation.
#
# Below, we follow Option (3). We will request netCDF files directly from the data server. The drawback to this option is that it requires an intermediate step of downloading the netCDF files directly to your harddisk. While this notebook has a built in purge of the downloaded datasets after processing to avoid eating up disk space, for certain instruments (BOTPT, hydrophone data, some spectral instruments) this is suboptimal. However, this approach is faster than working with the THREDDS server (because the data becomes local for operations), and the downloaded netCDF files are CF-conforming and thus easily manipulated with xarray.

# Completely new set of functions to download the datasets in a 

# +
from xml.dom import minidom
from urllib.request import urlopen
from urllib.request import urlretrieve

def get_elements(url, tag_name, attribute_name):
    """Get elements from an XML file"""
    # usock = urllib2.urlopen(url)
    usock = urlopen(url)
    xmldoc = minidom.parse(usock)
    usock.close()
    tags = xmldoc.getElementsByTagName(tag_name)
    attributes=[]
    for tag in tags:
        attribute = tag.getAttribute(attribute_name)
        attributes.append(attribute)
    return attributes


def get_thredds_url(data_request_url, min_time=None, max_time=None, username=None, token=None):
    """
    Return the url for the THREDDS server for the desired dataset(s).

    Args:
        data_request_url - this is the OOINet url with the platform/node/sensor/
            method/stream information
        min_time - optional to limit the data request to only data after a
            particular date. May be None.
        max_time - optional to limit the data request to only data before a
            particular date. May be None.
        username - your OOINet username
        token - your OOINet authentication token

    Returns:
        thredds_url - a url to the OOI Thredds server which contains the desired
            datasets
    """

    # Ensure proper datetime format for the request
    if min_time is not None:
        min_time = pd.to_datetime(min_time).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        max_time = pd.to_datetime(max_time).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    # Build the query
    params = {
        'beginDT': min_time,
        'endDT': max_time,
    }

    # Request the data
    r = requests.get(data_request_url, params=params, auth=(username, token))
    if r.status_code == 200:
        data_urls = r.json()
    else:
        print(r.reason)
        return None

    # The asynchronous data request is contained in the 'allURLs' key,
    # in which we want to find the url to the thredds server
    for d in data_urls['allURLs']:
        if 'thredds' in d:
            thredds_url = d

    return thredds_url


def get_thredds_catalog(thredds_url):
    """Get the dataset catalog for the requested data stream."""
    
    # ==========================================================
    # Parse out the dataset_id from the thredds url
    server_url = 'https://opendap.oceanobservatories.org/thredds/'
    dataset_id = re.findall(r'(ooi/.*)/catalog', thredds_url)[0]
    
    # ==========================================================
    # This block of code checks the status of the request until
    # the datasets are read; will timeout if longer than 8 mins
    status_url = thredds_url + '?dataset=' + dataset_id + '/status.txt'
    status = requests.get(status_url)
    start_time = time.time()
    while status.status_code != requests.codes.ok:
        elapsed_time = time.time() - start_time
        status = requests.get(status_url)
        if elapsed_time > 10*60:
            print(f'Request time out for {thredds_url}')
            return None
        time.sleep(5)

    # ============================================================
    # Parse the datasets from the catalog for the requests url
    catalog_url = server_url + dataset_id + '/catalog.xml'
    catalog = get_elements(catalog_url, 'dataset', 'urlPath')
    
    return catalog


def parse_catalog(catalog, exclude=[]):
    """
    Parses the THREDDS catalog for the netCDF files. The exclude
    argument takes in a list of strings to check a given catalog
    item against and, if in the item, not return it.
    """
    datasets = [citem for citem in catalog if citem.endswith('.nc')]
    if type(exclude) is not list:
        raise ValueError(f'arg exclude must be a list')
    for ex in exclude:
        if type(ex) is not str:
            raise ValueError(f'Element {ex} of exclude must be a string.')
        datasets = [dset for dset in datasets if ex not in dset]
    return datasets


def download_netCDF_files(datasets, save_dir=None):
    """
    Download netCDF files for given netCDF datasets. If no path
    is specified for the save directory, will download the files to
    the current working directory.
    """
    
    # Specify the server url
    server_url = 'https://opendap.oceanobservatories.org/thredds/'
    
    # ===========================================================
    # Specify and make the relevant save directory
    if save_dir is not None:
        # Make the save directory if it doesn't exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = os.getcwd()
        
    # ===========================================================
    # Download and save the netCDF files from the HTTPServer
    # to the save directory
    count = 0
    for dset in datasets:
        # Check that the datasets are netCDF
        if not dset.endswith('.nc'):
            raise ValueError(f'Dataset {dset} not netCDF.')
        count += 1
        file_url = server_url + 'fileServer/' + dset
        filename = file_url.split('/')[-1]
        print(f'Downloading file {count} of {len(datasets)}: {dset} \n')
        a = urlretrieve(file_url, '/'.join((save_dir,filename)))


# -

# #### Need to develop an improved method for getting the netCDF files (can't do json because only for synchronous requests)
# Turns out, you can't just download all the available data sets because this entire system is a fucking joke, and if _any_ datasets overlap you can't load them into xarray because rather than use time as the interpretive index they use "obs." Fuck this shit.
#
# Actually, it doesn't matter _what_ the index is, it won't merge if the time bases are even slightly different. So that is a major fucking problem depending on what the 
#
# #### Single-data stream
# If we want to download the datasets for a single datastream, we can use the following approach:

# List of the given reference designators
reference_designators = sensor_parameters['refdes'].unique()
sorted(reference_designators)

# Enter your reference designator
refdes = reference_designators[0]

# Select a single reference designator
array, node, sensor = refdes.split('-', 2)
array, node, sensor

# Get the data streams for the given reference designator
refdes_parameters = sensor_parameters[sensor_parameters['refdes'] == refdes]
# Filter out "bad" data streams
good = refdes_parameters['method'].apply(lambda x: False if 'bad' in x else True)
refdes_parameters = refdes_parameters[good]
streams = refdes_parameters.groupby('stream').agg(['unique'])
streams = streams.droplevel(level=1, axis=1)
streams

# Select your method and stream
method = 'recovered_host'
stream = 'pco2a_a_dcl_instrument_water_recovered'
array, node, sensor, method, stream

# #### Download the data:
# 1. Build the data request url based on the data_url for OOINet, the array, node, sensor, method, and stream for the given dataset(s) you want.
# 2. Get the url to the THREDDS server which has your desired dataset(s)
# 3. Get the THREDDS server catalog of datasets for your request
# 4. Parse the catalog for the desired netCDF dataset(s)
# 5. Download the netCDF dataset(s) to your computer

data_request_url = '/'.join((data_url, array, node, sensor, method, stream))
data_request_url

thredds_url = get_thredds_url(data_request_url, None, None, username, token)
thredds_url

catalog = get_thredds_catalog(thredds_url)
catalog

datasets = parse_catalog(catalog, exclude=['ENG', 'gps', 'blank'])
datasets

save_dir = '/'.join(('/media/andrew/Files/Instrument_Data', inst, refdes, method))
save_dir

if os.path.exists(save_dir):
    pass
else:
    os.makedirs(save_dir)

download_netCDF_files(datasets, save_dir=save_dir)

# **==============================================================================================================**
# #### Instrument data download
# The following code block will cycle through the available reference designators for a given instrument-class and return 

# +
# First, load the data stream parameters
parameters = pd.read_csv('/media/andrew/Files/OOINet/parameters.csv')

# Second, filter for a specific sensor
inst = 'CTDMO'
mask = parameters['sensor'].apply(lambda x: True if inst in x else False)
sensor_parameters = parameters[mask]

# Next, remove bad data streams
good = sensor_parameters["method"].apply(lambda x: False if 'bad' in x else True)
sensor_parameters = sensor_parameters[good]

# Now, select a specifi array
# sensor_parameters
# -

# List the availabe arrays for the given sensors
np.unique(sensor_parameters['array'])

# Limit the reference designators to a specific array/platform
array = 'GS03FLMB'
sensor_parameters = sensor_parameters[sensor_parameters['array'] == array]

# List the available reference designators
reference_designators = np.unique(sensor_parameters['refdes'])
reference_designators

# Script this to download all of the DOSTA data stream data
for refdes in sorted(reference_designators):

    # Select a single reference designator
    array, node, sensor = refdes.split('-', 2)
    
    # =======================================================
    # Identify the relevant data streams and methods
    # Get the data streams for the given reference designator
    refdes_parameters = sensor_parameters[sensor_parameters['refdes'] == refdes]
    # Filter out "bad" data streams
    good = refdes_parameters['method'].apply(lambda x: False if 'bad' in x else True)
    refdes_parameters = refdes_parameters[good]
    streams = refdes_parameters.groupby('stream').agg(['unique'])
    streams = streams.droplevel(level=1, axis=1)
    
    # =======================================================
    # Iterate through each refdes data stream and download
    for stream in streams.index:
        
        # For PCO2W, don't want to download all of the "blank" data streams
        if 'blank' in stream:
            continue   
        
        # Get the method with the data stream
        method = streams['method'].loc[stream][0]
        
        save_dir = '/'.join(('/media/andrew/Files/Instrument_Data', inst, refdes, method, stream))

        # Build the data request
        data_request_url = '/'.join((data_url, array, node, sensor, method, stream))

        # Query the THREDDS server for the url
        thredds_url = get_thredds_url(data_request_url, None, None, username, token)

        # Query the THREDDs server catalog
        catalog = get_thredds_catalog(thredds_url)
        if catalog is None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(save_dir + '/no_data.txt', 'w') as f:
                f.write(f'No data returned for {thredds_url}.')
            continue
            
        # Parse the server catalog for the netCDF datasets of the relevant sensor
        datasets = parse_catalog(catalog, exclude=['ENG', 'gps', 'blank'])

        # Download the netCDF datasets to a specified directory
        download_netCDF_files(datasets, save_dir=save_dir)

# **==================================================================================================================**
# ## Process Data
# Now, we want to load the netCDF datasets which we previously downloaded into an xarray dataset. We are using xarray because it most closely recreates the netCDF data model while having powerful built-in pandas functionality under-the-hood. This will allow us to apply split-apply-combine workflows to more rapidly process the data and derive our desired quantities.

# {sensor}/{refdes}/{method}
sorted(os.listdir('PCO2W/CP01CNSM-MFD35-05-PCO2WB000/'))

refdes = 'CP01CNSM-MFD35-05-PCO2WB000'
method = 'recovered_host'
array, node, sensor = refdes.split('-', 2)
array, node, sensor, method

# This attempt will load each dataset individually, select the subset consistent with the particleKeys, convert to a pandas dataframe, and then append each dataset to eachother

# Get the data streams for the given reference designator
refdes_parameters = sensor_parameters[sensor_parameters['refdes'] == refdes]
refdes_parameters

mask = refdes_parameters['stream'].apply(lambda x: False if 'blank' in x else True)
refdes_parameters = refdes_parameters[mask]
refdes_parameters

pKeys = refdes_parameters['particleKey'].unique()
pKeys = np.append(pKeys, 'deployment')
pKeys

# Next, how to load all of the data
os.listdir(f'PCO2W/{refdes}')

files = sorted(os.listdir(f'PCO2W/{refdes}/recovered_inst/'))
files

path = f'PCO2W/{refdes}'
path

# +
# Delete xarray datasets from memory to avoid killing kernel
try:
    del ds
except:
    pass

# Set the deployment to look for
depNum = 'deployment'+str(1).zfill(4)

# This block of code finds all the netCDF files associated with a specific deployment
datasets = []
for (dirpath, dirnames, filenames) in os.walk(path):
    for file in filenames:
        if depNum in file:
            datasets.append('/'.join((dirpath, file)))
# -

datasets

# Next, want to load and combine the datasets to get the most complete dataset available
for i,dset in enumerate(datasets):
    if i==0:
        ds = xr.open_dataset(dset)
        ds = ds.swap_dims({'obs':'time'})
        ds = ds.sortby('time')
        ds = ds[pKeys]
    else:
        ds2 = xr.open_dataset(dset)
        ds2 = ds2.swap_dims({'obs':'time'})
        ds2 = ds2.sortby('time')
        ds2 = ds2[pKeys]
        # And combine with the first dataset
        ds = ds.combine_first(ds2)
        del ds2

ds

# This block of code 
try:
    data = data.append(ds.to_dataframe())
except:
    data = ds.to_dataframe()

data

# Next, want to incorporate all the relevant data so that missing data gets fused into a single datastream. Can accomplish this the following way:
# 1. Load both the telemetered and recovered datasets into an xarray dataset
# 2. Use combine_first method to fill in missing spaces in recovered dataset with telemetered data
# 3. Then, 

array

deployments = pd.read_csv(f'/home/andrew/Documents/OOI-CGSN/asset-management/deployment/{array}_Deploy.csv')
deployments

refdes_deployments = deployments[deployments['Reference Designator'] == refdes]
refdes_deployments.set_index(keys='deploymentNumber', inplace=True)
refdes_deployments

# Set the parent path
path = '/'.join((os.getcwd(), 'PCO2W', refdes))
results = {}
for depNum in refdes_deployments.index:
    
    # Delete xarray datasets from memory to avoid killing kernel
    try:
        del ds
    except:
        pass
      
    # Set the deployment to look for
    depNum = 'deployment'+str(depNum).zfill(4)
    
    # This block of code finds all the netCDF files associated with a specific deployment
    datasets = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for file in filenames:
            if depNum in file:
                datasets.append('/'.join((dirpath, file)))
    # Check that there are datasets. If not, can move on to next deployment
    if len(datasets) == 0:
        continue
    # Next, want to load and combine the datasets to get the most complete dataset available
    for i,dset in enumerate(datasets):
        if i==0:
            ds = xr.open_dataset(dset)
            ds = ds.swap_dims({'obs':'time'})
            ds = ds.sortby('time')
            ds = ds[pKeys]
        else:
            ds2 = xr.open_dataset(dset)
            ds2 = ds2.swap_dims({'obs':'time'})
            ds2 = ds2.sortby('time')
            ds2 = ds2[pKeys]
            # And combine with the first dataset
            ds = ds.combine_first(ds2)
            del ds2
    
    # This block of code 
    try:
        data = data.append(ds.to_dataframe())
    except:
        data = ds.to_dataframe()





















# For PCO2W calibrated only over 200 - 1000: going to limit to those ranges due to 
data[(data['pco2_seawater'] > 1000) | (data['pco2_seawater'] < 200)]

data.sort_index(inplace=True)
# Now, calculate the mean and standard deviation
results = {}
for col in data.columns:
    avg = data[col].mean()
    std = data[col].std()
    lower = avg-3*std
    upper = avg+3*std
    results.update({col: (lower, upper)})

data['pco2_seawater'].std()

# Next, need a way to merge this results
rdf = pd.DataFrame.from_dict(results).T
rdf

rdf.drop(labels=['deployment','obs'], inplace=True)
#rdf.drop(labels=['pressure_depth','lat','lon'], inplace=True)
rdf

# Write the results out 
outpath = '/'.join((os.getcwd(),'DOSTA','Results',f'{refdes}_results.csv'))
outpath

rdf.to_csv(outpath)

# **==================================================================================================================**

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,8))
ax.scatter(data.index, data['pco2_seawater'])

subset_data = data.loc['2017-07-01':'2018-07-01']

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,8))
ax.scatter(subset_data.index, subset_data['pco2_seawater'])


def get_asyn_netCDF_datasets(async_url):
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
        if elapsed_time > 8*60:
            print(f'Request time out {async_url}')
        time.sleep(1)

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
    r = requests.get(data_request_url, params=params, auth=(username, token))
    if r.status_code == requests.codes.ok:
        urls = r.json()
        async_url = urls['allURLs'][1]
        return async_url
    else:
        return None
