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

# # Location Test
#
# #### Purpose
# The purpose of this notebook is to identify the parameters for the location test for each CGSN data stream

# Import libraries
import os, shutil, sys, time, re, requests, csv, datetime, pytz
import yaml
import pandas as pd
import numpy as np
import netCDF4 as nc
import xarray as xr
import warnings
warnings.filterwarnings("ignore")

# Import user info for accessing UFrame
userinfo = yaml.load(open('../user_info.yaml'))
username = userinfo['apiname']
token = userinfo['apikey']

# Define the relevant UFrame api paths
data_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv'
anno_url = 'https://ooinet.oceanobservatories.org/api/m2m/12580/anno/find'
vocab_url = 'https://ooinet.oceanobservatories.org/api/m2m/12586/vocab/inv'
asset_url = 'https://ooinet.oceanobservatories.org/api/m2m/12587'
deploy_url = asset_url + '/events/deployment/inv'
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


# +
# Define functions for getting and filtering sensor metadata
def get_sensor_metadata(metadata_url, username=username, token=token):
    
    # Parse out the reference designator from the metadata url
    
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
    else:
        return False


# -

# **==================================================================================================================**
# #### Load the Data Streams

data_streams = pd.read_csv('Results/cgsn_data_streams.csv')
data_streams

# #### Generate the reference designator

data_streams['refdes'] = data_streams['array'] + '-' + data_streams['node'] + '-' + data_streams['sensor']
data_streams

# #### Select a specific instrument

mask = data_streams['sensor'].apply(lambda x: True if 'CTD' in x else False)
ctd_streams = data_streams[mask]
ctd_streams

# **==================================================================================================================**
# #### Load deployment sheets

for sheet in os.listdir('/home/andrew/Documents/OOI-CGSN/asset-management/deployment/'):
    if sheet.startswith(('CP','G')):
        try:
            deployments = deployments.append(pd.read_csv('/home/andrew/Documents/OOI-CGSN/asset-management/deployment/'+sheet))
        except:
            deployments = pd.read_csv('/home/andrew/Documents/OOI-CGSN/asset-management/deployment/'+sheet)

# Iterate over the reference designators to get the values
location_data = pd.DataFrame(columns=['refdes','lat','lon','depth'])
for refdes in np.unique(data_streams['refdes']):
    
    # Select a subset of the deployment data which corresponds to a given reference designator
    deploy_data = deployments[deployments['Reference Designator'] == refdes]
    
    # Calculate the min and max for the lat/lon/depth values
    lat = (np.floor(deploy_data['lat'].min()*100)/100, np.ceil(deploy_data['lat'].max()*100)/100)
    lon = (np.floor(deploy_data['lon'].min()*100)/100, np.ceil(deploy_data['lon'].max()*100)/100)
    depth = (np.floor(deploy_data['deployment_depth'].min()*100)/100, np.ceil(deploy_data['deployment_depth'].max()*100)/100)
    
    # Update the data dictionary
    location_data = location_data.append({
        'refdes': refdes,
        'lat': lat,
        'lon': lon,
        'depth': depth
    }, ignore_index = True)

data_streams.merge(location_data, left_on='refdes', right_on='refdes')

refdes = ctd_streams['refdes'].iloc[0]
refdes

subset = deployments[deployments['Reference Designator'] == refdes]
subset

# Calculate the min and max for the lat, lon, and depth
lat = (np.floor(subset['lat'].min()*100)/100, np.ceil(subset['lat'].max()*100)/100)
lon = (np.floor(subset['lon'].min()*100)/100, np.ceil(subset['lon'].max()*100)/100)
depth = (np.floor(subset['deployment_depth'].min()*100)/100, np.ceil(subset['deployment_depth'].max()*100)/100)

lon

np.unique(subset['lon'].min())

np.ceil(subset['lon'].max()*100)/100


