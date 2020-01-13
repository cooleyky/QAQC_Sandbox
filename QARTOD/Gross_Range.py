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

# # Gross Range Test
# The purpose of this notebook is to determine the gross range and user-defined ranges for a given instrument to implement the gross_range QARTOD test.

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

# **====================================================================================================================**
#

# Load in the relevant data parameters
qartod = pd.read_csv('Results/CTD_parameters.csv')
qartod.drop(columns='Unnamed: 0', inplace=True)
qartod.head()

refdes = np.unique(qartod['refdes'])
refdes[0:10]

refdes = 'CP01CNSM-RID27-03-CTDBPC000'
subset = qartod[qartod['refdes'] == refdes]
subset

# Problem, the different data streams have different parameter keys. I don't want to accidentally ignore/miss some
# parameter keys based on the different methods/streams.
subset.drop_duplicates(subset=['refdes','stream','pdId'])

stuff = subset.groupby(by='pdId')['method'].apply(list)
stuff = stuff.apply(np.unique)
stuff

p_dict = {}
for i,j in stuff.items():
    for m in ('inst','host','recov','tele'):
        preffered_method = [x for x in j if m in x]
        if len(preffered_method) > 0:
            break
        else:
            pass
    preffered_method = preffered_method[0]
    p_dict.update({i: preffered_method})
    print(preffered_method)

subset['preferred_method'] = subset['pdId'].apply(lambda x: p_dict.get(x))
subset

stuff = subset.groupby(by='method')['pdId'].apply(list)
stuff

stuff.apply(np.unique)



# Remove the bad data streams in the subset
mask = subset['method'].apply(lambda x: False if 'bad' in x else True)
subset = subset[mask]
subset

# Remove the bad data streams in the subset
mask = subset['method'].apply(lambda x: False if 'bad' in x else True)
subset = subset[mask]
subset

subset2 = subset.drop_duplicates(subset=['method','stream','pdId'])


