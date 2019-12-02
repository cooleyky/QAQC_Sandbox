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

# # OMS++ Data Availability
# This notebook is attempt number 1 in an attempt to adapt the Oregon State 

# ##### Recovered Host Master Script Example

import os, shutil, sys, time, re, requests, csv, pytz
import pandas as pd
import numpy as np
import netCDF4 as nc
import xarray as xr
import warnings
import yaml
warnings.filterwarnings("ignore")

# ERDDAP Access: OOI-Net

from erddapy import ERDDAP


def to_df(url):
    import pandas as pd
    return pd.read_csv(url)


erd = ERDDAP(
    server='https://erddap-uncabled.oceanobservatories.org/uncabled/erddap',
    protocol='tabledap',
)

url = erd.get_search_url(search_for='CP01CNSM ctdbp', response='csv')
url

datasets = to_df(url)['Dataset ID']
datasets

# Get a specific dataset info:

datasets[7]

# ### OMS++ Data Availability
#

from erddapy import ERDDAP


def to_df(url):
    import pandas as pd
    return pd.read_csv(url)


def show_iframe(src):
    """Helper function to show HTML returns."""
    from IPython.display import HTML
    iframe = f'<iframe src="{src}" width="100%" height="950"></iframe>'
    return HTML(iframe)


# Initialize the erddap server response
erd = ERDDAP(
    server='http://ooivm1.whoi.net/erddap',
    protocol = 'tabledap'
)

# Check what datasets are available on the OMS erddap server:

datasets = erd.get_search_url(search_for='all', response='csv')

arrays = to_df(datasets)['Dataset ID'].apply(lambda x: x.split('-')[0] )

arrays = [x for x in np.unique(arrays) if not x == 'allDatasets']

arrays

# Select the Coast Pioneer Central Surface Mooring
url = erd.get_search_url(search_for=arrays[1], response='csv')
datasets = to_df(url)['Dataset ID']
datasets

# This returns all of the datasets available for the Coastal Pioneer Surface Mooring. The three available nodes are:
# * BUOY (surface buoy)
# * MFN (multifunction node - on the bottom of the ocean)
# * NSIF (near-surface instrument frame - located at 7 m depth)
#
# First, lets try the CTDBP on the NSIF:

url = erd.get_search_url(search_for='"CP01CNSM NSIF CTDBP"', response='csv')

datasets = to_df(url)['Dataset ID']
datasets

erd.dataset_id = datasets[0]

# Check what variables are available on the dataset:

info_url = erd.get_info_url(response='html')
show_iframe(info_url)

info_url = erd.get_info_url(response='csv')

info_df = to_df(info_url)
info_df

info_df[info_df['Row Type'] == 'variable']

# Take a look at the variables with standard names:

variables = erd.get_var_by_attr(standard_name=lambda v: v is not None)
variables

# These are the standard variables for the CTDBP instrument - specifically for the CP01CNSM-NSIF-CTDBP. Next, lets query the server for _all_ available data from the CP01CNSM-NSIF-CTDBP.

erd.variables = variables

erd.get_download_url()

# Put it all into a dataframe:

data = erd.to_pandas()

# +
# Plot a basic time-series of the conductivity 
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")
# -

data[data['time (UTC)'].isnull()]

data['time (UTC)'] = data['time (UTC)'].apply(lambda x: pd.to_datetime(x))

data.set_index(keys='time (UTC)', inplace=True)

data[data['deploy_id (1)'].isnull()]

subset = data.loc[min_time:max_time]
subset

sns.set(rc={'figure.figsize':(11,4)})

ax = subset['conductivity (S/m)'].plot(marker='.', alpha=0.5, linestyle='None')
ax.set_ylabel('Conductivity (S/m)')
ax.set_title(erd.dataset_id)



erd.dataset_id

# **====================================================================================================================**
# ### Deployment Information
# To properly identify and calculate the data availability on a deployment level, need to pull in the deployment information for each array/platform. This can be ingested from the deployment sheets.

deploy_info = pd.read_csv('/home/andrew/Documents/OOI-CGSN/asset-management/deployment/CP01CNSM_Deploy.csv')

deploy_info

# +
# This chunck of code identifies the start and stop times of the deployments from the associated
# deploy csv from ooi asset management. This is necessary due to the presence of NaNs in ERDDAP
# deploy_id field.
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
    
deploy_df = pd.DataFrame(data=zip(deploy_num, startDateTime, stopDateTime), columns=['deploymentNumber','startDateTime','stopDateTime'])
# -

deploy_df


# With the deployment time periods, I can split the deployment time periods into equal spaced days (from midnight to midnight). After creating a period of days, I can then query ERDDAP for a particular dataset within the particular day. 

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


def get_fill_values(erd):
    """
    Function which returns a dictionary of fill
    values for the variables contained in the
    erddap dataset
    """
    # Logically index the fill values to get results by variable name
    info_df = to_df(erd.get_info_url(response='csv'))
    fill_df = info_df[info_df['Attribute Name'] == '_FillValue']
    # Generate a dictionary of variable name:fill value pairs
    fill_dict = {}
    for var in erd.variables:
        fill_val = fill_df[fill_df['Variable Name'] == var]['Value']
        if fill_val.empty:
            fill_val = np.nan
        else:
            fill_val = list(fill_val)[0]
        fill_dict.update({var: fill_val})
        
    return fill_dict


def get_erddap_data(num_trys, erd):
    """
    Function to get and return the relevant erddap data
    """
    def return_null_df():
        return pd.DataFrame(np.full(len(erd.variables), np.nan), index=[var for var in erd.variables]).T
    
    # This prevents an endless loop of requests for data from the ERDDAP server
    if num_trys < 5:
        
        try:
            data = erd.to_pandas()
        except Exception as exc:
            # Check for error code
            if exc.response.status_code == 500:
                # Check for a specific error which suggests retrying the request
                if 'internal server error' in exc.args[0]:
                    # Retry the request
                    num_trys = num_trys+1
                    data = get_erddap_data(num_trys, erd)
                else:
                    # No data availabe in time period, create a dataframe of all NaNs
                    data = pd.DataFrame(np.full(len(erd.variables), np.nan), index=[var for var in erd.variables]).T
            else:
                # Similarly, generate a dummy dataframe of NaNs
                data = return_null_df()
    else:
        data = return_null_df()
    # Return the data
    return data


# Iterate through the deployment information for a particular dataset id
results = pd.DataFrame(columns=erd.variables)
for dn, ts, te in deploy_df[['deploymentNumber', 'startDateTime', 'stopDateTime']].values:
    # Generate an array of days to request from the erddap server
    days = time_periods(ts, te)
    
    # Get the data availability
    data_availability = get_data_availability(erd, days)
    
    # Save the data availability stats
    filename = f'deployment_{dn}_data_availability.csv'
    save_path = '/'.join((save_dir, array, erd.dataset_id))
    make_dirs(save_path)
    data_availability.to_csv('/'.join((save_path, filename)))
    
    # Generate the statistics on the data
    data_stats = data_availability.describe()
    
    # Add in the deployment number and reindex
    data_stats['deployment'] = dn
    data_stats.set_index('deployment', append=True, inplace=True)
    data_stats = data_stats.reorder_levels(['deployment',None])
    
    # Finally, put everything into a results dataframe
    try:
        results = results.append(data_stats)
    except:
        results = data_stats

results.set_index(keys='index',inplace=True)

results.index = pd.MultiIndex.from_tuples(results.index)
results

filename = '_'.join((erd.dataset_id, 'data','availability',pd.datetime.now().strftime('%Y-%m-%d'),'.csv'))
filename

results.to_csv(filename)

r2.index = pd.MultiIndex.from_tuples(r2.index)

import shutil

save_dir = '/home/andrew/Documents/OOI-CGSN/QAQC_Sandbox/Data_Review/Data Availability/Results'


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


# Need to put everything together into a dataframe
data_availability = pd.DataFrame(columns=erd.variables)
for i in range(len(days)-1):
    
    # Get the time constraints
    min_time = days[i]
    max_time = days[i+1]
    erd.constraints = {
        'time>=': min_time,
        'time<=': max_time
    }
    
    # Query the relevant data and put into a dataframe
    data_subset = get_erddap_data(0, erd)
        
    # Rename the columns to be consistent with variable names
    for col in data_subset.columns:
        data_subset.rename(columns={col: col.split()[0]}, inplace=True)
            
    # Get the fill_values
    fill_values = get_fill_values(erd)
        
    # Calculate the % available data for each variable in the day time period
    available = {}
    for var in erd.variables:
        nans = len(data_subset[data_subset[var].isnull()])
        fills = len(data_subset[data_subset[var] == fill_values.get(var)])
        good = len(data_subset) - (nans+fills)
        percent_good = good/len(data_subset)*100
        available.update({var: percent_good})
    
    # Next append the availability data to the availability dataframe
    data_availability = data_availability.append(available, ignore_index=True)


def get_data_availability(erd, days):
    """
    Function which takes in the instantiated erddap object
    and an array of the binned days and returns the data
    availability of the particular erddap dataset
    """
    data_availability = pd.DataFrame(columns=erd.variables)
    for i in range(len(days)-1):
    
        # Get the time constraints
        min_time = days[i]
        max_time = days[i+1]
        erd.constraints = {
            'time>=': min_time,
            'time<=': max_time
        }
    
        # Query the relevant data and put into a dataframe
        data_subset = get_erddap_data(0, erd)
        
        # Rename the columns to be consistent with variable names
        for col in data_subset.columns:
            data_subset.rename(columns={col: col.split()[0]}, inplace=True)
            
        # Get the fill_values
        fill_values = get_fill_values(erd)
        
        # Calculate the % available data for each variable in the day time period
        available = {}
        for var in erd.variables:
            nans = len(data_subset[data_subset[var].isnull()])
            fills = len(data_subset[data_subset[var] == fill_values.get(var)])
            good = len(data_subset) - (nans+fills)
            percent_good = good/len(data_subset)*100
            available.update({var: percent_good})
    
        # Next append the availability data to the availability dataframe
        data_availability = data_availability.append(available, ignore_index=True)
    
    return data_availability


# **====================================================================================================================**
# ## UFrame Data Availability

import yaml
import datetime

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

# Function to make an API request and print the results
def get_and_print_api(url):
    r = requests.get(url, auth=(username, token))
    data = r.json()
    for d in data:
        print(d)
    
    # Return the data
    return data


user_info = yaml.load(open('../user_info.yaml'))
username = user_info['apiname']
token = user_info['apikey']
array = 'CP01CNSM'

deploy_df

url = f'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/{array}'

nodes = get_and_print_api(url)

node = nodes[5]
node

url = f'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/{array}/{node}'

insts = get_and_print_api(url)

inst = insts[3]
inst

url = f'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/{array}/{node}/{inst}'
methods = get_and_print_api(url)

# +
min_time = pd.to_datetime(deploy_df[deploy_df['deploymentNumber'] == 4]['startDateTime']).iloc[0]
min_time = min_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

max_time = pd.to_datetime(deploy_df[deploy_df['deploymentNumber'] == 4]['stopDateTime']).iloc[0]
max_time = max_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

min_time, max_time

# +
data_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv'
array = 'CP01CNSM'
node = 'RID27'
sensor = '03-CTDBPC000'
method = 'recovered_inst'
stream = 'ctdbp_cdef_instrument_recovered'

# Setup the data API requests
data_request_url = '/'.join((data_url, array, node, sensor, method, stream))
params = {
    'beginDT':min_time,
    'endDT':max_time,
    'include_provenance':'true',
}
r = requests.get(data_request_url, params=params, auth=(username, token))
if r.status_code == 200:
    data_urls = r.json()
else:
    print(r.reason)
# -

for url in data_urls['allURLs']:
    if 'thredds' in url:
        thredds_url = url
thredds_url


def get_netcdf_datasets(thredds_url):
    import time
    datasets = []
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
            time.sleep(10)
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
        


datasets = get_netcdf_datasets(thredds_url)

datasets

ds = load_netcdf_datasets(datasets)

# Convert the xarray dataset to a pandas dataframe for ease of use
df = ds.to_dataframe()
# Strip the non-relevant columns and variables from the dataframe
df.drop(columns=[x for x in df.columns if 'qc' in x], inplace=True)


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


metadata = get_sensor_metadata('/'.join((data_url, array, node, sensor, 'metadata')), username, token)
metadata.head()


# Filter the metadata for the relevant variable info
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


fillValues = get_UFrame_fillValues(df, metadata, stream)
fillValues

ds

ds = ds.swap_dims({'obs':'time'})
ds = ds.sortby('time')

df = ds.to_dataframe()
df.head()

# Need to check the column for fill values and NaNs
data_availability = {}
for col in df.columns:
    
    # Check for NaNs in each col
    nans = len(df[df[col].isnull()][col])
    
    # Check for values with fill values
    fv = fill_values.get(col)
    if fv is not None:
        if np.isnan(fv):
            fills = 0
        else:
            fills = len(df[df[col] == fv][col])
    else:
        fills = 0
        
    # Get the length of the whole dataframe
    num_data = len(df[col])
    
    # Calculate the statistics for the nans, fills, and length
    num_bad = nans + fills
    num_good = num_data - num_bad
    per_good = (num_good/num_data)*100
    
    print(col + ': ' + str(per_good))
    data_availability.update({
        col: per_good
    })

pd.DataFrame().from_dict(data_availability, orient='index').T

days = time_periods(min_time, max_time)
days

# +
# Now, need to split the data into distinct days in order to calculate the statistics
# Have to strip the 'Z' from the day labels
startDate = days[0].strip('Z')
endDate = days[1].strip('Z')

# Slice the dataframe based on the start and end dates
data_df = 
for i in range(len(days)-1):
    
    # Get the start and stop dates, remove Z to make timezone agnostic
    startDate = days[i].strip('Z')
    stopDate = days[i+1].strip('Z')
    
    # Slice the dataframe for only data within the time range
    subset_data = data.loc[startDate:stopDate]
    
    # Calcualte the data availability for a particular day
    data_availability = calc_UFrame_data_availability(subset_data, fillValues)
    data_availability.update({'day': i})
    
    #  
    
    
    
# -

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
    
        # Calculate the statistics for the nans, fills, and length
        num_bad = nans + fills
        num_good = num_data - num_bad
        per_good = (num_good/num_data)*100
    
        # Update the dictionary with the stats for a particular variable
        data_availability.update({
            col: per_good
        })
        
    return data_availability


