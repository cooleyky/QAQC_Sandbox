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

# # QA/QC Process Development - CTDBP Example
#
# ## Summary
# This is a practice of developing a review process, review, and report on the CTD. The goal here is to help develop processes for both Quality Assurance of the data, and algorithms for Quality Control. First, this will start from established methods, processes, and examples. We will want to follow on some of the steps before:
#
# 1. Data Availability
#     * What data are available?
#     * Is the data relevant?
# 2. Metadata
#     * What metadata is available?
#     * Is the metadata complete?
#     * What does it tell you about the dataset (for good or bad)?
# 3. Understand the context
#     * Plot a large range of data. Does it look right based on what you would expect?
#     * What are the ranges?
#     * Do the ranges make sense?
# 4. Focus on one or more smaller subsets of data
#     * Plot some smaller periods (in time or space) to see if they look correct or have issues
# 5. Environmental Comparisons
#     * Compare the instrument with independent datasets (such as from CTD casts, satellites, gliders, etc.)
#     * How do they compare?
#     * Is there anything wrong?

import os, re, requests
import netCDF4 as nc
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import datetime
import pytz

username = 'OOIAPI-C9OSZAQABG1H3U'
token = 'JA48WUQVG7F'

# ## Step 1. Set up the sensor names, url names, etc.

# Lay out the different api urls to use for data requests:

data_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv'
anno_url = 'https://ooinet.oceanobservatories.org/api/m2m/12580/anno/find'
vocab_url = 'https://ooinet.oceanobservatories.org/api/m2m/12586/vocab/inv'
asset_url = 'https://ooinet.oceanobservatories.org/api/m2m/12587'
deploy_url = asset_url + '/events/deployment/query'
cal_url = asset_url + '/asset/cal'

# ### Access the data:
# The first step is to access the data in a systematic, automatic way, using the M2M interface with the apis. The key is to identify where and what I want to access. I can look at the inventory using the port 12576/sensor/inv systematically find and drill down into the data directories. 
#
# I want to utilize the **CTDBP** on the Coastal Pioneer Central Surface Mooring **CP01CNSM**, which is mounted on the Near-Surface Instrument Frame **RID27**.

# List the site, node, instrument names
site = 'CP01CNSM'
node = 'RID27'
sensor = '03-CTDBPC000'
method = 'recovered_host' # 'recovered_inst' 'telemetered'


# Function to make an API request and print the results
def get_and_print_api(url):
    r = requests.get(url, auth=(username, token))
    data = r.json()
    for d in data:
        print(d)


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

convert_time(1449014400000)

# ### Vocabulary Metadata:
# Check out basic instrument vocab (metadata), which will return the reference designator, and allow us to make sure we have the correct instrument:

get_and_print_api(data_url+'/'+site)


def get_api(url):
    r = requests.get(url, auth=(username, token))
    data = r.json()
    return data


data = get_api(data_url)

data

# Retrieve vocabulary information for a given instrument
request_url = '/'.join((vocab_url, site, node, sensor))
r = requests.get(request_url, auth=(username, token))
data = r.json()

data

# So the instrument on the CP01CNSM NSIR is a SBE 16plusV2 at 7 meters of depth.

refdes = data[0]['refdes']
refdes

# ### Deployment Information
# Next, we will get some of the information about the deployments for this instrument. We will get all the deployments available in the system, and output the following: date ranges, latitude/longitude, asset ID, and sensor ID for each. Note that the **reference designator** specified above represents the geographical location of an instrument across all deployments (e.g. the CTD on the Pioneer Central Surface Mooring), the **Sensor ID** (and its Asset ID equivalent) represents the specific instrument used for a given deployment (i.e. a unique make, model, and serial numbered instrument).

# +
# Set up the API request url
deploy_request_url = deploy_url
params = {
    'refdes':refdes,
}

# Get the information from the server
r = requests.get(deploy_request_url, params=params, auth=(username, token))
deploy_data = r.json()


# -

def reformat_deployment_data(deploy_data):
    df = pd.DataFrame()
    for d in deploy_data:
        df = df.append({
            'deployment': d['deploymentNumber'],
            'start': convert_time(d['eventStartTime']),
            'stop': convert_time(d['eventStopTime']),
            'latitude': d['location']['latitude'],
            'longitude': d['location']['longitude'],
            'sensor': d['sensor']['assetId'],
        }, ignore_index=True)
    return df


deploy_df = reformat_deployment_data(deploy_data)
deploy_df

np.unique(deploy_df['sensor'])

# According to the deployment information, there have been 10 deployments of  4 different CTDs on the CP01CNSM NSIF, with asset IDs of 1451, 2059, 2345, 2659. The deployments started on 

str(deploy_df['stop'].iloc[9])


# +
# Develop a function to plot a timeline of deployments
def plot_deployment_timeline(df):
    import matplotlib.dates as mdates
    levels = np.array([-5, 5, -3, 3, -1, 1])
    fig, (ax) = plt.subplots(figsize=(12,6))
    
    # Create a baseline for plotting
    start = min(df['start'])
    stop  = max(df['stop'])
    ax.plot((start, stop), (0, 0), 'k', alpha=0.5)
    
    # Now, iterate through the dates in order to plot and annotate
    deployment = df['deployment']
    asset = df['sensor']
    xdates = df['start']
    ydates = df['stop']
    
    
    for ii, (iname, idate, jdate) in enumerate(zip(deployment, xdates, ydates)):
        # Set some plotting parameters
        level = levels[ii % 6]      # Not sure why/what this is doing
        vert = 'top' if level < 0 else 'bottom'
#        vert2 = 'bottom' if level < 0 else 'top'
        
        # Plot!!!
#        if str(jdate) == 'NaT':
#            pass
#        else:
#            # Plot the stop points
#            ax.scatter(jdate, 0, s=100, marker='s', facecolor='k', edgecolor='k', zorder = 999)
#            # Plot a line to the text
#            ax.plot((jdate, jdate), (0, -level), c='k', alpha=1.0)
#            # Align the stop text properly
#            ax.text(jdate, -level, iname, horizontalalignment='right', verticalalignment=vert2, fontsize=16)
        ax.scatter(idate, 0, s=100, facecolor='w', edgecolor='k',  zorder=9999)
        # Plot a line to the text
        ax.plot((idate, idate), (0, level), c='r', alpha=1.0)
        # Align the text properly
        ax.text(idate, level, iname,
                horizontalalignment='right', verticalalignment=vert, fontsize=16)
    
    ax.set(title='Deployments')
    ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=3))
    ax.get_xaxis().set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()
    
    # Remove components for easier read
    plt.setp((ax.get_yticklabels() + ax.get_yticklines() + list(ax.spines.values())), visible=False)
    plt.show()
        
    
# -

plot_deployment_timeline(deploy_df)

# From the table and timeline above about the deployments, there are some time periods where deployments overlap. How to deal with this issue, because it will cause a problem when loading the data due to conflicting "obs" coordinates.

# ### Calibration Information
# When Uframe delivers data, it often uses a number of calibration coefficients to generate derived data products. 

# Set up the API request
cal_request_url = cal_url
params = {
    'refdes':refdes,
}

# Get the information from the server
r = requests.get(cal_request_url, params=params, auth=(username, token))
cal_data = r.json()


def reformat_cal_data(cal_data):
    df = pd.DataFrame()
    for d in cal_data:
        for dd in d['sensor']['calibration']:
            for ddd in dd['calData']:
                df = df.append({
                    'value': ddd['value'],
                    'start': convert_time(ddd['eventStartTime']),
                    'stop': convert_time(ddd['eventStopTime']),
                    'name': ddd['eventName'],
                    'assetUid': ddd['assetUid'],
                    'calSheet': ddd['dataSource'],
                }, ignore_index=True)
    df = df.sort_values(by=['start','name'])
    return df


cal_df = reformat_cal_data(cal_data)

cal_df.head(10)


# ## Asynchronous Data Requests
# * Return netCDF data from the desired sensor for a desired time period

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


# Build an URL parser that only keeps your
def parse_dataset_names(dataset_names,refdes):
    for x in dataset_names:
        if x.count(refdes) > 1:
            pass
        else:
            dataset_names.remove(x)
    return dataset_names


# The vocab requests doesn't seem to be working
# Try the instrument information
site = 'CP01CNSM'
node = 'RID27'
sensor = '03-CTDBPC000'
method = 'recovered_inst' # recovered_inst, telemetered
stream = 'ctdbp_cdef_instrument_recovered'

get_and_print_api(data_url+'/'+site+'/'+node+'/'+sensor)

# Setup the data API requests
data_request_url = '/'.join((data_url, site, node, sensor, method, stream))
params = {
    'include_provenance':'true',
    'include_annotations':'true',
}
r = requests.get(data_request_url, params=params, auth=(username, token))
if r.status_code == 200:
    data_urls = r.json()
else:
    print(r.reason)

data_urls['allURLs'][0]

datasets = get_netcdf_datasets(data_urls['allURLs'][0])

datasets

datasets = parse_dataset_names(datasets,refdes)

datasets

ctd_ds = xr.open_mfdataset(datasets)

ctd_ds = ctd_ds.swap_dims({'obs':'time'})

ctd_ds.var

np.unique(ctd_ds.deployment.values)

# Create a function to plot where I have data and where I do not. Will need two things
ctd_ds.data_vars['ctdbp_seawater_pressure']

ctd_ds.data_vars['temperature']

# do some checks on the difference between the different salinity and temperature fields
ctd_ds.data_vars['preferred_timestamp']

ctd_ds.coords['time'].values

# ## Preliminary Data Exploration
# With the different CTDBP deployment data loaded into datasets, now we'll take a preliminary look at some of the data. This includes plotting L0, L1, and L2 data products for visual comparison and compute some basic statistics.

import seaborn as sns

# +
# First, lets plot the raw conductivity, temperature, and pressure
import matplotlib.dates as mdates
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 6))

# Plot the raw conductivity
ax1.plot_date(x=ctd_ds.coords['time'].values, y=ctd_ds.data_vars['conductivity'].values, marker = '.', color = 'blue')
ax1.set_ylabel('Conductivity')
ax1.grid()
ax1.set_title('Raw Data (Counts) from CP01CNSM NSIF CTDBP')
# Plot the raw temperature
ax2.plot_date(x=ctd_ds.coords['time'].values, y=ctd_ds.data_vars['temperature'].values, marker='.', color = 'red')
ax2.set_ylabel('Temperature')
ax2.grid()
# Plot the raw pressure
ax3.plot_date(x=ctd_ds.coords['time'].values, y=ctd_ds.coords['pressure'].values, marker='.', color='black')
ax3.set_ylabel('Pressure')
ax3.grid()

ax3.get_xaxis().set_major_locator(mdates.MonthLocator(interval=3))
ax3.get_xaxis().set_major_formatter(mdates.DateFormatter('%b %Y'))
fig.autofmt_xdate()
# -

cond = ctd_ds.data_vars['conductivity']
cond

cond = ctd_ds.data_vars['conductivity'].to_dataframe()

cond.sort_index(inplace=True)
cond.head()

cond.index

cond_rolling_mean=cond.rolling(window='365D').mean()

fig, ax = plt.subplots(figsize=(12,6))
ax.plot_date(x=cond.index, y=cond['conductivity'], marker='.', color='blue')
ax.plot_date(x=cond_rolling_mean.index, y=cond_rolling_mean['conductivity'], marker='.', color='red')
ax.grid()



