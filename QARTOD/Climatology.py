# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Climatology
# This is the development workbook for CGSN 

# Import libraries
import os, shutil, sys, time, re, requests, csv, datetime, pytz
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
userinfo = yaml.load(open('../../../user_info.yaml'))
username = userinfo['apiname']
token = userinfo['apikey']

# #### Initialize the connection to OOINet

OOI = M2M(username, token)

# ---
# ## Data Streams
# This section is necessary to identify all of the data stream associated with a specific instrument. This can be done by querying UFrame and iteratively walking through all of the API endpoints. The results are saved into a csv file so this step doesn't have to be repeated each time.

datasets = OOI.search_datasets(array="GA01SUMO", instrument="CTD", English_names=True)

datasets = datasets.sort_values(by="array")
datasets

# Save the datasets locally so this process doesn't have to be re-run
datasets.to_csv("Results/pco2w_datasets.csv", index=False)

# #### Datasets already downloaded
# If the datasets for the given instrument have already been identified and saved, we can just load the data from the locally saved csv file.

datastreams = pd.read_csv("../Results/pco2w_datasets.csv")
datastreams

# ---
# ## Download Data
# Next, we need to access and download the 

reference_designators = sorted(datasets["refdes"].unique())
reference_designators

# Select a reference designator
refdes = "GA01SUMO-RII11-02-CTDBPP032"

# Look at the vocab info for the reference designator
vocab = OOI.get_vocab(refdes)
vocab

# Get the deployment information for the reference designator
deployments = OOI.get_deployments(refdes)
deployments

# Get the datastreams
datastreams = OOI.get_datastreams(refdes)
datastreams

method = "recovered_inst"
stream = "ctdbp_cdef_instrument_recovered"

thredds_url = OOI.get_thredds_url(refdes, method, stream)
thredds_url

catalog = OOI.get_thredds_catalog(thredds_url)
catalog

netcdf_files = OOI.parse_catalog(catalog, exclude=["gps"])
netcdf_files

inst = refdes.split("-")[-1][0:5]
save_dir = f"/media/andrew/Files/Instrument_Data/{inst}/{refdes}/{method}/{stream}"
save_dir

OOI.download_netCDF_files(netcdf_files, save_dir=save_dir)

# Iterate through each datastream and 
for _, method, stream in datastreams.values:
    # 


# #### Download Data
# If the

# ---
# ## Load Data
# If the data has been previously downloaded, we can go ahead and load the data directly from the local directory.

def process_dataset(ds):
    
    # Remove the *_qartod_executed variables
    qartod_pattern = re.compile(r"^.+_qartod_executed.+$")
    for v in ds.variables:
        if qartod_pattern.match(v):
            # the shape of the QARTOD executed should compare to the provenance variable
            if ds[v].shape[0] != ds["provenance"].shape[0]:
                ds = ds.drop_vars(v)
                
    # Reset the dimensions and coordinates
    ds = ds.swap_dims({"obs": "time"})
    ds = ds.reset_coords()
    keys = ["obs", "id", "provenance", "driver_timestamp", "ingestion_timestamp",
            'port_timestamp', 'preferred_timestamp']
    for key in keys:
        if key in ds.variables:
            ds = ds.drop_vars(key)
    ds = ds.sortby('time')

    # clear-up some global attributes we will no longer be using
    keys = ['DODS.strlen', 'DODS.dimName', 'DODS_EXTRA.Unlimited_Dimension', '_NCProperties', 'feature_Type']
    for key in keys:
        if key in ds.attrs:
            del(ds.attrs[key])
            
    # Fix the dimension encoding
    if ds.encoding['unlimited_dims']:
        del ds.encoding['unlimited_dims']
    
    # resetting cdm_data_type from Point to Station and the featureType from point to timeSeries
    ds.attrs['cdm_data_type'] = 'Station'
    ds.attrs['featureType'] = 'timeSeries'

    # update some of the global attributes
    ds.attrs['acknowledgement'] = 'National Science Foundation'
    ds.attrs['comment'] = 'Data collected from the OOI M2M API and reworked for use in locally stored NetCDF files.'

    return ds


def load_datasets(datasets, ds=None):
    
    while len(datasets) > 0:
        
        dset = datasets.pop()
        new_ds = xr.open_dataset(dset)
        new_ds = process_dataset(new_ds)
        
        if ds is None:
            ds = new_ds
        else:
            ds = xr.concat([new_ds, ds], dim="time")
            
        ds = load_datasets(datasets, ds)
        
    return ds


inst_dir = f"/media/andrew/Files/Instrument_Data/CTDBP/"
sorted(os.listdir(inst_dir))

refdes = "GA01SUMO-RII11-02-CTDBPP033"

OOI.get_datastreams(refdes)

method = "telemetered"
stream = "ctdbp_p_dcl_instrument"

file_dir = f"{inst_dir}/{refdes}/{method}/{stream}"
sorted(os.listdir(file_dir))

netCDF_files = ["/".join((file_dir, dset)) for dset in sorted(os.listdir(file_dir)) if dset.endswith(".nc")]
netCDF_files

ds = load_datasets(netCDF_files)
ds

# Need to move 
np.unique(ds.deployment)

# ---
# ## Explore the Data
#

# +
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(15,10))

ax0.plot(ds.time, ds.temp, linestyle="", marker=".", color="tab:red")
ax0.set_ylabel("Temperature")
ax0.grid()

ax1.plot(ds.time, ds.practical_salinity, linestyle="", marker=".", color="tab:blue")
ax1.set_ylabel("Salinity")
ax1.grid()

ax2.plot(ds.time, ds.pressure, linestyle="", marker=".", color="tab:grey")
ax2.set_ylabel("Pressure")
ax2.grid()
ax2.set_xlabel("Time")

fig.autofmt_xdate()


# -

# ## Fit Data
# Next, want to fit the data with the harmonic fit.

def calc_monthly_means(ds, param):
    
    df = ds[param].to_dataframe()
    da = xr.DataArray(df.resample("M").mean())
    
    return da


def fit_monthly_data(time_series, freq=1/12, lin_trend=False, cycles=2):
    
    # Rename some of the data variables
    ts = time_series
    N = len(ts)
    t = np.arange(0, N, 1)
    new_t = t
    f = freq
    
    # Drop NaNs from the fit
    mask = np.isnan(ts)
    ts = ts[mask == False]
    t = t[mask == False]
    N = len(t)
    
    arr0 = np.ones(N)
    if cycles == 1:
        arr1 = np.sin(2*np.pi*f*t)
        arr2 = np.cos(2*np.pi*f*t)
        if lin_trend:
            x = np.stack([arr0, arr1, arr2, t])
        else:
            x = np.stack([arr0, arr1, arr2])
    else:
        arr1 = np.sin(2*np.pi*f*t)
        arr2 = np.cos(2*np.pi*f*t)
        arr3 = np.sin(4*np.pi*f*t)
        arr4 = np.cos(4*np.pi*f*t)
        if lin_trend:
            x = np.stack([arr0, arr1, arr2, arr3, arr4, t])
        else:
            x = np.stack([arr0, arr1, arr2, arr3, arr4])
    
    # Fit the coefficients using OLS
    beta, _, _, _ = np.linalg.lstsq(x.T, ts)
    
    # Now fit a new timeseries with the coefficients of best fit
    if cycles == 1:
        if lin_trend:
            seasonal_cycle = beta[0] + beta[1]*np.sin(2*np.pi*f*new_t) + beta[2]*np.cos(2*np.pi*f*new_t)
            + beta[-1]*new_t
        else:
            seasonal_cycle = beta[0] + beta[1]*np.sin(2*np.pi*f*new_t) + beta[2]*np.cos(2*np.pi*f*new_t)
    else:
        if lin_trend:
            seasonal_cycle = beta[0] + beta[1]*np.sin(2*np.pi*f*new_t)
            + beta[2]*np.cos(2*np.pi*f*new_t) + beta[3]*np.sin(4*np.pi*f*new_t)
            + beta[4]*np.cos(4*np.pi*f*new_t) + beta[-1]*new_t
        else:
            seasonal_cycle = beta[0] + beta[1]*np.sin(2*np.pi*f*new_t)
            + beta[2]*np.cos(2*np.pi*f*new_t) + beta[3]*np.sin(4*np.pi*f*new_t)
            + beta[4]*np.cos(4*np.pi*f*new_t)
            
    # Now calculate the standard deviation of the time series
    sigma = np.sqrt((1/(len(ts)-1))*np.sum(np.square(ts - seasonal_cycle[mask == False])))
    
    return seasonal_cycle, beta, sigma


ds

temp = calc_monthly_means(ds, "temp")
sal = calc_monthly_means(ds, "practical_salinity")
pres = calc_monthly_means(ds, "pressure")

temp_cycle, temp_beta, temp_sigma = fit_monthly_data(temp.values.reshape(-1), cycles=1)
pres_cycle, pres_beta, pres_sigma = fit_monthly_data(pres.values.reshape(-1), cycles=1)
sal_cycle, sal_beta, sal_sigma = fit_monthly_data(sal.values.reshape(-1), cycles=1)

# +
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(15,10))

ax0.plot(temp.time, temp.values, marker="o", color="tab:red")
ax0.set_ylabel("Temperature")
ax0.grid()

ax1.plot(sal.time, sal.values, marker="o", color="tab:blue")
ax1.set_ylabel("Salinity")
ax1.grid()

ax2.plot(pres.time, pres.values, marker="o", color="tab:grey")
ax2.set_ylabel("Pressure")
ax2.grid()
ax2.set_xlabel("Time")

fig.autofmt_xdate()
# -

temp_climatology = pd.Series(temp_cycle, index=temp.time.values)
sal_climatology = pd.Series(sal_cycle, index=sal.time.values)
pres_climatology = pd.Series(pres_cycle, index=pres.time.values)

# +
temp_lower = np.round(temp_climatology-temp_sigma*3, decimals=2)
temp_upper = np.round(temp_climatology+temp_sigma*3, decimals=2)

sal_lower = np.round(sal_climatology-sal_sigma*3, decimals=2)
sal_upper = np.round(sal_climatology+sal_sigma*3, decimals=2)

pres_lower = np.round(pres_climatology-pres_sigma*3, decimals=2)
pres_upper = np.round(pres_climatology+pres_sigma*3, decimals=2)

# +
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(15,10))

ax0.plot(ds.time, ds.temp, linestyle="", marker=".", color="tab:red")
ax0.plot(temp.time, temp_cycle, color="black")
ax0.fill_between(temp.time, temp_lower, temp_upper, color="tab:red", alpha=0.3)
ax0.set_ylabel("Temperature")
ax0.grid()

ax1.plot(ds.time, ds.practical_salinity, linestyle="", marker=".", color="tab:blue")
ax1.plot(sal.time, sal_cycle, color="black")
ax1.fill_between(sal.time, sal_lower, sal_upper, color="tab:blue", alpha=0.3)
ax1.set_ylabel("Salinity")
ax1.grid()

ax2.plot(ds.time, ds.pressure, linestyle="", marker=".", color="tab:grey")
ax2.plot(pres.time, pres_cycle, color="black")
ax2.fill_between(pres.time, pres_lower, pres_upper, color="tab:grey", alpha=0.3)

ax2.set_ylabel("Pressure")
ax2.grid()
ax2.set_xlabel("Time")

fig.autofmt_xdate()

# +
temp_results = pd.Series(data=temp_cycle, index=temp.time.values)
sal_results = pd.Series(data=sal_cycle, index=sal.time.values)

temp_qc = qcConfig(temp_results, temp_sigma)
sal_qc = qcConfig(sal_results, sal_sigma)
# -

temp_qc

sal_qc


def qcConfig(series, sigma):
    
    config = {
        "qartod": {
            "climatology": {
                "config": make_config(series, sigma)
            }
        }
    }
    
    return config


def make_config(series, sigma):
    """Function to make the config dictionary for climatology"""
    
    config = []
    
    months = np.arange(1, 13, 1)
    
    for month in months:
        val = series[series.index.month == month]
        if len(val) == 0:
            val = np.nan
        else:
            val = val.mean()
        
        # Get the min/max values
        vmin = np.round(val-sigma*3, 2)
        vmax = np.round(val+sigma*3, 2)

        # Record the results
        tspan = [month-1, month]
        vspan = [vmin, vmax]

        # Add in the 
        config.append({
            "tspan":tspan,
            "vspan":vspan,
            "period":"month"
        })
        
    return config


# ### Output the Results

array, node, instrument = refdes.split("-",2)
print(" ".join((array, node, instrument, stream)))

print(temp_qc)

print(sal_qc)

# ---
# ### Methods Development
# ---

# #### Find the OOINet data streams 

# Load all of the data streams
data_streams = pd.read_csv('/media/andrew/Files/OOINet/data_streams.csv')
data_streams.head()

# #### Load the metadata

metadata = pd.read_csv('/media/andrew/Files/OOINet/metadata.csv')
metadata.head()

# #### Load the data stream parameters

parameters = pd.read_csv('/media/andrew/Files/OOINet/parameters.csv')
parameters.head()

# #### Load the data streams for the ctds

ga01sumo = data_streams[data_streams['array'] == 'GA01SUMO']

# Get the ctd data streams
ctdbps = ga01sumo['sensor'].apply(lambda x: True if 'CTDBP' in x else False)
ga01sumo[ctdbps]
# Drop any "bad" streams
ga01sumo_ctdbp = ga01sumo[ctdbps]
good = ga01sumo_ctdbp['method'].apply(lambda x: False if 'bad' in x else True)
ga01sumo_ctdbp = ga01sumo_ctdbp[good]
ga01sumo_ctdbp

for row in ga01sumo_ctdbp.values[1:]:
    
    # Construct the data request url
    array, node, sensor, method, stream = list(row)
    data_request_url = '/'.join((data_url, array, node, sensor, method, stream))

    # Get the THREDDS server url
    thredds_url = get_thredds_url(data_request_url, None, None, username, token)
    
    # List the catalog on the THREDDS server
    catalog = get_thredds_catalog(thredds_url)
    
    # Parse the THREDDS catalog to get the datasets
    datasets = parse_catalog(catalog, exclude=['ENG', 'gps', 'blank'])
    
    # Construct the reference designator and make sure the save directory exists
    refdes = '-'.join((array, node, sensor))
    save_dir = '/'.join(('/media/andrew/Files/Instrument_Data', 'CTDBP', refdes, method, stream))
    if os.path.exists(save_dir):
        pass
    else:
        os.makedirs(save_dir)
        
    # Download the datasets to the save directory
    download_netCDF_files(datasets, save_dir=save_dir)



# ---
# ### Process the Data
# ---

# Clean up the parameters by dropping the duplicated columns
parameters = parameters.drop_duplicates()
any(parameters.duplicated())

# What is the available metadata for the given data streams
# Get the GA01SUMO
ga01sumo_params = parameters[parameters['array'] == 'GA01SUMO']
ga01sumo_params.head()

# List all of the available reference designators
ctdbp = ga01sumo_params['refdes'].apply(lambda x: True if 'CTDBP' in x else False)
sorted(np.unique(ga01sumo_params['refdes'][ctdbp]))

refdes = 'GA01SUMO-RID16-03-CTDBPF000'
refdes_params = ga01sumo_params[ga01sumo_params['refdes'] == 'GA01SUMO-RID16-03-CTDBPF000']
refdes_params

os.listdir('/media/andrew/Files/Instrument_Data/CTDBP/')

methods = np.unique(refdes_params['method'])

method = 'recovered_host'
method_params = refdes_params[refdes_params['method'] == method]
method_params

stream = 'ctdbp_cdef_dcl_instrument_recovered'
root = "/media/andrew/Files/Instrument_Data/CTDBP"
refdes, method, stream

os.listdir(root)

reference_designators = [rd for rd in os.listdir(root) if 'CP' in rd]
reference_designators

# +
# First, load the parameters info
parameters = pd.read_csv('/media/andrew/Files/OOINet/parameters.csv')

# Clean up the parameters by dropping the duplicated columns
parameters = parameters.drop_duplicates()

#for refdes in os.listdir(root):
for refdes in reference_designators:
    # Select the associated parameters for the given refdes
    refdes_parameters = parameters[parameters['refdes'] == refdes]
    
    # Iterate through the available methods
    for method in os.listdir("/".join((root, refdes))):
        # Select the associated parameters for the give refdes-parameters
        method_parameters = refdes_parameters[refdes_parameters["method"] == method]
        
        for stream in os.listdir("/".join((root, refdes, method))):
            # Select the parameters associated with the given data stream
            stream_parameters = method_parameters[method_parameters["stream"] == stream]
            
            # Load the data sets for the given refdes-method-stream
            path = "/".join((root, refdes, method, stream))
            datasets = ["/".join((path, dset)) for dset in sorted(os.listdir(path))]
            
            # Next, load the data sets into a xarray dataset
            data = xr.open_mfdataset(datasets)
            data = data.swap_dims({'obs':'time'})
            data = data.sortby('time')
            
            # With the loaded data, need to iterate over the given particleKeys to get the important data values
            particleKeys = sorted(np.unique(stream_parameters['particleKey']))
            
            # Iterate over the given particle Keys to calculate the individual climatologies
            for pKey in particleKeys:
                # First, get the data array for the particle Key from the dataset
                da = data[pKey]
                
                # Next, fit the climatology to it
                climatology = qartod_climatology(da)
                
                # Finally, construct the data frame with the results
                results = results.append({
                    'Reference Designator': refdes,
                    'Stream': stream,
                    'Delivery Method': method,
                    'Parameter': pKey,
                    'Time Parameter': 'time',
                    'Depth Parameter': None,
                    'Climatology': climatology
                }, ignore_index=True)
# -

results.tail(5)

results.to_csv('/home/andrew/Documents/OOI-CGSN/QAQC_Sandbox/QARTOD/Results/Climatology/CTDBP.csv', index=False)

del results

columns=['Reference Designator', 'Stream', 'Delivery Method', 'Parameter', 'Time Parameter', 'Depth Parameter', 'Climatology']
results = pd.DataFrame(columns=columns)
#results

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

OOI.get_metadata()


def calc_regression_climatology(time_series, freq=1/12, lin_trend=False):
    """
    Calculate a harmonic linear regression following Ax=b.
    
    Args:
        time_series - a numpy array of monthly mean data
        freq - default 1/12 (monthly)
        lin_trend - if a linear trend should be added to the fit. Default False.
    
    Returns:
        seasonal_cycle - the fitted time series with the least-square fit coefficients
        beta - the least-squares fitted coefficients
        
    
    """
    ts = time_series
    N = len(ts)
    t = np.arange(0,N,1)
    new_t = t
    f = freq
    
    # Drop NaNs from the fit
    mask = np.isnan(ts)
    ts = ts[mask==False]
    t = t[mask==False]
    N = len(t)
    
    # Build the linear coefficients
    if lin_trend:
        arr0 = np.ones(N)
        arr1 = np.sin(2*np.pi*f*t)
        arr2 = np.cos(2*np.pi*f*t)
        arr3 = np.sin(4*np.pi*f*t)
        arr4 = np.cos(4*np.pi*f*t)
        x = np.stack([arr0, arr1, arr2, arr3, arr4, t])
    else:
        arr0 = np.ones(N)
        arr1 = np.sin(2*np.pi*f*t)
        arr2 = np.cos(2*np.pi*f*t)
        arr3 = np.sin(4*np.pi*f*t)
        arr4 = np.cos(4*np.pi*f*t)
        x = np.stack([arr0, arr1, arr2, arr3, arr4])
        
    # Next, fit the coefficients
    beta, _, _, _ = np.linalg.lstsq(x.T, ts)
    
    # Now fit a new timeseries
    if lin_trend:
        seasonal_cycle = beta[0] + beta[1]*np.sin(2*np.pi*f*new_t)+beta[2]*np.cos(2*np.pi*f*new_t) + beta[3]*np.sin(4*np.pi*f*new_t) + beta[4]*np.cos(4*np.pi*f*new_t) + beta[-1]*new_t
    else:
        seasonal_cycle = beta[0] + beta[1]*np.sin(2*np.pi*f*new_t)+beta[2]*np.cos(2*np.pi*f*new_t) + beta[3]*np.sin(4*np.pi*f*new_t) + beta[4]*np.cos(4*np.pi*f*new_t)
        
    # Now calculate the standard deviation of the time series
    sigma = np.sqrt( (1/(len(ts)-1)) * np.sum( np.square(ts - seasonal_cycle[mask==False])))
        
    return seasonal_cycle, beta, sigma


def qartod_climatology(ds):
    """
    Calculate the monthly QARTOD Climatology for a time series
    """
    # Calculate the monthly means of the dataset
    monthly = ds.resample(time="M").mean()
    
    # Fit the regression for the monthly harmonic
    cycle, beta, sigma = calc_regression_climatology(monthly.values)
    
    # Calculate the monthly means, take a look at the seasonal cycle values
    climatology = pd.Series(cycle, index=monthly.time.values)
    climatology = climatology.groupby(climatology.index.month).mean()
    
    # Now add the standard deviations to get the range of data
    lower = np.round(climatology-sigma*2, decimals=2)
    upper = np.round(climatology+sigma*2, decimals=2)
    
    # This generates the results tuple
    results = []
    for month in climatology.index:
        tup = (month, None, [lower[month], upper[month]], None)
        results.append(tup)
    
    return results


directory = "/media/andrew/Files/Instrument_Data/CTDBP/GA01SUMO-RID16-03-CTDBPF000/recovered_inst/ctdbp_cdef_instrument_recovered/"

datasets = ["/".join((directory, dset)) for dset in os.listdir(directory) if dset.endswith(".nc")]
datasets

with xr.open_mfdataset(datasets) as ds:
    ds = ds.swap_dims({"obs":"time"})
    ds = ds.sortby("time")

ds = ds.resample(time="M").mean()

ds.to_netcdf("GA01SUMO-RID16-03-CTDBPF000-recovered_inst.nc")

ds

# +
# Plot the results
fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(15,10))
ax0.plot(data.time, data.ctdbp_seawater_temperature, c='tab:blue', label='Data')
ax0.plot(monthly.time, temp_cycle, c='tab:red', label='Harmonic Fit')
ax0.scatter(monthly.time, monthly.ctdbp_seawater_temperature, c='tab:red', label='Monthly Mean')
ax0.fill_between(monthly.time, temp_cycle+sigma_temp*1.96, temp_cycle-sigma_temp*1.96, alpha=0.25, color='tab:red')
ax0.grid()
ax0.set_ylabel('Temperature ($^{\circ}$C)', fontsize=12)
ax0.legend(fontsize=12)
ax0.set_title('GA01SUMO NSIF CTDBP', fontsize=16)

ax1.plot(data.time, data.ctdbp_seawater_conductivity, c='tab:green', label='Data')
ax1.plot(monthly.time, cond_cycle, c='tab:red', label='Harmonic Fit')
ax1.scatter(monthly.time, monthly.ctdbp_seawater_conductivity, c='tab:red', label='Monthly Mean')
ax1.fill_between(monthly.time, cond_cycle+sigma_cond*1.96, cond_cycle-sigma_cond*1.96, alpha=0.25, color='tab:red')
ax1.grid()
ax1.set_ylabel('Conductivity', fontsize=12)
ax1.legend(fontsize=12)

fig.autofmt_xdate()
# -

# To calculate the monthly means, take a look at the seasonal cycle values
results = pd.Series(seasonal_cycle, index=monthly.time.values)
climatology = results.groupby(results.index.month).mean()
climatology

lower, upper = climatology-sigma_temp*3, climatology+sigma_temp*3

# This generates the 
results = []
for month in climatology.index:
    tup = (month, None, [lower[month], upper[month]], None)
    results.append(tup)


results

pd.DataFrame(data=[climatology-sigma_temp*3, climatology+sigma_temp*3], index=['lower','upper'])



results.ctdbp_seawater_conductivity.values

monthly.ctdbp_seawater_conductivity.values

(monthly.ctdbp_seawater_temperature.values - temp_cycle).std()

beta_temp

temp_cycle

# Try resampling with frequency
monthly.time.values

# **==================================================================================================================**
# ### Methods Development
#

# #### Attempt an FFT analysis of the time series

from scipy.optimize import leastsq

# Get the data
data = temp_daily['ctdbp_seawater_temperature'].values
date = temp_daily.index
N = len(data)
N

# Now compute the Fourier Transform and the spectral density of the signal
temp_fft = sp.fftpack.fft(data)
temp_psd = np.abs(temp_fft)**2

# Get the frequencies corresponding to the values of the PSD
fftfreq = sp.fftpack.fftfreq(len(temp_psd), 1/365.25)

# fftfreq returns the positive and negative frequencies. We only want positivie frequencies
i = fftfreq > 0

# Plot the power spectral density
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.scatter(fftfreq[i], 10 * np.log10(temp_psd[i]))
ax.set_xlim(0,30)
ax.grid()
ax.set_xlabel('Frequency (1/year)')
ax.set_ylabel('PSD (dB)')

temp_psd[10*np.log10(temp_psd) > 43]

# Fit the fft results
temp_fft_bis = temp_fft.copy()
temp_fft_bis[10*np.log10(temp_psd) < 30] = 0
temp_slow = np.real(sp.fftpack.ifft(temp_fft_bis))

# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(12,8))
ax.plot(date, data, lw=0.5)
ax.plot_date(date, temp_slow, '-')
fig.autofmt_xdate()
ax.set_xlabel('Date')
ax.set_ylabel('Temperature')




# +
# Try fitting a single harmonic
N = len(temp_daily) # Number of data points
t = np.linspace(0, 4*np.pi, N)

guess_mean = temp_daily['ctdbp_seawater_temperature'].mean()
guess_std = (3*temp_daily['ctdbp_seawater_temperature'].std())/np.sqrt(2)
guess_phase = 0
guess_freq = 1
guess_amp = 1

data_first_guess = guess_std*np.sin(t+guess_phase) + guess_mean
optimize_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) + x[3] - temp_daily['ctdbp_seawater_temperature']
est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]

# recreate the fitted curve using the optimized parameters
data_fit = est_amp*np.sin(est_freq*t+est_phase) + est_mean

fine_t = np.arange(0,max(t),0.1)
data_fit=est_amp*np.sin(est_freq*t+est_phase)+est_mean

plt.plot(t, temp_daily['ctdbp_seawater_temperature'], '.')
plt.plot(t, data_first_guess, label='first guess')
plt.plot(t, data_fit, label='after fitting')
plt.legend()
plt.show()
# -

est_freq

fig = plt.figure(figsize=(12,8))
plt.scatter(pd.to_datetime(daily_bins), temp_daily['ctdbp_seawater_temperature'], c=temp_daily['deployment'], label='Daily Means')
plt.plot(daily_bins, data_fit, c='red', label='Harmonic fit' )
#plt.scatter(monthly_temp.index, monthly_temp['ctdbp_seawater_temperature'])
plt.legend()
plt.grid()
plt.xlabel('Datetime')
plt.ylabel('Temperature')
plt.title(refdes)
fig.autofmt_xdate()

# +
# Calculate the residuals
residuals = temp_daily['ctdbp_seawater_temperature'] - data_fit

# Plot the residuals
fig = plt.figure(figsize=(12,8))
plt.scatter(pd.to_datetime(daily_bins), residuals, c=temp_daily['deployment'], label='residuals')
plt.legend()
plt.grid()
plt.xlabel('Datetime')
plt.ylabel('Temperature')
plt.title(refdes)
fig.autofmt_xdate()

# +
# Try fitting the residuals with another harmonic

resid_mean = residuals.mean()
resid_std = (3*residuals.std())/np.sqrt(2)
resid_phase = 0
resid_freq = 4
resid_amp = 1

resid_first_guess = resid_std*np.sin(t+resid_phase) + resid_mean
optimize_func2 = lambda x: x[0]*np.sin(x[1]*t+x[2]) + x[3] - residuals
resid_amp, resid_freq, resid_phase, resid_mean = leastsq(optimize_func2, [resid_amp, resid_freq, resid_phase, resid_mean])[0]

# recreate the fitted curve using the optimized parameter
resid_fit = resid_amp*np.sin(resid_freq*t+resid_phase) + resid_mean

# Plot the fit to the residuals
plt.plot(t, residuals, '.')
plt.plot(t, resid_first_guess, label='first guess')
plt.plot(t, resid_fit, label='after fitting')
plt.legend()
plt.show()
# -



# +
# Take a look at the FFT  - start with a toy problem of two-years of data and some random, normally distributed 
# noise
t = np.arange(0,365*2,1)
toy_data = 12*np.sin(1/365*2*np.pi*t) + 3*np.sin(12/365*2*np.pi*t) + -2 + np.random.normal(0,3,t.shape)

fig = plt.figure(figsize=(12,8))
plt.plot(t, toy_data)
plt.xlabel('Time [days]')
plt.ylabel('Amplitude')
plt.grid()
#fig.autofmt_xdate()
# -

tt = np.linspace(0, 10, N)
ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))
fft = abs(np.fft.fft(toy_data))
guess_freq = abs(ff[np.argmax(fft[1:])+1])
guess_amp = np.std(toy_data)
guess_offset = np.mean(toy_data)
guess = np.array([guess_amp, 2*np.pi*guess_freq, 0, guess_offset])


def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c


popt, pcov = sp.optimize.curve_fit(sinfunc, tt, toy_data, p0=guess)

A, w, p, c = popt

f = w/(2*np.pi)
fitfunc = lambda t: A*np.sin(w*t + p) + c

guess_freq

# +
# Try fitting a single harmonic
N = len(temp_daily) # Number of data points
t = np.linspace(0, 4*np.pi, N)

guess_mean = temp_daily['ctdbp_seawater_temperature'].mean()
guess_std = (3*temp_daily['ctdbp_seawater_temperature'].std())/np.sqrt(2)
guess_phase = 0
guess_freq = 1
guess_amp = 1

data_first_guess = guess_std*np.sin(t+guess_phase) + guess_mean
optimize_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) + x[3] - temp_daily['ctdbp_seawater_temperature']
est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]

# recreate the fitted curve using the optimized parameters
data_fit = est_amp*np.sin(est_freq*t+est_phase) + est_mean

fine_t = np.arange(0,max(t),0.1)
data_fit=est_amp*np.sin(est_freq*t+est_phase)+est_mean

plt.plot(t, temp_daily['ctdbp_seawater_temperature'], '.')
plt.plot(t, data_first_guess, label='first guess')
plt.plot(t, data_fit, label='after fitting')
plt.legend()
plt.show()

# +
# Try fitting the residuals with another harmonic
toy_mean = toy_data.mean()
toy_std = (3*toy_data.std())/np.sqrt(2)

# Start with the first harmonic
toy_phase1 = 0
toy_freq1 = 1
toy_amp1 = 12

# Second harmonic
toy_phase2 = 0
toy_freq2 = 10
toy_amp2 = 3

N = len(toy_dates)
t = np.linspace(0,4*np.pi,N)

toy_first_guess = toy_amp1*np.sin(toy_freq1*t+toy_phase1) + toy_amp2*np.sin(toy_freq2*t+toy_phase2) + toy_mean
optimize_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) + x[3]*np.sin(x[4]*t+x[5]) + x[6] - toy_data
toy_amp1, toy_freq1, toy_phase1, toy_amp2, toy_freq2, toy_phase2, toy_mean = leastsq(optimize_func, [toy_amp1,toy_freq1,toy_phase1,toy_amp2,toy_freq2,toy_phase2,toy_mean])[0]

# recreate the fitted curve using the optimized parameter
toy_fit = toy_amp1*np.sin(toy_freq1*t+toy_phase1) + toy_amp2*np.sin(toy_freq2*t+toy_phase2) + toy_mean

# Plot the fit to the residuals
fig = plt.figure(figsize=(12,8))
plt.plot(toy_dates, toy_data)
plt.plot(toy_dates, toy_first_guess, label='first guess')
plt.plot(toy_dates, toy_fit, label='after fitting')
plt.grid()
plt.legend()
fig.autofmt_xdate()
plt.show()
# -

toy_data.mean()

# +
# Okay, try basic curve fitting approach on the toy data
# Try fitting the residuals with another harmonic
toy_mean = toy_data.mean()
toy_std = (3*toy_data.std())/np.sqrt(2)
toy_phase = 0
toy_freq = 1
toy_amp = 12

N = len(toy_dates)
t = np.linspace(0,4*np.pi,N)

toy_first_guess = toy_std*np.sin(t+toy_phase) + toy_mean
optimize_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) + x[3] - toy_data
toy_amp, toy_freq, toy_phase, toy_mean = leastsq(optimize_func, [toy_amp, toy_freq, toy_phase, toy_mean])[0]

# recreate the fitted curve using the optimized parameter
toy_fit = toy_amp*np.sin(toy_freq*t+toy_phase) + toy_mean

# Plot the fit to the residuals
fig = plt.figure(figsize=(12,8))
plt.plot(toy_dates, toy_data)
plt.plot(toy_dates, toy_first_guess, label='first guess')
plt.plot(toy_dates, toy_fit, label='after fitting')
plt.grid()
plt.legend()
fig.autofmt_xdate()
plt.show()
# -

toy_fit

toy_data

toy_dates = pd.date_range(start='1/1/2017',end='12/31/2018')
toy_dates

toy_data = pd.DataFrame(data=toy_data, index=toy_dates)

import scipy as sp

# Compute the Fourier transform of the toy_data
toy_fft = sp.fftpack.fft(toy_data)

# Take the square of the absolute value to ge the power-spectral-density (PSD)
toy_psd = np.abs(toy_fft)**2

# Next, ge the grequencies corresponding to the values of the PSD.
fftfreq = sp.fftpack.fftfreq(len(toy_psd), 1/365)

i = fftfreq > 0

fig, ax = plt.subplots(1, 1, figsize=(12,8))
ax.plot(fftfreq[i], 10*np.log10(toy_psd[i]))
ax.set_xlabel('Frequency (1/year)')
ax.grid()
ax.set_ylabel('PSD (dB)')

# +
# Calculate the spectrum
fft = np.fft.fft(toy_data)
T = t[1]-t[0]
N = t.size

# 1/T = frequency
f = np.linspace(0, 1/T, N)
# -



T

plt.bar(f[:N // 2], np.abs(fft)[:N // 2] *1/N, width=1.5)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")

# +
# Try fitting a two-cycle harmonic
# -












X = list(temp_daily.index)
y = temp_daily['ctdbp_seawater_temperature'].values
degree = 4
coef = np.polyfit(X, y, degree)
print('Coefficients: {}'.format(coef))

# Create curve
curve = list()
for i in range(len(X)):
    value = coef[-1]
    for d in range(degree):
        value += X[i]**(degree-d) * coef[d]
    curve.append(value)

# Plot curve over original data
plt.plot(temp_daily['ctdbp_seawater_temperature'].values)
plt.plot(curve, color='red', linewidth=3)

# Above I've generated daily mean data, so f=1/365
f = 1/365
beta_values = np.zeros(len(temp_daily))

np.linalg.lstsq()

temp_daily

# Resample monthly data
monthly_temp = temp.resample('M').mean()
monthly_temp

temp.index.dayofyear

fig = plt.figure(figsize=(12,8))
plt.scatter(x=df.index, y=df['ctdbp_seawater_conductivity'])
fig.autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('Conductivity S/m')
plt.title(f'{refdes}')
plt.grid()

df.columns

resampled = df[['ctdbp_seawater_conductivity','ctdbp_seawater_temperature','ctdbp_seawater_pressure']].resample('M', how='mean')

# +
fig, axes = plt.subplots(3,1,figsize=(12,8))

axes[0].scatter(resampled.index,resampled['ctdbp_seawater_conductivity'])
axes[0].grid()

axes[1].scatter(resampled.index,resampled['ctdbp_seawater_temperature'])
axes[1].grid()

axes[2].scatter(resampled.index,resampled['ctdbp_seawater_pressure'])
axes[2].grid()

fig.autofmt_xdate()
# -


