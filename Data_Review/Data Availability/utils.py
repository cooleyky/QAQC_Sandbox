import datetime
import os
import re
import requests
import numpy as np
import pandas as pd
import xarray as xr


def ntp_seconds_to_datetime(ntp_seconds):
    # Set some constants needed for time conversion
    ntp_epoch = datetime.datetime(1900, 1, 1)
    unix_epoch = datetime.datetime(1970, 1, 1)
    ntp_delta = (unix_epoch - ntp_epoch).total_seconds()

    # Convert the datetime
    dt = datetime.datetime.utcfromtimestamp(ntp_seconds - ntp_delta).replace(microseconds=0)
    return dt


def convert_time(ms):
    if ms is None:
        return None
    else:
        return datetime.datetime.utcfromtimestamp(ms/1000)


def get_and_print_api(url, username, token):
    # Request the api output
    r = requests.get(url, auth=(username, token))
    data = r.json()

    # Print the results
    for d in data:
        print(d)

    # Return the data
    return data


def get_thredds_url(data_request_url, min_time, max_time, username, token):
    """
    Returns the associated thredds url for a desired dataset(s) from the
    OOI api

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

    # The asynchronous data request is contained in the 'allURLs' key,
    # in which we want to find the url to the thredds server
    for d in data_urls['allURLs']:
        if 'thredds' in d:
            thredds_url = d

    return thredds_url


def get_netcdf_datasets(thredds_url):
    """
    Function which returns the netcdf datasets from a given OOI thredds server.

    Args:
        thredds_url - a url to the OOI Thredds server which contains your desired
            datasets

    Returns:
        datasets - a list of netcdf dataset urls
    """

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
            time.sleep(10)
    return datasets


def load_netcdf_datasets(datasets):
    """
    Function which opens and loads netcdf files from OOI thredds server
    into an xarray dataset. May accept more than one netcdf file to open, but
    all the netcdf files must be for the same sensor (some sensors automatically
    download companion sensor datasets if they are used in computing a derived
    data product).

    Also recommended to downgrade python package libnetcdf to <= 4.6.1 due to
    strict fill value matching requirements in most recent libnetcdf release
    causing loading errors with OOI netcdf files.

    Args:
        datasets - a list of OOI netcdf datasets

    Returns:
        ds - a sorted xarray dataset with primary dimension of time
    """
    if len(datasets) > 1:
        try:
            ds = xr.open_mfdataset(datasets)
        except Exception as exc:
            if '_FillValue type mismatch' in exc.args[1]:
                # Retry the request with #fillmismatch argument
                ds = xr.open_mfdataset([x+'#fillmismatch' for x in datasets])

        # Need to switch dimensions and sort by the time dimension
        ds = ds.swap_dims({'obs': 'time'})
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
        ds = ds.swap_dims({'obs': 'time'})
        ds = ds.sortby('time')

        return ds

    # If there are no available datasets on the thredds server, let user know
    else:
        print('No datasets to open')
        return None


def request_UFrame_data(url, array, node, sensor, method, stream, min_time, max_time, username, token):
    """
    Function which requests a netCDF dataset(s) from OOI UFrame
    and returns a pandas dataframe with the requested dataset(s)

    Args:
        url - the base OOI API for requesting data
        array - OOI array (e.g. CP01CNSM) from a reference designator
        node - OOI node (e.g. RID27) from a reference designator
        sensor - OOI sensor (e.g. 04-DOSTAD000) from a reference designator
        method - telemetered/host/instrument methods for a given sensor
        stream - a particular stream for a given reference designator
        min_time - optional to limit the data request to only data after a
            particular date. May be None.
        max_time - optional to limit the data request to only data before a
            particular date. May be None.
        username - your OOINet username
        token - your OOINet authentication token

    Calls:
        get_thredds_url
        get_netcdf_datasets
        load_netcdf_datasets

    Returns:
        df - a dataframe with the requested dataset(s) loaded
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


# Define a function to return the sensor metadata
def get_sensor_metadata(metadata_url, username=username, token=token):
    """
    Function which gets the metadata for a given sensor from OOI Net
    """

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
    """
    Function which returns the fill values for a particular data set
    based on that dataset's metadata information

    Args:
        data - a dataframe which contains the data from a given sensor
            stream
        metadata - a dataframe which contains the metadata information
            for the given data
        stream - the particular instrument stream from which the data
            was requested from

    Returns:
        fillValues - a dictionary with key:value pair of variable names
            from the data stream : fill values
    """

    # Filter the metadata down to a particular sensor stream
    mdf = metadata[metadata['stream'] == stream]

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
