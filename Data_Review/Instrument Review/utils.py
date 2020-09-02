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
#    for d in data:
#        print(d)

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
    if max_time is not None:
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

    # Define the nested function
    def request_datasets(thredds_url):

        counter = 0
        datasets = []
        tds_url = 'https://opendap.oceanobservatories.org/thredds/dodsC/'

        # Request the datasets and wait until something is returned
        while not datasets:
            datasets = requests.get(thredds_url).text
            urls = re.findall(r'(ooi/.*?.nc)', datasets)
            for i in urls:
                if i.endswith('.nc') == False:
                    urls.remove(i)
            for i in urls:
                try:
                    float(i[-4])
                except:
                    urls.remove(i)
            datasets = [os.path.join(tds_url, i) for i in urls]
            if not datasets:
                print(f'Re-requesting data: {counter}')
                counter = counter + 1
                time.sleep(15)

        return datasets

    # Initialize the data request
    datasets = request_datasets(thredds_url)

    # Now, recursively request the data until the total datasets stop changing
    done = False
    while not done:
        time.sleep(15)
        datasets2 = request_datasets(thredds_url)
        if datasets == datasets2:
            done = True
        else:
            datasets = datasets2

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
    if thredds_url is None:
        return None

    # Find and return the netCDF datasets from the thredds url
    datasets = get_netcdf_datasets(thredds_url)

    # Filter to remove companion data streams that aren't the target stream
    datasets = [d for d in datasets if 'ENG' not in d]
    if 'CTD' not in sensor:
        datasets = [d for d in datasets if 'CTD' not in d]

    # Load the netCDF files from the datasets
    if len(datasets) == 0 or datasets == None:
        return None
    else:
        ds = load_netcdf_datasets(datasets)

    # Convert the xarray dataset to a pandas dataframe for ease of use
    df = ds.to_dataframe()

    # Return the dataframe
    return df


# Define a function to return the sensor metadata
def get_sensor_metadata(metadata_url, username, token):
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


# Define a function to calculate the data availability for a day
def calc_UFrame_data_availability(subset_data, fillValues):
    """
    Function which calculates the data availability for a particular dataset.

    Args:
        subset_data - a pandas dataframe with time as the primary index
        fillValues - a dictionary of key:value pairs with keys which correspond
            to the subset_data column headers and values which correspond to the
            associated fill values for that column
    Returns:
        data_availability - a dictionary with keys corresponding to the
            subset_data column headers and values with the percent data available
    """

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
        print(f'"{path}" already exists')
