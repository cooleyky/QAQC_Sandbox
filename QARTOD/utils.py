import os
import numpy as np
import pandas as pd
import xarray as xr


def calc_regression_climatology(time_series, freq=1/12, lin_trend=False):
    """
    Calculate a two-cycle harmonic linear regression following Ax=b.
    
    This is an Ordinary-Least Squares regression fit for a two-cycle harmonic
    for OOI climatology data. 

    Parameters
    ----------
    time_series: (numpy.array)
        A numpy array of monthly-binned mean data to be fitted with a harmonic cycle.
    freq: (float)
        The frequency of the fit (default = 1/12).
    lin_trend: (boolean)
        Switch to determine if a linear trends should be added to the
        climatological fit (default=False).
        
    Returns
    -------
    seasonal_cycle: (numpy.array)
        A numpy array of the monthly-best fit values fitted with the OLS-regressed
            harmonic cycle. Note this is NOT robust to significant outliers.
    beta: (numpy.array)
        A numpy array of the coefficients of best-fit
    sigma: (float)
        The standard deviation of the monthly-best fit values against the input
            time series.    
    """
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

    # Build the linear coefficients as a stacked array (this is the matrix A)
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

    # Fit the coefficients using OLS
    beta, _, _, _ = np.linalg.lstsq(x.T, ts)

    # Now fit a new timeseries
    if lin_trend:
        seasonal_cycle = beta[0] + beta[1]*np.sin(2*np.pi*f*new_t)
        + beta[2]*np.cos(2*np.pi*f*new_t) + beta[3]*np.sin(4*np.pi*f*new_t)
        + beta[4]*np.cos(4*np.pi*f*new_t) + beta[-1]*new_t
    else:
        seasonal_cycle = beta[0] + beta[1]*np.sin(2*np.pi*f*new_t)
        + beta[2]*np.cos(2*np.pi*f*new_t) + beta[3]*np.sin(4*np.pi*f*new_t)
        + beta[4]*np.cos(4*np.pi*f*new_t)

    # Now calculate the standard deviation of the time series
    sigma = np.sqrt((1/(len(ts)-1))
                    * np.sum(np.square(ts - seasonal_cycle[mask == False])))

    return seasonal_cycle, beta, sigma


def qartod_climatology(ds):
    """
    Calculate the monthly QARTOD Climatology for a time series.

    Parameters
    ----------
    ds: (xarray.dataArray)
        An xarray data array with main dimension of time.

    Returns
    -------
    results: (list of tuples)
        A list of tuples in the format of
        (numerical month, None, [lower bound, upper bound], month)
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
