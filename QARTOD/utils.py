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

import os
import re
import datetime
import numpy as np
import pandas as pd
import xarray as xr


class GrossRange():
    """Gross Range fitting process for QARTOD.

    For a given parameter in a dataset, calculate the gross range QARTOD values
    for the data, and fromat the data into a qcConfig object optimized for use
    with Axiom-implemented QARTOD gross range test.

    Example
    -------
    from qartod.gross_range import GrossRange
    gross_range = GrossRange(fail_min=200, fail_max=1200)
    gross_range.fit(pco2_dataset, "pco2_seawater")
    gross_range.make_qcConfig()
    gross_range.qcConfig = {'qartod':
        {'gross_range_test':
            {'suspect_span': [200, 767.5], 'fail_span': [200, 1200]}}}
    """

    def __init__(self, fail_min, fail_max):
        """Init the Gross Range with the relevant fail min/max.

        Parameters
        ----------
        fail_min: (float)
            The minimum value for the given dataset parameter below which the
            data is fail.
        fail_max: (float)
            The maximum value for the given dataset parameter above which the
            data is fail.
        """
        self.fail_min = fail_min
        self.fail_max = fail_max

    def fit(self, ds, param, sigma=5):
        """Fit suspect range with specified standard deviation.

        Parameters
        ----------
        ds: (xarray.DataSet)
            An xarray datasets containing the given data variable to be fitted,
            with a primary dimension of time.
        param: (string)
            The name of the data variable from the given dataset to fit
        sigma: (float)
            The number of standard deviations for calculating the suspect range

        """
        # First, filter out data which falls outside of the fail ranges
        ds = self.filter_fail_range(ds, param)

        # Calculate the mean and standard deviation
        avg = np.nanmean(ds[param])
        std = np.nanstd(ds[param])

        # Calculate the suspect range
        suspect_min = avg-sigma*std
        suspect_max = avg+sigma*std

        # If the suspect ranges are outside the fail ranges, set
        # suspect ranges to the fail_ranges
        if suspect_min < self.fail_min:
            suspect_min = self.fail_min
        if suspect_max > self.fail_max:
            suspect_max = self.suspect_max

        # Save the results
        self.suspect_min = np.round(suspect_min, decimals=2)
        self.suspect_max = np.round(suspect_max, decimals=2)

    def filter_fail_range(self, ds, param):
        """Filter out values which fall outside the fail range."""
        ds = ds.where((ds[param] < self.fail_max) &
                      (ds[param] > self.fail_min), drop=True)
        return ds

    def make_qcConfig(self):
        """Build properly formatted qcConfig object for qartod gross range."""
        self.qcConfig = {
            "qartod": {
                "gross_range_test": {
                    "suspect_span": [self.suspect_min, self.suspect_max],
                    "fail_span": [self.fail_min, self.fail_max]
                }
            }
        }


class Climatology():
    
    def std(self, ds, param):
        """Calculate the standard deviation of grouped-monthly data.
        
        Calculates the standard deviation for a calendar-month from all
        of the observations for a given calendar-month.
        
        Parameters
        ----------
        ds: (xarray.DataSet)
            DataSet of the original time series observations
        param: (str)
            A string corresponding to the variable in the DataSet which is fit.
            
        Attributes
        ----------
        monthly_std: (pandas.Series)
            The standard deviation for a calendar month calculated from all of the
            observations for a given calendar-month.
        """
        
        da = ds[param].groupby(ds.time.dt.month).std()
        self.monthly_std = pd.Series(da.values, index=da.month)
        
        # Fill missing std values
        ind = np.arange(1,13,1)
        self.monthly_std = self.monthly_std.reindex(index=ind)
        self.monthly_std = self.monthly_std.fillna(value=self.monthly_std.mean())
            
    def fit(self, ds, param):
        """Calculate the climatological fit and monthly standard deviations.
        
        Calculates the climatological fit for a time series. First, the data 
        are binned by month and averaged. Next, a two-cycle harmonic is fitted
        via OLS-regression. The climatological expected value for each month
        is then calculated from the regression coefficients. Finally, the 
        standard deviation is derived using the observations for a given month
        and the climatological fit for that month as the expected value.
        
        Parameters
        ----------
        ds: (xarray.DataSet)
            DataSet of the original time series observations
        param: (str)
            A string corresponding to the variable in the DataSet to fit.
            
        Attributes
        -------
        fitted_data: (pandas.Series)
            The climatological monthly expectation calculated from the 
            regression, indexed by the year-month
        regression: (dict)
            A dictionary containing the OLS-regression values for
            * beta: Least-squares solution.
            * residuals: Sums of residuals; squared Euclidean 2-norm
            * rank: rank of the input matrix
            * singular_values: The singular values of input matrix
        monthly_fit: (pandas.Series)
            The climatological expectation for each calendar month of a year
            
        Example
        -------
        from qartod.climatology import Climatology
        climatology = Climatology()
        climatology.fit(ctdbp_data, "ctdbp_seawater_temperature")
        """
        
        # Resample the data to monthly means
        mu = ds[param].resample(time="M").mean()

        # Next, build the model
        ts = mu.values
        f = 1/12
        N = len(ts)
        t_in = np.arange(0, N, 1)
        t_out = t_in

        # Drop NaNs from the fit
        mask = np.isnan(ts)
        ts = ts[mask == False]
        t_in = t_in[mask == False]
        n = len(t_in)

        # Build the 2-cycle model
        X = [np.ones(n), np.sin(2*np.pi*f*t_in), np.cos(2*np.pi*f*t_in), np.sin(4*np.pi*f*t_in), np.cos(4*np.pi*f*t_in)]
        [beta, resid, rank, s] = np.linalg.lstsq(np.transpose(X), ts)
        self.regression = {
            "beta": beta,
            "residuals": resid,
            "rank": rank,
            "singular_values": s
        }

        # Calculate the two-cycle fitted data
        fitted_data = beta[0] + beta[1]*np.sin(2*np.pi*f*t_out) + beta[2]*np.cos(2*np.pi*f*t_out) + beta[3]*np.sin(4*np.pi*f*t_out) + beta[4]*np.cos(4*np.pi*f*t_out)
        fitted_data = pd.Series(fitted_data, index=mu.get_index("time"))
        self.fitted_data = fitted_data
        
        # Return the monthly_avg
        self.monthly_fit = self.fitted_data.groupby(self.fitted_data.index.month).mean()
        
        # Return the monthly_std
        self.std(ds, param)


# +
def reprocess_dataset(ds):
    """Reprocess the netCDF dataset to conform to CF-standards.

    Parameters
    ----------
    ds: (xarray.DataSet)
        An opened xarray dataset of the netCDF file.

    Returns
    -------
    ds: (xarray.DataSet)
        Reprocessed xarray DataSet
    """ 
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
    keys = ['DODS.strlen', 'DODS.dimName', 'DODS_EXTRA.Unlimited_Dimension',
            '_NCProperties', 'feature_Type']
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



def preprocess(ds):
    """
    This function trims overlapping deployment times to allow
    the dataset to be opened as a single dataset. If you want the
    overlapping time periods, the relevant netCDF files must be
    opened separately.
    """
    
    # First, reprocess the dataset
    ds = reprocess_dataset(ds)
    
    deployments = OOINet.get_deployments(refdes)
    
    # --------------------------------
    # Second, get the deployment times
    deployments = deployments.sort_values(by="deploymentNumber")
    deployments = deployments.set_index(keys="deploymentNumber")
    # Shift the start times by (-1) 
    deployEnd = deployments["deployStart"].shift(-1)
    # Find where the deployEnd times are earlier than the deployStart times
    mask = deployments["deployEnd"] > deployEnd
    # Wherever the deployEnd times occur after the shifted deployStart times, replace those deployEnd times
    deployments["deployEnd"][mask] = deployEnd[mask]
    deployments["deployEnd"] = deployments["deployEnd"].apply(lambda x: pd.to_datetime(x))
    
    # ---------------------------------
    # With the deployments info, can write a preprocess function to filter 
    # the data based on the deployment number
    depNum = np.unique(ds["deployment"])
    deployInfo = deployments.loc[depNum]
    deployStart = deployInfo["deployStart"].values[0]
    deployEnd = deployInfo["deployEnd"].values[0]
    
    # Select the dataset data which falls within the specified time range
    ds = ds.sel(time=slice(deployStart, deployEnd))
    
    return ds


def load_netCDF(thredds_catalog):
    """Load remote netCDF files from OOINet."""

    # Get the OpenDAP server
    opendap_url = "https://opendap.oceanobservatories.org/thredds/dodsC"

    # Add the OpenDAP url to the netCDF dataset names
    netCDF_datasets = ["/".join((opendap_url, dset)) for dset in thredds_catalog]
    netCDF_datasets = [dset+"#fillmismatch" for dset in netCDF_datasets]
    
    # Open the datasets
    ds = xr.open_mfdataset(netCDF_datasets, preprocess=preprocess)
    ds = ds.sortby("time")
    
    return ds
