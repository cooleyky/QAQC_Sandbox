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

import xarray as xr
import re


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

        
def load_datasets(datasets, ds=None):
    """Load and reprocess netCDF datasets recursively."""
    while len(datasets) > 0:

        dset = datasets.pop()
        new_ds = xr.open_dataset(dset)
        new_ds = reprocess_dataset(new_ds)

        if ds is None:
            ds = new_ds
        else:
            ds = xr.concat([new_ds, ds], dim="time")

        ds = load_datasets(datasets, ds)

    return ds


class Climatology():
    
    def resample(self, ds, param, period):
        """Resample a data variable from the dataset to a desired period.
        
        Parameters
        ----------
        ds: (xarray.DataSet)
            An xarray datasets containing the given data variable to be resampled and a
            primary dimension of time.
        param: (string)
            The name of the data variable to resample from the given dataset
        period: (string)
            The time period (e.g. month = "M", day = "D") to resample and take the mean of
            
        Returns
        -------
        da: (xarray.DataArray)
            An xarray DataArray with the resampled mean values for the given param
        """
        
        df = ds[param].to_dataframe()
        da = xr.DataArray(df.resample(period).mean())
        
        return da
    
    
    def fit(self, ds, param, period="M", cycles=1, lin_trend=False):
        """Fit the climatology
        Parameters
        ----------
        ds: (xarray.DataSet)
            An xarray datasets containing the given data variable to be fitted, with a
            primary dimension of time.
        param: (string)
            The name of the data variable from the given dataset to fit
        period: (string)
            The time period (e.g. month = "M", day = "D") to bin the data, 
            which will correspond to the fitted result
        cycle: (int: 1)
            The number of cycles per year to fit the data with
        lin_trend: (bool: False)
            Whether to include a monotonic linear trend in the fitted data
        """
                       
        # Resample the data
        da = self.resample(ds, param, period)
        
        # Calculate the frequency from the period
        if period=="M":
            freq=1/12
        elif period=="D":
            freq=1/365
        else:
            pass
        
        # Get the time series
        time_series = da.values.reshape(-1)
                
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
                fitted_data = beta[0] + beta[1]*np.sin(2*np.pi*f*new_t) + beta[2]*np.cos(2*np.pi*f*new_t)
                + beta[-1]*new_t
            else:
                fitted_data = beta[0] + beta[1]*np.sin(2*np.pi*f*new_t) + beta[2]*np.cos(2*np.pi*f*new_t)
        else:
            if lin_trend:
                fitted_data = beta[0] + beta[1]*np.sin(2*np.pi*f*new_t)
                + beta[2]*np.cos(2*np.pi*f*new_t) + beta[3]*np.sin(4*np.pi*f*new_t)
                + beta[4]*np.cos(4*np.pi*f*new_t) + beta[-1]*new_t
            else:
                fitted_data = beta[0] + beta[1]*np.sin(2*np.pi*f*new_t)
                + beta[2]*np.cos(2*np.pi*f*new_t) + beta[3]*np.sin(4*np.pi*f*new_t)
                + beta[4]*np.cos(4*np.pi*f*new_t)

        # Now calculate the standard deviation of the time series
        sigma = np.sqrt((1/(len(ts)-1))*np.sum(np.square(ts - fitted_data[mask == False])))
        sigma = np.round(sigma, decimals=2)
        
        # Reformat the fitted data into a pandas series indexed by the time and store the period information
        fitted_data = pd.Series(data=np.round(fitted_data, decimals=2), index=da.time.values)
        fitted_data.index.freq = period
        
        # Reformat
        beta = np.round(beta, decimals=2)
        
        # Save the results as attributes of the object
        self.fitted_data = fitted_data
        self.sigma = sigma
        self.beta = beta
        
        
    def make_config(self):
        """Function to make the config dictionary for climatology"""

        config = []

        months = np.arange(1, 13, 1)

        for month in months:
            val = self.fitted_data[self.fitted_data.index.month == month]
            if len(val) == 0:
                val = np.nan
            else:
                val = val.mean()

            # Get the min/max values
            vmin = np.round(val-self.sigma*3, 2)
            vmax = np.round(val+self.sigma*3, 2)

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
    
    
    def make_qcConfig(self):
    
        config = {
            "qartod": {
                "climatology": {
                    "config": self.make_config()
                }
            }
        }

        self.qcConfig = config
