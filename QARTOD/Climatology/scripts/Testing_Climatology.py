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

# # Climatology Method Intercomparison
# **Author**: Andrew Reed <br>
# **Date**: 2012-04-12 <br>
#
# ---
# ### Purpose
# This notebook examines the different approaches to calculating the standard deviation envelope around a monthly "climatological" fit.
#
# ---
# ### Methods
#
# #### Climatological Fit
# The climatological fit is determined by fitting a two-cycle harmonic model to our time-series data via Ordinary-Least-Squares. However, instead of fitting each individual observation, we first bin the data by each year & month and calculate the average. These monthly averages are the values which will be fitted with the 2-cycle harmonic model.
#
# The climatology model can generally be modeled as a linear regression fitting some p number of parameters:
# <br>
# $$ Y_{i}  = \beta _{0} + \beta_{1}*X_{i,1} + \beta_{2}*X_{i,2} + ... + \beta_{p}*X_{i,p} + \epsilon_{i} $$
# <br>
# where $N$ is the number of observations $i$. For our two-cycle harmonic fit with no-linear trend in the time-series, this is modeled as:
# <br>
# $$ Y_{i,t}  = \beta _{0} + \beta_{1,t}*sin(2*\pi*f*t) + \beta_{2,t}*cos(2*\pi*f*t) + \beta_{3,t}*sin(4*\pi*f*t) + \beta_{4,t}*cos(4*\pi*f*t) + \epsilon_{i,t} $$ 
# <br>
# The challenge is how to best determine the values $\beta_{p}$. We choosed our best-estimate $\hat{\beta}_{p}$ so as to minimize the sum-of-squared-residuals, also known as Ordinary-Least-Squares:
# <br>
# $$SSR = \sum^{N}_{i}\hat{\epsilon}^{2}_{i} = \sum^{N}_{i}(Y_{i}-\hat{\beta}_{0}-\hat{\beta}_{p}X_{i,p})^2$$
# <br>
#
#
# #### Standard Deviations/Variability
# Next, there are three different appoaches which have been discussed for calculating the standard deviation value which is used to determine the range of accepted values for each month. There are different methods taking into account whether to use the monthly averages used to fit the climatological fits, the climatological fit, or just the observations.
#
#
# ##### Residuals
# The first method calculate the standard deviation for the time series from the residuals of the harmonic-fit:
# <br>
# $$\sigma = \sqrt{\frac{1}{N}\sum^{N}_{i}(\bar{X}_{i} - \mu_{i})^{2}}$$
# <br>
# where $\mu_{i}$ is the value for month $i$ determined from the climatological fit, $\bar{X}_{i}$ is the mean of month $i$, and $N$ is the length of the monthly-binned time series.
# <br>
# ##### Observations & Fit
# The second approach is to use all of the observations for a given calendar month (e.g. January, February, etc.) along with the climatological fit for the given month to determine the standard deviation for calendar month $i$:
# <br>
# $$\sigma_{i} = \sqrt{\frac{1}{N}\sum^{N_{i}}_{k}(X_{k,i} - \mu_{i})^{2}}$$
# <br>
# where $\mu_{i}$ is the model fit for a given calendar month $i$, $N_{i}$ are the number of observations $k$ for a calendar month $i$, and $X_{k,i}$ is an observation $k$ from calendar month $i$.
# <br>
# ##### Observations only
# The third method of determining the standard deviation is to group all the observations for a calendar month and calculate the standard deviation using just the observations, without taking into account the model fit:
# <br>
# $$\sigma_{i} = \sqrt{\frac{1}{N}\sum^{N_{i}}_{k}(X_{k,i} - \bar{X}_{i})^{2}}$$
# <br>
# where $N_{i}$ are the number of observations $k$ for a calendar month $i$, $X_{k,i}$ is an observation $k$ from calendar month $i$, and $\bar{X}_{i}$  is the mean of all observations for a calendar month $i$.
# <br>

# ---
# ### Data
# This analysis makes use of the ```ctdbp_seawater_temperature``` and ```practical_salinity``` parameters from the Argentine Basin Surface Mooring Near-Surface Instrument Frame.

# This notebook is to test the climatology.py function on a datasets
#
# **Dataset**: GA01SUMO-RID16-03-CTDBPF000-recovered_inst.nc

# Import libraries
import pandas as pd
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings("ignore")

# Import plotting libraries
import matplotlib.pyplot as plt
# %matplotlib inline

# ### Load local data file

data = xr.open_dataset("../data/GA01SUMO_RID16_03_CTDBPF000.nc")
data = data.sortby("time")
data

# +
# Plot the data
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(15,10))

ax0.plot(data.time, data.ctdbp_seawater_temperature, marker=".", linestyle="", color="tab:red")
ax0.set_ylabel(data.ctdbp_seawater_temperature.attrs["long_name"], fontsize=16)
ax0.grid()

ax1.plot(data.time, data.practical_salinity, marker=".", linestyle="", color="tab:blue")
ax1.set_ylabel(data.practical_salinity.attrs["long_name"], fontsize=16)
ax1.set_ylim((34.0, 35.6))
ax1.grid()

ax2.plot(data.time, data.ctdbp_seawater_pressure, marker=".", linestyle="", color="black")
ax2.set_ylabel(data.ctdbp_seawater_pressure.attrs["long_name"], fontsize=16)
ax2.grid()
ax2.invert_yaxis()

fig.autofmt_xdate()


# -

# ---
# ## Methods

# ### Climatological Fit

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


# Fit the ```ctdbp_seawater_temperature``` and ```practical_salinity``` variables

temperature = Climatology()
temperature.fit(data, "ctdbp_seawater_temperature")
temperature.monthly_fit

temperature.monthly_std

temperature.regression

salinity = Climatology()
salinity.fit(data, "practical_salinity")
salinity.mu_i

# Can access the regression coefficients in the ```regression``` attribute:

temperature.regression

salinity.regression


# ### Method 1

def method1(obj):
    N = len(obj.fitted_data)
    std = np.sqrt(obj.regression["residuals"]/N)
    return std


temperature.method1 = method1(temperature)
temperature.method1

salinity.method1 = method1(salinity)
salinity.method1


# ### Method 2

def method2(obj, ds, param):
    
    def _std(x):
        """Calculate the grouped standard deviations."""
        N = len(x)
        std = np.sqrt(np.sum(x**2)/N)
        return std

    # Monthly expectations
    mu = xr.DataArray(obj.fitted_data)
    mu = mu.groupby(mu.time.dt.month).mean()
    
    # Group the original observations by month
    X = ds[param].groupby(ds.time.dt.month)
    
    # Calculate the difference between the obs and expectation
    diff = X - mu
    
    # Calculate the standard deviation for each month
    std = diff.groupby("month").apply(_std)
    std = pd.Series(std.values, index=std.month)
    
    return std




temperature.method2 = method2(temperature, data, "ctdbp_seawater_temperature")
temperature.method2

salinity.method2 = method2(salinity, data, "practical_salinity")
salinity.method2


# ### Method 3

def method3(ds, param):
    da = ds[param].groupby(ds.time.dt.month).std()
    return pd.Series(da.values, index=da.month)


temperature.method3 = method3(data, "ctdbp_seawater_temperature")
temperature.method3

salinity.method3 = method3(data, "practical_salinity")
salinity.method3


# ### Monthly Averages
# For comparison, we'll also calculate the average for each month as well as the mean for each calendar month

def year_month_means(ds, param):
    "Bin by year-month and calcualte mean"
    avg = ds[param].resample(time="M").mean()
    return pd.Series(avg.values, index=avg.time.values)


temperature.Xbar_m = year_month_means(data, "ctdbp_seawater_temperature")
temperature.Xbar_m

salinity.Xbar_m = year_month_means(data, "practical_salinity")
salinity.Xbar_m


def calendar_month_means(ds, param):
    """Calculate the calendar-month average value"""
    avg = ds[param].groupby(ds.time.dt.month).mean()
    return pd.Series(avg.values, avg.month)


temperature.Xbar_i = calendar_month_means(data, "ctdbp_seawater_temperature")
temperature.Xbar_i

salinity.Xbar_i = calendar_month_means(data, "practical_salinity")
salinity.Xbar_i

# Plot the temperature data with the monthly variable standard deviations

# ---
# ## Plot the Results

# #### Method 1

# +
fig, ax = plt.subplots(figsize=(12, 8))

# Observations
ax.plot(data.time, data.ctdbp_seawater_temperature, marker=".", linestyle="", color="tab:red", zorder=0, label="Observations")

# Standard Deviation +/- 3
for t in temperature.fitted_data.index:
    t0 = pd.Timestamp(year=t.year, month=t.month, day=1)
    mu = temperature.fitted_data.loc[t]
    std = temperature.method1[0]
    ax.hlines(mu, t0, t, color="black", linewidth=3, label="Climatological Fit")
    ax.fill_between([t0, t], [mu+3*std, mu+3*std], [mu-3*std, mu-3*std], color="tab:red", alpha=0.3, label="3*$\sigma$")
ax.grid()

# Add legend and labels
handles, labels = ax.get_legend_handles_labels()[0][0:3], ax.get_legend_handles_labels()[1][0:3]
ax.legend(handles, labels, fontsize=12)
ax.set_title("-".join((data.attrs["id"].split("-")[0:4])), fontsize=16)
ax.set_ylabel("Temperature", fontsize=16)
fig.autofmt_xdate()

# +
fig, ax = plt.subplots(figsize=(12, 8))

# Observations
ax.plot(data.time, data.practical_salinity, marker=".", linestyle="", color="tab:blue", zorder=0, label="Observations")

# Standard Deviation +/- 3
for t in salinity.fitted_data.index:
    t0 = pd.Timestamp(year=t.year, month=t.month, day=1)
    mu = salinity.fitted_data.loc[t]
    std = salinity.method1[0]
    ax.hlines(mu, t0, t, color="black", linewidth=3, label="Climatological Fit")
    ax.fill_between([t0, t], [mu+3*std, mu+3*std], [mu-3*std, mu-3*std], color="tab:blue", alpha=0.3, label="3*$\sigma$")
ax.grid()
ax.set_ylim((34.0, 35.6))

# Add legend and labels
handles, labels = ax.get_legend_handles_labels()[0][0:3], ax.get_legend_handles_labels()[1][0:3]
ax.legend(handles, labels, fontsize=12)
ax.set_title("-".join((data.attrs["id"].split("-")[0:4])), fontsize=16)
ax.set_ylabel("Practical Salinity", fontsize=16)
fig.autofmt_xdate()
# -

# #### Method 2

# +
fig, ax = plt.subplots(figsize=(12, 8))

# Observations
ax.plot(data.time, data.ctdbp_seawater_temperature, marker=".", linestyle="", color="tab:red", zorder=0, label="Observations")

# Standard Deviation +/- 3
for t in temperature.fitted_data.index:
    t0 = pd.Timestamp(year=t.year, month=t.month, day=1)
    mu = temperature.fitted_data.loc[t]
    std = temperature.method2.loc[t.month]
    ax.hlines(mu, t0, t, color="black", linewidth=3, label="Climatological Fit")
    ax.fill_between([t0, t], [mu+3*std, mu+3*std], [mu-3*std, mu-3*std], color="tab:red", alpha=0.3, label="3*$\sigma_{i}$")
ax.grid()

# Add legend and labels
handles, labels = ax.get_legend_handles_labels()[0][0:3], ax.get_legend_handles_labels()[1][0:3]
ax.legend(handles, labels, fontsize=12)
ax.set_title("-".join((data.attrs["id"].split("-")[0:4])), fontsize=16)
ax.set_ylabel("Temperature", fontsize=16)
fig.autofmt_xdate()

# +
fig, ax = plt.subplots(figsize=(12, 8))

# Observations
ax.plot(data.time, data.practical_salinity, marker=".", linestyle="", color="tab:blue", zorder=0, label="Observations")

# Standard Deviation +/- 3
for t in salinity.fitted_data.index:
    t0 = pd.Timestamp(year=t.year, month=t.month, day=1)
    mu = salinity.fitted_data.loc[t]
    std = salinity.method2.loc[t.month]
    ax.hlines(mu, t0, t, color="black", linewidth=3, label="Climatological Fit")
    ax.fill_between([t0, t], [mu+3*std, mu+3*std], [mu-3*std, mu-3*std], color="tab:blue", alpha=0.3, label="3*$\sigma_{i}$")
ax.grid()
ax.set_ylim((33.7, 35.7))

# Add legend and labels
handles, labels = ax.get_legend_handles_labels()[0][0:3], ax.get_legend_handles_labels()[1][0:3]
ax.legend(handles, labels, fontsize=12)
ax.set_title("-".join((data.attrs["id"].split("-")[0:4])), fontsize=16)
ax.set_ylabel("Practical Salinity", fontsize=16)
fig.autofmt_xdate()
# -

# #### Method 3

# +
fig, ax = plt.subplots(figsize=(12, 8))

# Observations
ax.plot(data.time, data.ctdbp_seawater_temperature, marker=".", linestyle="", color="tab:red", zorder=0, label="Observations")

# Standard Deviation +/- 3
for t in temperature.fitted_data.index:
    t0 = pd.Timestamp(year=t.year, month=t.month, day=1)
    mu = temperature.fitted_data.loc[t]
    std = temperature.method3.loc[t.month]
    ax.hlines(mu, t0, t, color="black", linewidth=3, label="Climatological Fit")
    ax.fill_between([t0, t], [mu+3*std, mu+3*std], [mu-3*std, mu-3*std], color="tab:red", alpha=0.3, label="3*$\sigma_{i}$")
ax.grid()

# Add legend and labels
handles, labels = ax.get_legend_handles_labels()[0][0:3], ax.get_legend_handles_labels()[1][0:3]
ax.legend(handles, labels, fontsize=12)
ax.set_title("-".join((data.attrs["id"].split("-")[0:4])), fontsize=16)
ax.set_ylabel("Temperature", fontsize=16)
fig.autofmt_xdate()

# +
fig, ax = plt.subplots(figsize=(12, 8))

# Observations
ax.plot(data.time, data.practical_salinity, marker=".", linestyle="", color="tab:blue", zorder=0, label="Observations")

# Standard Deviation +/- 3
for t in salinity.fitted_data.index:
    t0 = pd.Timestamp(year=t.year, month=t.month, day=1)
    mu = salinity.fitted_data.loc[t]
    std = salinity.method3.loc[t.month]
    ax.hlines(mu, t0, t, color="black", linewidth=3, label="Climatological Fit")
    ax.fill_between([t0, t], [mu+3*std, mu+3*std], [mu-3*std, mu-3*std], color="tab:blue", alpha=0.3, label="3*$\sigma_{i}$")
ax.grid()
ax.set_ylim((33.7, 35.7))

# Add legend and labels
handles, labels = ax.get_legend_handles_labels()[0][0:3], ax.get_legend_handles_labels()[1][0:3]
ax.legend(handles, labels, fontsize=12)
ax.set_title("-".join((data.attrs["id"].split("-")[0:4])), fontsize=16)
ax.set_ylabel("Practical Salinity", fontsize=16)
fig.autofmt_xdate()
# -

salinity.method3

# +
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(data.time, data.ctdbp_seawater_temperature, marker=".", linestyle="", color="tab:red", zorder=0, label="Observations")
# Plot the monthly fit +/- 3-sigma
for t in temperature.fitted_data.index:
    t0 = pd.Timestamp(year=t.year, month=t.month, day=1)
    avg = temperature.Xbar_i.loc[t.month]
    std = temperature.Xbar_i.loc[t.month]
    ax.hlines(avg, t0, t, color="black", linewidth=3, label="Climatological Fit")
    ax.fill_between([t0, t], [avg+3*std, avg+3*std], [avg-3*std, avg-3*std], color="tab:red", alpha=0.3, label="3 STD")
ax.grid()
handles, labels = ax.get_legend_handles_labels()[0][0:3], ax.get_legend_handles_labels()[1][0:3]
ax.legend(handles, labels, fontsize=12)
ax.set_title("-".join((data.attrs["id"].split("-")[0:4])), fontsize=16)
ax.set_ylabel("Temperature", fontsize=16)
fig.autofmt_xdate()

# +
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(data.time, data.practical_salinity, marker=".", linestyle="", color="tab:blue", zorder=0, label="Observations")
# Plot the monthly fit +/- 3-sigma
for t in temperature.fitted_data.index:
    t0 = pd.Timestamp(year=t.year, month=t.month, day=1)
    avg = salinity.monthly_avg.loc[t.month]
    std = salinity.monthly_std.loc[t.month]
    ax.hlines(avg, t0, t, color="black", linewidth=3, label="Climatology Fit")
    ax.fill_between([t0, t], [avg+3*std, avg+3*std], [avg-3*std, avg-3*std], color="tab:blue", alpha=0.3, label="3 STD")
ax.grid()
ax.set_ylim((33.6, 35.6))
handles, labels = ax.get_legend_handles_labels()[0][0:3], ax.get_legend_handles_labels()[1][0:3]
ax.legend(handles, labels, fontsize=12)
ax.set_title("-".join((data.attrs["id"].split("-")[0:4])), fontsize=16)
ax.set_ylabel("Practical Salinity", fontsize=16)
fig.autofmt_xdate()
# -


