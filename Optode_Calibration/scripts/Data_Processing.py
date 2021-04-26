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

# # Oxygen Optode Calibration Notebook
#
# Author: Andrew Reed <br>
# Version: 0.1 <br>
# Date: 2020/08/24 <br>
#
# ### Purpose
# The notebook outlines the method for processing the data from the two-point oxygen calibrations for the Aanderaa oxygen optodes for OOI. This includes parsing the optode and barometric data, corrections for changes in temperature and pressure, and determining the saturated and zero oxygen points via curve fitting.

import os
import re
from scipy import signal
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
# %matplotlib inline

# Import functions specific to this workbook:

from utils import *

# ## Load Data
# First, load in the relevant optode and barometeric data. 
#
# #### Optode
# We can load the optode data using the **load_optode** function. This will parse the data into a pandas dataframe with parsed timestamps.

os.listdir("../data")

optode = load_optode("../data/optode/2020-08-17.log")
optode.head()

# #### Barometric Data
#
# Load the barometric data. A quirk with how the barometric data is recorded means you have week's worth of data saved into a single csv which is overwritten each time by the barometer. Best practice is to rename the file each time you download and save to the corresponding time stamp of the optode files its associated with.

barometer = load_barometer("../data/barometer/F843527A.CSV")
barometer.head()

# **Barometer timestamp is in Eastern Local Time. Need to adjust +4 hours to match UTC**

barometer["timestamp"] = barometer["timestamp"].apply(lambda x: x + pd.to_timedelta(4, unit="H"))
barometer.head()

# Get the barometric readings relevant to the optode readings by filtering for barometer timestamps which fall within the range of the optode timestamps:

tstart = optode["timestamp"].min()
tstop = optode["timestamp"].max()
tstart, tstop

barometer = barometer[(barometer["timestamp"] >= tstart) & (barometer["timestamp"] <= tstop)]
barometer.head()

# Next, calculate the water vapor pressure from the actual barometric pressure, relative humidity, and temperature:

from GasSolubities import WaterVapor

WaterVapor = WaterVapor()

barometer["water vapor"] = WaterVapor.RH_to_vapor_pressure(barometer["relative humidity"], barometer["temperature"], barometer["pressure"])
barometer.head()

# ## Reprocess Optode Data
# Next, we need to apply the most recent calibration information to the oxygen optode to calculate the correct oxygen concentrations. The concentrations in the optode output may not be the most recent calibrations. We'll use the relevant ion-functions ported into this notebook in order to achieve this. This keeps our processing as close to the OOINet processing as possible.
#
# First, need to import the existing calibration coefficients for the optode being calibrated:

print(optode["serial number"].unique())

CAL_DIR = "/home/andrew/Documents/OOI-CGSN/asset-management/calibration/DOSTAD/"
CAL_FILE = CAL_DIR + "CGINS-DOSTAD-00507__20170821.csv"

# Load the calibration file:

calibration = pd.read_csv(CAL_FILE)
calibration["value"] = calibration["value"].apply(lambda x: json.loads(x))
calibration

csv = calibration[calibration["name"] == "CC_csv"]["value"].iloc[0]
conc_coef = calibration[calibration["name"] == "CC_conc_coef"]["value"].iloc[0]

DO = do2_SVU(optode["calibrated phase"], optode["temperature"], csv, [0, 1])

# +
fig, ax = plt.subplots(figsize=(16,8))

ax = plt.plot(optode["timestamp"], optode["oxygen concentration"], linestyle="", marker=".", color="tab:blue")
ax = plt.plot(optode["timestamp"], DO, linestyle="", marker=".", color="tab:green")

ax2 = plt.twinx()
ax2 = plt.plot(optode["timestamp"], optode["temperature"], linestyle="", marker=".", color="tab:red")
# -

# ---
# ## Bokeh Example



# ---
# ## Curve-Fitting Example

from scipy.optimize import curve_fit


# #### Exponential Fitting

def exponential(x, a, b):
    return a*np.exp(b*x)


# Generate dummy datasets
x_dummy = np.linspace(start=5, stop=15, num=50)

y_dummy = exponential(x_dummy, 0.5, 0.5)
noise = 5*np.random.normal(size=y_dummy.size)
y_dummy = y_dummy + noise





# ### Calculate the solutilities

from GasSolubities import O2sol

O2sol()



# #### Temperature normalization
#
# Next, we need to normalize for changes or variation in temperature







dO2 = O2 - O2[0]

tau = 30
win = 1/tau*np.exp(-t/tau)

O2opt = O2[0] + signal.convolve(dO2, win, mode="full")/sum(win)
O2opt = O2opt[0:len(O2)]

# +

fig, (ax_orig, ax_win, ax_filt) = plt.subplots(3, 1, sharex=True)

ax_orig.plot(O2)

ax_orig.set_title('Original pulse')

ax_orig.margins(0, 0.1)

ax_win.plot(win)

ax_win.set_title('Filter impulse response')

ax_win.margins(0, 0.1)

ax_filt.plot(O2opt)

ax_filt.set_title('Filtered signal')

ax_filt.margins(0, 0.1)

fig.tight_layout()

fig.show()
# -

O2opt.max()

# Try **deconvolution** from the filtered signal:

# List the available files in the data directory:

optode_files = os.listdir("Data/")
optode_files


# Select a file, load, and parse:

# +
def parse_optode(data):
    """Parse the loaded optode data into a dataframe with column headers."""
    
    columns = ["timestamp", "model", "serial number", "oxygen concentration", "oxygen saturation",
              "temperature", "calibrated phase", "temp-compensated calibrated phase", "blue phase",
              "red phase", "blue amplitude", "red amplitude", "raw temperature"]
    df = pd.DataFrame(columns=columns)
    for line in data.splitlines():
        # Now need to parse the lines
        # Get the timestamp of the line
        timeindex = re.search("\[[^\]]*\]", line)
        timestamp = pd.to_datetime(line[timeindex.start()+1:timeindex.end()-1])
        line = line[timeindex.end():].strip()
        # Next, split the data
        model, sn, o2con, o2sat, temp, cal_phase, tcal_phase, blue_phase, red_phase, blue_amp, red_amp, raw_temp = line.split("\t")
        # Put the data into a dataframe
        df = df.append({
            "timestamp": timestamp,
            "model": int(model.strip("!")),
            "serial number": int(sn),
            "oxygen concentration": float(o2con),
            "oxygen saturation": float(o2sat),
            "temperature": float(temp),
            "calibrated phase": float(cal_phase),
            "temp-compensated calibrated phase": float(tcal_phase),
            "blue phase": float(blue_phase),
            "red phase": float(red_phase),
            "blue amplitude": float(blue_amp),
            "red amplitude": float(red_amp),
            "raw temperature": float(raw_temp)
        }, ignore_index=True)
        
    return df

def load_optode(file):
    """Open, load, and parse a file from the Aanderaa oxygen optode."""
    
    # Open and load the optode file
    with open(file) as f:
        data = f.read()
        data = data.strip("\n")
        data = data.strip("!")
    
    # Parse the data into a dataframe with column headers
    df = parse_optode(data)
    
    return df


# -

optode = load_optode("Data/2020-08-17.log")
optode


# +
def parse_barometer(data):
    """Parse the barometric info into a pandas dataframe."""
    
    columns = ["timestamp","temperature","pressure","relative humidity"]
    df = pd.DataFrame(columns=columns)

    for line in data:
        # Skip the lines without actual data
        if "+" not in line:
            continue

        # If the line has data, split into its different measurements
        line = line.strip("\x00")
        line = line.replace("+","")
        rh, temp, pres, time, date = line.split(",")

        # Make a datetime object out of the date and time
        datetime = " ".join((date, time))
        datetime = pd.to_datetime(datetime)

        # Parse the measurements into a dataframe
        df = df.append({
            "timestamp": datetime,
            "temperature": float(temp),
            "pressure": float(pres),
            "relative humidity": float(rh)        
        }, ignore_index=True)
        
    return df


def load_barometer(file):
    """Open, read, and parse barometric data into a dataframe."""
    # Open and read in the barometric data
    with open(file) as f:
        data = f.read()
        data = data.strip("\n")
        data = data.splitlines()
        
    # Parse the data
    df = parse_barometer(data)
    
    return df


# -

df = df.set_index(keys="timestamp")

# +
fig, ax = plt.subplots(figsize=(12,8))

ax.plot(df.index, df["temperature"], linestyle="", marker=".")
ax.set_ylabel("Temperature", fontsize=12)
ax.set_xlabel("Time", fontsize=12)

ax1 = ax.twinx()
ax1.plot(df.index, df["oxygen saturation"], linestyle="", marker=".", color="tab:red")

fig.autofmt_xdate()

# -
import gsw


gsw.
