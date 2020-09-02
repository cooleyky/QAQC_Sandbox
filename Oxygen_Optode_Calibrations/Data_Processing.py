# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
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
import matplotlib.pyplot as plt
# %matplotlib inline

# Import functions specific to this workbook:

from utils import *

# ## Load Data
# First, load in the relevant
#
# #### Optode
# We can load the optode data using the **load_optode** function. This will parse the data into a pandas dataframe with parsed timestamps.

optode = load_optode("Data/2020-08-17.log")
optode.head()

# #### Barometric Data
#
# Load the barometric data. A quirk with how the barometric data is recorded means you have week's worth of data saved into a single csv which is overwritten each time by the barometer. Best practice is to rename the file each time you download and save to the corresponding time stamp of the optode files its associated with.

barometer = load_barometer("Barometer/F843527A.CSV")
barometer.head()

# **Barometer timestamp is in Eastern Local Time. Need to adjust +4 hours to match UTC**

barometer["timestamp"] = barometer["timestamp"].apply(lambda x: x + pd.to_timedelta(4, unit="H"))
barometer.head()

# Get the barometric readings relevant to the optode readings by filtering for barometer timestamps which fall within the range of the optode timestamps:

tstart = optode["timestamp"].min()
tstop = optode["timestamp"].max()
tstart, tstop

met_data = barometer[(barometer["timestamp"] >= tstart) & (barometer["timestamp"] <= tstop)]
met_data.head()

# ## Reprocess Optode Data
# Next, we need to apply the most recent calibration information to the oxygen optode to calculate the correct oxygen concentrations. The concentrations in the optode output may not be the most recent calibrations.







# ## Plot data
# With the data loaded, the first step is to plot the data in order to check that (1) the data was parsed correctly and (2) the data matches the lab log. Below we plot the 

# +
fig, ax = plt.subplots(figsize=(12,8))

l1 = ax.plot(optode["timestamp"], optode["oxygen concentration"], linestyle="", marker=".", color="tab:blue", label="Oxygen")
ax.set_ylabel("Oxygen Concentration", fontsize=12)
ax.set_xlabel("Time", fontsize=12)
ax.grid("x")

ax1 = ax.twinx()
l2 = ax1.plot(optode["timestamp"], optode["temperature"], linestyle="", marker=".", color="tab:red", label="Temp")
ax1.set_ylabel("Optode Temperature", fontsize=12)

lns = l1 + l2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc="center right")

fig.autofmt_xdate()
# -



# +
def f(x,A,t,mu,sigma):
    y1 = A*np.exp(-x/t)
    y2 = A*np.exp(-0.5*(x-mu)**2/sigma**2)
    return signal.convolve(y1,y2,'same')/ sum(y2)

x = np.arange(-10,10,0.01)
# -

from scipy import optimize

t = np.arange(0,1000,1)

O2 = np.ones(t.shape)
O2[300:701] = 10

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


