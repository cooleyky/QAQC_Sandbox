# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Import libraries that will be used
import os, shutil, sys, time, re, requests, csv, datetime, pytz
import time
import yaml
import pandas as pd
import numpy as np
import netCDF4 as nc
import xarray as xr
import warnings
warnings.filterwarnings("ignore")

# +
# Import all files in a directory and append them to eachother to make one file
data = ''
for log in sorted(os.listdir('Data/CP04OSSM/D00010/')):
    if 'zero_reset' in log:
        continue
    # Open and read the log file
    with open(f'Data/CP04OSSM/D00010/{log}', errors='ignore') as file:
        f = file.read()
    # 
    data += f

# Write the appended data together
with open('Data/CP04OSSM/D00010/All_Data.pco2a.log', 'w') as file:
    file.write(data)

# ================================================================
# This block of code cleans the appended data
with open('Data/CP04OSSM/D00010/All_Data.pco2a.log') as file:
    data = file.readlines()

# Clean the data
cleaned = []
for line in data:
    if 'A M' in line or 'W M' in line:
        cleaned.append(line)

# Write out the cleaned data
with open('Data/CP04OSSM/D00010/All_Data_Cleaned.pco2a.log', 'w') as file:
    file.writelines(cleaned)


# -

# **==================================================================================================================**
# ### Load the Data

# Create a function to read in data to a dataframe
def load_pco2a(path):
    
    # Load the data into a pandas dataframe
    columns = ['DCL','Year','Month','Day','Hour','Minute','Second','Zero Counts','pCO2 Counts','pCO2 [ppm]','IRGA Temp [C]','Humidity [mbar]','Humidity Temp [C]','Gas Tension [mbar]','IRGA Detector Temp [C]','IRGA Source Temp [C]','Battery Voltage [V]']
    data = pd.read_csv(path, names=columns, header=None)
    # Drop NaNs from the data
    data.dropna(inplace=True)
    
    # Function which converts the dcl timestamp and cleans up errors
    def convert_dcl_timestamp(x):
        if type(x) is str:
            x = x.replace('>','')
        if x.endswith('60.000'):
            t = x[:-6] + '59.999'
            t = pd.to_datetime(t)
        else:
            t = pd.to_datetime(x)
        return t
    
    # Parse the DCL data into a DCL Timestamp and Measurement type
    data[['DCL Date','DCL Time','Measurement Type']] = data['DCL'].str.split(' ', n=2, expand=True)
    data['DCL Timestamp'] = data['DCL Date'] + ' ' + data['DCL Time']
    data.drop(columns={'DCL','DCL Date','DCL Time'}, inplace=True)
    data['DCL Timestamp'] = data['DCL Timestamp'].apply(lambda x: convert_dcl_timestamp(x))
    
    # Convert index to datetime
    data.set_index(keys='DCL Timestamp', inplace=True)
    
    return data


# Load the CP01CNSM data
CNSM_telemetered = load_pco2a('Data/CP01CNSM/D00011/All_Data_Cleaned.pco2a.log')
CNSM_recovered = load_pco2a('Data/CP01CNSM/R00011/All_Data_Cleaned.pco2a.log')

# Load the CP03ISSM data
ISSM_telemetered = load_pco2a('Data/CP03ISSM/D00010/All_Data_Cleaned.pco2a.log')
ISSM_recovered = load_pco2a('Data/CP03ISSM/R00010/All_Data_Cleaned.pco2a.log')

# Load the CP04OSSM data
OSSM_telemetered = load_pco2a('Data/CP04OSSM/D00010/All_Data_Cleaned.pco2a.log')
OSSM_recovered = load_pco2a('Data/CP04OSSM/R00010/All_Data_Cleaned.pco2a.log')

# #### Load the Corrected pCO2 Data

# Load the corrected pCO2 data
CNSM_corrected = pd.read_excel('Data/CNSM_33-182-50A_MainLog_corrected.xlsx')
ISSM_corrected = pd.read_excel('Data/ISSM_33-154-50A_MainLog_corrected.xlsx')
OSSM_corrected = pd.read_excel('Data/OSSM_34-231-50A_MainLog_corrected.xlsx')

ISSM_corrected

ISSM_recovered['Year'] = ISSM_recovered['Year'].apply(lambda x: int(x))
ISSM_recovered['Month'] = ISSM_recovered['Month'].apply(lambda x: int(x))
ISSM_recovered['Day'] = ISSM_recovered['Day'].apply(lambda x: int(x))
ISSM_recovered['Hour'] = ISSM_recovered['Hour'].apply(lambda x: int(x))
ISSM_recovered['Minute'] = ISSM_recovered['Minute'].apply(lambda x: int(x))
ISSM_recovered['Second'] = ISSM_recovered['Second'].apply(lambda x: int(x))


# #### Merge the corrected data with the original data

CNSM = CNSM_recovered.reset_index().merge(CNSM_corrected[['Year','Month','Day','Hour','Minute','Second','Recalculated CO2']], how='left',
                        left_on=['Year','Month','Day','Hour','Minute','Second'],
                        right_on=['Year','Month','Day','Hour','Minute','Second'])
CNSM.set_index(keys='DCL Timestamp', inplace=True)

ISSM = ISSM_recovered.reset_index().merge(ISSM_corrected[['Year','Month','Day','Hour','Minute','Second','Recalculated CO2']], how='left',
                        left_on=['Year','Month','Day','Hour','Minute','Second'],
                        right_on=['Year','Month','Day','Hour','Minute','Second'])
ISSM.set_index(keys='DCL Timestamp', inplace=True)

OSSM = OSSM_recovered.reset_index().merge(OSSM_corrected[['Year','Month','Day','Hour','Minute','Second','Recalculated CO2']], how='left',
                        left_on=['Year','Month','Day','Hour','Minute','Second'],
                        right_on=['Year','Month','Day','Hour','Minute','Second'])
OSSM.set_index(keys='DCL Timestamp', inplace=True)

# ### Exploratory Data Analysis

# +
# take a look at the 
CNSM_air = CNSM[CNSM['Measurement Type'] == 'A M']
CNSM_water = CNSM[CNSM['Measurement Type'] == 'W M']

ISSM_air = ISSM[ISSM['Measurement Type'] == 'A M']
ISSM_water = ISSM[ISSM['Measurement Type'] == 'W M']

OSSM_air = OSSM[OSSM['Measurement Type'] == 'A M']
OSSM_water = OSSM[OSSM['Measurement Type'] == 'W M']
# -

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(CNSM_air.index, CNSM_air['pCO2 [ppm]'], label='CNSM')
ax.scatter(ISSM_air.index, ISSM_air['pCO2 [ppm]'], label='ISSM')
ax.scatter(OSSM_air.index, OSSM_air['pCO2 [ppm]'], label='OSSM')
ax.grid()
ax.set_ylim((200, 600))
ax.set_title('Non-corrected data')
ax.set_ylabel('pCO2 [ppm] of Air')
ax.legend()
fig.autofmt_xdate()

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(CNSM_air.index, CNSM_air['Recalculated CO2'], label='CNSM')
ax.scatter(ISSM_air.index, ISSM_air['Recalculated CO2'], label='ISSM')
ax.scatter(OSSM_air.index, OSSM_air['Recalculated CO2'], label='OSSM')
ax.grid()
ax.set_ylim((200, 600))
ax.set_title('Corrected data')
ax.set_ylabel('pCO2 [ppm] of Air')
ax.legend()
fig.autofmt_xdate()

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(CNSM_air.index, CNSM_air['Recalculated CO2'], label='CNSM')
ax.scatter(ISSM_air.index, ISSM_air['Recalculated CO2'], label='ISSM')
ax.scatter(OSSM_air.index, OSSM_air['Recalculated CO2'], label='OSSM')
ax.grid()
ax.set_ylim((200, 600))
ax.set_title('Corrected data')
ax.set_ylabel('pCO2 [ppm] of air')
ax.legend()
fig.autofmt_xdate()

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(CNSM_water.index, CNSM_water['Recalculated CO2'], label='CNSM')
ax.scatter(ISSM_water.index, ISSM_water['Recalculated CO2'], label='ISSM')
ax.scatter(OSSM_water.index, OSSM_water['Recalculated CO2'], label='OSSM')
ax.grid()
ax.set_ylim((200, 600))
ax.set_title('Corrected data')
ax.set_ylabel('pCO2 [ppm] of water')
ax.legend()
fig.autofmt_xdate()

# Resample air and water to daily 
CNSM_water_daily = CNSM_water.resample('d').mean()
CNSM_air_daily = CNSM_air.resample('d').mean()

# +
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(CNSM_water_daily.index, CNSM_water_daily['Recalculated CO2'] - CNSM_air_daily['Recalculated CO2'], label='CNSM')

ax.grid()
ax.set_ylim((-120,20))
ax.set_title('Corrected data')
ax.set_ylabel('pCO2 [ppm] of Water - Air')
ax.legend()
fig.autofmt_xdate()
# -



# #### Plot the uncorrected vs corrected for each mooring sensor
#

# +
# Set the min/max for the xaxis
xmin = np.min([CNSM_air.index.min(), ISSM_air.index.min(), OSSM_air.index.min()])
xmax = np.max([CNSM_air.index.max(), ISSM_air.index.max(), OSSM_air.index.max()])

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(15,10))
ax0.scatter(CNSM_air.index, CNSM_air['pCO2 [ppm]'], label='Uncorrected')
ax0.scatter(CNSM_air.index, CNSM_air['Recalculated CO2'], label='Corrected')
ax0.set_title('CNSM')
ax0.set_ylabel('pCO2 [ppm] air')
ax0.set_xlim((xmin, xmax))
ax0.grid()
ax0.set_ylim((200, 600))
ax0.legend(loc='lower left')

ax1.scatter(ISSM_air.index, ISSM_air['pCO2 [ppm]'], label='Uncorrected')
ax1.scatter(ISSM_air.index, ISSM_air['Recalculated CO2'], label='Corrected')
ax1.set_title('ISSM')
ax1.set_ylabel('pCO2 [ppm] air')
ax1.set_xlim((xmin, xmax))
ax1.grid()
ax1.set_ylim((200, 600))
ax1.legend(loc='lower left')

ax2.scatter(OSSM_air.index, OSSM_air['pCO2 [ppm]'], label='Uncorrected')
ax2.scatter(OSSM_air.index, OSSM_air['Recalculated CO2'], label='Corrected')
ax2.set_title('OSSM')
ax2.set_ylabel('pCO2 [ppm] air')
ax2.grid()
ax2.set_ylim((200, 600))
ax2.set_xlim((xmin, xmax))
ax2.legend(loc='lower left')

fig.autofmt_xdate()
# -

# #### Calculate and plot the differences between the corrected and uncorrected air pCO2

# +
# Set the min/max for the xaxis
xmin = np.min([CNSM_air.index.min(), ISSM_air.index.min(), OSSM_air.index.min()])
xmax = np.max([CNSM_air.index.max(), ISSM_air.index.max(), OSSM_air.index.max()])

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(15,10))
ax0.scatter(CNSM_air.index, CNSM_air['Recalculated CO2']-CNSM_air['pCO2 [ppm]'])
ax0.set_title('CNSM')
ax0.set_ylabel('pCO2 [ppm] air')
ax0.set_xlim((xmin, xmax))
ax0.grid()
#ax0.set_ylim((200, 600))

ax1.scatter(ISSM_air.index, ISSM_air['Recalculated CO2']-ISSM_air['pCO2 [ppm]'])
ax1.set_title('ISSM')
ax1.set_ylabel('pCO2 [ppm] air')
ax1.set_xlim((xmin, xmax))
ax1.grid()
#ax1.set_ylim((200, 600))

ax2.scatter(OSSM_air.index, OSSM_air['Recalculated CO2']-OSSM_air['pCO2 [ppm]'])
ax2.set_title('OSSM')
ax2.set_ylabel('pCO2 [ppm] air')
ax2.grid()
#ax2.set_ylim((200, 600))
ax2.set_xlim((xmin, xmax))

fig.autofmt_xdate()
# -

# Burst sampling
CNSM_air_burst = CNSM_air.reset_index().groupby(by=['Year','Month','Day','Hour','Minute'], as_index=False).mean()
CNSM_water_burst = CNSM_water.reset_index().groupby(by=['Year','Month','Day','Hour','Minute'], as_index=False).mean()

CNSM_water_burst

check = CNSM.reset_index()
check

check['DCL Timestamp']

# Try hourly data
CNSM_hourly = CNSM.resample('H').mean()
CNSM_hourly

CNSM_hourly.interpolate(method=)


