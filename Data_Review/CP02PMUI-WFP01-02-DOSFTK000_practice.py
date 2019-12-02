# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Notebook to work through an example QA/QC Data Problem
# **Author:** Andrew Reed
#
# This notebook is going to serve as a dev notebook for RedMine Ticket #9445. I will be testing the DOSTA on the Coastal Pioneer (CP) Site #2 (02) Pioneer Upstream Inshore Wire Fllowing Profiler Mooring (PMUI) - Wire Following Profiler (WFP) #1 (01) - Port Number 2 (02) - Dissolved Oxygen Fast Response Series K number 1 (DOSFTK0001):
#
# CP02PMUI-WFP01-02-DOSFTK0001
#
#

import os, re, requests
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import netCDF4 as nc
import datetime

# ## Request OOI Data

# API information
username = 'OOIAPI-C9OSZAQABG1H3U'
token = 'JA48WUQVG7F'
data_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv'
vocab_url = 'https://ooinet.oceanobservatories.org/api/m2m/12586/vocab/inv'
asset_url = 'https://ooinet.oceanobservatories.org/api/m2m/12587'


# Function to make an API request and print the results
def get_and_print_api(url):
    r = requests.get(url, auth=(username, token))
    data = r.json()
    if r.status_code == 200:
        for d in data:
            print(d)
    else:
        print('message: ' + r.reason + '\n' + 'status_code: ' + str(r.status_code) )


# Site, node, method, and streamed data
site = 'CP02PMUI'
node = 'WFP01'
sensor = '02-DOFSTK000'
method = 'recovered'
stream = 'dofst_k_wfp_instrument_recovered'

get_and_print_api('/'.join( (vocab_url, site, node, sensor) ) )

# +
# Vocab metadata
vocab_request_url = '/'.join((vocab_url, site, node, sensor))
params = {
    'beginDT':'2014-03-14T00:00:02',
    'endDT':'2014-03-19T22:37:54'
}

# Go get the data from the server
r = requests.get(vocab_request_url, params=params, auth=(username, token))
vocab_data = r.json()[0]
vocab_data
# -

vocab_data['refdes']

# +
# Okay, now grab deployment info for matching cruise ctds
refdes = vocab_data['refdes']
deployment_request_url = asset_url + '/events/deployment/query'
params = {
    'beginDT':'2014-03-14T00:00:02.000Z',
    'endDT':'2014-03-19T22:37:54.000Z',
    'refdes':refdes
}

# Get the deployment information 
r = requests.get(deployment_request_url, params=params, auth=(username, token))
deployment_data = r.json()


# -

def convert_time(ms):
    if ms != None:
        return datetime.datetime.utcfromtimestamp(ms/1000)
    else:
        return None


def reformat_cal_data(data):
    df = pd.DataFrame()
    for d in data[0]['sensor']['calibration']:
        for dd in d['calData']:
            df = df.append({
                'value': dd['value'],
                'start': dd['eventStartTime'],
                'stop':  dd['eventStopTime'],
                'name':  dd['eventName'],
                'CalSheet': dd['dataSource'],
                'assetUid': dd['assetUid'],
            }, ignore_index=True)
    return df


# +
# Calibration information
cal_request_url = asset_url + '/asset/cal'
params = {
    'beginDT':'2014-03-14T00:00:02.000Z',
    'endDT':'2014-03-19T22:37:54.000Z',
    'refdes':refdes
}

r = requests.get(cal_request_url, params=params, auth=(username, token))
cal_data = r.json()
# -

df_cal = reformat_cal_data(cal_data)

df_cal = df_cal.sort_values(by=['start','name'])
df_cal['start'] = df_cal['start'].apply(convert_time)

df_cal.head()

# ### With Calibration data -> check against the existing calibration file as a basic QA check

cal_dir = 'C:/Users/areed/Documents/OOI-CGSN/OOI-Integration/asset-management/calibration/DOFSTK/'

df = pd.DataFrame()
for c in list(df_cal['CalSheet'].unique()):
    dfnew = pd.read_csv(cal_dir + c.split('_C')[0] + '.csv')
    dfnew['start'] = pd.to_datetime(c.split('_')[2])
    df = df.append(dfnew)

df.rename(columns={'value':'calsheet_value'}, inplace=True)

df_cal = df_cal.merge(df, on=['start','name']).sort_values(by='start')

df_cal

#

cal_checks = (df_cal['value']==df_cal['calsheet_value'])
def check_calibrations(df,inst_cal='value',csv_cal='calsheet_value'):
    checks = (df[inst_cal] == df[csv_cal]) 
    for i,j in enumerate(checks):
        if j is False:
            print('Cal Name: {}, Inst. Cal: {}, .csv cal: {}'.format(df['name'][i], df[inst_cal][i], df[csv_cal][i]))
        else:
            pass
    df['cal_check'] = checks
    return df


check_calibrations(df_cal)



# +
# Okay, the calibration values look okay to me. What is the source of the offset?
# Lets plot some of the data to see what the oxygen is doing with respect to time.
# First, find and get the THREDDS server url
method = 'recovered_wfp'
data_request_url = '/'.join((data_url,site,node,sensor,method,stream))

params = {
    'beginDT':'2014-03-14T00:00:00.000Z',
    'endDT':'2014-03-20T00:00:00.000Z',
}

r = requests.get(data_request_url, params=params, auth=(username,token))
data = r.json()
# -

data['allURLs']

url = data['allURLs'][0]
tds_url = 'https://opendap.oceanobservatories.org/thredds/dodsC/'
datasets = requests.get(url).text

nc_urls = re.findall(r'href=[\'"]?([^\'" >]+)', datasets)
nc = re.findall(r'(ooi/.*?.nc)', datasets)
for i in nc:
    if i.endswith('.nc') == False:
        nc.remove(i)
for i in nc:
    try:
        float(i[-4])
    except:
        nc.remove(i)
nc_datasets = [os.path.join(tds_url, i) for i in nc]
nc_datasets

ctd_data = xr.open_mfdataset([nc_datasets[0]])
dofstk_data = xr.open_mfdataset([nc_datasets[1]])                           

dofstk_data

dofstk_data = dofstk_data.swap_dims({'obs':'time'})
dofstk_data = dofstk_data.chunk({'time':100})
dofstk_data = dofstk_data.sortby('time')

max(dofstk_data.int_ctd_pressure.values)

# %matplotlib inline
fig, ax = plt.subplots(1)
fig.set_size_inches(16, 6)
dofstk_data['dofst_k_oxygen_l2'].plot(linestyle = 'None', marker='.', markersize=1, ax=ax)
ax.grid()

min(dofstk_data['dofst_k_oxygen_l2'].values)


