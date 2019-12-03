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

import os, shutil, sys, time, re, requests, csv, datetime, pytz
import pandas as pd
import numpy as np
import netCDF4 as nc
import xarray as xr

username = 'OOIAPI-C9OSZAQABG1H3U'
token = 'JA48WUQVG7F'

data_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv'
anno_url = 'https://ooinet.oceanobservatories.org/api/m2m/12580/anno/find'
vocab_url = 'https://ooinet.oceanobservatories.org/api/m2m/12586/vocab/inv'
asset_url = 'https://ooinet.oceanobservatories.org/api/m2m/12587'
deploy_url = asset_url + '/events/deployment/query'
cal_url = asset_url + '/asset/cal'

r = requests.get('/'.join((asset_url,'asset')), auth=(username, token)).json()

r

r = requests.get('/'.join((vocab_url, 'CP01CNSM', 'RID27', '03-CTDBPC000')), auth=(username, token)).json()
r

gross_range_filepath = '/home/andrew/Documents/OOI-CGSN/ooi-integration/qc-lookup/data_qc_global_range_values.csv'


# +
# Function to make an API request 
def get_api(url):
    r = requests.get(url, auth=(username, token))
    data = r.json()
    return data

# Function to make an API request and print the results
def get_and_print_api(url):
    r = requests.get(url, auth=(username, token))
    data = r.json()
    for d in data:
        print(d)
        
# Specify some functions to convert timestamps
ntp_epoch = datetime.datetime(1900, 1, 1)
unix_epoch = datetime.datetime(1970, 1, 1)
ntp_delta = (unix_epoch - ntp_epoch).total_seconds()

def ntp_seconds_to_datetime(ntp_seconds):
    return datetime.datetime.utcfromtimestamp(ntp_seconds - ntp_delta).replace(microsecond=0)
  
def convert_time(ms):
    if ms is None:
        return None
    else:
        return datetime.datetime.utcfromtimestamp(ms/1000)


# -

convert_time(1479859200000)

gross_range_lookup_table = pd.read_csv(gross_range_filepath)
gross_range_lookup_table.head()


def refdes_match(rfd):
    if rfd.startswith('CE') or rfd.startswith('RS') or rfd.startswith('SS') or rfd.endswith('MOAS'):
        return False
    else:
        return True


def sensor_match(sensor):
    if 'CTDBP' in sensor:
        return True
    else:
        return False


gross_range_lookup_table['refdes match'] = gross_range_lookup_table['_Array ID'].apply(lambda x: refdes_match(x))

gross_range_lookup_table['sensor match'] = gross_range_lookup_table['ReferenceDesignator'].apply(lambda x: sensor_match(x))

metadata_url = 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/CP01CNSM/RID27/03-CTDBPC000/recovered_inst/ctdbp_cdef_instrument_recovered'

data = requests.get(metadata_url, auth=(username, token)).json()

data



# Filter out the non-WHOI responsible 
all_sites = get_api(data_url)
sites = []
for x in all_sites:
    if x.startswith('CE') or x.startswith('RS') or x.startswith('SS') or x.endswith('MOAS'):
        pass
    else:
        sites.append(x)
sites



# +
df = pd.DataFrame(columns=['RefDes','Method','Stream'])

for site in sites:
    nodes = get_api('/'.join((data_url,site)))
    for node in nodes:
        sensors = get_api('/'.join((data_url,site,node)))
        for sensor in sensors:
            # Now check for the CTDBP in the system
            if 'CTDBP' in sensor:
                # Get the reference designator of the sensor
                request_url = '/'.join((vocab_url, site, node, sensor))
                r = requests.get(request_url, auth=(username, token))
                data = r.json()
                rfd = data[0]['refdes']
                # Get the metadata and parameters for each reference designator
                request_url = '/'.join((vocab_url, site, node, sensor, 'metadata', 'parameters'))
                r = requests.get(request_url, auth=(username, token))
                data = r.json()              
                methods = get_api('/'.join((data_url,site,node,sensor)))
                for method in methods:
                    streams = get_api('/'.join((data_url,site,node,sensor,method)))
                    for stream in streams:
                        # Now update the dataframe
                        df = df.append( {
                            'RefDes':rfd,
                            'Method':method,
                            'Stream':stream,
                        }, ignore_index = True)
            else:
                pass            
# -

sensors

df.head()

# # Resolve the Streams
# This is a port of the resolve_streams.py from the preload database to Python 3.7 with some adaptations

toc = requests.get('/'.join((data_url,'toc')), auth=(username, token)).json()

toc['parameter_definitions']

pdId = {}
for stream in set(df['Stream']):
    stream_params = toc['parameters_by_stream'][stream]
    pdId.update({stream:stream_params})
    

pdId_df = pd.DataFrame(columns=['stream','pdId'])
for key in pdId.keys():
    value = pdId.get(key)
    pdId_df = pdId_df.append( {
        'stream':key,
        'pdId':value
    }, ignore_index=True )
    

df = df.merge(pdId_df, how='left', left_on=['Stream'], right_on=['stream'])

df.drop(columns='stream',inplace=True)

df.head()

# have to get the parameter information from the ParameterDefs.csv file stored in the preload-database
filepath = '../../Ocean Observatories Initiative/preload-database/csv/ParameterDefs.csv'

os.listdir('../../Ocean Observatories Initiative/preload-database/csv/')

param_defs = pd.read_csv(filepath)

param_defs[param_defs['id'] == 'PD8']

param_defs.columns.values


def get_param_name(pdId, param_defs):
    df = param_defs[param_defs['id'] == pdId]
    return list(df['name'])[0]


def get_param_units(pdId, param_defs):
    df = param_defs[param_defs['id'] == pdId]
    return list(df['unitofmeasure'])[0]


def get_param_level(pdId, param_defs):
    df = param_defs[param_defs['id'] == pdId]
    return list(df['datalevel'])[0]


def get_param_ident(pdId, param_defs):
    df = param_defs[param_defs['id'] == pdId]
    return list(df['dataproductidentifier'])[0]


df = df.merge(ids, left_index=True, right_index=True).drop('pdId',axis=1).melt(id_vars=['RefDes','Method','Stream'], value_name = 'pdId').drop('variable',axis=1).dropna()

df['Parameter Name'] = df['pdId'].apply(lambda x: get_param_name(x, param_defs))

df['Units'] = df['pdId'].apply(lambda x: get_param_units(x, param_defs))

df['Data Product Identifier'] = df['pdId'].apply(lambda x: get_param_ident(x, param_defs))

df['Level'] = df['pdId'].apply(lambda x: get_param_level(x, param_defs))

df

# Now we need to check if we have the data in the look up table
gross_range_lookup_table


def check_lookup_table(refdes, param_name, table):
    table = table[table['ReferenceDesignator'] == refdes]
    table = table[table['ParameterID_R'] == param_name]
    if len(table) == 0:
        gross_min = None
        gross_max = None
    else:
        gross_min = table['GlobalRangeMin'].iloc[0]
        gross_max = table['GlobalRangeMax'].iloc[0]
    
    return (gross_min, gross_max)


refdes = 'CP01CNSM-RID17-03-CTDBPC000'
param_name = 'conductivity'

df['Gross Range Values'] = df.apply(lambda x: check_lookup_table(x['RefDes'], x['Parameter Name'], gross_range_lookup_table), axis=1)

df2 = df
for x in df2['Parameter Name']:
    if 'time' in x:
        df2 = df2.drop(df2[df2['Parameter Name'] == x].index)
    elif 'ingestion' in x:
        df2 = df2.drop(df2[df2['Parameter Name'] == x].index)
    elif x == 'serial_number':
        df2 = df2.drop(df2[df2['Parameter Name'] == x].index)
    else:
        pass

set(df2['Parameter Name'])

df[df['pdId'] == 'PD1'].index

df2.to_csv('CTDBP_Gross_Range.csv')

table = gross_range_lookup_table[gross_range_lookup_table['ReferenceDesignator'] == refdes]
table = table[table['ParameterID_R'] == param_name]
gross_min = list(table['GlobalRangeMax'])
gross_min

gross_min

x2 = df['pdId'].apply(lambda x: get_param_name(x, param_defs) )

df.columns.values


