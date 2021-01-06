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

# # GitHub Miner

import os, shutil, sys, time, re, requests, csv, datetime, pytz
import pandas as pd
import numpy as np
import netCDF4 as nc
import xarray as xr
import warnings
import yaml
import csv
from github import Github
warnings.filterwarnings("ignore")

# Import the github user info
userinfo = yaml.load(open('github_user.yaml'))
username = userinfo['apiname']
token = userinfo['apitoken']

g = Github(username, token)

repo = g.get_repo("ooi-integration/asset-management")
repo

# Enter the ooi urls and user info:

base_url = 'https://ooinet.oceanobservatories.org/api/m2m'
sensor_url = '12576/sensor/inv'
asset_url = '12587/asset'

ooi_userinfo = yaml.load(open('../../user_info.yaml'))

ooi_user = ooi_userinfo['apiname']
ooi_token = ooi_userinfo['apikey']

# **====================================================================================================================**
# ## Pull Request Approach

# Get a particular pull request #
pr = repo.get_pull(703)


def is_data_affecting(file):
    """
    This function tests a file patch diff for if the change
    is only in the notes column which is not data affecting
    """
    for line in file.patch.splitlines():
        line = line.strip()
        if line.startswith('@@'):
            # This is the metadata about the patch changes
            metadata = line
        elif line.startswith('serial'):
            # This is the title line
            columns = line.split(',')
            new_file = pd.DataFrame(columns=columns)
            old_file = pd.DataFrame(columns=columns)
        elif line.startswith('+'):
            line = line.replace('+','',1)
            line = [x for x in csv.reader([line])][0]
            new_file = new_file.append({
                'serial':line[0],
                'name':line[1],
                'value':line[2],
                'notes':line[3],
            }, ignore_index = True)
        elif line.startswith('-'):
            line = line.replace('-','',1)
            # This is the old file data
            line = [x for x in csv.reader([line])][0]
            old_file = old_file.append({
                'serial':line[0],
                'name':line[1],
                'value':line[2],
                'notes':line[3],
            }, ignore_index = True)
        else:
            # Append the line to both the new and old files
            line = [x for x in csv.reader([line])][0]
            new_file = new_file.append({
                'serial':line[0],
                'name':line[1],
                'value':line[2],
                'notes':line[3],
            }, ignore_index = True)
            old_file = old_file.append({
                'serial':line[0],
                'name':line[1],
                'value':line[2],
                'notes':line[3],
            }, ignore_index = True)

    # Now, test that the dataframes are similar
    test = new_file == old_file

    # Determine if only the notes have changed
    for col in test.columns:
        result = all(test[col])
        if result == False and col != 'notes':
            data_affecting = True
            break
        else:
            data_affecting = False
    
    return data_affecting


gitHub_df = pd.DataFrame(columns=['file','URL','changeType', 'gitHub changeDate'])
for file in list(pr.get_files()):
    if 'CGINS' not in file.filename:
        continue
    # Get useful metadata
    URL = file.blob_url
    changeDate = pd.to_datetime(pr.merged_at).strftime('%Y-%m-%d')
    filename = file.filename.split('/')[-1]
    # Now test the type of change
    if file.status == 'modified':
        changeType = 'Calibration coefficients were modified '
    elif file.status == 'added':
        changeType = 'Missing file added '
    elif file.status == 'renamed':
        changeType = 'File renamed with correct date '
        # First, chec if it was only a rename
        if file.patch == None:
            # Then its just a rename
            pass
        elif file.filename.endswith('.ext'):
            changeType = changeType + 'Calibration coefficients were modified '
        else:
            # Otherwise, the file itself was also modified
            # Need to check if the modification was data affecting
            data_affecting = is_data_affecting(file)
            if data_affecting:
                changeType = changeType + 'Calibration coefficients were modified '
            else:
                # Then the modification was an addition to only notes field
                pass
    else:
        continue
    # Now, append the data to a dataframe
    gitHub_df = gitHub_df.append({
        'file':filename,
        'URL':URL,
        'changeType':changeType,
        'gitHub changeDate':changeDate
    }, ignore_index = True)


np.unique(gitHub_df['changeType'])

gitHub_df[gitHub_df['changeType']!= 'File deleted']



# **====================================================================================================================**
# ### UFrame Info
# Next, get the deployment information from UFrame
#

# Now, deal with 
gitHub_df['UID'] = gitHub_df['file'].apply(lambda x: x.split('__')[0])
gitHub_df


def get_deployData(uid, username, token):
    """
    Query and return the deployment data from OOINet
    for a particular instrument uid
    """
    url = '/'.join((base_url,'12587','asset','deployments',uid+'?editphase=ALL'))
    data = requests.get(url, auth=(username, token)).json()
    if len(data) == 0:
        return None
    else:
        df = pd.DataFrame(data)
        df.sort_values(by='deploymentNumber', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


def get_calData(uid, deployData, username, token):
    """
    This function takes in the instrument uid and a dataframe of the
    deployment information for the uid, and loops through all of the
    instrument deployments to return the calibration data for the
    instrument for each individual deployment.
    """
    
    startTime = deployData['startTime']
    dt = 8.64E7 # microseconds in a day
    
    # Initialize tuples for non-mutable storage of data
    dataSource = ()
    lastModifiedTimestamp = ()
    instrument = ()
    serialNumber = ()
    
    # Loop over the deployment startTime and get the data
    for t in startTime:
        T1 = convert_ooi_time(t).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        T2 = convert_ooi_time(t+dt).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        # Generate the url and get the calibration data for a single deployment by querying a single day
        url = '/'.join((base_url,'12587','asset','cal?uid='+uid+'&beginDT={}&endDT={}'.format(T1,T2)))
        calData = requests.get(url, auth=(username, token)).json()
        # Fill out the data tuples
        instrument = instrument + (calData['description'],)
        serialNumber = serialNumber + (calData['serialNumber'],)
        dataSource = dataSource + (calData['calibration'][0]['calData'][0]['dataSource'],)
        lastModifiedTimestamp = lastModifiedTimestamp + (calData['calibration'][0]['calData'][0]['lastModifiedTimestamp'],)
        
    # Now, put the data tuples into the deploy data dataframe
    deployData['dataSource'] = dataSource
    deployData['lastModifiedTimestamp'] = lastModifiedTimestamp
    deployData['instrument'] = instrument
    deployData['serialNumber'] = serialNumber
    
    # Return the expanded deployment data
    return deployData


# +
# Specify some functions to convert timestamps
ntp_epoch = datetime.datetime(1900, 1, 1)
unix_epoch = datetime.datetime(1970, 1, 1)
ntp_delta = (unix_epoch - ntp_epoch).total_seconds()

def ntp_seconds_to_datetime(ntp_seconds):
    return datetime.datetime.utcfromtimestamp(ntp_seconds - ntp_delta).replace(microsecond=0)
  
def convert_ooi_time(ms):
    if ms is None:
        return None
    elif np.isnan(ms):
        return None
    else:
        return datetime.datetime.utcfromtimestamp(ms/1000)


# -

# Establish the order of columns for the Metadata Communications Spreadsheet
cols = ('Array','Platform','Node','Instrument','RefDes','Asset ID','Serial Number','deployment','gitHub changeDate',
        'file','URL','changeType','dateRangeStart','dateRangeEnd','annotation')

# Map the names from the OOINet json fields to the Metadata Communication Spreadsheet columns
name_map = {
    'Array':None,
    'Platform':'subsite',
    'Node':'node',
    'Instrument':'instrument',
    'RefDes':'RefDes',
    'Asset ID':'UID',
    'Serial Number':'serialNumber',
    'deployment':'deploymentNumber',
    'gitHub changeDate':'gitHub changeDate',
    'file':'file',
    'URL':'URL',
    'changeType':'changeType',
    'dateRangeStart':'startTime',
    'dateRangeEnd':'endTime',
    'annotation':None,
}

metadata_communications = pd.DataFrame(columns=cols)
missing_files = []
for i in range(len(gitHub_df)):
    
    # Get the gitHub info for a specific file
    gitHub_data = pd.DataFrame(gitHub_df.iloc[i]).T
    uid = gitHub_data['UID'][i] 
    filename = gitHub_data['file'][i]
    
    # Get the deployment data
    deployData = get_deployData(uid, ooi_user, ooi_token)
    if deployData is None:
        missing_files.append(filename)
        print(filename + ' not in Uframe')
        continue
    
    # With the deployment data, can query the calibration data
    calData = get_calData(uid, deployData, ooi_user, ooi_token)
    
    # Rename the calibration data file to conform to the github csv nomenclature
    calData['calFile'] = calData['dataSource'].apply(lambda x: x.replace('_Cal_Info.xlsx','.csv'))    
    
    # Merge the gitHub file data and the UFrame file data
    gitHub_data = gitHub_data.merge(calData, left_on='file', right_on='calFile')
    
    if len(gitHub_data) == 0:
        missing_files.append(filename)
        print(filename + ' not in UFrame')
        continue
        
    # Need to generate the Reference Designator from the 
    refdes = '-'.join((gitHub_data['subsite'][0], gitHub_data['node'][0], gitHub_data['sensor'][0]))
    gitHub_data['RefDes'] = refdes
    
    # Loop through the data and map from gitHub/UFrame naming to metadata communications naming
    data = {}
    for key in name_map:
        name = name_map.get(key)
        if name is not None:
            data.update({key: gitHub_data[name][0]})
        else:
            data.update({key: None})
            
    # Update the metadata communications dataframe
    metadata_communications = metadata_communications.append(data, ignore_index=True)


metadata_communications

missing_files

metadata = metadata_communications


def generate_arrayName(x):
    if 'GA' in x:
        arrayName = 'Global Argentine Basin'
    elif 'GI' in x:
        arrayName = 'Global Irminger Sea'
    elif 'GP' in x:
        arrayName = 'Global Station Papa'
    elif 'GS' in x:
        arrayName = 'Global Southern Ocean'
    elif 'CP' in x:
        arrayName = 'Coastal Pioneer'
    else:
        arrayName = np.nan
    return arrayName


metadata_communications['Array'] = metadata_communications['Platform'].apply(generate_arrayName)

metadata_communications

# **===================================================================================================================**
# ### Add Annotation Text

# +
with open('Annotation Text/annotationMissing.txt') as file:
    anno_missing = file.read()
    anno_missing = anno_missing.strip('\n')

with open('Annotation Text/annotationModify.txt') as file:
    anno_modify = file.read()
    anno_modify = anno_modify.strip('\n')
    
with open('Annotation Text/annotationTruncated.txt') as file:
    anno_truncated = file.read()
# -

for i in range(len(metadata_communications)):
    
    # Collect relevant data for the annotations
    refdes = metadata_communications['RefDes'].iloc[i]
    deployment = metadata_communications['deployment'].iloc[i]
    gitHub_changeDate = metadata_communications['gitHub changeDate'].iloc[i]
    dateRangeStart = metadata_communications['dateRangeStart'].iloc[i]
    dateRangeEnd = metadata_communications['dateRangeEnd'].iloc[i]
    if type(dateRangeEnd) == float:
        dateRangeEnd = 'now'
    URL = metadata_communications['URL'].iloc[i]
    
    # Set flags
    missing = False
    modify = False
    renamed = False
    
    # Determine which annotation to use based on the change type
    if 'missing' in metadata_communications['changeType'].iloc[i].lower():
        missing = True
    elif 'modified' in metadata_communications['changeType'].iloc[i].lower() or 'renamed' in metadata_communications['changeType'].iloc[i].lower():
        modify = True
    else:
        continue
    
    # Now fill in the annotation strings
    if missing:
        metadata_communications['annotation'].iloc[i] = anno_missing.format(refdes,
                                                                           deployment,
                                                                           gitHub_changeDate,
                                                                           deployment,
                                                                           dateRangeStart,
                                                                           dateRangeEnd,
                                                                           URL)
    elif modify:
        metadata_communications['annotation'].iloc[i] = anno_modify.format(refdes,
                                                                          deployment,
                                                                          gitHub_changeDate,
                                                                          deployment,
                                                                          dateRangeStart,
                                                                          dateRangeEnd,
                                                                          URL)
    else:
        pass

metadata_communications

# **===================================================================================================================**
# ### Apply the downstream affected streams

import sys
sys.path

sys.path.append('/home/andrew/Documents/OOI-CGSN/Ocean Observatories Initiative/ooi-data/')
sys.path.append('/home/andrew/Documents/OOI-CGSN/Ocean Observatories Initiative/preload-database/')

# +
import yaml
from ooi_data.postgres.model import *

from tools.m2m import MachineToMachine
from database import create_engine_from_url, create_scoped_session

# +
engine = create_engine_from_url(None)
session = create_scoped_session(engine)

MetadataBase.query = session.query_property()


# -

def build_dpi_map():
    """
    Build a map from a specific data product identifier to a set of parameters which fulfill it
    :return:
    """
    dpi_map = {}
    for p in Parameter.query:
        if p.data_product_identifier:
            dpi_map.setdefault(p.data_product_identifier, set()).add(p)
    return dpi_map


def build_affects_map():
    """
    Build a map from parameter to the set of parameters *directly* affected by it
    :return:
    """
    dpi_map = build_dpi_map()
    affects_map = {}
    for p in Parameter.query:
        if p.is_function:
            pmap = p.parameter_function_map
            for key in pmap:
                values = pmap[key]
                if not isinstance(values, list):
                    values = [values]
                for value in values:
                    if isinstance(value, Number):
                        continue
                    if value.startswith('CC'):
                        continue
                    if value.startswith('dpi_'):
                        value = value.split('dpi_')[-1]
                        for param in dpi_map.get(value, []):
                            affects_map.setdefault(param, set()).add(p)

                    if 'PD' in value:
                        pdid = int(value.split('PD')[-1])
                        param = Parameter.query.get(pdid)
                        affects_map.setdefault(param, set()).add(p)
    return affects_map


def parameter_affects(pdid, affects_map):
    """
    Given a specific parameter and a map of parameter to the set of its directly affected parameters,
    traverse the given graph to determine all possible affected parameters for the given parameter.
    Return the map of stream_name to affected parameters.
    :param pdid:
    :param affects_map:
    :return:
    """
    p = Parameter.query.get(pdid)

    affected = {p}
    to_visit = affects_map[p]

    while to_visit:
        p = to_visit.pop()
        affected.add(p)
        for param in affects_map.get(p, []):
            if param in affected:
                continue
            affected.add(param)
            to_visit.add(param)

    streams = {}
    for p in affected:
        for stream in p.streams:
            streams.setdefault(stream.name, set()).add(p)

    return streams


def find_affected(affected_streams, subsite, node, toc):
    """
    Given a map of affected streams for a parameter, traverse the TOC and identify all instrument streams
    with the same subsite and node which are affected. For each affected stream, print the affected parameters.
    :param affected_streams:
    :param subsite:
    :param node:
    :param toc:
    :return:
    """

    out = []
    for each in toc['instruments']:
        if each['platform_code'] == subsite and each['mooring_code'] == node:
            for stream in each['streams']:
                name = stream['stream']
                for parameter in affected_streams.get(name, []):
                    print('{refdes} {stream} {pid} {pname}'.format(
                    refdes=each['reference_designator'],
                    stream=stream['stream'],
                    pid=parameter.id,
                    pname=parameter.name)
                         )
                        
                    out.append(' '.join((each['reference_designator'], 
                               stream['stream'], 
                               str(parameter.id), 
                               parameter.name))
                              )
    return out


config = yaml.load(open('m2m_config.yml'))
m2m = MachineToMachine(config['url'], config['apiname'], config['apikey'])
toc = m2m.toc()
affects_map = build_affects_map()

# Now, we need to go over the available arrays and nodes for CTDs to check what downstream sensors changes to the CTD calibration files may have:

# Filter for only the CTD instruments, since thye are the only streams which affect downstream instruments
id_filter = metadata_communications['Asset ID'].apply(lambda x: True if 'CTD' in x else False)
ctd = metadata_communications[id_filter]
error_filter = ctd['changeType'].apply(lambda x: False if 'No' in x else True)
ctd = ctd[error_filter]
ctd

# +
# Get the downstream_metadata info 
downstream_metadata = pd.DataFrame(columns=metadata_communications.columns)
downstream_metadata['Upstream'] = ''

# Now, iterate over all of the Platform and Node combinations to query the affected downstream instruments
pid = [193, 194, 195]
for row in ctd.values:

    # Get the row values
    array = row[0]
    platform = row[1] 
    node = row[2]
    up_instrument = row[3]
    up_refdes = row[4]
    up_asset_id = row[5]
    up_serial_num = row[6]
    deployment = row[7]
    gitHub_changeDate = row[8] 
    up_csv = row[9]
    up_url = row[10]
    up_changeType = row[11]
    up_startTime = row[12]
    up_endTime = row[13]
    
    print('\n')
    print('===========================================')
    print(' '.join((up_refdes, platform, node, str(deployment), str(up_startTime), str(up_endTime), '\n')))
    
    # Next step is to get the affected_sensors 
    affected_sensors = []
    for p in pid:
        # Query the m2m api and get the instrument and parameter maps
        config = yaml.load(open('m2m_config.yml'))
        m2m = MachineToMachine(config['url'], config['apiname'], config['apikey'])
        toc = m2m.toc()
        affects_map = build_affects_map()
        affected_streams = parameter_affects(p, affects_map)
        affected_sensors.append(find_affected(affected_streams, platform, node, toc))
        
    # Now, I need to filter out and only get the unique refdes of the affected_streams
    downstream_refdes = []
    for sensor in affected_sensors:
        for inst in sensor:
            refdes, stream, p_id, pname = inst.split()
            print(' '.join((refdes, stream, p_id, pname, '\n')))
            if refdes not in downstream_refdes and refdes != up_refdes:
                downstream_refdes.append(refdes)
            
    # For each unique refdes, I now need to put into a dataframe the important information that I need
    for refdes in downstream_refdes:
        downstream_metadata = downstream_metadata.append({
            'Array':array,
            'Platform':platform,
            'Node':node,
            'Instrument':None,
            'RefDes':refdes,
            'Asset ID':None,
            'deployment':deployment,
            'gitHub changeDate':gitHub_changeDate,
            'OOI changeDate':None,
            'file':up_csv,
            'URL':up_url,
            'changeType':up_changeType,
            'dateRangeStart':up_startTime,
            'dateRangeEnd':up_endTime,
            'Upstream':up_refdes            
        }, ignore_index=True)
    
# -

# Drop the CTD's from the downstream list
ctd_filter = downstream_metadata['RefDes'].apply(lambda x: False if 'CTD' in x else True)
downstream_metadata = downstream_metadata[ctd_filter]
downstream_metadata

# With the downstream sensors identified, I now need to requery OOINet to get the **Instrument**, **Asset ID**, and **Serial Number** of the downstream sensor.

for i in range(len(downstream_metadata)):
    # Request the calibration data for the specific reference data for all deployments
    refdes = downstream_metadata['RefDes'].iloc[i]
    url = '/'.join((base_url,'12587','asset','cal?refdes='+refdes))
    calData = requests.get(url, auth=(ooi_user, ooi_token)).json()
    
    # Only want the data associated with a specific deployment number
    for deployment in calData:
        if deployment['deploymentNumber'] != downstream_metadata['deployment'].iloc[i]:
            continue
        else:
            data = deployment
    
    # Now, get the relevant data from the queried calibration info
    try:
        downstream_metadata['Instrument'].iloc[i] = data['sensor']['description']
        downstream_metadata['Asset ID'].iloc[i] = data['sensor']['uid']
        downstream_metadata['Serial Number'].iloc[i] = data['sensor']['serialNumber']
    except:
        pass


downstream_metadata

# **===================================================================================================================**
# ### Add in the Downstream Sensor Annotation

# +
with open('Annotation Text/annotationDownstream.txt') as file:
    downstream_anno = file.read()
    downstream_anno = downstream_anno.strip('\n')
    
for i in range(len(downstream_metadata)):
    refdes = downstream_metadata['RefDes'].iloc[i]
    upstream = downstream_metadata['Upstream'].iloc[i]
    gitHub_changeDate = downstream_metadata['gitHub changeDate'].iloc[i]
    dateRangeStart = downstream_metadata['dateRangeStart'].iloc[i]
    dateRangeEnd = downstream_metadata['dateRangeEnd'].iloc[i]
    if type(dateRangeEnd) == float:
        dateRangeEnd = 'now'
    URL = downstream_metadata['URL'].iloc[i]

    downstream_metadata['annotation'].iloc[i] = downstream_anno.format(refdes,
                                                                       upstream,
                                                                       refdes,
                                                                       gitHub_changeDate,
                                                                       refdes,
                                                                       dateRangeStart,
                                                                       dateRangeEnd,
                                                                       upstream,
                                                                       URL)
# -

# **===================================================================================================================**
# ### Concatenate the Metadata Review and Downstream DataFrames

# First drop the upstream column from the Downstream DataFrame
downstream_metadata.drop(columns='Upstream', inplace=True)

downstream_metadata

metadata_communications = metadata_communications.append(downstream_metadata)

metadata = pd.DataFrame(columns=cols)
for name in cols:
    metadata[name] = metadata_communications[name]

metadata

metadata['dateRangeStart'] = metadata['dateRangeStart'].apply(convert_ooi_time)

metadata['dateRangeEnd'] = metadata['dateRangeEnd'].apply(convert_ooi_time)

metadata

pr.number

savefile = f'Pull_request_{pr.number}_metadata_communications.csv'
savefile

metadata.to_csv("Output/" + savefile, index=False)

pd.DataFrame(missing_files).to_csv("Output/" + f'Pull_request_{pr.number}_missing_files.csv', index=False)



# Load the existing metadata communications
cgsn_metadata_communications = pd.read_csv('Output/CGSN_Metadata_Communications.csv')
cgsn_metadata_communications.drop(columns='Unnamed: 0', inplace=True)
cgsn_metadata_communications

pr_698 = pd.read_csv('Output/Pull_request_698_metadata_communications.csv')
pr_698

cgsn_metadata_communications = cgsn_metadata_communications.append(pr_698)
cgsn_metadata_communications

pr_703 = pd.read_csv('Output/Pull_request_703_metadata_communications.csv')
pr_703

cgsn_metadata_communications = cgsn_metadata_communications.append(pr_703)
cgsn_metadata_communications

pr_708 = pd.read_csv('Output/Pull_request_708_metadata_communications.csv')
pr_708

cgsn_metadata_communications = cgsn_metadata_communications.append(pr_708)
cgsn_metadata_communications

pr_725 = pd.read_csv('Output/Pull_request_725_metadata_communications.csv')
pr_725

cgsn_metadata_communications = cgsn_metadata_communications.append(pr_725)
cgsn_metadata_communications

pr_729 = pd.read_csv('Output/Pull_request_729_metadata_communications.csv')
pr_729

cgsn_metadata_communications = cgsn_metadata_communications.append(pr_729)
cgsn_metadata_communications

cgsn_metadata_communications.drop_duplicates(inplace=True)

cgsn_metadata_communications = cgsn_metadata_communications[[x for x in cols]]

cgsn_metadata_communications.to_csv('Output/CGSN_Metadata_Communication_2019-12-10.csv',index=False)



# Fuck this stupid shit. I'm so goddamn tired of this goddamn crap. 

cgsn_metadata_communications = pd.read_csv('Output/CGSN_Metadata_Communication_2019-12-10.csv')

vocab = pd.read_csv('/home/andrew/Documents/OOI-CGSN/ooi-integration/asset-management/vocab/vocab.csv')

vocab

cgsn_metadata_communications


# +
def get_platform_vocab(vocab, refdes):
    
    vocab = vocab[vocab['Reference_Designator'] == refdes]
    vocab.reset_index(drop=True, inplace=True)
    platform = vocab['TOC_L2'].iloc[0]
    
    return platform

def get_node_vocab(vocab, refdes):
    
    vocab = vocab[vocab['Reference_Designator'] == refdes]
    vocab.reset_index(drop=True, inplace=True)
    node = vocab['TOC_L3'].iloc[0]
    
    return node


# -

cgsn_metadata_communications['Platform'] = cgsn_metadata_communications['RefDes'].apply(lambda x: get_platform_vocab(vocab, x))
cgsn_metadata_communications['Node'] = cgsn_metadata_communications['RefDes'].apply(lambda x: get_node_vocab(vocab, x))

cgsn_metadata_communications

mask = cgsn_metadata_communications['RefDes'].apply(lambda x: False if x.startswith('CE') else True)
mask

cgsn_metadata_communications = cgsn_metadata_communications[mask]
cgsn_metadata_communications


def filter_change_type(x):
    
    x = x.strip()
    if x == 'File renamed with correct date':
        return False
    else:
        return True


mask = cgsn_metadata_communications['changeType'].apply(lambda x: filter_change_type(x))
mask

cgsn_metadata_communications = cgsn_metadata_communications[mask]

cgsn_metadata_communications.to_csv('Output/CGSN_Metadata_Communication_2019-12-11.csv',index=False)

pulls = repo.get_pulls(state='all', sort='merged', base='master')
for pr in pulls:
    if pr.merged:
        print(str(pr.number) + ': ' + pr.merged_at.strftime('%Y-%m-%d'))



pulls = repo.get_pulls(state='all', sort='merged', base='master')

# +
prNum = ()
prDate = ()
prFiles = ()

for pr in pulls:
    if pr.merged:
        if pr.merged_at > pd.to_datetime('2019-10-01'):
            prNum = prNum + (str(pr.number),)
            prDate = prDate + (pr.merged_at.strftime('%Y-%m-%d'),)
            files = []
            for file in pr.get_files():
                files.append(file.filename)
            prFiles = prFiles + (files,)
# -

df = pd.DataFrame(data=zip(prNum, prDate, prFiles), columns=['Pull Request #', 'Merge Date', 'Files'])

df

df.sort_values(by='Pull Request #', ascending=False, inplace = True)

df['Files'] = df['Files'].apply(lambda x: [y.split('/')[-1] for y in x])

lst = ['CE','AT','RS']

df['Files'] = df['Files'].apply(lambda x: [y for y in x if y[:2] not in lst])

df

df['Num of Files'] = df['Files'].apply(lambda x: len(x))

df

np.sum(df['Num of Files'])

df.to_csv('ooi_integration_pull_requests.csv')

files = []
for file in pr.get_files():
    files.append(file.filename)

files

for file in pr.get_files():
    print(file.filename)

file.sha

file.changes

file.deletions

file.contents_url

pr.

repo.get(sha=file.sha)


