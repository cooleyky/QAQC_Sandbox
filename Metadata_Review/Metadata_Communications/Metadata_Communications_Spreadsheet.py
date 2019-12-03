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

# # CGSN Metadata Communications
# Author: Andrew Reed
#
# Date: 2019-08-21
#
# Ver: 1.0
#
# This notebook lays out the development process for querying the relevant data fields necessary for CGSN to fill out the UW Metadata Changes & Communications spreadsheet. The goal is to explore the possible sources of the necessary information by M2M calls to OOINet based upon the available information recorded in CGSN's Metadata Tracking Spreadsheet. Eventually, the exploratory and development process laid out below will be transitioned into an automated function which fills out the requisite information with manual execution of the relevant scripts and code.
#
# **===================================================================================================================**
# Date: 2019-09-17  
# Ver: 1.10
#
# Updates:
# * Implemented GitHub mining to capture the **gitHub changeDate** value. This value is based on the merge date of the pull request containing the impacted csv file from CGSN's fork of ooi-integration/asset-management to ooi-integration.
# * Implemented Affected Downstream Sensors by porting the script **affected.py** from ooi-preload. This required the following steps:
#     * Cloning ooi-preload repository, creating a local environment to run the repo, and running the preload-setup script. This set up the preload Postgres database needed to identify affected parameter streams.
#     * Cloning the ooi-data repository
#     * Running futurize on ooi-data repo scripts
#     * Port the functions in affected.py into jupyter notebooks after running futurize on affected.py
# * Implemented annotation text. Only change was to capture end dates for instruments currently deployed as "now"
#
# To Do:
# * Redownload CGSN Metadata Review Tracking spreadsheet
# * Re-run the notebook to capture changes implemented since the last download of the CGSN Metadata Review Tracking spreadsheet
# * Reformat spreadsheet containing CTDMO changes inorder to process it using this notebook
#
# **===================================================================================================================**
# Date 2019-09-18
# Ver: 1.11
#
# Updates:
# * Redownloaded and captured metadata communications for CGSN instruments, excluding ADCPs
# * Ran the notebook and captured changes for CTDMOs
# * Fixed the gitHub mining so if a file has been changed multiple times, it captures only the latest change, OR if a change has been captured on Metadata Review but not merged it captures that as well
#
# To Do:
# * Run the WFP instrument changes
# * Implement CRUD so the entire sheet isn't redone each time the notebook is run
# * Clean up the MOAS metadata changes so they can be run in the notebook
#
# **===================================================================================================================**
# Date 2019-09-19
# Ver: 1.20
#
# Updates:
# * Processed the WFP and the MOAS instruments.
# * Created an unified spreadsheet with all captured and reviewed instrument metadata
#
# To Do:
# * Implement a CRUD/diff checker to avoid rerunning program for all files
#
# **===================================================================================================================**
# Date: 2019-09-27
# Ver: 1.21
#
# Updates:
# * Rerun code to include changes pushed in the ooi-integration:asset_management release tag 20190920-1630 which contains the following updates from Release 20190906-1930:
#     * Instruments: DOFSTK/DOSTAD, FLORTD, NUTNRB/NUTNRJ, OPTAAC/OPTAAD, PARADK, PCO2WB/C, PHSEND, SPKIRB
#     * Deployments: CE06ISSP, CE01ISSP
#     * Cruise Information
#     * Platform Bulk Load
#     
# **===================================================================================================================**
# Date: 2019-10-25
# Ver: 1.30
#
# Updates:
# * Included code to compare the original filename date against the associated deployment date and determine if the change was data-affecting.
# * Rerun the code to include changes pushed to ooi-integration:asset_management release tags 20190920-1630, 20191004-1930, and 20191017-1530
#
#     
#     
#

import os, shutil, sys, time, re, requests, csv, datetime, pytz
import pandas as pd
import numpy as np
import netCDF4 as nc
import xarray as xr
import warnings
import yaml
warnings.filterwarnings("ignore")

# Load OOINet credentials:

os.listdir('../../')

user = yaml.load(open('../../user_info.yaml'))
username = user['apiname']
token = user['apikey']

# Declare the OOINet m2m api urls:

base_url = 'https://ooinet.oceanobservatories.org/api/m2m'
sensor_url = '12576/sensor/inv'
asset_url = '12587/asset'

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

# **====================================================================================================================**
# ### Metadata Review Tracking Spreadsheet
# Load and process the metadata review tracking spreadsheet used by CGSN (CGSN Metadata Review.xlsx), which contains the following sheets:
# * Cal Review Log - contains all of the mooring instruments reviewed EXCEPT the CTDMOs (due to size of CTDMO instrument class)
# * CTDMO Cal Review - contains the metadata review of the CTDMOs on moored instruments
# * WFP Cal Review - contains all of the wire-following-profilers review instrument metadata
# * MOAS Cal Review - contains all of the mobile asset (AUVs, Gliders) reviewed instrument metadata
#
# Initial processing is needed to remove some edgecases, eliminating the few edgecases (such as a couple of DOSTAs) that have Bad Calibrations (i.e. calibrations that can't be fixed) and any empty or null rows in the spreadsheet. Additionally, instruments marked with 'TODO' have not been reviewed and should also be dropped.

metadata_review = pd.read_excel('CGSN Metadata Review.xlsx',sheet_name='MOAS Cal Review')
metadata_review.dropna(subset=['CLASS-SERIES','Duplicate'], inplace=True)
metadata_review = metadata_review[metadata_review['Original Calibration CSV'] != 'Bad']
metadata_review.head()

# Drop all of the TODO instruments
todo_filter = metadata_review['Duplicate'].apply(lambda x: False if x.lower() == 'todo' else True)
metadata_review = metadata_review[todo_filter]

print(metadata_review.columns.values)

# Next, process the metadata review tracking spreadsheet (MRTS) to create the following information:
# 1. UID
# 2. New Calibration CSV filename: this is what files are renamed to if the original csv filename is found to be incorrect, such as when the wrong calibration date was used. This is important to know since, when querying calibration and deployment data from OOINet via M2M, the new csv names are the way files are identified. We can build the new CSV filenames from the instrument UID, which was previously bilt from the instrument class-series and serial number, and the correct/corrected calibration date.
# 3. Error Classification: this is how idenified errors in the calibration csvs are grouped
#     * Wrong cal date - if the calibration date in the csv filename was wrong
#     * Wrong cal coef - this is if a calibration coefficient in the csv was identified as being incorrect
#     * Is missing - if a calibration csv file that should be in asset management is missing
#     * Is duplicate - if a calibration csv in asset management is identified as a duplicate of another csv and should be deleted
#     * Is good - the calibration date and calibration coefficients were all correct and the file is not a duplicate.
#     
# The preceding information is done via a series of simple functions applied to the appropriate dataframe columns.

# Reformat the serial numbers to make sure they are strings
metadata_review['S/N'] = metadata_review['S/N'].apply(lambda x: str(int(x)) if type(x) == float else x)


def reformat_calDate(x):
    if type(x) is int:
        return pd.to_datetime(str(x))
    else:
        return x


metadata_review['Cal Date'] = metadata_review['Cal Date'].apply(reformat_calDate)


def generate_uid(inst, sn, whoi_inst=True):
    """
    Function which takes in instrument class - series and serial number to generate an instrument uid.
    The exception to the rule is the METBK instruments, which are classified as Loggers, and thus are
    recorded as METLGR.
    """
    
    # Clean the names of the class-series
    if '-' in inst:
        inst = inst.replace('-','')
        
    # Clean the serial numbers
    sn = str(sn)
    if '-' in sn:
        ind = sn.index('-')
        sn = sn[ind+1:].zfill(5)
    elif len(sn) < 5:
        sn = sn.zfill(5)
    else:
        pass
    
    # If the instrument is a METBK, have to handle differently
    if 'METBKA' in inst:
        inst = 'METLGR'
        if 'UNKNOWN' in sn:
            sn = sn.split('\n')[-1]
        else:
            sn = sn[3:].zfill(5)   
        
    # Generate the UID
    if whoi_inst == True:
        uid = '-'.join(('CGINS',inst,sn))
        
    return uid


metadata_review['UID'] = metadata_review.apply(lambda x: generate_uid(x['CLASS-SERIES'], x['S/N']), axis=1)


def wrong_cal_date(x):
    if type(x) == str:
        if 'no' in x.lower():
            return True
        else:
            return False
    else:
        return False


metadata_review['Wrong Date'] = metadata_review['Filename correct'].apply(wrong_cal_date)


def wrong_cal_coef(x):
    if type(x) == str:
        if 'yes' in x.lower():
            return False
        else:
            return True
    elif np.isnan(x):
        return False
    else:
        return False


metadata_review['Wrong cal'] = metadata_review['Cal coeff match'].apply(wrong_cal_coef)


def is_missing(x):
    if type(x) is str:
        if x.lower() == 'new':
            return True
        else:
            return False
    else:
        return False


metadata_review['Is missing'] = metadata_review['Duplicate'].apply(is_missing)


def is_duplicate(x):
    if type(x) is str:
        if x.lower() == 'yes':
            return True
        else:
            return False
    else:
        return False        


metadata_review['Is duplicate'] = metadata_review['Duplicate'].apply(is_duplicate)


def is_good(x):
    if any(x) == True:
        return False
    else:
        return True


metadata_review['Is good'] = metadata_review[['Wrong Date', 'Wrong cal', 'Is missing', 'Is duplicate']].apply(is_good, axis=1)

# Visually confirm that the classification is correct:

metadata_review[['Wrong Date','Wrong cal','Is missing','Is duplicate','Is good']].head(10)


# Generate the new csv filenames from the instrument class-series, serial number, and the correct/corrected calibration date:

def print_calDate(x):
    if type(x) == float or type(x) == pd._libs.tslibs.nattype.NaTType:
        return None
    elif x == 'U':
        return None
    else:
        return x.strftime('%Y%m%d')


metadata_review[metadata_review['Cal Date'].apply(type) == str]

metadata_review['Cal Date'] = metadata_review['Cal Date'].apply(print_calDate)


def new_csv_filename(x):
    """
    Function to generate the new calibration csv name from
    the original csv name with the updated calibration date
    """
    
    og_csv = x['Original Calibration CSV']
    
    if not og_csv.endswith('.csv') and og_csv != None:
        og_csv = og_csv + '.csv'
        x['Original Calibration CSV'] = og_csv
        
    if x['Is duplicate']:
        return np.nan
    elif x['Cal Date'] == None:
        return og_csv
    elif x['Wrong Date'] or x['Is missing']:
        new_csv = x['UID'] + '__' + x['Cal Date'] + '.csv'
        if new_csv == x['Original Calibration CSV'] and not x['Is missing']:
            print("Check calibration date for {} for errors.".format(x['Original Calibration CSV']))
        return new_csv
    else:
        return x['Original Calibration CSV']


metadata_review['New Calibration CSV'] = metadata_review.apply(new_csv_filename, axis=1)

# **====================================================================================================================**
# ## Metadata Communications Spreadsheet
# This section steps through generating the requisite information need to fill out the UW metadata communication spreadsheet based upon CGSN's metadata review approach. I need to gather the following information for the spreadsheet:
# * Array
# * Platform
# * Node
# * Instrument
# * RefDes
# * Asset UID -  
# * Serial
# * Deployment(s)
# * Github Change Date - I don't think this is necessary, since it doesn't affect the end user until a release and ingestion to OOINet
# * OOI Change Date - Question on 
# * CSV file name - this is the filename which is in the system (so changes which have not been pushed to OOI net should not be put on the spreadsheet?)
# * Github URL - Is this also necessary, when they can directly call (via M2M) or download the calibration information from the Portal
# * Change type - This is my 5 categories from above
# * dateRange Start
# * dateRange End
# * Annotation
#
# ### Gather Relevant Data from OOINet
#
# Starting with my Metadata tracking spreadsheet above, I want to be able to use a series of M2M calls to the OOI API in order to get the data necessary to fill out the spreadsheet above. A wrinkle is that _only_ csv files which have been merged, push to ooi-integration, and ingested into OOINet can be identified by M2M. That means taking an instrument-by-instrument approach following our metadata-branching system on gitHub is preferable, in order to avoid getting ahead of files with changes not yet ingested into OOINet.

metadata_review['CLASS-SERIES'] = metadata_review['CLASS-SERIES'].apply(lambda x: x.replace('-',''))


def get_deployData(uid):
    """
    Query and return the deployment data from OOINet
    for a particular instrument uid
    """
    url = '/'.join((base_url,'12587','asset','deployments',uid+'?editphase=ALL'))
    data = requests.get(url, auth=(username, token)).json()
    df = pd.DataFrame(data)
    df.sort_values(by='deploymentNumber', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_calData(uid, deployData):
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


def reformat_dataSource(x):
    new = x.replace('_Cal_Info.xlsx','.csv')
    return new


# Establish the order of columns for the Metadata Communications Spreadsheet
cols = ('Array','Platform','Node','Instrument','RefDes','Asset ID','Serial Number','deployment','gitHub changeDate',
        'OOI changeDate','file','URL','changeType','dateRangeStart','dateRangeEnd','annotation','Wrong Date',
       'Wrong cal','Is missing','Is duplicate','Is good')

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
    'gitHub changeDate':'Pull request #',
    'OOI changeDate':'lastModifiedTimestamp',
    'file':'dataSource',
    'dateRangeStart':'startTime',
    'dateRangeEnd':'endTime',
    'annotation':None,
    'Wrong Date':'Wrong Date',
    'Wrong cal':'Wrong cal',
    'Is missing':'Is missing',
    'Is duplicate':'Is duplicate',
    'Is good':'Is good'
}


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


def generate_gitHub_url(x):
    base_url = 'https://github.com/ooi-integration/asset-management/blob/master/calibration'
    inst = x.split('-')[1]
    full_url = '/'.join((base_url,inst,x))
    return full_url


def classify_changeType(x):
    statement = ''
    if x['Is good'] == True:
        return 'No errors found'
    elif x['Is missing'] == True:
        return 'Missing file added'
    elif x['Is duplicate'] == True:
        return 'File deleted'
    else:
        if len(statement) > 0:
            statement = statement + '; '
        if x['Wrong Date'] == True:
            statement = statement + ' File renamed with correct date'
        if x['Wrong cal'] == True:
            statement = statement + ' Calibration coefficients were modified'
        return statement.strip()


def reformat_comdf(comdf):
    comdf['Array'] = comdf['Platform'].apply(generate_arrayName)
    comdf['OOI changeDate'] = comdf['OOI changeDate'].apply(convert_ooi_time)
    comdf['dateRangeStart'] = comdf['dateRangeStart'].apply(convert_ooi_time)
    comdf['dateRangeEnd'] = comdf['dateRangeEnd'].apply(convert_ooi_time)
    comdf['URL'] = comdf['file'].apply(generate_gitHub_url)
    comdf['changeType'] = comdf[['Wrong Date','Wrong cal','Is missing','Is duplicate','Is good']].apply(classify_changeType, axis=1)
    comdf.drop(columns=['Wrong Date','Wrong cal','Is missing','Is duplicate','Is good'], inplace=True)
    return comdf


# Select an instrument CLASS-SERIES from the metadata_review, iterate over all of the instrument uids in that particular class, querying OOINet for the associated deployment and calibration information. This section was purposefully not automated for each CLASS-SERIES in order to avoid hitting api requests limits and timing-out.

print(np.unique(metadata_review['CLASS-SERIES']))

# Import all the pull requests for ooi-integration since the start of 2019:

git = pd.read_csv('ooi_integration_pull_requests.csv')
git.drop(columns='Unnamed: 0', inplace=True)

for instrument in np.unique(metadata_review['CLASS-SERIES']):
    
    # Initialize the metadata communications dataframe
    metadata_communications = pd.DataFrame(columns=cols).drop(columns=['Wrong Date','Wrong cal','Is missing','Is duplicate','Is good'])
    metadata_communications
    
    # Initialize a dataframe to store which instruments throw errors and where the errors occur
    error_df = pd.DataFrame(columns=['uid','step'])
    error_df
    
    # Now, iterate through all of the unique UIDs for a particular instrument class
    uids = np.unique(metadata_review[metadata_review['CLASS-SERIES'] == instrument]['UID'])
    for uid in uids:
        # Step 1: Get the deployment data
        try:
            deploydf = get_deployData(uid)
        except:
            edf = pd.DataFrame.from_dict({'uid':uid,'step':[1]})
            error_df = error_df.append(edf)
            continue
        # Step 2: Get the associated cal data for each deployment
        try:
            deploydf = get_calData(uid, deploydf)
        except:
            edf = pd.DataFrame.from_dict({'uid':uid,'step':[2]})
            error_df = error_df.append(edf)
            continue
        # Step 3: Reformat some of the deployment data to match OOI naming conventions
        try:
            deploydf['dataSource'] = deploydf['dataSource'].apply(reformat_dataSource)
            deploydf['RefDes'] = deploydf['subsite'] + '-' + deploydf['node'] + '-' + deploydf['sensor']
        except:
            edf = pd.DataFrame.from_dict({'uid':uid,'step':[3]})
            error_df = error_df.append(edf)
            continue
        # Step 4:
        try:
            udf = metadata_review[metadata_review['UID'] == uid]
        except:
            edf = pd.DataFrame.from_dict({'uid':uid,'step':[4]})
            error_df = error_df.append(edf)
            continue
        # Step 5:
        try:
            udf = udf.merge(deploydf, left_on='New Calibration CSV', right_on='dataSource')
        except:
            edf = pd.DataFrame.from_dict({'uid':uid,'step':[5]})
            error_df = error_df.append(edf)
            continue
        # Step 6: Generate the communications dataframe
        try:
            comdf = pd.DataFrame(columns=cols)
            for i in cols:
                if name_map.get(i) is not None:
                    comdf[i] = udf[name_map.get(i)]
        except:
            edf = pd.DataFrame.from_dict({'uid':uid,'step':[6]})
            error_df = error_df.append(edf)
            continue
        # Step 7: Reformat many of the fields in the communications dataframe
        try:
            comdf = reformat_comdf(comdf)
        except:
            edf = pd.DataFrame.from_dict({'uid':uid,'step':[7]})
            error_df = error_df.append(edf)
            continue
        # Step 8: Append the comdf dataframe to the metadata_communications dataframe
        try:
            metadata_communications = metadata_communications.append(comdf)
        except:
            edf = pd.DataFrame.from_dict({'uid':uid,'step':[8]})
            error_df = error_df.append(edf)
            continue
            
    # Now I need a function to parse out and identify the gitHub changeDate because I effed up
    for i in range(len(metadata_communications)):
        # Check that there actually is an error
        if 'No' in metadata_communications['changeType'].iloc[i]:
            continue
        else:
            file = metadata_communications['file'].iloc[i]
            changeDate = []
            for j in range(len(git)):
                files = git['Files'].iloc[j]
                if file in files:
                    changeDate.append(git['Merge Date'].iloc[j])
            if len(changeDate) == 0:
                changeDate.append('Not Yet Merged')
            elif len(changeDate) > 1:
                changeDate = list(max(changeDate))
            else:
                pass

            metadata_communications['gitHub changeDate'].iloc[i] = changeDate
            
    # Now, save the metadata communications spreadsheets and error spreadsheets
    filename = instrument + '_metadata_communications.csv'
    metadata_communications.to_csv('Output/' + filename, index=False)
    error_df.to_csv('Output/' + instrument + '_errors.csv' )


filename

# Initialize the metadata communications dataframe
metadata_communications = pd.DataFrame(columns=cols).drop(columns=['Wrong Date','Wrong cal','Is missing','Is duplicate','Is good'])
metadata_communications

# Initialize a dataframe to store which instruments throw errors and where the errors occur
error_df = pd.DataFrame(columns=['uid','step'])
error_df

uids = np.unique(metadata_review[metadata_review['CLASS-SERIES'] == instrument]['UID'])
for uid in uids:
    # Step 1: Get the deployment data
    try:
        deploydf = get_deployData(uid)
    except:
        edf = pd.DataFrame.from_dict({'uid':uid,'step':[1]})
        error_df = error_df.append(edf)
        continue
    # Step 2: Get the associated cal data for each deployment
    try:
        deploydf = get_calData(uid, deploydf)
    except:
        edf = pd.DataFrame.from_dict({'uid':uid,'step':[2]})
        error_df = error_df.append(edf)
        continue
    # Step 3: Reformat some of the deployment data to match OOI naming conventions
    try:
        deploydf['dataSource'] = deploydf['dataSource'].apply(reformat_dataSource)
        deploydf['RefDes'] = deploydf['subsite'] + '-' + deploydf['node'] + '-' + deploydf['sensor']
    except:
        edf = pd.DataFrame.from_dict({'uid':uid,'step':[3]})
        error_df = error_df.append(edf)
        continue
    # Step 4:
    try:
        udf = metadata_review[metadata_review['UID'] == uid]
    except:
        edf = pd.DataFrame.from_dict({'uid':uid,'step':[4]})
        error_df = error_df.append(edf)
        continue
    # Step 5:
    try:
        udf = udf.merge(deploydf, left_on='New Calibration CSV', right_on='dataSource')
    except:
        edf = pd.DataFrame.from_dict({'uid':uid,'step':[5]})
        error_df = error_df.append(edf)
        continue
    # Step 6: Generate the communications dataframe
    try:
        comdf = pd.DataFrame(columns=cols)
        for i in cols:
            if name_map.get(i) is not None:
                comdf[i] = udf[name_map.get(i)]
    except:
        edf = pd.DataFrame.from_dict({'uid':uid,'step':[6]})
        error_df = error_df.append(edf)
        continue
    # Step 7: Reformat many of the fields in the communications dataframe
    try:
        comdf = reformat_comdf(comdf)
    except:
        edf = pd.DataFrame.from_dict({'uid':uid,'step':[7]})
        error_df = error_df.append(edf)
        continue
    # Step 8: Append the comdf dataframe to the metadata_communications dataframe
    try:
        metadata_communications = metadata_communications.append(comdf)
    except:
        edf = pd.DataFrame.from_dict({'uid':uid,'step':[8]})
        error_df = error_df.append(edf)
        continue

error_df

# Import all the pull requests for ooi-integration since the start of 2019:

git = pd.read_csv('ooi_integration_pull_requests.csv')
git.drop(columns='Unnamed: 0', inplace=True)

# Parse the ooi-integration/asset-management pull request merge dates for when the metadata files were updated:

# Now I need a function to parse out and identify the gitHub changeDate because I effed up
for i in range(len(metadata_communications)):
    # Check that there actually is an error
    if 'No' in metadata_communications['changeType'].iloc[i]:
        continue
    else:
        file = metadata_communications['file'].iloc[i]
        changeDate = []
        for j in range(len(git)):
            files = git['Files'].iloc[j]
            if file in files:
                changeDate.append(git['Merge Date'].iloc[j])
        if len(changeDate) == 0:
            changeDate.append('Not Yet Merged')
        elif len(changeDate) > 1:
            changeDate = list(max(changeDate))
        else:
            pass

        metadata_communications['gitHub changeDate'].iloc[i] = changeDate

metadata_communications.head()

filename = instrument + '_metadata_communications.csv'
filename

metadata_communications.to_csv('Output/' + filename, index=False)

# **===================================================================================================================**
# ### Merge and Reprocess Communication Spreadsheets
# Iterate over all of the instrument specific metadata communication spreadsheets, concatenate them, and then reprocess to arrive at a single gitHub changeDate.

sorted(os.listdir('Output/'))

metadata_communications = pd.DataFrame()
for file in os.listdir('Output/'):
    if file.endswith('.csv'):
        if 'metadata_communications.csv' in file:
            metadata_communications = metadata_communications.append(pd.read_csv('Output/'+file))

np.unique(metadata_communications['Instrument'])


def reformat_changeDate(x):
    if type(x) == str:
        x = "".join((y for y in x if y not in "'[]'"))
        d = max(x.split(','))
        return d
    else:
        return x


metadata_communications['gitHub changeDate'] = metadata_communications['gitHub changeDate'].apply(reformat_changeDate)

metadata_communications.head()

metadata_communications.to_csv('Metadata_communications_spreadsheet.csv', index=False)

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

error_filter = metadata_communications['changeType'].apply(lambda x: False if 'No' in x else True)
metadata_communications = metadata_communications[error_filter]
todo_filter = metadata_communications['gitHub changeDate'].apply(lambda x: False if 'No' in x else True)
metadata_communications = metadata_communications[todo_filter]

np.unique(metadata_communications['changeType'])

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


cgsn_filter = metadata_communications['Platform'].apply(lambda x: False if x.startswith('CE') else True)
metadata_communications = metadata_communications[cgsn_filter]

metadata_communications

metadata_communications.to_csv('Metadata_Communications.csv', index=False)

# **===================================================================================================================**
# ### Apply the downstream affected streams

metadata_communications = pd.read_csv('Metadata_Communications.csv')
#metadata_communications.drop(columns=['Unnamed: 0','Unnamed: 0.1'], inplace=True)
metadata_communications

#metadata_communications.drop(columns=['Unnamed: 0'], inplace=True)
metadata_communications.head(10)

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
    up_csv = row[10]
    up_url = row[11]
    up_changeType = row[12]
    up_startTime = row[13]
    up_endTime = row[14]
    
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

# +
for i in range(len(downstream_metadata)):
    # Request the calibration data for the specific reference data for all deployments
    refdes = downstream_metadata['RefDes'].iloc[i]
    url = '/'.join((base_url,'12587','asset','cal?refdes='+refdes))
    calData = requests.get(url, auth=(username, token)).json()
    
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
    
    # FIN!
# -

downstream_metadata.head(10)

data

data['sensor']['uid']

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

downstream_metadata

downstream_metadata['annotation'].iloc[16]

# **===================================================================================================================**
# ### Concatenate the Metadata Review and Downstream DataFrames

# First drop the upstream column from the Downstream DataFrame
downstream_metadata.drop(columns='Upstream', inplace=True)

# Second, concatenate the two DataFrames
metadata_communications = metadata_communications.append(downstream_metadata)

# Check the output
metadata_communications

# Save the output as a csv file
metadata_communications.to_csv('Output/CGSN_Metadata_Communications.csv')

# **===================================================================================================================**
#
