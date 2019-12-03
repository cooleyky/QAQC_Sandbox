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

# # Exploratory Analysis for Metadata Review in OOI Asset Management System
#
# ### Motivation:
# The Asset Management system for OOI is primarly housed on GitHub in a variety of csv files. Until now, the calibration coefficients stored in the csv files have been manually entered. While we have utilized a "human-in-the-loop" review approach to catch errors, some errors have slipped through (e.g. truncation of significant figures).
#
# ### Approach:
# My goal is to develop an automated approach to catch possible errors which already exist within the asset management system. To accomplish this, I will compare the csv files loaded into the GitHub asset management system with the original vendor files as well as the QCT (quality control testing) documents which capture the coefficients loaded onto the instrument at the time of reception at WHOI from the vendor.
#
# ### Data Sources:
# * **GitHub**: CSV files containing the calibration coefficients. Directory organization by sensor+class. The files are named as "(CGINS)-(sensor+class)-(serial number)-(YYYYMMDD)" where YYYYMMDD is the calibration date.
# * **Vault**: Version-controlled storage location of the vendor calibrations, in the Records/Instrument Records/Instrument directories. Within the relevant directory, calibration files are stored as either .cal, .xmlcon, .pdf, or within zipped directories.
# * **Alfresco**: Version-controlled web-accessed. The calibrations loaded onto the instrument during the initial checkin-in upon receipt (the QCT process) are stored here as either .cap or .txt files. 

# Import likely important packages, etc.
import sys, os, csv, re
from wcmatch import fnmatch
import datetime
import time
import xml.etree.ElementTree as et
from zipfile import ZipFile
import numpy as np
import pandas as pd
import xarray as xr
import shutil

sys.path.append('/'.join((os.getcwd(),'Calibration','Parsers')))

# Import self-written functions from utils package:

from utils import *
from CTDCalibration import CTDCalibration

# ### WHOI Asset Tracking Spreadsheet
# First, I want to load and examine exactly what type of data is stored in the WHOI Asset Tracking Spreadsheet and what information it has that may be useful.

#excel_spreadsheet = 'C:/Users/areed/Documents/Project_Files/Documentation/System/System Notebook/WHOI_Asset_Tracking.xlsx'
excel_spreadsheet = '/media/andrew/OS/Users/areed/Documents/Project_Files/Documentation/System/System Notebook/WHOI_Asset_Tracking.xlsx'
sheet_name = 'Sensors'

# What are all the different series of CTDs?
ADCPA = whoi_asset_tracking(excel_spreadsheet,sheet_name,instrument_class='ADCPA',whoi=True)

ADCPA.dropna(subset=['UID'],inplace=True)

ADCPA

set(ADCPT['Series'])

len(set(ADCPT['UID']))

qct_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/ADCPT/ADCPT_Results'
cal_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/ADCPT/ADCPT_Cal'
asset_management_directory = '/home/andrew/Documents/OOI-CGSN/ooi-integration/asset-management/calibration/ADCPTG/'

csv_dict = load_asset_management(ADCPT, asset_management_directory)
csv_dict

len(csv_dict.keys())

#count = 0
for uid in csv_dict.keys():
    count = count + len(csv_dict[uid])
print(count)
    

# Load an example of the csv file to determine the number of calibration coefficients per instrument that need to be checked.

fpath = generate_file_path(asset_management_directory, csv_dict['CGINS-SPKIRB-00288'][0])

fpath

df = pd.read_csv(fpath)
df

len(df['name'])

# Get the unique identifiers (UID) of the instruments:

uids = list(set(DOSTA['UID']))

len(uids)

qct_dict = {}
for uid in uids:
    # Get the QCT Document numbers from the asset tracking sheet
    CTDBP['UID_match'] = CTDBP['UID'].apply(lambda x: True if uid in x else False)
    qct_series = CTDBP[CTDBP['UID_match'] == True]['QCT Testing']
    qct_series = list(qct_series.iloc[0].split('\n'))
    qct_dict.update({uid:qct_series})

serial_nums = get_serial_nums(CTDBP, uids)

# Get the calibration files:

calibration_files = {}
for uid in serial_nums.keys():
    sn = serial_nums.get(uid)
    sn = str(sn[:])
    files = []
    for file in os.listdir(cal_directory):
        if 'Calibration_File' in file:
            if sn in file:
                files.append(file)
    calibration_files.update({uid:files})
cal_dict = calibration_files

calibration_files

csv_dict

# Purge the temp directory
shutil.rmtree('/'.join((os.getcwd(),'temp')))

# Put the csv files into a similar temp directory for local working
for file in csv_dict[uid]:
    csv_savepath = '/'.join((os.getcwd(),'temp','csv'))
    ensure_dir('/'.join((os.getcwd(),'temp','csv')))
    # Now save the csv into the temp directory
    shutil.copy('/'.join((asset_management_directory,file)), csv_savepath)


def generate_file_path(dirpath, filename, ext=['.cap', '.txt', '.log'], exclude=['_V', '_Data_Workshop']):
    """
    Function which searches for the location of the given file and returns
    the full path to the file.

    Args:
        dirpath - parent directory path under which to search
        filename - the name of the file to search for
        ext - file endings to search for
        exclude - optional list which allows for excluding certain
            directories from the search
    Returns:
        fpath - the file path to the filename from the current
            working directory.
    """
    # Check if the input file name has an extension already
    # If it does, parse it for input into the search algo
    if '.' in filename:
        check = filename.split('.')
        filename = check[0]
        ext = ['.'+check[1]]

    for root, dirs, files in os.walk(dirpath):
        dirs[:] = [d for d in dirs if d not in exclude]
        for fname in files:
            if fnmatch.fnmatch(fname, [filename+'*'+x for x in ext]):
                fpath = os.path.join(root, fname)
                return fpath


for file in qct_dict[uid]:
    # Generate the full file path
    qct_path = generate_file_path(qct_directory, file)
    # Initialize a CTD object
    CTD = CTDCalibration(uid=uid)
    # Load the QCT information
    try:
        CTD.load_qct(qct_path)
        # Generate the save file path
        qct_savepath = '/'.join((os.getcwd(),'temp','qct'))
        ensure_dir('/'.join((os.getcwd(),'temp','qct')))
    except ValueError:
        print(f"QCT UID does not match serial in QCT")
    except:
        print(f'No QCT file found for: {file}')
    # Now save the qct info to a csv
    try:
        CTD.write_csv(qct_savepath)
    except:
        pass

for file in cal_dict[uid]:
    # Generate the full file path
    cal_path = generate_file_path(cal_directory, file, ext=[''])
    # Initialize a CTD object
    CTD = CTDCalibration(uid=uid)
    # Load the QCT information
    xml_savepath = '/'.join((os.getcwd(),'temp','xml'))
    ensure_dir(xml_savepath)
    try:
        CTD.load_xml(cal_path)
        # Generate the save file path
        xml_savepath = '/'.join((os.getcwd(),'temp','xml'))
        ensure_dir(xml_savepath)
        # Now save the qct info to a csv
    except:
        pass
    try:
        CTD.write_csv(xml_savepath)
    except ValueError:
        print(f'No xml file found for {file}')

for file in cal_dict[uid]:
    # Generate the full file path
    cal_path = generate_file_path(cal_directory, file, ext=[''])
    # Initialize a CTD object
    CTD = CTDCalibration(uid=uid)
    # Load the QCT information
    try:
        CTD.load_cal(cal_path)
        # Generate the save file path
        cal_savepath = '/'.join((os.getcwd(),'temp','cal'))
        ensure_dir(cal_savepath)
        # Now save the qct info to a csv
    except ValueError:
        raise ValueError
    except:
        pass
    try:
        CTD.write_csv(cal_savepath)
    except ValueError:
        print(f'No cal file found for {file}')

CTD.serial


# ### Checking instrument calibration values
# After loading the **WHOI Asset Tracking Sheet**, we now have the following critical data for checking calibration information:
# * Supplier Serial Number - this links back to the original **.cal**, **.xmlcon**, and vendor docs
# * OOI UID - this is the link between the instrument and the OOINet
# * QCT Document Number - this number links the instrument to the QCT screen capture of the calibration values loaded onto the instruments

# ### Process to load the **CSV** calibration file
# In order to check that the calibrations in asset management, I have to be able to load the asset management calibration csv files into a dataframe. 
# * First, get all the unique CTDBPCs in Asset Management
# * Next, parse the csv files in asset management to get the unique instrument serial numbers
# * With the serial numbers, find the associated instrument calibration csvs
# * For each calibration csv, load the data into a pandas dataframe

def get_file_date(x):
    x = str(x)
    ind1 = x.index('__')
    ind2 = x.index('.')
    return x[ind1+2:ind2]


# Now we want to compare dataframe
csv_files = pd.DataFrame(sorted(csv_dict[uid]),columns=['csv'])
csv_files['cal date'] = csv_files['csv'].apply(lambda x: get_file_date(x))
csv_files.set_index('cal date',inplace=True)

# Now we want to compare dataframe
cal_files = pd.DataFrame(sorted(os.listdir('temp/cal')),columns=['cal'])
cal_files['cal date'] = cal_files['cal'].apply(lambda x: get_file_date(x))
cal_files.set_index('cal date',inplace=True)

# Now we want to compare dataframe
xml_files = pd.DataFrame(sorted(os.listdir('temp/xml')),columns=['xml'])
xml_files['cal date'] = xml_files['xml'].apply(lambda x: get_file_date(x))
xml_files.set_index('cal date',inplace=True)

# Now we want to compare dataframe
qct_files = pd.DataFrame(sorted(os.listdir('temp/qct')),columns=['qct'])
qct_files['cal date'] = qct_files['qct'].apply(lambda x: get_file_date(x))
qct_files.set_index('cal date',inplace=True)

df_files = csv_files.join(cal_files,how='outer').join(xml_files,how='outer').join(qct_files,how='outer').fillna(value='-999')

df_files

uid

CTD.ctd_type + '-' + CTD.serial

df_files['csv']


def check_cal_coeffs(coeffs_dict):
    
    # Part 1: coeff by coeff comparison between each source of coefficients
    keys = list(coeffs_dict.keys())
    comparison = {}
    for i in range(len(keys)):
        names = (keys[i], keys[i - (len(keys)-1)])
        check = len(coeffs_dict.get(keys[i])['value']) == len(coeffs_dict.get(keys[i - (len(keys)-1)])['value'])
        if check:
            compare = np.isclose(coeffs_dict.get(keys[i])['value'], coeffs_dict.get(keys[i - (len(keys)-1)])['value'])
            comparison.update({names:compare})
        else:
            pass
        
    # Part 2: now do a logical_and comparison between the results from part 1
    keys = list(comparison.keys())
    i = 0
    mask = comparison.get(keys[i])
    while i < len(keys)-1:
        i = i + 1
        mask = np.logical_and(mask, comparison.get(keys[i]))
        print(i)
       
    return mask 


result = {}
for cal_date in df_files.index:
    # Part 1, load all of the csv files
    coeffs_dict = {}
    for source,fname in df_files.loc[cal_date].items():
        if fname != '-999':
            load_directory = '/'.join((os.getcwd(),'temp',source,fname))
            df_coeffs = pd.read_csv(load_directory)
            df_coeffs.set_index(keys='name',inplace=True)
            df_coeffs.sort_index(inplace=True)
            coeffs_dict.update({source:df_coeffs})
        else:
            pass
    
    # Part 2, now check the calibration coefficients
    mask = check_cal_coeffs(coeffs_dict)
    
    # Part 3: get the calibration coefficients are wrong
    # and show them
    fname = df_files.loc[cal_date]['csv']
    if fname == '-999':
        incorrect = 'No csv file.'
    else:
        incorrect = coeffs_dict['csv'][mask == False]
    result.update({fname:incorrect})

result

CSV = pd.read_csv('/'.join((asset_management_directory,df_files.loc[indices[0]].loc['csv'])))
CSV.set_index(keys='name',inplace=True)
CSV.sort_index(inplace=True)
CSV

QCT = pd.read_csv('/'.join((os.getcwd(),'temp','qct',df_files.loc[indices[0]].loc['qct'])))
QCT.set_index(keys='name',inplace=True)
QCT.sort_index(inplace=True)
QCT

XML = pd.read_csv('/'.join((os.getcwd(),'temp','xml',df_files.loc[indices[0]].loc['xml'])))
XML.set_index(keys='name',inplace=True)
XML.sort_index(inplace=True)
XML

csv_xml = np.isclose(CSV['value'],XML['value'])
csv_qct = np.isclose(CSV['value'],QCT['value'])

csv_xml

mask = (csv_xml | csv_qct)

mask

CSV[mask == False]

file

# Now we have successfully loaded the csv calibrations into a pandas dataframe that allows for easy comparison between calibrations based on the calibration date for each calibration coefficient.

# ### Load the QCT values
# The next step is to take the capture files from the QCT and load them into a comparable pandas dataframe. This involves several steps:
# * Get the QCT document numbers from the WHOI Asset Tracking Sheet for each individual instrument
# * Find where the QCT documents are stored
# * Load the QCT documents
# * Parse the QCT documents
# * Translate the parsed QCT values into a pandas dataframe

uids = sorted( list( set( CTDBPP['UID'])))


# +
# Now I need to load the all of the csv files based on their UID
def load_csv_info(csv_dict,filepath):
    """
    Loads the calibration coefficient information contained in asset management
    
    Args:
        csv_dict - a dictionary which associates an instrument UID to the
            calibration csv files in asset management
        filepath - the path to the directory containing the calibration csv files
    Returns:
        csv_cals - a dictionary which associates an instrument UID to a pandas
            dataframe which contains the calibration coefficients. The dataframes
            are indexed by the date of calibration
    """
    
    # Load the calibration data into pandas dataframes, which are then placed into
    # a dictionary by the UID
    csv_cals = {}
    for uid in csv_dict:
        cals = pd.DataFrame()
        for file in csv_dict[uid]:
            data = pd.read_csv(filepath+file)
            date = file.split('__')[1].split('.')[0]
            data['CAL DATE'] = pd.to_datetime(date)
            cals = cals.append(data)
        csv_cals.update({uid:cals})
        
    # Pivot the dataframe to be sorted based on calibration date
    for uid in csv_cals:
        csv_cals[uid] = csv_cals[uid].pivot(index=csv_cals[uid]['CAL DATE'], columns='name')['value']
        
    return csv_cals


# -

qct_dict = {}
for uid in uids:
    # Get the QCT Document numbers from the asset tracking sheet
    CTDBPP['UID_match'] = CTDBPP['UID'].apply(lambda x: True if uid in x else False)
    qct_series = CTDBPP[CTDBPP['UID_match'] == True]['QCT Testing']
    qct_series = list(qct_series.iloc[0].split('\n'))
    qct_dict.update({uid:qct_series})

qct_dict


# Try building a function to do the file path generator
def generate_file_path(dirpath,filename,ext=['.cap','.txt','.log'],exclude=['_V','_Data_Workshop']):
    """
    Function which searches for the location of the given file and returns
    the full path to the file.
    
    Args:
        dirpath - parent directory path under which to search
        filename - the name of the file to search for
        ext - 
        exclude - optional list which allows for excluding certain
            directories from the search
    Returns:
        fpath - the file path to the filename from the current
            working directory.
    """
    for root, dirs, files in os.walk(dirpath):
        dirs[:] = [d for d in dirs if d not in exclude]
        for fname in files:
            if fnmatch.fnmatch(fname, [filename+'*'+x for x in ext]):
                fpath = os.path.join(root, fname)
                return fpath


qct_filepath

CTD = CTDCalibration(uid=uids[0])

CTD.coefficients

CTD.date

CTD.load_qct(qct_filepath)

CTD.coefficients

qct_filepath = generate_file_path(dirpath,qcts[0])
qct_filepath

CTD = CTDCalibration(uid=uids[0])

CTD.load_qct('/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/3305-00102-00019-A.txt')

CTD.serial

for root, dirs, files in os.walk(dirpath):
    dirs[:] = [d for d in dirs if d not in exclude]
    for fname in files:
        if fnmatch.fnmatch(fname, [])


# Now to develop an automated approach to load all the QCT documents, parse them
# into a dictionary, and convert the dictionary into a pandas dataframe
def load_qct_data(qct_dict,coefficient_name_map,dirpath='../../../Documents/Project_Files/'):
    qct = {}
    qct_missing = {}
    for uid in qct_dict:
        print(uid)
        capture_data = {}
        missing = []
        for capfile in qct_dict[uid]:
            # First, find and return the path to the capture file which
            # matches the capture file indentifier
            cappath = generate_file_path(dirpath, capfile)
            
            # Function to pull out the coefficients from the capture files. This is a naive implementation
            # and splits only on either a ":" or "=", it doesn't do any comprehension of the file
            if cappath is None:
                missing.append(capfile)
            else:
                coeffs = {}
                with open(cappath) as filename:
                    data = filename.read()
                    for line in data.splitlines():
                        items = re.split(': | =',line)
                        key = items[0].strip()
                        value = items[-1].strip()
                        coeffs.update({key:value})
                    
                # The best way to do this is to use the CTD name mapping to only get the important values
                capture = {}
                # With the capture coefficients, now map it to the CTD coefficients
                for key in coeffs.keys():
                    if key in coefficient_name_map.keys():
                        capture[coefficient_name_map[key]] = coeffs[key]
            
                # Get the calibration date
                caldate = coeffs['conductivity']
            
                # Update the capture file to include the calibration date
                capture['CAL DATE'] = pd.to_datetime(caldate)
            
                # Now, update the parent dictionary
                capture_data.update({capfile:capture})
            
        df = pd.DataFrame.from_dict({i: capture_data[i] for i in capture_data.keys()}, orient='index')
        qct.update({uid:df})
        qct_missing.update({uid:missing})
        
    return qct, qct_missing   


qct, qct_missing = load_qct_data(qct_dict,coefficient_name_map,dirpath='../../../../Documents/Project_Files/')

qct

qct_missing

# Reset the index to the calibration date
for uid in qct:
    qct[uid].set_index('CAL DATE', drop=True, inplace=True)

qct

# ### Vendor Calibration values: .cal and .xmlcon
# This next step is to load the CTD .cal and .xmlcon files in order to compare the

serial_nums = get_serial_nums(CTDBPC, uids)

serial_nums



vendor_files = {}
for uid,sn in serial_nums.items():
    files = []
    for file in os.listdir('../../../../Documents/Project_Files/Records/Instrument_Records/CTDBP/'):
        if sn in file:
            if 'Calibration_File' in file:
                files.append(file)
            else:
                pass
        else:
            pass
    vendor_files.update({uid:files})

cal_dict = get_calibration_files(serial_nums,'/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/CTDBP')

cal = {}
cal_missing = {}
filepath = '../../../../Documents/Project_Files/Records/Instrument_Records/CTDBP/'
for uid,files in vendor_files.items():
    cal_coeffs, missing = load_cal_coeffs(files,filepath,coefficient_name_map,o2_coefficients_map)
    cal_df = pd.DataFrame.from_dict({i: cal_coeffs[i] for i in cal_coeffs.keys()}, orient='index')
    cal_df.index = pd.to_datetime(cal_df.index)
    cal.update({uid:cal_df})
    cal_missing.update({uid:missing})

cal

cal_missing

# #### Repeat the above process with the .xmlcon file

xml = {}
xml_missing = {}
filepath = '../../../../Documents/Project_Files/Records/Instrument_Records/CTDBP/'
for uid,files in vendor_files.items():
    xml_coeffs, missing = load_xml_coeffs(files,filepath,coefficient_name_map,o2_coefficients_map)
    xml_df = pd.DataFrame.from_dict({i: xml_coeffs[i] for i in xml_coeffs.keys()}, orient='index')
    xml_df.drop(columns=[None],axis=1,inplace=True)
    xml_df.index = pd.to_datetime(xml_df.index)
    xml.update({uid:xml_df})
    xml_missing.update({uid:missing})

xml

xml_missing

# ### Comparisons
# Now that I have .cal, .xmlcon, the qct capture files, and the csv files from asset management, I can begin comparison of the calibration coefficients between the different files. The goal is that the dates, values, and coefficients all match.

CSV

qct

cal

xml

# First, I need to reindex all of the different dataframes such that they all have two indices:
# A dataset index and a datetime index, and set them to uniform name (for concatenation)
for uid in uids:
    try:
        CSV[uid]['Dataset'] = 'CSV'
        CSV[uid].set_index(['Dataset',CSV[uid].index],inplace=True)
        CSV[uid].index.set_names(['Dataset','Cal Date'],inplace=True)
    except:
        pass
CSV

qct

for uid in uids:
    qct[uid]['Dataset'] = 'QCT'
    qct[uid].set_index(['Dataset',qct[uid].index],inplace=True)
    qct[uid].index.set_names(['Dataset','Cal Date'],inplace=True)
qct

for uid in uids:
    cal[uid]['Dataset'] = 'CAL'
    cal[uid].set_index(['Dataset',cal[uid].index],inplace=True)
    cal[uid].index.set_names(['Dataset','Cal Date'],inplace=True)
cal

for uid in uids:
    xml[uid]['Dataset'] = 'XML'
    xml[uid].set_index(['Dataset',xml[uid].index],inplace=True)
    xml[uid].index.set_names(['Dataset','Cal Date'],inplace=True)
xml

# All four possible sources of calibration coefficients available for an instrument - the calibration **CSV** loaded into asset management, the calibration coefficients loaded onto the instrument during check-in (**QCT**), the **.cal** file provided by the vendor, and the **XML** file provided by the vendor. 
#
# The next step is to concatenate the different instruments into a single dataframe and to sort by calibration date. This will allow for comparison based on the date of the calibration.

comparison = {}
for uid in uids:
    comparison.update({uid:pd.concat([CSV.get(uid), cal.get(uid), xml.get(uid), qct.get(uid)])})
    comparison[uid].reset_index(level='Cal Date',inplace=True)
    comparison[uid].sort_values(by='Cal Date',inplace=True)
comparison


def convert_type(x):
    if type(x) is str:
        return float(x)
    else:
        return x


for uid in uids:
    comparison[uid] = comparison[uid].applymap(convert_type)
comparison


def all_the_same(elements):
    """
    This function checks which values in an array are all the same.
    
    Args:
        elements - an array of values
    Returns:
        error - an array of length (m-1) which checks if
    
    """
    if len(elements) < 1:
        return True
    el = iter(elements)
    first = next(el, None)
    #check = [element == first for element in el]
    error = [np.isclose(element,first) for element in el]
    return error


def locate_cal_error(array):
    """
    This function locates which source file (e.g. xmlcon vs csv vs cal)
    have calibration values that are different from the others. It does
    NOT identify which is correct, only which is different.
    
    Args:
        array - A numpy array which contains the values for a specific
                calibration coefficient for a specific date from all of
                the calibration source files
    Returns:
        dataset - a list containing which calibration sources are different
                from the other files
        True - if all of the calibration values are the same
        False - if the first calibration value is different
    """
    # Call the function to check if there are any differences between each of
    # calibration values from the different sheets
    error = all_the_same(array)
    # If they are all the same, return True
    if all(error):
        return True
    # If there is a mixture of True/False, find the false and return them
    elif any(error) == True:
        indices = [i+1 for i, j in enumerate(error) if j == False]
        dataset = list(array.index[indices])
        return dataset
    # Last, if all are false, that means the first value 
    else:
        return False


# With all the functions set up, now go through all of the data
def search_for_errors(df):
    """
    This function is designed to search through a pandas dataframe
    which contains all of the calibration coefficients from all of
    the files, and check for differences.
    
    Args: 
        df - A dataframe which contains all fo the calibration coefficients
        from the asset management csv, qct checkout, and the vendor
        files (.cal and .xmlcon)
    Returns:
        cal_errors - A nested dictionary containing the calibration timestamp, the
        relevant calibration coefficient, and which file(s) have the
        erroneous calibration file.
    """
    
    cal_errors = {}
    for date in np.unique(df['Cal Date']):
        df2 = df[df['Cal Date'] == date]
        wrong_cals = {}
        for column in df2.columns.values:
            array = df2[column]
            array.sort_index()
            if array.dtype == 'datetime64[ns]':
                pass
            else:
                error = locate_cal_error(array)
                if error == False:
                    wrong_cals.update({column:array.index[0]})
                elif error == True:
                    pass
                else:
                    wrong_cals.update({column:error})
        
        if len(wrong_cals) < 1:
            cal_errors.update({str(date).split('T')[0]:'No Errors'})
        else:
            cal_errors.update({str(date).split('T')[0]:wrong_cals})
    
    return cal_errors


cal_errors = {}
for uid in uids:
    ce = search_for_errors(comparison[uid])
    cal_errors.update({uid:ce})
    

cal_errors

pd.DataFrame.from_dict(cal_errors)

df2=pd.DataFrame.from_dict({i: cal_errors[i] for i in cal_errors.keys()}, orient='index')

df2

df2.to_csv('CTDBPP_Errors.csv')

# Generate a dataframe of the missing files
df_missing = pd.DataFrame(index=uids)

df_missing['.CAL FILES'] = cal_missing.values()
df_missing

df_missing['.XML FILES'] = xml_missing.values()
df_missing

df_missing['.QCT FILES'] = qct_missing.values()
df_missing

df_missing.to_csv('CTDBPP_Missing_Files.csv')

# ### Check which CTDBP-C Calibration files are not correctly named
# In order to check the calibration values, need to have the correctly named calibration csv files. We can check this by comparison of deployment dates with the CTDBPC calibration dates. This requires loading both the deployment csv and parsing all the file names, flagging the file names THAT MATCH, and then revisiting them in order to correct the name.

# Load the deployment csvs fo
# Parse for all WHOI CG Deployment Sheets based on 'CP' or CG
# Easier to check for non-CG 
deploy_csvs = []
for file in os.listdir('../../GitHub/OOI-Integration/asset-management/deployment/'):
    if file[0:2] == 'RS' or file[0:2] == 'CE':
        pass
    elif 'MOAS' in file:
        pass
    else:
        deploy_csvs.append(file)
        print(file)

# Get the Deployment History from the WHOI Asset Tracking System
CTDBPF_Deploy = CTDBPF['Deployment History']

CTDBPF_Deploy

# Split the string at the newline to generate a list of deployments for each CTDBP-C
CTDBPF_Deploy = CTDBPF['Deployment History'].apply(lambda x: x.split('\n'))

CTDBPF_Deploy

# List out all the individual deployments
deploy_list = []
for i in range(0,len(CTDBPF_Deploy)):
    for item in CTDBPF_Deploy.iloc[i]:
        if '-' in item:
            deploy_list.append(item)
        else:
            pass

deploy_list

# So I now have a list of the deployments all the CTDBP-Cs were used on.
# Now, parse the name of the array to
array = list( set( [x.split('-')[0] for x in deploy_list] ) )
array

# With the list of array names, I can now parse the deployment file names to find
# the relevant deployment sheets which match where the CTDBP-Cs were deployed
deploy_csvs = []
for file in os.listdir('../../GitHub/OOI-Integration/asset-management/deployment/'):
    if file.split('_')[0] in array:
        deploy_csvs.append(file)
deploy_csvs

# Using the identified deployment csvs, can now load the deployment csvs into
# a pandas dataframe
deployments = pd.DataFrame()
for file in deploy_csvs:
    deployments = deployments.append(pd.read_csv('../../GitHub/OOI-Integration/asset-management/deployment/'+file))
deployments.head()

# Get the CTDBPF sensor uids
sensor_uids = list( set( CTDBPF['UID'] ) )
sensor_uids

# Find in the deployment spreadsheets the matching entry for the CTDBP-Cs that I'm looking for
deployments['CTDBPF'] = deployments['sensor.uid'].apply(lambda x: True if x in sensor_uids else False)
deployments = deployments[deployments['CTDBPF'] == True]

deployments.head()

# Now, parse out the date string in the format of YYYYMMDD from the startDateTime
# in order to compare with the date in the calibration file names
deploy_dates = deployments['startDateTime'].apply(lambda x: x.replace('-','').split('T')[0])
deploy_dates = list(set(deploy_dates))
deploy_dates

cal_csvs = []
for file in os.listdir('../../GitHub/OOI-Integration/asset-management/calibration/CTDBPF/'):
    date = file.split('__')[1].split('.')[0]
    print(date)
    if date in deploy_dates:
        cal_csvs = cal_csvs.append(file)
print(cal_csvs)
        

cal_csvs

# Great! None of the CTDBP-C have calibration dates which match deployment dates. That is a good sign - it means that the dates in the calibration file name *should* match the calibration dates in the calibration info.
#
# However, that is no guarantee that the date in the file name matches the date in the calibration data. This can be check in a future step by comparing the calibration date in the vendor docs, QCT info, and the .cal and .xmlcon file info.

# +
# Now, using the "deploy" csvs for each node in the various arrays,
# need to load into a large pandas dataframe for easy handling
import pandas as pd

deployments = pd.DataFrame()
for file in deploy_csvs:
    deployments = deployments.append(pd.read_csv('../GitHub/OOI-Integration/asset-management/deployment/'+file))
# -

deployments

# Get all the unique deployment dates from the deployment csvs and put into the form of 
# YYYYMMDD. 
deploy_dates = deployments['startDateTime'].apply(lambda x: x.split('T')[0].replace('-',''))

deploy_dates = list(set(deploy_dates))
deploy_dates[0:10]

len(deploy_dates)

check_files = []
for root, dirs, files in os.walk('../GitHub/OOI-Integration/asset-management/calibration/'):
    for name in files:
        if 'CGINS' in name:
            cal_date = name.split('__')[1].split('.')[0]
            if cal_date in deploy_dates:
                check_files.append(name)

# +
import datetime
import re
import os
from wcmatch import fnmatch
import xml.etree.ElementTree as et
import pandas as pd
import numpy as np
import string
from zipfile import ZipFile
import csv
import PyPDF2
from nltk.tokenize import word_tokenize
import json

class CTDCalibration():
    # Class that stores calibration values for CTDs.

    def __init__(self, uid):
        self.serial = ''
        self.uid = uid
        self.ctd_type = uid
        self.coefficients = {}
        self.date = {}

        self.coefficient_name_map = {
            'TA0': 'CC_a0',
            'TA1': 'CC_a1',
            'TA2': 'CC_a2',
            'TA3': 'CC_a3',
            'CPCOR': 'CC_cpcor',
            'CTCOR': 'CC_ctcor',
            'CG': 'CC_g',
            'CH': 'CC_h',
            'CI': 'CC_i',
            'CJ': 'CC_j',
            'G': 'CC_g',
            'H': 'CC_h',
            'I': 'CC_i',
            'J': 'CC_j',
            'PA0': 'CC_pa0',
            'PA1': 'CC_pa1',
            'PA2': 'CC_pa2',
            'PTEMPA0': 'CC_ptempa0',
            'PTEMPA1': 'CC_ptempa1',
            'PTEMPA2': 'CC_ptempa2',
            'PTCA0': 'CC_ptca0',
            'PTCA1': 'CC_ptca1',
            'PTCA2': 'CC_ptca2',
            'PTCB0': 'CC_ptcb0',
            'PTCB1': 'CC_ptcb1',
            'PTCB2': 'CC_ptcb2',
            # additional types for series O
            'C1': 'CC_C1',
            'C2': 'CC_C2',
            'C3': 'CC_C3',
            'D1': 'CC_D1',
            'D2': 'CC_D2',
            'T1': 'CC_T1',
            'T2': 'CC_T2',
            'T3': 'CC_T3',
            'T4': 'CC_T4',
            'T5': 'CC_T5',
        }

        # Name mapping for the MO-type CTDs (when reading from pdfs)
        self.mo_coefficient_name_map = {
            'ptcb1': 'CC_ptcb1',
            'pa2': 'CC_pa2',
            'a3': 'CC_a3',
            'pa0': 'CC_pa0',
            'wbotc': 'CC_wbotc',
            'ptcb0': 'CC_ptcb0',
            'g': 'CC_g',
            'ptempa1': 'CC_ptempa1',
            'ptcb2': 'CC_ptcb2',
            'a0': 'CC_a0',
            'h': 'CC_h',
            'ptca0': 'CC_ptca0',
            'a2': 'CC_a2',
            'cpcor': 'CC_cpcor',
            'i': 'CC_i',
            'ptempa0': 'CC_ptempa0',
            'prange': 'CC_p_range',
            'ctcor': 'CC_ctcor',
            'a1': 'CC_a1',
            'j': 'CC_j',
            'ptempa2': 'CC_ptempa2',
            'pa1': 'CC_pa1',
            'ptca1': 'CC_ptca1',
            'ptca2': 'CC_ptca2',
        }

        self.o2_coefficients_map = {
            'A': 'CC_residual_temperature_correction_factor_a',
            'B': 'CC_residual_temperature_correction_factor_b',
            'C': 'CC_residual_temperature_correction_factor_c',
            'E': 'CC_residual_temperature_correction_factor_e',
            'SOC': 'CC_oxygen_signal_slope',
            'OFFSET': 'CC_frequency_offset'
        }

    @property
    def uid(self):
        return self._uid

    @uid.setter
    def uid(self, d):
        r = re.compile('.{5}-.{6}-.{5}')
        if r.match(d) is not None:
            self.serial = d.split('-')[2]
            self._uid = d
        else:
            raise Exception(f"The instrument uid {d} is not a valid uid. Please check.")

    @property
    def ctd_type(self):
        return self._ctd_type

    @ctd_type.setter
    def ctd_type(self, d):
        if 'MO' in d:
            self._ctd_type = '37'
        elif 'BP' in d:
            self._ctd_type = '16'
        else:
            self._ctd_type = ''

    def load_pdf(self, filepath):
        """
        This function opens and loads a pdf into a parseable format.

        Args:
            filepath - full directory path with filename
        Raises:
            IOError - error reading or loading text from the pdf object
        Returns:
            text - a dictionary with page numbers as keys and the pdf text as items
        """

        # Open and read the pdf file
        pdfFileObj = open(filepath, 'rb')
        # Create a reader to be parsed
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        # Now, enumerate through the pdf and decode the text
        num_pages = pdfReader.numPages
        count = 0
        text = {}

        while count < num_pages:
            pageObj = pdfReader.getPage(count)
            count = count + 1
            text.update({count: pageObj.extractText()})

        # Run a check that text was actually extracted
        if len(text) == 0:
            raise(IOError(f'No text was parsed from the pdf file {filepath}'))
        else:
            return text

    def read_pdf(self, filepath):
        """
        Function which parses the opened and loaded pdf file into the
        relevant calibration coefficient data. This function works if
        the calibration pdfs have been split based on sensor as well as
        for combined pdfs.

        Args:
            text - the opened and loaded pdf text returned from load_pdf
        Raises:
            Exception - thrown when a relevant calibration information is
                missing from the text
        Returns:
            date - the calibration dates of the temperature, conductivity,
                and pressure sensors of the CTDMO in a dictionary object
            serial - populated serial number of the CTDMO
            coefficients - populated dictionary of the calibration coefficients
                as keys and associated values as items.
        """
        text = self.load_pdf(filepath)

        for page_num in text.keys():
            # Search for the temperature calibration data
            if 'SBE 37 TEMPERATURE CALIBRATION DATA' in text[page_num]:
                tokens = word_tokenize(text[page_num])
                data = [word.lower() for word in tokens if not word in string.punctuation]
                # Now, find and record the calibration date
                if 'calibration' and 'date' in data:
                    cal_ind = data.index('calibration')
                    date_ind = data.index('date')
                    # Run check they are in order
                    if date_ind == cal_ind+1:
                        date = pd.to_datetime(data[date_ind+1]).strftime('%Y%m%d')
                        self.date.update({'TCAL':date})
                    else:
                        raise Exception(f"Can't locate temp calibration date.")
                else:
                    raise Exception(f"Can't locate temp calibration date.")

                # Check for the serial number
                if 'serial' and 'number' in data and len(self.serial) == 0:
                    ser_ind = data.index('serial')
                    num_ind = data.index('number')
                    if num_ind == ser_ind+1:
                        self.serial = data[num_ind+1]
                    else:
                        pass

                # Now, get the calibration coefficients
                for key in self.mo_coefficient_name_map.keys():
                    if key in data:
                        ind = data.index(key)
                        self.coefficients.update({self.mo_coefficient_name_map[key]: data[ind+1]})
                    else:
                        pass

            # Search for the conductivity calibration data
            elif 'SBE 37 CONDUCTIVITY CALIBRATION DATA' in text[page_num]:
                # tokenize the text data and extract only key words
                tokens = word_tokenize(text[page_num])
                data = [word.lower() for word in tokens if not word in string.punctuation]

                # Now, find and record the calibration date
                if 'calibration' and 'date' in data:
                    cal_ind = data.index('calibration')
                    date_ind = data.index('date')
                    # Run check they are in order
                    if date_ind == cal_ind+1:
                        date = pd.to_datetime(data[date_ind+1]).strftime('%Y%m%d')
                        self.date.update({'CCAL': date})
                    else:
                        raise Exception(f"Can't locate conductivity calibration date.")
                else:
                    raise Exception(f"Can't locate conductivity calibration date.")

                # Check for the serial number
                if 'serial' and 'number' in data and len(self.serial) == 0:
                    ser_ind = data.index('serial')
                    num_ind = data.index('number')
                    if num_ind == ser_ind+1:
                        self.serial = data[num_ind+1]
                    else:
                        pass

                # Now, get the calibration coefficients
                for key in self.mo_coefficient_name_map.keys():
                    if key in data:
                        ind = data.index(key)
                        self.coefficients.update({self.mo_coefficient_name_map[key]: data[ind+1]})
                    else:
                        pass

            elif 'SBE 37 PRESSURE CALIBRATION DATA' in text[page_num]:
                # tokenize the text data and extract only key words
                tokens = word_tokenize(text[page_num])
                data = [word.lower() for word in tokens if not word in string.punctuation]

                # Now, find and record the calibration date
                if 'calibration' and 'date' in data:
                    cal_ind = data.index('calibration')
                    date_ind = data.index('date')
                    # Run check they are in order
                    if date_ind == cal_ind+1:
                        date = pd.to_datetime(data[date_ind+1]).strftime('%Y%m%d')
                        self.date.update({'PCAL': date})
                    else:
                        raise Exception(f"Can't locate pressure calibration date.")
                else:
                    raise Exception(f"Can't locate pressure calibration date.")

                # Check for the serial number
                if 'serial' and 'number' in data and len(self.serial) == 0:
                    ser_ind = data.index('serial')
                    num_ind = data.index('number')
                    if num_ind == ser_ind+1:
                        self.serial = data[num_ind+1]
                    else:
                        pass

                # Now, get the calibration coefficients
                for key in self.mo_coefficient_name_map.keys():
                    if key in data:
                        ind = data.index(key)
                        self.coefficients.update({self.mo_coefficient_name_map[key]: data[ind+1]})
                    else:
                        pass

            # Now check for other important information
            else:
                tokens = word_tokenize(text[page_num])
                data = [word.lower() for word in tokens if not word in string.punctuation]

                # Now, find the sensor rating
                if 'sensor' and 'rating' in data:
                    ind = data.index('rating')
                    self.coefficients.update({self.mo_coefficient_name_map['prange']: data[ind+1]})

    def read_cal(self, data):
        """
        Function which reads and parses the CTDBP calibration values stored
        in a .cal file.

        Args:
            filename - the name of the calibration (.cal) file to load. If the
                cal file is not located in the same directory as this script, the
                full filepath also needs to be specified.
        Returns:
            self.coefficients - populated coefficients dictionary
            self.date - the calibration dates associated with the calibration values
            self.type - the type (i.e. 16+/37-IM) of the CTD
            self.serial - populates the 5-digit serial number of the instrument
        """

        for line in data.splitlines():
            key, value = line.replace(" ", "").split('=')

            if key == 'INSTRUMENT_TYPE':
                if value == 'SEACATPLUS':
                    ctd_type = '16'
                elif value == '37SBE':
                    ctd_type = '37'
                else:
                    ctd_type = ''
                if self.ctd_type != ctd_type:
                    raise ValueError(f'CTD type in cal file {ctd_type} does not match the UID type {self.ctd_type}')

            elif key == 'SERIALNO':
                if self.serial != value.zfill(5):
                    raise Exception(f'Serial number {value.zfill(5)} stored in cal file does not match {self.serial} from the UID.')

            elif 'CALDATE' in key:
                self.date.update({key: datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})

            else:
                if self.ctd_type == '16':
                    name = self.coefficient_name_map.get(key)
                elif self.ctd_type == '37':
                    name = self.mo_coefficient_name_map.get(key)
                else:
                    pass

                if not name or name is None:
                    continue
                else:
                    self.coefficients.update({name: value})

    def load_cal(self, filepath):
        """
        Loads all of the calibration coefficients from the vendor cal files for
        a given CTD instrument class.

        Args:
            filepath - directory path to where the zipfiles are stored locally
        Raises:
            FileExistsError - Checks the given filepath that a .cal file exists
        Returns:
            self.coefficients - populated coefficients dictionary
            self.date - the calibration dates associated with the calibration values
            self.type - the type (i.e. 16+/37-IM) of the CTD
            self.serial - populates the 5-digit serial number of the instrument
        """

        if filepath.endswith('.zip'):
            with ZipFile(filepath) as zfile:
                filename = [name for name in zfile.namelist() if '.cal' in name]
                if len(filename) > 0:
                    data = zfile.read(filename[0]).decode('ASCII')
                    self.read_cal(data)
                else:
                    FileExistsError(f"No .cal file found in {filepath}.")

        elif filepath.endswith('.cal'):
            with open(filepath) as filename:
                data = filename.read()
                self.read_cal(data)

        else:
            FileExistsError(f"No .cal file found in {filepath}.")

    def read_xml(self, data):
        """
        Function which reads and parses the CTDBP calibration values stored
        in the xmlcon file.

        Args:
            data - the data string to parse
        Returns:
            self.coefficients - populated coefficients dictionary
            self.date - the calibration dates associated with the calibration values
            self.type - the type (i.e. 16+/37-IM) of the CTD
            self.serial - populates the 5-digit serial number of the instrument
        """

        Tflag  = False
        Cflag  = False
        O2flag = False

        for child in data.iter():
            key = child.tag.upper()
            value = child.text.upper()

            if key == 'NAME':
                if '16PLUS' in value:
                    ctd_type = '16'
                    if self.ctd_type != ctd_type:
                        raise ValueError(f'CTD type in xmlcon file {ctd_type} does not match the UID type {self.ctd_type}')

            # Check if we are processing the calibration values for the temperature sensor
            # If we already have parsed the Temp data, need to turn the flag off
            if key == 'TEMPERATURESENSOR':
                Tflag = True
            elif 'SENSOR' in key and Tflag == True:
                Tflag = False
            else:
                pass

            # Check on if we are now parsing the conductivity data
            if key == 'CONDUCTIVITYSENSOR':
                Cflag = True
            elif 'SENSOR' in key and Cflag == True:
                Cflag = False
            else:
                pass

            # Check if an oxygen sensor has been appended to the CTD configuration
            if key == 'OXYGENSENSOR':
                O2flag = True

            # Check that the serial number in the xmlcon file matches the serial
            # number from the UID
            if key == 'SERIALNUMBER':
                if self.serial != value.zfill(5):
                    raise Exception(f'Serial number {value.zfill(5)} stored in xmlcon file does not match {self.serial} from the UID.')

            # Parse the calibration dates of the different sensors
            if key == 'CALIBRATIONDATE':
                if Tflag:
                    self.date.update({'TCALDATE':datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})
                elif Cflag:
                    self.date.update({'CCALDATE': datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})
                else:
                    self.date.update({'PCALDATE': datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})

            # Now, we get to parse the actual calibration values, but it is necessary to make sure the
            # key names are correct
            if Tflag:
                key = 'T'+key

            name = self.coefficient_name_map.get(key)
            if not name or name is None:
                if O2flag:
                    name = self.o2_coefficients_map.get(key)
                    self.coefficients.update({name: value})
                else:
                    pass
            else:
                self.coefficients.update({name: value})

    def load_xml(self, filepath):
        """
        Loads all of the calibration coefficients from the vendor xmlcon files for
        a given CTD instrument class.

        Args:
            filepath - the name of the xmlcon file to load and parse. If the
                xmlcon file is not located in the same directory as this script,
                the full filepath also needs to be specified. May point to a zipfile.
        Raises:
            FileExistsError - Checks the given filepath that an xmlcon file exists
        Returns:
            self.coefficients - populated coefficients dictionary
            self.date - the calibration dates associated with the calibration values
            self.type - the type (i.e. 16+/37-IM) of the CTD
            self.serial - populates the 5-digit serial number of the instrument
        """

        if filepath.endswith('.zip'):
            with ZipFile(filepath) as zfile:
                filename = [name for name in zfile.namelist() if '.xmlcon' in name]
                if len(filename) > 0:
                    data = et.parse(zfile.open(filename[0]))
                    self.read_xml(data)
                else:
                    FileExistsError(f"No .cal file found in {filepath}.")

        elif filepath.endswith('.xmlcon'):
            with open(filepath) as file:
                data = et.parse(file)
                self.read_xml(data)

        else:
            FileExistsError(f"No .cal file found in {filepath}.")

    def load_qct(self, filepath):
        """
        Function which parses the output from the QCT check-in and loads them into
        the CTD object.

        Args:
            filepath - the full directory path and filename
        Raises:
            ValueError - checks if the serial number parsed from the UID matches the
                the serial number stored in the file.
        Returns:
            self.coefficients - populated coefficients dictionary
            self.date - the calibration dates associated with the calibration values
            self.type - the type (i.e. 16+/37-IM) of the CTD
            self.serial - populates the 5-digit serial number of the instrument
        """

        with open(filepath,errors='ignore') as filename:
            data = filename.read()

        if self.ctd_type == '37':
            data = data.replace('<', ' ').replace('>', ' ')
            for line in data.splitlines():
                keys = list(self.mo_coefficient_name_map.keys())
                if any([word for word in line.split() if word.lower() in keys]):
                    name = self.mo_coefficient_name_map.get(line.split()[0])
                    value = line.split()[-1]
                    self.coefficients.update({name: value})

        elif self.ctd_type == '16':
            if 'CTDBPP' in self.uid:
                # Okay, the CTDBPP (inductive mooring ones) have to be handled differently
                data = data.replace('<', ' ').replace('>', ' ')
                for line in data.splitlines():
                    keys = list(self.coefficient_name_map.keys())
                    
                    if any([word for word in line.split() if word in keys]):
                        name = self.coefficient_name_map.get(line.split()[0])
                        value = line.split()[1]
                        self.coefficients.update({name: value})
                        
                    elif 'SERIAL NO.' in line:
                        ind = line.split().index('NO.')
                        serial_num = line.split()[ind+1]
                        serial_num = serial_num.zfill(5)
                        if not self.serial == serial_num:
                            raise ValueError(f'UID serial number {self.serial} does not match the QCT serial num {serial_num}')
                    
                    elif 'CalDate' in line:
                        words = line.split()
                        self.date.update({str(len(self.date)): pd.to_datetime(words[1]).strftime('%Y%m%d')})
                    
                    else:
                        pass
            
            else:
                for line in data.splitlines():
                    keys = list(self.coefficient_name_map.keys())
                    if any([word for word in line.split() if word in keys]):
                        name = self.coefficient_name_map.get(line.split()[0])
                        value = line.split()[-1]
                        self.coefficients.update({name: value})

                    if 'temperature:' in line:
                        self.date.update({'TCAL': pd.to_datetime(line.split()[-1]).strftime('%Y%m%d')})
                    elif 'conductivity:' in line:
                        self.date.update({'CCAL': pd.to_datetime(line.split()[-1]).strftime('%Y%m%d')})
                    elif 'pressure S/N' in line:
                        self.date.update({'PCAL': pd.to_datetime(line.split()[-1]).strftime('%Y%m%d')})
                    else:
                        pass

                    if 'SERIAL NO.' in line:
                        ind = line.split().index('NO.')
                        serial_num = line.split()[ind+1]
                        serial_num = serial_num.zfill(5)
                        if not self.serial == serial_num:
                            raise ValueError(f'UID serial number {self.serial} does not match the QCT serial num {serial_num}')

                if 'SBE 16Plus' in line:
                    if self.ctd_type is not '16':
                        raise TypeError(f'CTD type {self.ctd_type} does not match the qct.')

        else:
            pass

    def write_csv(self, outpath):
        """
        This function writes the correctly named csv file for the ctd to the
        specified directory.

        Args:
            outpath - directory path of where to write the csv file
        Raises:
            ValueError - raised if the CTD object's coefficient dictionary
                has not been populated
        Returns:
            self.to_csv - a csv of the calibration coefficients which is
                written to the specified directory from the outpath.
        """

        # Run a check that the coefficients have actually been loaded
        if len(self.coefficients) == 0:
            raise ValueError('No calibration coefficients have been loaded.')

        # Create a dataframe to write to the csv
        data = {'serial': [self.ctd_type + '-' + self.serial]*len(self.coefficients),
                'name': list(self.coefficients.keys()),
                'value': list(self.coefficients.values()),
                'notes': ['']*len(self.coefficients)
                }
        df = pd.DataFrame().from_dict(data)

        # Generate the csv name
        cal_date = max(self.date.values())
        csv_name = self.uid + '__' + cal_date + '.csv'

        # Write the dataframe to a csv file
        # check = input(f"Write {csv_name} to {outpath}? [y/n]: ")
        check = 'y'
        if check.lower().strip() == 'y':
            df.to_csv(outpath+'/'+csv_name, index=False)
# -


