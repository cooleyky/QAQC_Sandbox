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

# # PRESF Metadata Review
#
# This notebook describes the process for reviewing the calibration coefficients for the PRESF SBE 26plus. The purpose is to check the calibration coefficients contained in the CSVs stored within the asset management repository on GitHub, which are the coefficients utilized by OOI-net for calculating data products, against the different available sources of calibration information to identify when errors were made during entering the calibration csvs. This includes checking the following information:
# 1. The calibration date - this information is stored in the filename of the csv
# 2. Calibration source - identifying all the possible sources of calibration information, and determine which file should supply the calibration info
# 3. Calibration coeffs - checking the accuracy and precision of the numbers stored in the calibration coefficients
#
# The PRESF contains 18 different calibration coefficients to check, two of which are fixed constants. The possible calibration sources for the PRESF are vendor PDFs and QCT check-ins. However, calibrations from the vendor PDFs are split across multiple documents and many are missing either coefficients or PDFs. Consequently, we utilize the QCT check-in as the source of calibration coefficients. The relevant file stored within the QCTs are .hex files.
#
# **========================================================================================================================**

from utils import *

import os, re, sys
import shutil
import pandas as pd
import numpy as np
from zipfile import ZipFile

# ========================================================================================================================
# ### Directories
# **Define the main directories where important information is stored.**

qct_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/PRESF/PRESF_Results'
cal_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/PRESF/PRESF_Cal'
asset_management_directory = '/home/andrew/Documents/OOI-CGSN/ooi-integration/asset-management/calibration/PRESFC'

excel_spreadsheet = '/media/andrew/OS/Users/areed/Documents/Project_Files/Documentation/System/System Notebook/WHOI_Asset_Tracking.xlsx'
sheet_name = 'Sensors'

PRESF = whoi_asset_tracking(excel_spreadsheet,sheet_name,instrument_class='PRESF',whoi=True)
PRESF

# **Identify the QCT Testing documents associated with each individual instrument (the UID)**

qct_dict = get_qct_files(PRESF, qct_directory)
qct_dict

# **Identify the calibration csvs stored in asset management which correspond to a particular instrument.**

csv_dict = load_asset_management(PRESF, asset_management_directory)
csv_dict

uids = sorted(list(csv_dict.keys()))

serial_nums = get_serial_nums(PRESF, uids)
serial_nums

cal_dict = get_calibration_files(serial_nums, cal_directory)
cal_dict

# ========================================================================================================================
# **Now, need to get all the files for a particular CTDMO UID:**

uid = sorted(uids)[2]
uid

cal_files = sorted(cal_dict[uid])
for file in cal_files:
    print(file)

csv_files = sorted(csv_dict[uid])
for file in csv_files:
    print(file)

qct_files = sorted(qct_dict[uid])
for file in qct_files:
    print(file)

csv_path = []
for cf in csv_files:
    path = generate_file_path(asset_management_directory, cf)
    csv_path.append(path)
csv_path

cal_path = []
for cf in cal_files:
    path = generate_file_path(cal_directory, cf)
    cal_path.append(path)
cal_path

qct_path = []
for qf in qct_files:
    path = generate_file_path(qct_directory, qf, ext=['.log','.txt','.zip'])
    qct_path.append(path)
qct_path


# ========================================================================================================================
# ### Now develop code to load the calibration coeffs from the capture files
# The **PRESFCalibration** object below is an object designed to load, parse, and write the respective PRESF calibration csvs. The calibration coefficients are stored in the object as attributes.

class PRESFCalibration():
    # Class that stores calibration values for CTDs.

    def __init__(self, uid):
        self.serial = ''
        self.uid = uid
        self.coefficients = {
            'CC_offset_correction_factor':'0',
            'CC_slope_correction_factor':'1',
        }
        self.date = {}
        self.notes = {}

        # Name mapping for the MO-type CTDs (when reading from pdfs)
        self.coefficient_name_map = {
            'U0':'CC_u0',
            'Y1':'CC_y1',
            'Y2':'CC_y2',
            'Y3':'CC_y3',
            'C1':'CC_c1',
            'C2':'CC_c2',
            'C3':'CC_c3',
            'D1':'CC_d1',
            'D2':'CC_d2',
            'T1':'CC_t1',
            'T2':'CC_t2',
            'T3':'CC_t3',
            'T4':'CC_t4',
            'M':'CC_m',
            'B':'CC_b',
            'OFFSET':'CC_pressure_offset_calibration_coefficient'
        }

    @property
    def uid(self):
        return self._uid

    @uid.setter
    def uid(self, d):
        r = re.compile('.{5}-.{6}-.{5}')
        if r.match(d) is not None:
            self.serial = '26-' + d.split('-')[2].lstrip('0')
            self._uid = d
        else:
            raise Exception(f"The instrument uid {d} is not a valid uid. Please check.")
            

    def parse_qct(self, filepath):
        """
        Parses the QCT data in ascii-format.
        
        Args:
            filepath - the full directory to either the parent 
                directory or the full path with filename of the
                QCT file to parse
        Returns:
            self.coefficients - a dictionary which contains the
                calibration coefficients names as key with associated
                values as the key-entries
        """
        
        data = self.open_qct(filepath)
        Calflag = False
        for line in data.splitlines():
    
            line = line.replace('*','').strip()
    
            if 'Pressure coefficients' in line:
                _, cal_date = line.split(':')
                cal_date = pd.to_datetime(cal_date.strip()).strftime('%Y%m%d')
                self.date = cal_date
                # Turn on the flag
                Calflag = True
                # And move on to the next line
                continue
            elif 'Temperature coefficients' in line:
                # Turn the flag off
                Calflag = False
            else:
                pass
        
            if Calflag:
                key,_,value = line.split()
                name = self.coefficient_name_map.get(key)
                self.coefficients.update({name:value})
            
            
    def open_qct(self, filepath):
        """
        Function which opens and reads in the QCT data into a 
        format which is parseable.
        
        Args:
            filepath - the full directory to either the parent 
                directory or the full path with filename of the
                QCT file to parse
        Returns:
            data - the data in ascii-format from the QCT file
        """
        
        if filepath.endswith('.zip'):
            with ZipFile(filepath) as zfile:
                for name in zfile.namelist():
                    if fnmatch.fnmatch(name,'*.hex'):
                        fname = name
                data = zfile.read(fname).decode('ascii')

        elif os.path.isdir(filepath):
            for file in os.listdir(filepath):
                if fnmatch.fnmatch(name,'*.hex'):
                    fname = file
            with open(fname) as file:
                data = file.read().decode('ascii')
                
        else:
            with open(filepath) as file:
                data = file.read().decode('ascii')
        
        return data


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
        data = {
            'serial': [self.serial]*len(self.coefficients),
            'name': list(self.coefficients.keys()),
            'value': list(self.coefficients.values())
        }
        df = pd.DataFrame().from_dict(data)

        # Define a function to reformat the notes into an uniform system
        def reformat_notes(x):
            # First, get rid of 
            try:
                np.isnan(x)
                x = ''
            except:
                x = str(x).replace('[','').replace(']','')
            return x
        
        # Now merge the coefficients dataframe with the notes
        if len(self.notes) > 0:
            notes = pd.DataFrame().from_dict({
                'name':list(self.notes.keys()),
                'notes':list(self.notes.values())
            })
            df = df.merge(notes, how='outer', left_on='name', right_on='name')
        else:
            df['notes'] = ''
        
        # Sort the data by the coefficient name
        df = df.sort_values(by='name')

        # Generate the csv name
        cal_date = self.date
        csv_name = self.uid + '__' + cal_date + '.csv'

        # Write the dataframe to a csv file
        # check = input(f"Write {csv_name} to {outpath}? [y/n]: ")
        check = 'y'
        if check.lower().strip() == 'y':
            df.to_csv(outpath+'/'+csv_name, index=False)

# **Initialize the PRESFCalibration object using the instrument uid.**

presf = PRESFCalibration(uid)

# **Check that the serial number has been correctly parsed.**

presf.serial

# **Load the PRESF calibration coefficients based on the QCT file.**

presf.parse_qct(qct_path[2])

# **Check that the calibration coefficients loaded successfully.**

presf.coefficients

qct_dict[uid]

# **Now, if you want to add any notes to the calibration csv, they can be added using a dictionary to the notes attribute, based on the calibration coefficient name by writing.**

presf.notes = {
    'CC_b': 'Source file is QCT document number 3305-00105-00043.',
    'CC_m': 'I think that this is a constant value.'
}

# **For right now, write the file to a temporary local directory.**

temp_directory = '/'.join((os.getcwd(),'temp'))
temp_directory
shutil.rmtree(temp_directory)

temp_path = '/'.join((temp_directory,'qct'))
ensure_dir(temp_path)

# **Write the PRESF calibration object using the standardized naming format to the temporary directory in a format that can be ingested by UFrame.**

presf.write_csv(temp_path)

# **Check that it wrote.**

os.listdir(temp_path)

# ========================================================================================================================
# ## Metadata Comparison
# Now the goal is to compare the calibration csvs contained in asset management against the calibration coefficients stored in the QCT files.

# **First, need to copy the calibration csvs from asset management to the local temp directory.**

shutil.rmtree('/'.join((os.getcwd(),'temp')))

for file in csv_path:
    savedir = '/'.join((os.getcwd(),'temp','csv'))
    ensure_dir(savedir)
    shutil.copy(file, savedir)

os.listdir(savedir)

# **Next, write all the QCT files to the temp directory in the appropriate csv format. This will print out any QCT files which don't parse.**

ensure_dir(temp_path)
for qct in qct_path:
    try:
        presf = PRESFCalibration(uid=uid)
        presf.parse_qct(qct)
        presf.write_csv(temp_path)
    except:
        print(qct)

os.listdir(temp_path)


# ========================================================================================================================
# ### Compare results
# Now, with QCT files parsed into csvs which follow the UFrame format, I can load both the QCT and the calibration csvs into pandas dataframes, which will allow element by element comparison in relatively few lines of code.

def get_file_date(x):
    x = str(x)
    ind1 = x.index('__')
    ind2 = x.index('.')
    return x[ind1+2:ind2]


# **Load the calibration csvs:**

# Now we want to compare dataframe
csv_files = pd.DataFrame(sorted(csv_dict[uid]),columns=['csv'])
csv_files['cal date'] = csv_files['csv'].apply(lambda x: get_file_date(x))
csv_files.set_index('cal date',inplace=True)
csv_files

# Now we want to compare dataframe
qct_files = pd.DataFrame(sorted(os.listdir('temp/qct')),columns=['qct'])
qct_files['cal date'] = qct_files['qct'].apply(lambda x: get_file_date(x))
qct_files.set_index('cal date',inplace=True)
qct_files

df_files = csv_files.join(qct_files,how='outer').fillna(value='-999')
df_files

# **The above dataframe shows the names of the csv files both pulled from asset management (csv) and from the qct. When they don't match based on the calibration date (cal date), that suggests that the date in the csv filename is likely incorrect.**

# **If the filename is wrong, the calibration coefficient checker will not manage to compare the results. Consequently, we'll make a local copy of the wrong file to a new file with the correct name, and then run the calibration coefficient checker. Do this for all the incorrectly named files.**

a = 'temp/csv/' + 'CGINS-PRESFC-01401__20171217.csv'
b = 'temp/csv/' + 'CGINS-PRESFC-01401__20171212.csv'
shutil.copy(a,b)

# !rm 'temp/csv/CGINS-PRESFC-01401__20171217.csv'

csv_files = pd.DataFrame(sorted(os.listdir('temp/csv')),columns=['csv'])
csv_files['cal date'] = csv_files['csv'].apply(lambda x: get_file_date(x))
csv_files.set_index('cal date',inplace=True)
csv_files


# **Define a function to check the calibration coefficients between the asset management csv and the csv generated from the QCT file. This function checks based on the relative difference, which is set to 0.001% threshold.**

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
            for i in list(set(df_coeffs['serial'])):
                print(source + '-' + fname + ': ' + str(i))
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


