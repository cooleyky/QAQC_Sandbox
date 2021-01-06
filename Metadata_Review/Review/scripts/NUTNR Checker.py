# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # NUTNRB Checker
#
# This notebook is designed to check the NUTNR csv calibration file in pull request. The process I follow is:
# 1. Read in the NUTNR csv from the pull request into a pandas dataframe
# 2. Identify the source file of the calibration coefficients
# 3. Parse the calibration coefficients directly from the source file
# 4. Compare the NUTNR csv from the pull request with the csv parsed from the source file
#
# **====================================================================================================================**
#
# The first step is to load relevant packages:

import csv
import re
import os
import shutil
import numpy as np
import pandas as pd

from utils import *

from zipfile import ZipFile
import string

sys.path.append('/home/andrew/Documents/OOI-CGSN/QAQC_Sandbox/Calibration/Parsers/')

from Parsers.NUTNRCalibration import NUTNRCalibration

# **====================================================================================================================**
# Define the directories where the **csv** file to check is stored, and where the **source** file is stored. Make sure to check the following information on your machine via your terminal first:
# 1. The branch of your local asset-management repository matches the location of the OPTAA cals.
# 2. Your local asset-management repository has the requisite **csv** file to check
# 3. You have downloaded the **source** of the csv file

doc_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/NUTNR/NUTNR_Results/'
cal_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/NUTNR/NUTNR_Cal/'
asset_management_directory = '/home/andrew/Documents/OOI-CGSN/ooi-integration/asset-management/calibration/NUTNRB/'
glider_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Platform_Records/Gliders/Instruments/NUTNR-M/'

# **====================================================================================================================**
# ### Find & Parse the source file
# Now, we want to find the source file of the calibration coefficients, parse the data using the optaa parser, and read the data into a pandas dataframe. The key pieces of data needed for the parser are:
# 1. Instrument UID: This is needed to initialize the OPTAA parser
# 2. Source file: This is the full path to the source file. Zip files are acceptable input.

# If the predeployment file is not listed in asset tracking, need to hunt through all the predeployment files for the possible candidates:

sn = '1089'

cal_file = '3305-00527-00064-A_SN_NTR-1088_Recovery_NUTNR-B.zip'

# Initialize the parser:

for file in os.listdir(cal_directory):
    if cal_file in file:
        print(cal_file)

for file in os.listdir(doc_directory):
    if '3305-00527-00035-A' in file:
        print(file)
        zfile = file

#filepath = cal_directory + '/' + cal_file
#filepath = doc_directory +'/' + zfile
filepath = '/home/andrew/Downloads/'+'3305-00527-00065-A.zip'
#filepath = '/home/andrew/Documents/OOI-CGSN/QAQC_Sandbox/Metadata_Review/Review/temp/'+'3305-00527-00062-A'+'.zip'

sn='1103'

nutnr = NUTNRCalibration('CGINS-NUTNRB-'+sn.zfill(5))

# Read in the calibration coefficients:

nutnr.load_cal(filepath)

nutnr.coefficients

# Write the csv to a temporary local folder:

temp_directory = '/'.join((os.getcwd(),'temp'))
# Check if the temp directory exists; if it already does, purge and rewrite
if os.path.exists(temp_directory):
    shutil.rmtree(temp_directory)
    ensure_dir(temp_directory)
else:
    ensure_dir(temp_directory)

nutnr.write_csv(temp_directory)

(nutnr.uid, nutnr.serial, nutnr.date)

nutnr.source


# **====================================================================================================================**
# ### Check the data
# Now, we have generated local csv and ext files from the data. We can now reload that data into python as a pandas dataframe, which will allow for a direct comparison with the existing data. 

def reformat_arrays(array):
    # First, need to strip extraneous characters from the array
    array = array.replace("'","").replace('[','').replace(']','')
    # Next, split the array into a list
    array = array.split(',')
    # Now, need to eliminate any white space surrounding the individual coeffs
    array = [num.strip() for num in array]
    # Next, float the nums
    array = [float(num) for num in array]
    # Check if the array is len == 1; if so, can just return the number
    if len(array) == 1:
        array = array[0]
    # Now we are done
    return array


#sn = nutnr.serial.zfill(5)
dt = max(nutnr.date)
dt

sn.zfill(5)

source_path = '/home/andrew/Documents/OOI-CGSN/QAQC_Sandbox/Metadata_Review/Review/temp/'
source_csv = pd.read_csv(source_path+'CGINS-NUTNRB-'+sn.zfill(5)+'__'+dt+'.csv')
source_csv['value'] = source_csv['value'].apply(lambda x: reformat_arrays(x))
#source_csv['serial'] = 1029
source_csv

path = '/home/andrew/Documents/OOI-CGSN/asset-management/calibration/NUTNRB/CGINS-NUTNRB-'+sn.zfill(5)+'__'+dt+'.csv'
path

am_csv = pd.read_csv(path)
am_csv['value'] = am_csv['value'].apply(lambda x: reformat_arrays(x))
am_csv

am_csv['notes'].iloc[0]

source_csv['notes'].iloc[0]

source_csv == am_csv



result = {}
for k,val in enumerate(am_csv['value'].iloc[1]):
    check = source_csv['value'].iloc[1][k] == val
    if not check:
        result.update({k:val})
result

for key in result:
    print(am_csv['value'].iloc[1][key], source_csv['value'].iloc[1][key])

result = {}
for k,val in enumerate(am_csv['value'].iloc[3]):
    check = source_csv['value'].iloc[3][k] == val
    if not check:
        result.update({k:val})
result

source_csv['value'].iloc[0] - am_csv['value'].iloc[0]

stuff



# +
import re
import pandas as pd
import numpy as np
from zipfile import ZipFile

class NUTNRCalibration():
    # Class that stores calibration values for CTDs.

    def __init__(self, uid):
        self.serial = None
        self.uid = uid
        self.coefficients = {
            'CC_cal_temp':[],
            'CC_di':[],
            'CC_eno3':[],
            'CC_eswa':[],
            'CC_lower_wavelength_limit_for_spectra_fit':'217',
            'CC_upper_wavelength_limit_for_spectra_fit':'240',
            'CC_wl':[]
        }
        self.date = []
        self.notes = {
            'CC_cal_temp':'',
            'CC_di':'',
            'CC_eno3':'',
            'CC_eswa':'',
            'CC_lower_wavelength_limit_for_spectra_fit':'217',
            'CC_upper_wavelength_limit_for_spectra_fit':'240',
            'CC_wl':''
        }

    @property
    def uid(self):
        return self._uid

    @uid.setter
    def uid(self, d):
        r = re.compile('.{5}-.{6}-.{5}')
        if r.match(d) is not None:
            self._uid = d
        else:
            raise Exception(f"The instrument uid {d} is not a valid uid. Please check.")
            
    def load_cal(self, filepath):
        """
        Wrapper function to load all of the calibration coefficients
        
        Args:
            filepath - path to the directory with filename which has the
                calibration coefficients to be parsed and loaded
        Calls:
            open_cal
            parse_cal
        """
        
        data = self.open_cal(filepath)
        
        self.parse_cal(data)
    
    
    def open_cal(self, filepath):
        """
        Function that opens and reads in cal file
        information for a NUTNR. Zipfiles are acceptable inputs.
        """
        
        if filepath.endswith('.zip'):
            with ZipFile(filepath) as zfile:
                # Check if ISUS or SUNA to get the appropriate name
                filename = [name for name in zfile.namelist()
                            if name.lower().endswith('.cal') and 'z' not in name.lower()]
                
                # Get and open the latest calibration file
                if len(filename) == 1:
                    data = zfile.read(filename[0]).decode('ascii')
                    self.source_file(filepath, filename[0])
                    
                elif len(filename) > 1:
                    filename = [max(filename)]
                    data = zfile.read(filename[0]).decode('ascii')
                    self.source_file(filepath, filename[0])

                else:
                    FileExistsError(f"No .cal file found in {filepath}")
                        
        elif filepath.lower().endswith('.cal'):
            if 'z' not in filepath.lower().split('/')[-1]:
                with open(filepath) as file:
                    data = file.read()
                self.source_file(filepath, file)
        else:
            pass
        
        return data
            
        
    def source_file(self, filepath, filename):
        """
        Routine which parses out the source file and filename
        where the calibration coefficients are sourced from.
        """
        
        if filepath.lower().endswith('.cal'):
            dcn = filepath.split('/')[-2]
            filename = filepath.split('/')[-1]
        else:
            dcn = filepath.split('/')[-1]
        
        self.source = f'Source file: {dcn} > {filename}'
        
    
    def parse_cal(self, data):
        
        for k,line in enumerate(data.splitlines()):
            
            if line.startswith('H'):
                _, info, *ignore = line.split(',')
                
                # The first line of the cal file contains the serial number
                if k == 0:
                    _, sn, *ignore = info.split()
                    if 'SUNA' in info:
                        self.serial = 'NTR-' + sn
                    else:
                        self.serial = sn
                    
                
                # File creation time is when the instrument was calibrated.
                # May be multiple times for different cal coeffs
                if 'file creation time' in info.lower():
                    _, _, _, date, time = info.split()
                    date_time = pd.to_datetime(date + ' ' + time).strftime('%Y%m%d')
                    self.date.append(date_time)
                    
                # The temperature at which it was calibrated
                if 't_cal_swa' in info.lower() or 't_cal' in info.lower():
                    _, cal_temp = info.split()
                    self.coefficients['CC_cal_temp'] = cal_temp
                    
            # Now parse the calibration coefficients
            if line.startswith('E'):
                _, wl, eno3, eswa, _, di = line.split(',')
                
                self.coefficients['CC_wl'].append(float(wl))
                self.coefficients['CC_di'].append(float(di))
                self.coefficients['CC_eno3'].append(float(eno3))
                self.coefficients['CC_eswa'].append(float(eswa))
                
                
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
        if len(self.coefficients.values()) <= 2:
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
            
        # Add in the source file
        df['notes'].iloc[0] = df['notes'].iloc[0] + ' ' + self.source
        
        # Sort the data by the coefficient name
        df = df.sort_values(by='name')

        # Generate the csv name
        cal_date = max(self.date)
        csv_name = self.uid + '__' + cal_date + '.csv'

        # Write the dataframe to a csv file
        check = input(f"Write {csv_name} to {outpath}? [y/n]: ")
        # check = 'y'
        if check.lower().strip() == 'y':
            df.to_csv(outpath+'/'+csv_name, index=False)
# -


