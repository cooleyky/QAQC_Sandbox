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

# # OPTAA Checker
#
# This notebook is designed to check the SPKIR csv calibration file in pull request. The process I follow is:
# 1. Read in the OPTAA csv from the pull request into a pandas dataframe
# 2. Identify the source file of the calibration coefficients
# 3. Parse the calibration coefficients directly from the source file
# 4. Compare the OPTAA csv from the pull request with the csv parsed from the source file
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

from Parsers.OPTAACalibration import OPTAACalibration

# **====================================================================================================================**
# Define the directories where the **csv** file to check is stored, and where the **source** file is stored. Make sure to check the following information on your machine via your terminal first:
# 1. The branch of your local asset-management repository matches the location of the OPTAA cals.
# 2. Your local asset-management repository has the requisite **csv** file to check
# 3. You have downloaded the **source** of the csv file

csv_dir = '/home/andrew/Documents/OOI-CGSN/asset-management/'
#source_dir = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/OPTAA/OPTAA_Cal/'
source_dir = '/home/andrew/Downloads/'

# **====================================================================================================================**
# ### Find & Parse the source file
# Now, we want to find the source file of the calibration coefficients, parse the data using the optaa parser, and read the data into a pandas dataframe. The key pieces of data needed for the parser are:
# 1. Instrument UID: This is needed to initialize the OPTAA parser
# 2. Source file: This is the full path to the source file. Zip files are acceptable input.

source_name = '257'
for file in os.listdir(source_dir):
    if source_name in file:
        source_file = file
        print(source_file)

source_file = 'OPTAA-D_AC-S_SN_257_Calibration_Files_2019-10-29.zip'

# Initialize the parser:

optaa = OPTAACalibration('CGINS-OPTAAD-00257')

# Read in the calibration coefficients:

optaa.load_cal(source_dir+source_file)

optaa.coefficients['CC_tcal']

# Write the csv to a temporary local folder:

temp_directory = '/'.join((os.getcwd(),'temp'))
# Check if the temp directory exists; if it already does, purge and rewrite
if os.path.exists(temp_directory):
    shutil.rmtree(temp_directory)
    ensure_dir(temp_directory)
else:
    ensure_dir(temp_directory)

optaa.write_csv(temp_directory)

os.listdir(temp_directory)

optaa.uid, optaa.serial, optaa.date

# **====================================================================================================================**
# ### Check the data
# Now, we have generated local csv and ext files from the data. We can now reload that data into python as a pandas dataframe, which will allow for a direct comparison with the existing data. 

sn = optaa.serial.split('-')[1].zfill(5)
dt = optaa.date

source_csv = pd.read_csv(temp_directory+'/CGINS-OPTAAD-'+sn+'__'+dt+'.csv')
source_csv


def reformat_arrays(array):
    if 'SheetRef:CC_taarray' == array or 'SheetRef:CC_tcarray' == array:
        return array
    else:
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


source_csv['value'] = source_csv['value'].apply(reformat_arrays)

source_csv['notes'].iloc[0]

source_taarray = pd.read_csv(temp_directory+'/CGINS-OPTAAD-'+sn+'__'+dt+'__CC_taarray.ext',header=None)
source_taarray.head()

source_tcarray = pd.read_csv(temp_directory+'/CGINS-OPTAAD-'+sn+'__'+dt+'__CC_tcarray.ext',header=None)
source_tcarray.head()

# **====================================================================================================================**
# Load the csv from asset management in order to compare

csv_filename = 'calibration/OPTAAD/CGINS-OPTAAD-00257__20191029.csv'
csv_file = pd.read_csv(csv_dir+csv_filename)

csv_file.sort_values(by='name', inplace=True)

csv_file.reset_index(inplace=True, drop=True)

csv_file['value'] = csv_file['value'].apply(reformat_arrays)

taarray = pd.read_csv(csv_dir + 'calibration/OPTAAD/CGINS-OPTAAD-00257__20191029__CC_taarray.ext',header=None)
tcarray = pd.read_csv(csv_dir + 'calibration/OPTAAD/CGINS-OPTAAD-00257__20191029__CC_tcarray.ext',header=None)

taarray

csv_file

source_csv

source_csv == csv_file

source_taarray == taarray

source_taarray[source_taarray != taarray].dropna(how='all').dropna(how='all',axis=1)

taarray[source_taarray != taarray].dropna(how='all').dropna(how='all',axis=1)

# **====================================================================================================================**
# # OPTAA Parser
# Below is a parser for the OPTAA calibration file. The following methods are available as part of the OPTAACalibration class:
# * **OPTAACalibration.load_cal**:
#         
#          Wrapper function to load all of the calibration coefficients
#         
#          Args:
#             filepath - path to the directory with filename which has the
#                 calibration coefficients to be parsed and loaded
#          Calls:
#             open_cal
#             parse_cal
#             
# * **OPTAACalibration.load_qct**:
#
#         Wrapper function to load the calibration coefficients from
#         the QCT checkin.
#             
#
# It is used as follows:
# 1. Initialize the OPTAA class using the **UID** for the OPTAA with the following code: OPTAA = OPTAACalibration(UID)
# 2. 

from zipfile import ZipFile
class OPTAACalibration():
    # Class that stores calibration values for CTDs.

    def __init__(self, uid):
        self.serial = None
        self.nbins = None
        self.uid = uid
        self.sigfig = 6
        self.date = []
        self.coefficients = {
            'CC_acwo': [],
            'CC_awlngth': [],
            'CC_ccwo': [],
            'CC_cwlngth': [],
            'CC_taarray': 'SheetRef:CC_taarray',
            'CC_tbins': [],
            'CC_tcal': [],
            'CC_tcarray': 'SheetRef:CC_tcarray'
        }
        self.tcarray = []
        self.taarray = []
        self.notes = {
            'CC_acwo': '',
            'CC_awlngth': '',
            'CC_ccwo': '',
            'CC_cwlngth': '',
            'CC_taarray': '',
            'CC_tbins': '',
            'CC_tcal': '',
            'CC_taarray': ''
        }

    @property
    def uid(self):
        return self._uid

    @uid.setter
    def uid(self, d):
        r = re.compile('.{5}-.{6}-.{5}')
        if r.match(d) is not None:
            self._uid = d
            serial = d.split('-')[-1].lstrip('0')
            self.serial = 'ACS-' + serial
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
        
        data = self.open_dev(filepath)
        
        self.parse_dev(data)
        
        
    def load_qct(self, filepath):
        """
        Wrapper function to load the calibration coefficients from
        the QCT checkin.
        """
        
        data = self.open_dev(filepath)
        
        self.parse_qct(data)
    
    
    def open_dev(self, filepath):
        """
        Function that opens and reads in cal file
        information for a OPTAA. Zipfiles are acceptable inputs.
        """
        
        if filepath.endswith('.zip'):
            with ZipFile(filepath) as zfile:
                # Check if OPTAA has the .dev file
                filename = [name for name in zfile.namelist() if name.lower().endswith('.dev')]
                
                # Get and open the latest calibration file
                if len(filename) == 1:
                    data = zfile.read(filename[0]).decode('ascii')
                    self.source_file(filepath, filename[0])
                    
                elif len(filename) > 1:
                    raise FileExistsError(f"Multiple .dev files found in {filepath}.")

                else:
                    raise FileNotFoundError(f"No .dev file found in {filepath}.")
                        
        elif filepath.lower().endswith('.dev'):
            with open(filepath) as file:
                data = file.read()
            self.source_file(filepath, file)
                
        elif filepath.lower().endswith('.dat'):
            with open(filepath) as file:
                data = file.read()
            self.source_file(filepath, file)
            
        else:
            raise FileNotFoundError(f"No .dev file found in {filepath}.")
        
        return data


    def source_file(self, filepath, filename):
        """
        Routine which parses out the source file and filename
        where the calibration coefficients are sourced from.
        """
        
        if filepath.lower().endswith('.dev'):
            dcn = filepath.split('/')[-2]
            filename = filepath.split('/')[-1]
        else:
            dcn = filepath.split('/')[-1]
        
        self.source = f'Source file: {dcn} > {filename}'
        

    def parse_dev(self, data):
        """
        Function to parse the .dev file in order to load the
        calibration coefficients for the OPTAA.
        
        Args:
            data - opened .dev file in ascii-format
        """
        
        for line in data.splitlines():
            # Split the data based on data -> header split
            parts = line.split(';')
                # If the len isn't number 2, 
            if len(parts) is not 2:
                # Find the calibration temperature and date
                if 'tcal' in line.lower():
                    line = ''.join((x for x in line if x not in [y for y in string.punctuation if y is not '/']))
                    parts = line.split()
                    # Calibration temperature
                    tcal = parts[1].replace('C','')
                    tcal = float(tcal)/10
                    self.coefficients['CC_tcal'] = tcal
                    # Calibration date
                    date = parts[-1].strip(string.punctuation)
                    self.date = pd.to_datetime(date).strftime('%Y%m%d')
        
            else:
                info, comment = parts
                
                if comment.strip().startswith('temperature bins'):
                    tbins = [float(x) for x in info.split()]
                    self.coefficients['CC_tbins'] = tbins
                    
                elif comment.strip().startswith('number'):
                    self.nbins = int(float(info.strip()))
                    
                elif comment.strip().startswith('C'):
                    if self.nbins is None:
                        raise AttributeError(f'Failed to load number of temperature bins.')
                        
                    # Parse out the different calibration coefficients
                    parts = info.split()
                    cwlngth = float(parts[0][1:])
                    awlngth = float(parts[1][1:])
                    ccwo = float(parts[3])
                    acwo = float(parts[4])
                    tcrow = [float(x) for x in parts[5:self.nbins+5]]
                    acrow = [float(x) for x in parts[self.nbins+5:2*self.nbins+5]]
                
                    # Now put the coefficients into the coefficients dictionary
                    self.coefficients['CC_acwo'].append(acwo)
                    self.coefficients['CC_awlngth'].append(awlngth)
                    self.coefficients['CC_ccwo'].append(ccwo)
                    self.coefficients['CC_cwlngth'].append(cwlngth)
                    self.tcarray.append(tcrow)
                    self.taarray.append(acrow)
                    
                    
    def parse_qct(self, data):
        """
        This function is designed to parse the QCT file, which contains the
        calibration data in slightly different format than the .dev file
        
        
        """
        
        for line in data.splitlines():
            if 'WetView' in line:
                _, _, _, date, time = line.split()
                try:
                    date_time = date + ' ' + time
                    self.date = pd.to_datetime(date_time).strftime('%Y%m%d')
                except:
                    date_time = from_excel_ordinal(float(date) + float(time))
                    self.date = pd.to_datetime(date_time).strftime('%Y%m%d')
                continue
                
            parts = line.split(';')
            
            if len(parts) == 2:
                if comment.strip().startswith('temperature bins'):
                    tbins = [float(x) for x in info.split()]
                    self.coefficients['CC_tbins'] = tbins
                    
                elif comment.strip().startswith('number'):
                    self.nbins = int(float(info.strip()))
                    
                elif comment.strip().startswith('C'):
                    if self.nbins is None:
                        raise AttributeError(f'Failed to load number of temperature bins.')
                    # Parse out the different calibration coefficients
                    parts = info.split()
                    cwlngth = float(parts[0][1:])
                    awlngth = float(parts[1][1:])
                    ccwo = float(parts[3])
                    acwo = float(parts[4])
                    tcrow = [float(x) for x in parts[5:self.nbins+5]]
                    acrow = [float(x) for x in parts[self.nbins+5:(2*self.nbins)+5]]
                    
                    # Now put the coefficients into the coefficients dictionary
                    self.coefficients['CC_acwo'].append(acwo)
                    self.coefficients['CC_awlngth'].append(awlngth)
                    self.coefficients['CC_ccwo'].append(ccwo)
                    self.coefficients['CC_cwlngth'].append(cwlngth)
                    self.tcarray.append(tcrow)
                    self.taarray.append(acrow)                
    
                        
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
      
        # Now merge the coefficients dataframe with the notes
        notes = pd.DataFrame().from_dict({
            'name':list(self.notes.keys()),
            'notes':list(self.notes.values())
        })
        df = df.merge(notes, how='outer', left_on='name', right_on='name')
            
        # Add in the source file
        df['notes'].iloc[0] = df['notes'].iloc[0] + ' ' + self.source
        
        # Sort the data by the coefficient name
        df = df.sort_values(by='name')

        # Generate the csv names
        csv_name = self.uid + '__' + self.date + '.csv'
        tca_name = self.uid + '__' + self.date + '__' + 'CC_tcarray.ext'
        taa_name = self.uid + '__' + self.date + '__' + 'CC_taarray.ext'
        
        def write_array(filename, cal_array):
            with open(filename, 'w') as out:
                array_writer = csv.writer(out)
                array_writer.writerows(cal_array)

        # Write the dataframe to a csv file
        check = input(f"Write {csv_name} to {outpath}? [y/n]: ")
        # check = 'y'
        if check.lower().strip() == 'y':
            df.to_csv(outpath+'/'+csv_name, index=False)
            write_array(outpath+'/'+tca_name, self.tcarray)
            write_array(outpath+'/'+taa_name, self.taarray)


