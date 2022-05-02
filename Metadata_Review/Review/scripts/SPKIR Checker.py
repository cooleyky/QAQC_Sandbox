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

# # SPKIR Checker
#
# This notebook is designed to check the SPKIR csv calibration file in pull request. The process I follow is:
# 1. Read in the SPKIR csv from the pull request into a pandas dataframe
# 2. Identify the source file of the calibration coefficients
# 3. Parse the calibration coefficients directly from the source file
# 4. Compare the SPKIR csv from the pull request with the csv parsed from the source file
#
# **====================================================================================================================**
#
# The first step is to load relevant packages:

import csv
import re
import os
import numpy as np
import pandas as pd

from utils import *

from zipfile import ZipFile
import string

sys.path.append('/home/andrew/Documents/OOI-CGSN/QAQC_Sandbox/Calibration/Parsers/')

from Parsers.SPKIRCalibration import SPKIRCalibration

# **====================================================================================================================**
# Define the directories where the **csv** file to check is stored, and where the **source** file is stored. Make sure to check the following information on your machine via your terminal first:
# 1. The branch of your local asset-management repository matches the location of the SPKIR cals
# 2. Your local asset-management repository has the requisite **csv** file to check
# 3. You have downloaded the **source** of the csv file

am_file = '/home/andrew/Documents/OOI-CGSN/asset-management/calibration/SPKIRB/CGINS-SPKIRB-00300__20200128.csv'
source_file = '/home/andrew/Downloads/SPKIR-B_OCR-507_SN_300_Calibration_Files_2020-01-28.zip'

# Read in the pull request csv into a pandas dataframe:

pr_csv = pd.read_csv(am_file)
pr_csv

# **====================================================================================================================**
# ### Find & Parse the source file
# Now, we want to find the source file of the calibration coefficients, parse the data using the spkir parser, and read the data into a pandas dataframe. The key pieces of data needed for the parser are:
# 1. Instrument UID: This is needed to initialize the OPTAA parser
# 2. Source file: This is the full path to the source file. Zip files are acceptable input.

# Initialize the parser with the UID:

spkir = SPKIRCalibration('CGINS-SPKIRB-00300')

# Read in the calibration coefficients from the source file:

spkir.load_cal(source_file)

# Write the csv to a temporary local folder:

temp_directory = '/'.join((os.getcwd(),'temp'))
# Check if the temp directory exists; if it already does, purge and rewrite
if os.path.exists(temp_directory):
    shutil.rmtree(temp_directory)
    ensure_dir(temp_directory)
else:
    ensure_dir(temp_directory)

spkir.write_csv(temp_directory)

# Check that the instrument uid, serial number, and calibration date make sense:

spkir.uid, spkir.serial, spkir.date

# **====================================================================================================================**
# ### Compare the data sets
# With the data parsed from the source file and the csv from the pull request, we can now directly compare the between the two datasets and identify any inconsistencies or errors.

source_csv = pd.read_csv(temp_directory+'/'+'CGINS-SPKIRB-00300__20200128.csv')
source_csv

pr_csv


# Reformat the coefficient value arrays:

def reformat_arrays(array):
    # First, need to strip extraneous characters from the array
    array = array.replace("'","").replace('[','').replace(']','')
    # Next, split the array into a list
    array = array.split(',')
    # Now, need to eliminate any white space surrounding the individual coeffs
    array = [num.strip() for num in array]
    # Next, float the nums
    try:
        array = [float(num) for num in array]
        # Check if the array is len == 1; if so, can just return the number
        if len(array) == 1:
            array = array[0]
    except:
        pass
    # Now we are done
    return array


source_csv['value'] = source_csv['value'].apply(lambda x: reformat_arrays(x))

pr_csv['value'] = pr_csv['value'].apply(lambda x: reformat_arrays(x))

# Compare the two pandas dataframes for differences:

source_csv == pr_csv

# If any (besides notes) return false, iterate through the False position to identify which specific calibration coefficient is incorrect:

for n,m in enumerate(source_csv['value'].iloc[2]):
    print(m, pr_csv['value'].iloc[2][n])
    print(m == pr_csv['value'].iloc[2][n])


# **=======================================================================================================================**
# # Parsing Calibration Coefficients
# Above, we have worked through identifying and mapping the calibration files and QCT check-ins to the individual instruments through their UIDs and serial numbers. The next step is to open the relevant files and parse out the calibration coefficients. This will require writing a parser for the SPKIR.

class SPKIRCalibration():
    # Class that stores calibration values for CTDs.

    def __init__(self, uid):
        self.serial = None
        self.uid = uid
        self.date = []
        self.coefficients = {
            'CC_immersion_factor': [],
            'CC_offset': [],
            'CC_scale': []
        }
        self.notes = {
            'CC_immersion_factor': '',
            'CC_offset': '',
            'CC_scale': '',
        }

    @property
    def uid(self):
        return self._uid

    @uid.setter
    def uid(self, d):
        r = re.compile('.{5}-.{6}-.{5}')
        if r.match(d) is not None:
            self._uid = d
            self.serial = d.split('-')[-1].lstrip('0')
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
        information for a SPKIR. Zipfiles are acceptable inputs.
        
        Args:
            filepath - path to the directory with filename which has the
                calibration coefficients to be parsed and loaded
        
        Returns:
            data - opended calibration file that has been read into 
                memory but is not parsed
        """
        
        if filepath.endswith('.zip'):
            with ZipFile(filepath) as zfile:
                # Check if OPTAA has the .dev file
                filename = [name for name in zfile.namelist() if name.lower().endswith('.cal')]
                
                # Get and open the latest calibration file
                if len(filename) == 1:
                    data = zfile.read(filename[0]).decode('ascii')
                    self.source_file(filepath, filename[0])
                    
                elif len(filename) > 1:
                    raise FileExistsError(f"Multiple .cal files found in {filepath}.")

                else:
                    raise FileNotFoundError(f"No .cal file found in {filepath}.")
                        
        elif filepath.lower().endswith('.cal'):
            with open(filepath) as file:
                data = file.read()
            self.source_file(filepath, file)
          
        else:
            raise FileNotFoundError(f"No .cal file found in {filepath}.")
        
        return data
        
        
    def parse_cal(self, data):
        """
        Function which parses the calibration data and loads the calibration
        coefficients into the object structure.
        
        Args:
            data - calibration data which has been read and loaded into memory
        Raises:
            ValueError - raised if the serial number parsed from the calibration
                data does not match the UID
        Returns:
            self.coefficients - populated dictionary of calibration coefficient values
            self.date - all relevant calibration dates parsed into a dictionary
            self.serial - parsed serial data
        """
        
        flag = False
        for line in data.splitlines():
            if line.startswith('#'):
                parts = line.split('|')
                if len(parts) > 5 and 'Calibration' in parts[-1].strip():
                    cal_date = parts[0].replace('#','').strip()
                    self.date.append(pd.to_datetime(cal_date).strftime('%Y%m%d'))
                    
            elif line.startswith('SN'):
                parts = line.split()
                _, sn, *ignore = parts
                sn = sn.lstrip('0')
                if self.serial != sn:
                    raise ValueError(f'Instrument serial number {sn} does not match UID {self.uid}')
                    
            elif line.startswith('ED'):
                flag = True
                
            elif flag:
                offset, scale, immersion_factor = line.split()
                self.coefficients['CC_immersion_factor'].append(float(immersion_factor))
                self.coefficients['CC_offset'].append(float(offset))
                self.coefficients['CC_scale'].append(float(scale))
                flag = False
                
            else:
                continue
        
        
    def source_file(self, filepath):
        """
        Routine which parses out the source file and filename
        where the calibration coefficients are sourced from.
        
        Args:
            filepath - path to the directory with filename which has the
                calibration coefficients to be parsed and loaded
        Returns:
            self.source - string which contains the parent file and the
                filename of the calibration data source
        """
        
        if filepath.lower().endswith('.cal'):
            dcn = filepath.split('/')[-2]
            filename = filepath.split('/')[-1]
        else:
            dcn = filepath.split('/')[-1]
        
        self.source = f'Source file: {dcn} > {filename}'
        
        
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
        csv_name = self.uid + '__' + max(self.date) + '.csv'
        
        # Write the dataframe to a csv file
        check = input(f"Write {csv_name} to {outpath}? [y/n]: ")
        # check = 'y'
        if check.lower().strip() == 'y':
            df.to_csv(outpath+'/'+csv_name, index=False)


