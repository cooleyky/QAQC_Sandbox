# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # SPKIR Metadata Review
#
# This notebook describes the process for reviewing the calibration coefficients for the SPKIR. The purpose is to check the calibration coefficients contained in the CSVs stored within the asset management repository on GitHub, which are the coefficients utilized by OOI-net for calculating data products, against the different available sources of calibration information to identify when errors were made during entering the calibration csvs. This includes checking the following information:
# 1. The calibration date - this information is stored in the filename of the csv
# 2. Calibration source - identifying all the possible sources of calibration information, and determine which file should supply the calibration info
# 3. Calibration coeffs - checking the accuracy and precision of the numbers stored in the calibration coefficients
#
# The SPKIRs contains three different calibration coefficients to check. All three of the coefficients are arrays of seven values. The possible calibration sources for the SPKIR are vendor calibration (.cal) files. The QCT, pre- and post-deployment files do not contain the relevant calibration information needed to perform checking.

import csv
import re
import os
import numpy as np
import pandas as pd

from utils import *

# **====================================================================================================================**
# Define the directories where the QCT, Pre, and Post deployment document files are stored, where the vendor documents are stored, where asset tracking is located, and where the calibration csvs are located.

doc_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/SPKIR/SPKIR_Results/'
cal_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/SPKIR/SPKIR_Cal/'
asset_management_directory = '/home/andrew/Documents/OOI-CGSN/ooi-integration/asset-management/calibration/SPKIRB/'

excel_spreadsheet = '/media/andrew/OS/Users/areed/Documents/Project_Files/Documentation/System/System Notebook/WHOI_Asset_Tracking.xlsx'
sheet_name = 'Sensors'

SPKIR = whoi_asset_tracking(spreadsheet=excel_spreadsheet,sheet_name=sheet_name,instrument_class='SPKIR',series='B')
SPKIR

# **======================================================================================================================**
# Now, I want to load all the calibration csvs and group them by UID:

uids = sorted( list( set(SPKIR['UID']) ) )
uids

csv_dict = {}
asset_management = os.listdir(asset_management_directory)
for uid in uids:
    files = [file for file in asset_management if uid in file]
    csv_dict.update({uid: sorted(files)})

csv_paths = {}
for uid in sorted(csv_dict.keys()):
    paths = []
    for file in csv_dict.get(uid):
        path = generate_file_path(asset_management_directory, file, ext=['.csv','.ext'])
        paths.append(path)
    csv_paths.update({uid: paths})

csv_paths

# **=======================================================================================================================**
# The SPKIR QCT capture files are stored with the following Document Control Numbers (DCNs): 3305-00114-XXXXX. Most are storead as **.txt** or **.log** files. The problem is that the encoding of the data is not clear how the QCT is stored. Consequently, the QCT files aren't going to be used to check the SPKIR instrument calibration (for now).
#
#
#

qct_dict = get_qct_files(SPKIR, doc_directory)
qct_paths = {}
for uid in sorted(qct_dict.keys()):
    paths = []
    for file in qct_dict.get(uid):
        path = generate_file_path(doc_directory, file)
        paths.append(path)
    qct_paths.update({uid: paths})

# **=======================================================================================================================** Find and return the calibration files which contain vendor supplied calibration information. This is achieved by searching the calibration directories and matching serial numbers to UIDs:

serial_nums = get_serial_nums(SPKIR, uids)

serial_nums;

cal_dict = get_calibration_files(serial_nums, cal_directory)

# Retrieve and save the full directory path to the calibration files
cal_paths = {}
for uid in sorted(cal_dict.keys()):
    paths = []
    for file in cal_dict.get(uid):
        path = generate_file_path(cal_directory, file, ext=['.zip','.cap', '.txt', '.log'])
        paths.append(path)
    cal_paths.update({uid: paths})

cal_paths;


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
                self.coefficients['CC_immersion_factor'].append(immersion_factor)
                self.coefficients['CC_offset'].append(offset)
                self.coefficients['CC_scale'].append(scale)
                flag = False
                
            else:
                continue
        
        
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

# **=======================================================================================================================**
# # Source Loading of Calibration Coefficients
# With a SPKIR Calibration object created, we can now begin parsing the different calibration sources for each SPKIR. We will then compare all of the calibration values from each of the sources, checking for any discrepancies between them.

# Below, I plan on going through each of the SPKIR UIDs, and parse the data into csvs. For source files which may contain multiple calibrations or calibration sources, I plan on extracting each of the calibrations to a temporary folder using the following structure:
#
#     <local working directory>/<temp>/<source>/data/<calibration file>
#     
# The separate calibrations will be saved using the standard UFrame naming convention with the following directory structure:
#
#     <local working directory>/<temp>/<source>/<calibration csv>
#     
# The csvs themselves will also be copied to the temporary folder. This allows for the program to be looking into the same temp directory for every SPKIR check.

import shutil

uid = uids[20]
uid

# Make the local temp directory. If it already exists; purge it and rewrite:

temp_directory = '/'.join((os.getcwd(),'temp'))
if os.path.exists(temp_directory):
    shutil.rmtree(temp_directory)
    ensure_dir(temp_directory)

# Copy the existing csvs from asset management to the temp directory:

for path in csv_paths[uid]:
    savedir = '/'.join((temp_directory,'csv'))
    ensure_dir(savedir)
    savepath = '/'.join((savedir, path.split('/')[-1]))
    shutil.copyfile(path, savepath)

# **=======================================================================================================================**
# Load the calibration coefficients from the vendor calibration source files. Start by extracting or copying them to the source data folder in the temporary directory.

# Extract the calibration zip files to the local temp directory:

for path in cal_paths[uid]:
    with ZipFile(path) as zfile:
        files = [name for name in zfile.namelist() if name.lower().endswith('.cal')]
        for file in files:
            exdir = path.split('/')[-1].strip('.zip')
            expath = '/'.join((temp_directory,'cal','data',exdir))
            ensure_dir(expath)
            zfile.extract(file,path=expath)

# Write the vendor calibration files to csvs following the UFrame convention:

savedir = '/'.join((temp_directory,'cal'))
ensure_dir(savedir)
# Now parse the calibration coefficients
for dirpath, dirnames, filenames in os.walk('/'.join((temp_directory,'cal','data'))):
    for file in filenames:
        filepath = os.path.join(dirpath, file)
        # With the filepath for the given calibration retrived, I can now start an instance of the NUTNR Calibration
        # object and begin parsing the coefficients
        spkir = SPKIRCalibration(uid)
        spkir.load_cal(filepath)
        spkir.write_csv(savedir)


# **=======================================================================================================================**
# # Calibration Coefficient Comparison
# We have now successfully parsed the calibration files from all the possible sources: the vendor calibration files, the pre-deployments files, and the post-deployment files. Furthermore, we have saved csvs in the UFrame format for all of these calibrations. Now, we want to load those csvs into pandas dataframes, which allow for easy element-by-element comparison of calibration coefficients.

def get_file_date(x):
    x = str(x)
    ind1 = x.index('__')
    ind2 = x.index('.')
    return x[ind1+2:ind2]


# Now we want to compare dataframe
csv_files = [file for file in sorted(os.listdir('temp/csv')) if 'data' not in file]
csv_files = pd.DataFrame(csv_files, columns=['csv'])
csv_files['cal date'] = csv_files['csv'].apply(lambda x: get_file_date(x))
csv_files.set_index('cal date', inplace=True)

# Now we want to compare dataframe
cal_files = [file for file in sorted(os.listdir('temp/cal')) if 'data' not in file]
cal_files = pd.DataFrame(cal_files, columns=['cal'])
cal_files['cal date'] = cal_files['cal'].apply(lambda x: get_file_date(x))
cal_files.set_index('cal date', inplace=True)

df_files = csv_files.join(cal_files,how='outer').fillna(value='-999')
df_files

# Rename above CSV file names
sn = '00302'
d1 = '20170913'
d2 = '20170911'

src = 'temp/csv/' + f'CGINS-SPKIRB-{sn}__{d1}.csv'
dst = 'temp/csv/' + f'CGINS-SPKIRB-{sn}__{d2}.csv'
shutil.move(src, dst)

# Reload the csv files in order to perform the comparison:

# CSV files
csv_files = [file for file in sorted(os.listdir('temp/csv')) if 'data' not in file]
csv_files = pd.DataFrame(csv_files, columns=['csv'])
csv_files['cal date'] = csv_files['csv'].apply(lambda x: get_file_date(x))
csv_files.set_index('cal date', inplace=True)

# Calibration source files
cal_files = [file for file in sorted(os.listdir('temp/cal')) if 'data' not in file]
cal_files = pd.DataFrame(cal_files, columns=['cal'])
cal_files['cal date'] = cal_files['cal'].apply(lambda x: get_file_date(x))
cal_files.set_index('cal date', inplace=True)

df_files = csv_files.join(cal_files,how='outer').fillna(value='-999')

df_files


# **=======================================================================================================================**
# Now, with the csv files renamed to match their associated calibration dates following the OOI UFrame format, we can load the info into pandas dataframe which will allow for the direct comparison of calibration coefficients using built in array comparison tools from numpy. 
#
# A complication is that, when loading a csv using pandas, it reads the csv as strings. This includes characters such as **[]**. Consequently, we need to reformat the arrays in the dataframe and convert to 64-bit floating point numbers. 

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


# Load the calibration coefficients into pandas dataframes:

# Use fstring literals to allow on the fly file-renaming
dt = '20170911'
fname = f'CGINS-SPKIRB-{sn}__{dt}.csv'2717

CSV = pd.read_csv('temp/csv/'+fname)
CSV

CAL = pd.read_csv('temp/cal/'+fname)
CAL

# Reformat the arrays
CSV['value'] = CSV['value'].apply(lambda x: reformat_arrays(x))
CAL['value'] = CAL['value'].apply(lambda x: reformat_arrays(x))

# Check that the calibration coefficients agree
np.equal(CSV,CAL)

# Check the source file for the calibration coefficients
CAL['notes'].iloc[0]


