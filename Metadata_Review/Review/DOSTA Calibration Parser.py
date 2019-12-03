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

# # DOSTA Calibration Parser
#
# This script and notebook gives the code for loading the CTD calibration coefficients into a properly named calibration csv, as well as an example of how to use it. The calibration coefficients can be parsed from either the vendor calibration (.cal) file, the vendor xmlcon file, or from the capture file (either .cap, .log, or .txt) from the QCT check-in.
#

from utils import *

import csv
import datetime
import os
import shutil
import sys
import time
import re
import xml.etree.ElementTree as et
import pandas as pd
from zipfile import ZipFile


# ========================================================================================================================
# ### Define functions

def get_calibration_files(serial_nums,dirpath):
    """
    Function which gets all the calibration files associated with the
    instrument serial numbers.
    
    Args:
        serial_nums - serial numbers of the instruments
        dirpath - path to the directory containing the calibration files
    Returns:
        calibration_files - a dictionary of instrument uids with associated
            calibration files
    """
    calibration_files = {}
    for uid in serial_nums.keys():
        sn = serial_nums.get(uid)[0]
        sn = str(sn)
        files = []
        for file in os.listdir(dirpath):
            if sn in file:
                if 'Calibration' in file:
                    files.append(file)
                else:
                    pass
            else:
                pass
        
        calibration_files.update({uid:files})
        
    return calibration_files


def get_qct_files(df, qct_directory):
    qct_dict = {}
    uids = list(set(df['UID']))
    for uid in uids:
        df['UID_match'] = df['UID'].apply(lambda x: True if uid in x else False)
        qct_series = df[df['UID_match'] == True]['QCT Testing']
        qct_series = list(str(qct_series.iloc[0]).split('\n'))
        qct_dict.update({uid:qct_series})
    return qct_dict


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


def splitDataFrameList(df,target_column):
    ''' 
    df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 
    The values in the other columns are duplicated across the newly divided rows.
    '''
    
    def splitListToRows(row,row_accumulator,target_column):
        split_row = row[target_column]
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)
            
    new_rows = []
    df.apply(splitListToRows,axis=1,args = (new_rows,target_column))
    new_df = pd.DataFrame(new_rows)
    return new_df


# ========================================================================================================================
# ### Directories
# **Define the main directories where important information is stored.**

qct_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/PRESF/PRESF_Results'
cal_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/PRESF/PRESF_Cal'
asset_management_directory = '/home/andrew/Documents/OOI-CGSN/ooi-integration/asset-management/calibration/PRESFB'

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

for uid in csv_dict.keys():
    for file in sorted(csv_dict[uid]):
        print(file)

uids = sorted(list(csv_dict.keys()))

serial_nums = get_serial_nums(DOSTA, uids)


cal_dict = get_calibration_files(serial_nums, cal_directory)
cal_dict

# ========================================================================================================================
# **Now, need to get all the files for a particular CTDMO UID:**

uid = sorted(uids)[0]
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
    path = generate_file_path(qct_directory, qf)
    qct_path.append(path)
qct_path

# ### What does the DOSTA Calibration file format look like?
#
# 1. The calibration files are all pdf files sent by SeaBird. They are most likely NOT machine readable, which presents a challenge in automating this approach.
#
# 2. Conclusion: I won't try to read the pdfs. Instead, I'll focus on writing the csv cal sheet from the qct checkin info.

import textract

ptext = textract.process(cal_path[1], method='tesseract', encoding='utf-8')
print(ptext.decode('utf-8'))

ptext = ptext.replace(b'\xe2\x80\x94',b'-')
ptext = ptext.decode('utf-8')

pd.read_csv(csv_path[0])





# ### What do the DOSTA QCT file format look like?

# +
#!/usr/bin/env python
import csv
import datetime
import os
import shutil
import sys
import time
import re
import xml.etree.ElementTree as et
import pandas as pd
from zipfile import ZipFile


class DOSTACalibration():
    # Class that stores calibration values for DOSTA's.

    def __init__(self, uid, calibration_date):
        self.serial = ''
        self.uid = uid
        self.date = pd.to_datetime(calibration_date).strftime('%Y%m%d')
        self.coefficients = {'CC_conc_coef':None,
                            'CC_csv':None}
        self.notes = {'CC_conc_coef':None,
                      'CC_csv':None}
                
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
            
            
    def generate_file_path(self,dirpath,filename,ext=['.cap','.txt','.log'],exclude=['_V','_Data_Workshop']):
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
                
    def load_qct(self, filepath):
        """
        Function which parses the output from the QCT check-in and loads them into
        the DOSTA object.
        
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
        
        data = {}
        with open(filepath, errors='ignore') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                data.update({reader.line_num:row})
                
        for key,info in data.items():
            # Find the serial number from the QCT check-in and compare to UID
            if 'serial number' in [x.lower() for x in info]:
                serial_num = info[-1].zfill(5)
                if self.serial != serial_num:
                    raise ValueError(f'Serial number {serial_num.zfill(5)} from the QCT file does not match {self.serial} from the UID.')
                else:
                    pass
                
            # Find the svu foil coefficients
            if 'svufoilcoef' in [x.lower() for x in info]:
                self.coefficients['CC_csv'] = [float(n) for n in info[3:]]
            
            # Find the concentration coefficients
            if 'conccoef' in [x.lower() for x in info]:
                self.coefficients['CC_conc_coef'] = [float(n) for n in info[3:]]
                
    def add_notes(self, notes):
        """
        This function adds notes to the calibration csv based on the 
        calibration coefficients.
        
        Args:
            notes - a dictionary with keys of the calibration coefficients
                which correspond to an entry of desired notes about the 
                corresponding coefficients
        Returns:
            self.notes - a dictionary with the entered notes.
        """
        keys = notes.keys()
        for key in keys:
            self.notes[key] = notes[key]


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
        for key in self.coefficients.keys():
            if self.coefficients[key] == None:
                raise ValueError(f'No coefficients for {key} have been loaded.')
            
        # Create a dataframe to write to the csv
        data = {'serial':[self.serial]*len(self.coefficients),
               'name':list(self.coefficients.keys()),
               'value':list(self.coefficients.values()),
               'notes':list(self.notes.values()) }
        df = pd.DataFrame().from_dict(data)
        
        # Generate the csv name
        csv_name = self.uid + '__' + self.date + '.csv'
        
        # Now write to 
        check = input(f"Write {csv_name} to {outpath}? [y/n]: ")
        if check.lower().strip() == 'y':
            df.to_csv(outpath+'/'+csv_name, index=False)
# -

DOSTA = DOSTACalibration(uid='CGINS-DOSTAD-00126',calibration_date='March 22, 2019')

DOSTA.date

filepath = DOSTA.generate_file_path(qct_directory,'3305-00115-00128')
filepath

DOSTA.load_qct(filepath)

DOSTA.coefficients

DOSTA.serial

DOSTA.add_notes({'CC_conc_coef':'[intercept,slope]'})

outdir = '/'.join((os.getcwd(),'temp'))

DOSTA.write_csv(outdir)

out = generate_file_path(asset_management_directory, csv_files[0].split('.')[0],ext=['.csv'])
print(out)

check = csv_files[0].split('.')
check

if type(check) is str:
    print('This logic works')

csv_files[0]

qct_filepath = generate_file_path(qct_directory, '3305-00115-00128')

data = {}
with open(qct_filepath, errors='ignore') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        data.update({reader.line_num:row})

for key,info in data.items():
    if 'serial number' in [x.lower() for x in info]:
        print(key)
        print(info)

data

for key,info in data.items():
    if 'svufoilcoef' in [x.lower() for x in info]:
        print(key)
        print(info)

info[3:]

for key,info in data.items():
    if 'conccoef' in [x.lower() for x in info]:
        print(key)
        print(info)

data


