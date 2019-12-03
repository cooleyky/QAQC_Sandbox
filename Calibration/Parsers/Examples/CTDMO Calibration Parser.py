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



from utils import *


def get_calibration_files(serial_nums,dirpath):
    calibration_files = {}
    for uid,sn in serial_nums.items():
        files = []
        for file in os.listdir(dirpath):
            if sn in file:
                if 'Calibration_File' in file:
                    files.append(file)
                else:
                    pass
            else:
                pass
        
        calibration_files.update({uid:files})
        
    return calibration_files


qct_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/'
cal_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/CTDMO/'
asset_management_directory = '/home/andrew/Documents/OOI-CGSN/ooi-integration/asset-management/calibration/CTDMOG'

excel_spreadsheet = '/media/andrew/OS/Users/areed/Documents/Project_Files/Documentation/System/System Notebook/WHOI_Asset_Tracking.xlsx'
sheet_name = 'Sensors'

CTDBP = whoi_asset_tracking(spreadsheet=excel_spreadsheet,sheet_name=sheet_name,instrument_class='CTDBP')

CTDBP

uids = list(CTDMO['UID'])

qct_dict = {}
for uid in uids:
    # Get the QCT Document numbers from the asset tracking sheet
    CTDMO['UID_match'] = CTDMO['UID'].apply(lambda x: True if uid in x else False)
    qct_series = CTDMO[CTDMO['UID_match'] == True]['QCT Testing']
    qct_series = list(qct_series.iloc[0].split('\n'))
    qct_dict.update({uid:qct_series})



qct_dict

serial_nums = get_serial_nums(CTDMO, uids)
serial_nums

cal_dict = get_calibration_files(serial_nums, cal_directory)
cal_dict

# +
# I want to check all of the different calibration coefficients in the csv system
# -

qct_dict[uid]

cal_dict[uid]

df = pd.DataFrame()
for uid in uids:
    csv_files = csv_dict[uid]
    for file in csv_files:
        filename = file.split('.')[0]
        fpath = generate_file_path(asset_management_directory, filename, ext=['.csv'])
        df = df.append(pd.read_csv(fpath))
        

coefficients = set(list(df['name']))

#fpath = generate_file_path(qct_directory, '3305-00101-00254')
fpath = generate_file_path(qct_directory, '3305-00102-00174')
fpath

# +
with open(fpath) as file:
    data = file.read()
    
data.splitlines()

# +
coefficients = {}
for line in data.splitlines():
    keys = list(coefficient_name_map.keys())
    if any([word for word in line.split() if word in keys]):
        coefficients.update({line.split()[0]:line.split()[-1]})

    if 'temperature:' in line:
        date.update({'TCAL':pd.to_datetime(line.split()[-1]).strftime('%Y%m%d')})
    elif 'conductivity:' in line:
        date.update({'CCAL':pd.to_datetime(line.split()[-1]).strftime('%Y%m%d')})
    elif 'pressure S/N' in line:
        date.update({'PCAL':pd.to_datetime(line.split()[-1]).strftime('%Y%m%d')})
    else:
        pass
    
    if 'SERIAL NO.' in line:
        ind = line.split().index('NO.')
        serial_num = line.split()[ind+1]
        if serial != serial_num:
            raise ValueError(f'UID serial number {serial} does not match the QCT serial num {serial_num}')

        
        
coefficients
# -

date

serial = '50062'

# +
with open(fpath) as file:
    data = file.read()

coefficients = {}
data = data.replace('<',' ').replace('>',' ')
for line in data.splitlines():
    keys = list(mo_coefficient_name_map.keys())
    if any([word for word in line.split() if word.lower() in keys]):
        coefficients.update({line.split()[0]:line.split()[1]})
        
coefficients
# -

any([word for word in line.split() if word.lower() in keys])

check = ['hour','sec','min','day','month']
check

any(check for word in line.split())

line.split()

CTD.coefficient_name_map

line = data.splitlines()[13]
line.split()[4]

fpath = generate_file_path(qct_directory, '3305-00101-00254')
fpath

with open(fpath) as file:
    data = file.read()

data.splitlines()



# +
# Now need to write a parser for the SBE 37 parser
if 'SBE37' in data:
    sbe37flag = True
    
for line in data.splitlines():
    
    # Check for the serial number
    if 'SERIAL NO.' in line:
        ind = line.split().index('NO.')
        serial_num = line.split()[ind+1]
        # Check if it matches the UID parsed serial num
        if len(self.serial) == 0:
            serial = serial_num
        if not self.serial == serial_num:
            raise ValueError(f"QCT serial number {serial_num} doesn't match UID serial number {self.serial}")
            
    if
        
    
# -

CTD.serial == line.split()[4]



csv = pd.read_csv(fpath)

csv

# +
import PyPDF2 

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# -

fpath = generate_file_path(cal_directory, 'CTDMO-G_SBE_37IM_SN_37-10214_Calibration_Files_2012-11-13', ext=['.pdf'])

fpath

cal_date

tokens = word_tokenize(text[5])
data = [word.lower() for word in tokens if not word in string.punctuation]
data.index('sensor')

import nltk
nltk.download('punkt')


CTD = CTDMOCalibration(uid=uids[0])

CTD.serial

CTD.uid

fpath

text = CTD.load_pdf(fpath)

CTD.read_pdf(text)

CTD.coefficients

max(CTD.date.values())

uid

serial = '10214'

coefficients = {}
ctd_type = '37'
date = {}
mo_coefficient_name_map = {
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

coefficient_name_map = {
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
import string
from zipfile import ZipFile


class CTDMOCalibration():
    # Class that stores calibration values for CTDs.

    def __init__(self, uid):
        self.serial = ''
        self.uid = uid
        self.coefficients = {}
        self.date = {}
        self.type = ''
                    
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
            

            
            
    def load_pdf(self,filepath):
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
        pdfFileObj = open(filepath,'rb')
        # Create a reader to be parsed
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        # Now, enumerate through the pdf and decode the text
        num_pages = pdfReader.numPages
        count = 0
        text = {}
    
        while count < num_pages:
            pageObj = pdfReader.getPage(count)
            count = count + 1
            text.update({count:pageObj.extractText()})
        
        # Run a check that text was actually extracted
        if len(text) == 0:
            raise(IOError(f'No text was parsed from the pdf file {filepath}'))
        else:
            return text
            
            
    def read_pdf(self,text):
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
                        coefficients.update({self.mo_coefficient_name_map[key]:data[ind+1]})
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
                        self.date.update({'CCAL':date})
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
                        self.coefficients.update({self.mo_coefficient_name_map[key]:data[ind+1]})
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
                        self.date.update({'PCAL':date})
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
                        self.coefficients.update({self.mo_coefficient_name_map[key]:data[ind+1]})
                    else:
                        pass
        
            # Now check for other important information
            else:
                tokens = word_tokenize(text[page_num])
                data = [word.lower() for word in tokens if not word in string.punctuation]

                # Now, find the sensor rating
                if 'sensor' and 'rating' in data:
                    ind = data.index('rating')
                    self.coefficients.update({self.mo_coefficient_name_map['prange']:data[ind+1]})

                        
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
        
        with open(filepath) as filename:
            data = filename.read()

        data.splitlines()
        for line in data.splitlines():
    
            if 'SBE 16Plus' in line:
                self.type = '16'
        
            elif 'SERIAL NO.' in line:
                items = line.split()
                ind = items.index('NO.')
                value = items[ind+1].strip().zfill(5)
                if self.serial != value:
                    raise ValueError(f'Serial number {value.zfill(5)} from the QCT file does not match {self.serial} from the UID.')
                else:
                    pass
        
            else:
                items = re.split(': | =',line)
                key = items[0].strip()
                value = items[-1].strip()
        
                if key == 'temperature':
                    self.date.update({'TCALDATE':datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})    
        
                elif key == 'conductivity':
                    self.date.update({'CCALDATE':datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})
                
                elif key == 'pressure S/N':
                    self.date.update({'PCALDATE':datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})
            
                else:
                    name = self.coefficient_name_map.get(key)
                    if not name or name is None:
                        pass
                    else:
                        self.coefficients.update({name:value})


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
        data = {'serial':[self.type + '-' + self.serial]*len(self.coefficients),
               'name':list(self.coefficients.keys()),
               'value':list(self.coefficients.values()),
               'notes':['']*len(self.coefficients) }
        df = pd.DataFrame().from_dict(data)
        
        # Generate the csv name
        cal_date = max(self.date.values())
        csv_name = self.uid + '__' + cal_date + '.csv'
        
        # Now write to 
        check = input(f"Write {csv_name} to {outpath}? [y/n]: ")
        if check.lower().strip() == 'y':
            df.to_csv(outpath+'/'+csv_name, index=False)
# -










