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

# # CTDMO Metadata Review
#
# This notebook describes the process for reviewing the calibration coefficients for the CTDMO IM-37. The purpose is to check the calibration coefficients contained in the CSVs stored within the asset management repository on GitHub, which are the coefficients utilized by OOI-net for calculating data products, against the different available sources of calibration information to identify when errors were made during entering the calibration csvs. This includes checking the following information:
# 1. The calibration date - this information is stored in the filename of the csv
# 2. Calibration source - identifying all the possible sources of calibration information, and determine which file should supply the calibration info
# 3. Calibration coeffs - checking the accuracy and precision of the numbers stored in the calibration coefficients
#
# The CTDMO contains 24 different calibration coefficients to check. The possible calibration sources for the CTDMOs are vendor PDFs, vendor .cal files, and QCT check-ins. A complication is that the vendor documents are principally available only as PDFs that are copies of images. This requires the use of Optical Character Recognition (OCR) in order to parse the PDFs. Unfortunately, OCR frequently misinterprets certain character combinations, since it utilizes Levenstein-distance to do character matching. 
#
# Furthermore, using OCR to read PDFs requires significant preprocessing of the PDFs to create individual PDFs with uniform metadata and encoding. Without this preprocessing, the OCR will not generate uniformly spaced characters, making parsing not amenable to repeatable automated parsing.

# Import likely important packages, etc.
import sys, os
import numpy as np
import pandas as pd
import shutil

from utils import *

# **====================================================================================================================**
# Define the directories where the QCT document files are stored as well as where the vendor documents are stored, where asset tracking is located, and where the calibration csvs are located.

qct_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/CTDMO/CTDMO_Results'
cal_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/CTDMO/CTDMO_Cal'
asset_management_directory = '/home/andrew/Documents/OOI-CGSN/ooi-integration/asset-management/calibration/CTDMOR'

excel_spreadsheet = '/media/andrew/OS/Users/areed/Documents/Project_Files/Documentation/System/System Notebook/WHOI_Asset_Tracking.xlsx'
sheet_name = 'Sensors'

CTDMO = whoi_asset_tracking(excel_spreadsheet,sheet_name,instrument_class='CTDMO',whoi=True,series='R')
CTDMO.head(10)

for file in sorted(os.listdir(asset_management_directory)):
    sn = '37-' + file[13:18]
    cd = file[20:28]
    print('CTDMO-G  ' + sn + '  ' + file + '  ' + cd)

# **======================================================================================================================**
#
# First, get all the unique CTDMO Instrument UIDs:

uids = sorted(list(set(CTDMO['UID'])))

# Identify the QCT Testing documents associated with each individual instrument (the UID):

qct_dict = get_qct_files(CTDMO, qct_directory)
qct_dict

# Identify the calibration csvs stored in asset management which correspond to a particular instrument:

csv_dict = load_asset_management(CTDMO, asset_management_directory)
csv_dict

# Get the serial numbers for each CTDMO, and use those serial numbers to search for and return all of the relevant vendor documents for a particular instrument:

serial_nums = get_serial_nums(CTDMO, uids)

cal_dict = get_calibration_files(serial_nums, cal_directory)
cal_dict



# **========================================================================================================================**
# Print all of the CTDMO CSV files in order to retrieve all of the relevant files that need to be checked:

for uid in sorted(csv_dict.keys()):
    files = sorted(csv_dict[uid])
    sn = serial_nums[uid]
    for f in files:
        print('CTDMO-G' + '  ' + '37-' + sn + '  ' + f)

# **========================================================================================================================**
# With the individual files identified for the CTDMO Vendor documents, QCTs, and CSVs, we next get the full directory path to the files. This is necessary to load them:

# CSV file paths:

csv_paths = {}
for uid in sorted(csv_dict.keys()):
    paths = []
    for file in csv_dict.get(uid):
        path = generate_file_path(asset_management_directory, file, ext=['.csv','.ext'])
        paths.append(path)
    csv_paths.update({uid: paths})

csv_paths

# CAL file paths:

# Retrieve and save the full directory path to the calibration files
cal_paths = {}
for uid in sorted(cal_dict.keys()):
    paths = []
    for file in cal_dict.get(uid):
        path = generate_file_path(cal_directory, file, ext=['.zip','.cap', '.txt', '.log'])
        paths.append(path)
    cal_paths.update({uid: paths})

cal_paths

# QCT file paths:

qct_paths = {}
for uid in sorted(qct_dict.keys()):
    paths = []
    for file in qct_dict.get(uid):
        path = generate_file_path(qct_directory, file)
        paths.append(path)
    qct_paths.update({uid: paths})

qct_paths

# **========================================================================================================================**
# # Processing and Parsing the Calibration Coefficients
# With the associated vendor documents (cal files/vendor pdfs), QCT checkins (qct files), and calibration csvs (csv files), I want to be able to compare the following:
# * **(1)** That the calibration date matches between the different documents
# * **(2)** The file name agrees with the CTDMO UID and the calibration date
# * **(3)** The calibration coefficients all agree between the different reference documents and calibration csvs
# * **(4)** Identify when a calibration coefficient is incorrect, where to find it, and how to correct it
#
# The first step is to define a CTDMO Calibration parsing object. This object contains the relevant attributes and the functions necessary to open, read, and parse the CTDMO calibration coefficients and date, and write the calibration info to a properly-named CSV file.

# +
import re
import os
import string
import pandas as pd
import numpy as np
from wcmatch import fnmatch
from zipfile import ZipFile
import textract

class CTDMOCalibration():
    # Class that stores calibration values for CTDs.

    def __init__(self, uid):
        self.serial = ''
        self.uid = uid
        self.ctd_type = uid
        self.coefficients = {}
        self.date = {}

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
            'pcor':'CC_cpcor',
            'i': 'CC_i',
            'ptempa0': 'CC_ptempa0',
            'prange': 'CC_p_range',
            'ctcor': 'CC_ctcor',
            'tcor':'CC_ctcor',
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

            
    def mo_parse_pdf(self, filepath):
        """
        This function extracts the text from a given pdf file.
        Depending on if the text concerns calibration for 
        temperature/conductivity or pressure, it calls a further
        function to parse out the individual calibration coeffs.
    
        Args:
            filepath - the full directory path to the pdf file
                which it to be extracted and parsed.
        Calls:
            mo_parse_p(text, filepath)
            mo_parse_ts(text)
        Returns:
            self - a CTDMO calibration object with calibration
                coefficients parsed into the object calibration
                dictionary
        """
    
        text = textract.process(filepath, encoding='utf-8')
        text = text.decode('utf-8')
    
        if 'PRESSURE CALIBRATION DATA' in text:
            self.mo_parse_p(filepath)
    
        elif 'TEMPERATURE CALIBRATION DATA' or 'CONDUCTIVITY CALIBRATION DATA' in text:
            self.mo_parse_ts(text)
        
        else:
            pass
    

    def mo_parse_ts(self, text):
        """
        This function parses text from a pdf and loads the appropriate calibration
        coefficients for the temperature and conductivity sensors into the CTDMO 
        calibration object.
    
        Args:
            text - extracted text from a pdf page
        Returns:
            self - a CTDMO calibration object with either temperature or conductivity
                calibration values filled in the calibration coefficients dictionary
        Raises:
            Exception - if the serial number in the pdf text does not match the
                serial number parsed from the UID
        """
    
        keys = self.mo_coefficient_name_map.keys()
        for line in text.splitlines():
    
            if 'CALIBRATION DATE' in line:
                *ignore, cal_date = line.split(':')
                cal_date = pd.to_datetime(cal_date).strftime('%Y%m%d')
                self.date.update({len(self.date): cal_date})
        
            elif 'SERIAL NUMBER' in line:
                *ignore, serial_num = line.split(':')
                serial_num = serial_num.strip()
                if serial_num != self.serial:
                    raise Exception(f'Instrument serial number {serial_num} does not match UID serial num {self.serial}')
           
            elif '=' in line:
                key, *ignore, value = line.split()
                name = self.mo_coefficient_name_map.get(key.strip().lower())
                if name is not None:
                    self.coefficients.update({name: value.strip()})
            else:
                continue
            
            
    def mo_parse_p(self,filepath):
        """
        Function to parse the pressure calibration information from a pdf. To parse
        the pressure cal info requires re-extracting the text from the pdf file using
        tesseract-ocr rather than the basic pdf2text converter.
    
        Args:
            text - extracted text from a pdf page using pdf2text
            filepath - full directory path to the pdf file containing the pressure
                calibration info. This is the file which will be re-extracted.
        Returns
            self - a CTDMO calibration object with pressure calibration values filled
                in the calibration coefficients dictionary
        """
    
        # Now, can reprocess using tesseract-ocr rather than pdftotext
        ptext = textract.process(filepath, method='tesseract', encoding='utf-8')
        ptext = ptext.replace(b'\xe2\x80\x94',b'-')
        ptext = ptext.decode('utf-8')
        keys = list(self.mo_coefficient_name_map.keys())
        
        # Get the calibration date:
        for line in ptext.splitlines():
            if 'CALIBRATION DATE' in line:
                items = line.split()
                ind = items.index('DATE:')
                cal_date = items[ind+1]
                cal_date = pd.to_datetime(cal_date).strftime('%Y%m%d')
                self.date.update({len(self.date):cal_date})
            
            if 'psia S/N' in line:
                items = line.split()
                ind = items.index('psia')
                prange = items[ind-1]
                name = self.mo_coefficient_name_map.get('prange')
                self.coefficients.update({name: prange})
    
            # Loop through each line looking for the lines which contain
            # calibration coefficients
            if '=' in line:
                # Tesseract-ocr misreads '0' as O, and 1 as IL
                line = line.replace('O','0').replace('IL','1').replace('=','').replace(',.','.').replace(',','.')
                line = line.replace('L','1').replace('@','0').replace('l','1').replace('--','-')
                if '11' in line and 'PA2' not in line:
                    line = line.replace('11','1')
                items = line.split()
                for n, k in enumerate(items):
                    if k.lower() in keys:
                        try:
                            float(items[n+1])
                            name = self.mo_coefficient_name_map.get(k.lower())
                            self.coefficients.update({name: items[n+1]})
                        except:
                            pass
        if 'CC_ptcb2' not in list(self.mo_coefficient_name_map.keys()):
            self.coefficients.update({'CC_ptcb2': '0.000000e+000'})


    def mo_parse_cal(self, filepath):
        """
        Function to parse the .cal file for the CTDMO when a .cal file
        is available.
        """
    
        if not filepath.endswith('.cal'):
            raise Exception(f'Not a .cal filetype.')
    
        with open(filepath) as file:
            data = file.read()
        
        for line in data.splitlines():
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()
        
            if 'SERIALNO' in key:
                sn = value
                if self.serial != sn:
                    raise Exception(f'File serial number {sn} does not match UID {self.uid}')
                
            elif 'CALDATE' in key:
                cal_date = pd.to_datetime(value).strftime('%Y%m%d')
                self.date.update({len(self.date): cal_date})
            
            elif 'INSTRUMENT_TYPE' in key:
                ctd_type = value[-2:]
                if self.ctd_type != ctd_type:
                    raise Exception(f'CTD type {ctd_type} does not match uid {self.uid}.')
                
            else:
                if key.startswith('T'):
                    key = key.replace('T','')
                if key.startswith('C') and len(key)==2:
                    key = key.replace('C','')
                name = self.mo_coefficient_name_map.get(key.lower())
                if name is not None:
                    self.coefficients.update({name: value})
                    
        # Now we need to add in the range of the sensor
        name = self.mo_coefficient_name_map.get('prange')
        self.coefficients.update({name: '1450'})

                    
    def mo_parse_qct(self, filepath):
        """
        This function reads and parses the QCT file into
        the CTDMO calibration object.
    
        Args:
            filepath - full directory path and filename of
                the QCT file
        Returns:
        
        """
        
        with open(filepath,errors='ignore') as file:
            data = file.read()

        data = data.replace('<',' ').replace('>',' ')
        keys = self.mo_coefficient_name_map.keys()

        for line in data.splitlines():
            items = line.split()
    
            # If the line is empty, go to next line
            if len(items) == 0:
                continue
    
            # Check the serial number from the instrument
            elif 'SERIAL NO' in line:
                ind = items.index('NO.')
                sn = items[ind+1]
                if sn != self.serial:
                    raise Exception(f'Serial number {sn} in QCT document does not match uid serial number {self.serial}')
        
            # Check if the line contains the calibration date
            elif 'CalDate' in line:
                cal_date = pd.to_datetime(items[1]).strftime('%Y%m%d')
                self.date.update({len(self.date): cal_date})
        
            # Get the coefficient names and values
            elif items[0].lower() in keys:
                name = self.mo_coefficient_name_map[items[0].lower()]
                self.coefficients.update({name: items[1]})
        
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

# **========================================================================================================================**
# Below, I plan on going through each of the CTDMO UIDs, and parse the data into csvs. For source files which may contain multiple calibrations or calibration sources, I plan on extracting each of the calibrations to a temporary folder using the following structure:
#
#     <local working directory>/<temp>/<source>/data/<calibration file>
#     
# The separate calibrations will be saved using the standard UFrame naming convention with the following directory structure:
#
#     <local working directory>/<temp>/<source>/<calibration csv>
#     
# The csvs themselves will also be copied to the temporary folder. This allows for the program to be looking into the same temp directory for every CTDMO check.

import shutil

uids

# **====================================================================================================================**
# # START HERE

i = 0
uid = sorted(uids)[i]
uid

i = i + 1
uid = sorted(uids)[i]
for cpath in sorted(cal_paths[uid]):
    print(cpath.split('/')[-1])
print()
for qpath in qct_paths[uid]:
    if qpath is not None:
        print(qpath.split('/')[-1].split('.')[0])

temp_directory = '/'.join((os.getcwd(),'temp'))
if os.path.exists(temp_directory):
    shutil.rmtree(temp_directory)
    ensure_dir(temp_directory)

# **=======================================================================================================================**
# Copy the existing CTDMO asset management csvs to the local temp directory:

for filepath in csv_paths[uid]:
    savedir = '/'.join((temp_directory,'csv'))
    ensure_dir(savedir)
    savepath = '/'.join((savedir, filepath.split('/')[-1]))
    shutil.copyfile(filepath, savepath)

# ========================================================================================================================
# ### Parse and process the vendor documents
# The next step is to read and parse the vendor documents. This is a more difficult challenge, since for CTDMOs the vendor documents are retained mostly as pdf files. While the pdf files are parseable, there is an added complication in that the forms have changed over time, with sometimes the T/S/P calibration pdfs combined into a single file, whereas other times they are separated into individual files. Furthermore, the files are often zipped into a single folder. So, I have the following possible vendor documents:
# * **(1)** A .cal file - this is the easiest to read and parse, in a similar format to the CTDBP .cal files
# * **(2)** A combinded pdf - this is the most difficult format. Need to separate out the different pages which each separately contain either the temperature calibration info, the conductivity calibration info, or the pressure calibration info.
# * **(3)** Separate pdfs - this is a simpler pdf reading schematic, where I know a priori which particular "page" will contain relevant calibration info. 

# There are a couple of different pdf readers that I can use:
# 1. PyPDF2
# 2. PDFMiner
# 3. Textract

import PyPDF2

# PyPDF2 does not work to extract text from the CTDMO combined pdf file document. Neither does the straightforward PDFMiner application. We will have to use OCR and textract to parse the pdf forms.

# When parsing the pdf file, it appears that the built-in method of pdf2text does the best job at parsing the forms, particularly the temperature and conductivity coefficients. The pressure calibration coefficients are not as well parsed, due to the positioning of the image.
#
# This means that I'm going to split and use two different methods for getting the calibration coefficients depending on what the calibration is for, i.e. T/S/P. For T and S, I'll use the built-in method for extracting text. For the pressure, I'll use the tesseract OCR approach.

# ========================================================================================================================
# ### Preprocessing the Vendor Files
# In order to automate the parsing of the CTDMO calibration coefficients from pdf files into csv files that can be read by Python requires a bit of preprocessing. In particular, the following steps are taken to make parsing the files:
# * **(1)** Copy or extract the vendor calibration files from the Vault location to a local temp directory
# * **(2)** Iterate over the available pdfs and split multipage pdfs into single page pdfs and append _page_ to the file
# * **(3)** Once the pdfs have been split, they are ready to be parsed by the CTDMO object parsers.

# Now, write a function to copy over the file
cal_paths[uid]

# Copy the vendor pdf files to a local temporary directory:

for filepath in cal_paths[uid]:
    folder, *ignore = filepath.split('/')[-1].split('.')
    savedir = '/'.join((temp_directory,'data',folder))
    ensure_dir(savedir)
    
    if filepath.endswith('.zip'):
        with ZipFile(filepath,'r') as zfile:
            for file in zfile.namelist():
                zfile.extract(file, path=savedir)
    else:
        shutil.copy(filepath, savedir)

for file in os.listdir('/'.join((temp_directory,'data',folder))):
    if os.path.isdir('/'.join((temp_directory,'data',folder,file))):
        for subfile in os.listdir('/'.join((temp_directory,'data',folder,file))):
            src = '/'.join((temp_directory,'data',folder,file,subfile))
            dst = '/'.join((temp_directory,'data',folder,subfile))
            shutil.move(src,dst)
        shutil.rmtree('/'.join((temp_directory,'data',folder,file)))

# +
folders = os.listdir('/'.join((os.getcwd(),'temp','data')))
rmfile = None
for folder in folders:
    filepath = '/'.join((os.getcwd(),'temp','data',folder))
    
    if any([file for file in os.listdir(filepath) if file.endswith('.cal')]):
        pass
    else:
        files = [file for file in os.listdir(filepath) if 'SERVICE REPORT' not in file]
        
        try:
            
            for file in files:
                trip = False
                inputpath = '/'.join((filepath,file))
                inputpdf = PyPDF2.PdfFileReader(inputpath, 'rb')

                for i in range(inputpdf.numPages):
                    output = PyPDF2.PdfFileWriter()
                    output.addPage(inputpdf.getPage(i))
                    filename = '_'.join((inputpath.split('.')[0], 'page', str(i)))
                    with open(filename+'.pdf', "wb") as outputStream:
                        output.write(outputStream)
        except:
            rmfile = filepath
            print(f'Cannot reformat {filepath}')
            
if rmfile is not None:
    shutil.rmtree(rmfile)
# -

os.listdir(temp_directory+'/data')

# The next step is to iterate over the vendor calibration files and extract the calibration coefficients from the files. This is done by starting an instance of the CTDMO calibration object, check if any of the calibration data is stored as a .cal file, if no .cal file loop over the other files looking for _page_ files which indicates that the pdf file has been prepped.

datadir = os.path.abspath('/'.join((os.getcwd(),'temp','data')))
for folder in os.listdir(datadir):
    # Okay, now start generating calibration csvs
    ctdmo = CTDMOCalibration(uid)
    files = [file for file in os.listdir('/'.join((datadir,folder)))]
    if any([file for file in files if file.endswith('.cal')]):
        for file in files:
            if file.endswith('.cal'):
                ctdmo.mo_parse_cal('/'.join((datadir,folder,file)))
    else:
        for file in files:
            if '_page_' in file:
                try:
                    ctdmo.mo_parse_pdf('/'.join((datadir,folder,file)))
                except:
                    print(f'Parsing failed for {file}')
                    
    savedir = '/'.join((os.getcwd(),'temp','cal'))
    ensure_dir(savedir)
    try:
        ctdmo.write_csv(savedir)
    except:
        pass

# Check that the calibration object properly loaded all of the calibration coefficients, serial number, calibration date, etc., and wrote the appropriate csv file.

os.listdir(temp_directory+'/cal')

# **=======================================================================================================================**
# Next, we need to parse the QCT files and check that they have been successfully saved to a csv file. There should be 24 coefficients. Similarly, check the instrument serial number, the calibration date (may be more than one b/c separate calibration dates for T, S, and P sensors), and the type (for CTDMOs should be 37).

for filepath in qct_paths[uid]:
    savedir = '/'.join((temp_directory,'qct'))
    ensure_dir(savedir)
    if filepath is not None:
        try:
            ctdmo = CTDMOCalibration(uid)
            ctdmo.mo_parse_qct(filepath)
            ctdmo.write_csv(savedir)
        except:
            print(f'Failed to parse {filepath}')
    else:
        pass

qct_paths[uid]

os.listdir('/'.join((temp_directory,'qct')))


# **========================================================================================================================**
# ### Compare results
# Now, with QCT files parsed into csvs which follow the UFrame format, I can load both the QCT and the calibratoin csvs into pandas dataframes, which will allow element by element comparison in relatively few lines of code.

def get_file_date(x):
    x = str(x)
    ind1 = x.index('__')
    ind2 = x.index('.')
    return x[ind1+2:ind2]


# Load the calibration csvs:

# Now we want to compare dataframe
csv_files = pd.DataFrame(sorted(os.listdir('temp/csv')),columns=['csv'])
csv_files['cal date'] = csv_files['csv'].apply(lambda x: get_file_date(x))
csv_files.set_index('cal date',inplace=True)
csv_files

# Load the QCT csvs:

# Now we want to compare dataframe
qct_files = pd.DataFrame(sorted(os.listdir('temp/qct')),columns=['qct'])
qct_files['cal date'] = qct_files['qct'].apply(lambda x: get_file_date(x))
qct_files.set_index('cal date',inplace=True)
qct_files

# Load the calibration csvs:

cal_files = pd.DataFrame(sorted(os.listdir('temp/cal')),columns=['cal'])
cal_files['cal date'] = cal_files['cal'].apply(lambda x: get_file_date(x))
cal_files.set_index('cal date',inplace=True)
cal_files

# Combine the dataframes into one in order to know which csv files to compare and check calibration dates.

df_files = csv_files.join(qct_files,how='outer').join(cal_files,how='outer').fillna(value='-999')
df_files

# If the filename is wrong, the calibration coefficient checker will not manage to compare the results. Consequently, we'll make a local copy of the wrong file to a new file with the correct name, and then run the calibration coefficient checker.

d1 = str(20161125)
d2 = str(20150618)

src = f'temp/csv/{uid}__{d1}.csv'
dst = f'temp/csv/{uid}__{d2}.csv'

shutil.move(src,dst)

os.listdir('temp/csv')

# Reload the data so that all files are uniformly named:

csv_files = pd.DataFrame(sorted(os.listdir('temp/csv')),columns=['csv'])
csv_files['cal date'] = csv_files['csv'].apply(lambda x: get_file_date(x))
csv_files.set_index('cal date',inplace=True)
csv_files

df_files = csv_files.join(qct_files,how='outer').join(cal_files,how='outer').fillna(value='-999')
df_files

caldates = df_files.index
for i in caldates:
    print(i)

for cpath in sorted(cal_paths[uid]):
    print(cpath.split('/')[-1])

for qpath in qct_paths[uid]:
    if qpath is not None:
        print(qpath.split('/')[-1].split('.')[0])

# With uniformly named csv files, we can now directly compare different calibration coefficient sources for the CTDMO.

# This table tells us that, for the csv CGINS-CTDMOG-11596__20150608.csv, I am missing a QCT document and vendor doc which could verify the calibration coefficients. Next, for the files I can compare, I want to go through and check each calibration coefficient.

# **========================================================================================================================**
# Okay, I want to check the following in the comparison between the CSV files contained in Asset Management, the QCT checkins, and the vendor docs:
# 1. Do the calibration coefficients match exactly?
# 2. Do the calibration coefficients match to within 0.001%?

files = sorted(os.listdir(temp_directory+'/csv'))
files

i = 0
fname = files[i]
dfcsv = pd.read_csv(temp_directory+'/csv/'+fname)
dfcsv.sort_values(by='name', inplace=True)
dfcsv.reset_index(inplace=True)
dfcsv.drop(columns='index',inplace=True)

dfcal = pd.read_csv(temp_directory+'/cal/'+fname)
dfcal.sort_values(by='name', inplace=True)
dfcal.reset_index(inplace=True)
dfcal.drop(columns='index',inplace=True)

dfqct = pd.read_csv(temp_directory+'/qct/'+fname)
dfqct.sort_values(by='name', inplace=True)
dfqct.reset_index(inplace=True)
dfqct.drop(columns='index', inplace=True)

cal_close = np.isclose(dfcsv['value'], dfcal['value'])
cal_check = dfcsv == dfcal

dfcsv[cal_check['value'] == False], dfcal[cal_check['value'] == False]

dfcsv[cal_close == False], dfcal[cal_close == False]

qct_check = dfcsv == dfqct
qct_close = np.isclose(dfcsv['value'], dfqct['value'])

dfcsv[qct_check['value'] == False], dfqct[qct_check['value'] == False]

set(dfqct[qct_check['value'] == False]['name'])


dfcsv[qct_close == False], dfqct[qct_close == False]


def check_exact_coeffs(coeffs_dict):
    
    # Part 1: coeff by coeff comparison between each source of coefficients
    keys = list(coeffs_dict.keys())
    comparison = {}
    for i in range(len(keys)):
        names = (keys[i], keys[i - (len(keys)-1)])
        check = len(coeffs_dict.get(keys[i])['value']) == len(coeffs_dict.get(keys[i - (len(keys)-1)])['value'])
        if check:
            compare = np.equal(coeffs_dict.get(keys[i])['value'], coeffs_dict.get(keys[i - (len(keys)-1)])['value'])
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


def check_relative_coeffs(coeffs_dict):
    
    # Part 1: coeff by coeff comparison between each source of coefficients
    keys = list(coeffs_dict.keys())
    comparison = {}
    for i in range(len(keys)):
        names = (keys[i], keys[i - (len(keys)-1)])
        check = len(coeffs_dict.get(keys[i])['value']) == len(coeffs_dict.get(keys[i - (len(keys)-1)])['value'])
        if check:
            compare = np.isclose(coeffs_dict.get(keys[i])['value'], coeffs_dict.get(keys[i - (len(keys)-1)])['value'], rtol=1e-5)
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


exact_match = {}
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
    mask = check_exact_coeffs(coeffs_dict)
    
    # Part 3: get the calibration coefficients are wrong
    # and show them
    fname = df_files.loc[cal_date]['csv']
    if fname == '-999':
        incorrect = 'No csv file.'
    else:
        incorrect = coeffs_dict['csv'][mask == False]
    exact_match.update({fname:incorrect})

relative_match = {}
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
    mask = check_relative_coeffs(coeffs_dict)
    
    # Part 3: get the calibration coefficients are wrong
    # and show them
    fname = df_files.loc[cal_date]['csv']
    if fname == '-999':
        incorrect = 'No csv file.'
    else:
        incorrect = coeffs_dict['csv'][mask == False]
    relative_match.update({fname:incorrect})

os.listdir(temp_directory+'/csv')

for key in sorted(exact_match.keys()):
    if key != '-999':
        print(', '.join((ind for ind in exact_match[key].index.values)))

for key in sorted(relative_match.keys()):
    if key != '-999':
        print(', '.join((ind for ind in relative_match[key].index.values)))

qct

# **========================================================================================================================**
# Now we need to check that the calibration coefficients for each CTDMO csv have the same number of significant digits as are reported on the vendor PDFs. For the CTDMO, the vendor reports to six significant figures.

csv_paths

uid = uids[0]
uid

CSV = pd.read_csv(csv_paths[uid][0])
CSV

for val in CSV['value']:
    print("{:.6e}".format(val))

print("{:.2e}".format(0.00253))

import math


def to_precision(x,p):
    """
    Returns a string representation of x formatted with a precision of p,
    following the toPrecision method from javascript. This implementation
    is based on example code from www.randlet.com.
    
    Args:
        x - number to format to a specified precision
        p - the specified precision for the number x
    Returns:
    
    """
    
    # First check if x is a string
    if type(x) is not float:
        x = float(x)
        
    # Next, check if p is an int and if not, convert to int
    if type(p) is not int:
        p = int(p)
    
        
    if x == 0.:
        return "0." + "0"*(p-1)
    
    out = []
    
    if x < 0:
        out.append("-")
        x = -x
        
    e = int(math.log10(x))
    tens = math.pow(10, e - p + 1)
    n = math.floor(x / tens)
    
    if n < math.pow(10, p - 1):
        e = e - 1
        tens = math.pow(10, e - p + 1)
        n = math.floor(x / tens)
        
    if abs((n + 1.) * tens - x) <= abs(n * tens - x):
        n = n + 1
        
    if n >= math.pow(10, p):
        n = n / 10.
        e = e + 1
        
    m = "%.*g" % (p, n)
    
    if e < -2 or e >= p:
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p - 1):
        out.append(m)
    elif e >= 0:
        out.append(m[:e+1])
        if (e + 1) < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)
        
    return "".join(out)
