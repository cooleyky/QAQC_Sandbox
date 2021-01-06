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

# # OPTAA METADATA REVIEW
#
# This notebook describes the process for reviewing the calibration coefficients for the OPTAAs. The purpose is to check the calibration coefficients contained in the CSVs stored within the asset management repository on GitHub, which are the coefficients utilized by OOI-net for calculating data products, against the different available sources of calibration information to identify when errors were made during entering the calibration csvs. This includes checking the following information:
# 1. The calibration date - this information is stored in the filename of the csv
# 2. Calibration source - identifying all the possible sources of calibration information, and determine which file should supply the calibration info
# 3. Calibration coeffs - checking the accuracy and precision of the numbers stored in the calibration coefficients
# 4. Calibration .ext files - arrays which are referenced by the main csv files and contain more calibration values
#
# The OPTAAs contains 8 different calibration coefficients to check. Five of the coefficients are arrays of varying lengths of values. Additionally, there are two .ext files which are referenced by the calibration csv. These .ext files are separate arrays of values whose name and values also need to be checked. The possible calibration source for the OPTAAs are vendor calibration (.dev) files. The QCT checkin, pre- and post-deployment files do not contain all the necessary calibration information in order to fully check the asset management csvs.
#
# **========================================================================================================================**
#
# The first step is to load relevant packages:

import csv
import re
import os
import shutil
import numpy as np
import pandas as pd

from utils import *

# **=======================================================================================================================**
# Define the directories where the QCT, Pre, and Post deployment document files are stored, where the vendor documents are stored, where asset tracking is located, and where the calibration csvs are located.

doc_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/OPTAA/OPTAA_Results/'
cal_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/OPTAA/OPTAA_Cal/'
asset_management_directory = '/home/andrew/Documents/OOI-CGSN/ooi-integration/asset-management/calibration/OPTAAD/'

excel_spreadsheet = '/media/andrew/OS/Users/areed/Documents/Project_Files/Documentation/System/System Notebook/WHOI_Asset_Tracking.xlsx'
sheet_name = 'Sensors'

OPTAA = whoi_asset_tracking(spreadsheet=excel_spreadsheet,sheet_name=sheet_name,instrument_class='OPTAA')

# **=======================================================================================================================**
# Now, I want to load all the calibration csvs and group them by UID:

uids = sorted( list( set(OPTAA['UID']) ) )

csv_dict = {}
asset_management = os.listdir(asset_management_directory)
for uid in uids:
    files = [file for file in asset_management if uid in file]
    csv_dict.update({uid: sorted(files)})

csv_dict;

# **=======================================================================================================================**
# Get the serial numbers of the instruments and match them to the UIDs:

serial_dict = {}
for uid in uids:
    sn = OPTAA[OPTAA['UID'] == uid]['Supplier\nSerial Number']
    serial_dict.update({uid: str(sn.iloc[0])})    

serial_dict;

# **=======================================================================================================================**
# The OPTAA QCT capture files are stored with the following Document Control Numbers (DCNs): 3305-00113-XXXXX. Most are storead as **.dat** files which are easy to parse and decode (same formatting as the **.dev** files). However, some are stored as Excel (**.xlsx**) files, which are much trickier to parse.
#
#
#

files = [file for file in os.listdir(doc_directory) if 'A' in file or 'B' in file]
qct_files = []
for file in files:
    if '113' in file:
        qct_files.append(file)
    else:
        pass

qct_dict = {}
for uid in uids:
    # Get the QCT Document numbers from the asset tracking sheet
    OPTAA['UID_match'] = OPTAA['UID'].apply(lambda x: True if uid in x else False)
    qct_series = OPTAA[OPTAA['UID_match'] == True]['QCT Testing']
    qct_series = list(qct_series.iloc[0].split('\n'))
    qct_dict.update({uid:qct_series})
qct_paths = {}
for uid in sorted(qct_dict.keys()):
    paths = []
    for file in qct_dict.get(uid):
        path = generate_file_path(doc_directory, file, ext=['.dat','.xlsx'])
        paths.append(path)
    qct_paths.update({uid: paths})

qct_paths;

# **=======================================================================================================================**
# Get the pre-deployment capture files which have the following DCN: 3305-00313-XXXXX. However, the OPTAA Predeployment procedure does not involve capturing any calibration information. Thus, we do not have any relevant calibration values to test the calibration csvs against.

csv_paths = {}
for uid in sorted(csv_dict.keys()):
    paths = []
    for file in csv_dict.get(uid):
        path = generate_file_path(asset_management_directory, file, ext=['.csv','.ext'])
        paths.append(path)
    csv_paths.update({uid: paths})

csv_paths;

# **=======================================================================================================================** Find and return the calibration files which contain vendor supplied calibration information. This is achieved by searching the calibration directories and matching serial numbers to UIDs:

serial_nums = get_serial_nums(OPTAA, uids)

cal_dict = get_calibration_files(serial_nums, cal_directory)

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
# Above, we have worked through identifying and mapping the calibration files, pre-deployment files, and post-deployment files to the individual instruments through their UIDs and serial numbers. The next step is to open the relevant files and parse out the calibration coefficients. This will require writing a parser for the NUTNRs, including sub-functions to handle the different characteristics of the ISUS and SUNA instruments.
#
# Start by opening the calibration files and read the data:

# +
from datetime import datetime, timedelta

def from_excel_ordinal(ordinal, _epoch0=datetime(1899, 12, 31)):
    if ordinal > 59:
        ordinal -= 1  # Excel leap year bug, 1900 is not a leap year!
    return (_epoch0 + timedelta(days=ordinal)).replace(microsecond=0)


# -

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

# **=======================================================================================================================**
# # Source Loading of Calibration Coefficients
# With an OPTAA Calibration object created, we can now begin parsing the different calibration sources for each OPTAA. We will then compare all of the calibration values from each of the sources, checking for any discrepancies between them.

# Below, I plan on going through each of the OPTAA UIDs, and parse the data into csvs. For sources which contain multiple sources, I plan on extracting each of the calibrations to a temporary folder using the following structure:
#
#     <local working directory>/<temp>/<source>/data/<calibration file>
#     
# The separate calibrations will be saved using the standard UFrame naming convention with the following directory structure:
#
#     <local working directory>/<temp>/<source>/<calibration csv>
#     
# The csvs themselves will also be copied to the temporary folder. This allows for the program to be looking into the same temp directory for every NUTNR check.

uid = uids[26]
uid

temp_directory = '/'.join((os.getcwd(),'temp'))
# Check if the temp directory exists; if it already does, purge and rewrite
if os.path.exists(temp_directory):
    shutil.rmtree(temp_directory)
    ensure_dir(temp_directory)

# Copy the existing csvs from asset management to the temp directory:

for path in csv_paths[uid]:
    savedir = '/'.join((temp_directory,'csv'))
    ensure_dir(savedir)
    savepath = '/'.join((savedir, path.split('/')[-1]))
    shutil.copyfile(path, savepath)

os.listdir(temp_directory+'/csv')

# **=======================================================================================================================**
# Load the calibration coefficients from the vendor calibration source files. Start by extracting or copying them to the source data folder in the temporary directory.

cal_paths[uid]

# Extract the calibration zip files to the local temp directory:

for path in cal_paths[uid]:
    with ZipFile(path) as zfile:
        files = [name for name in zfile.namelist() if name.lower().endswith('.dev')]
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
        optaa = OPTAACalibration(uid)
        optaa.load_cal(filepath)
        optaa.write_csv(savedir)

# **=======================================================================================================================**
# Load the QCT checkin for comparison with the calibration source files. Start by extracting or copying them to the source data folder in the temporary directory.

qct_paths[uid]

# For QCT documents which were saved as excel documents (**.xlsx**) files, I need to rewrite them to the local temp data directory instead as tab-delimited csv files rather than in excel workbook format. The function below handles the conversion.

# +
import xlrd
import csv

def csv_from_excel(excelpath, csvpath):

    if not excelpath.endswith('.xlsx'):
        raise FileExistsError("Must be an excel workbook.")
        
    wb = xlrd.open_workbook(excelpath)
    sh = wb.sheet_by_index(0)
    csv_file = open(csvpath, 'w')
    wr = csv.writer(csv_file, delimiter='\t')

    for rownum in range(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    csv_file.close()


# -

# With the above conversion function for excel to csv, we can iterate through the QCT documents for each OPTAA, and either copy them if in the **.dat** format or convert if in **.xlsx** format. Also note that the QCT documents have the following features:
# 1. No calibration temperature in the header
# 2. Acquisition data after the calibration data matrix. 
#
# This will require writing a separate parser for the QCT

for path in qct_paths[uid]:
    savedir = '/'.join((temp_directory,'qct','data'))
    ensure_dir(savedir)
    if path.endswith('.xlsx'):
        filename = path.split('/')[-1].replace('xlsx','dat')
        savepath = savedir + '/' + filename
        csv_from_excel(path, savepath)
    else:
        shutil.copy(path, savedir)

os.listdir(temp_directory+'/qct/data')

# Write the QCT calibration files to csvs following the UFrame convention:
#
# However, the QCT checkin does not contain the necessary information to produce the requisite calibration file. 

savedir = '/'.join((temp_directory,'qct'))
ensure_dir(savedir)
# Now parse the calibration coefficients
for dirpath, dirnames, filenames in os.walk('/'.join((temp_directory,'qct','data'))):
    for file in filenames:
        filepath = os.path.join(dirpath, file)
        # With the filepath for the given calibration retrived, I can now start an instance of the NUTNR Calibration
        # object and begin parsing the coefficients
        optaa = OPTAACalibration(uid)
        optaa.load_qct(filepath)
        optaa.write_csv(savedir)


# **=======================================================================================================================**
# # Calibration Coefficient Comparison
# We have now successfully parsed the calibration files from all the possible sources: the vendor calibration files, the pre-deployments files, and the post-deployment files. Furthermore, we have saved csvs in the UFrame format for all of these calibrations. Now, we want to load those csvs into pandas dataframes, which allow for easy element-by-element comparison of calibration coefficients.

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
cal_files

iloc = 6

cal_files.drop([iloc],inplace=True)

cal_files['cal date'] = cal_files['cal'].apply(lambda x: get_file_date(x))
cal_files.set_index('cal date',inplace=True)

df_files = csv_files.join(cal_files,how='outer').fillna(value='-999')

df_files

# Need to compare the dates in the CSV and QCT files against the **.dev** CAL files, which contain the date that the OPTAA itself was calibrated.

# CSV files:

sn = '00257'
d1 = '20161011'
d2 = '20160826'

src = 'temp/csv/' + f'CGINS-OPTAAD-{sn}__{d1}.csv'
dst = 'temp/csv/' + f'CGINS-OPTAAD-{sn}__{d2}.csv'
shutil.move(src, dst)

src = 'temp/csv/' + f'CGINS-OPTAAD-{sn}__{d1}__CC_taarray.ext'
dst = 'temp/csv/' + f'CGINS-OPTAAD-{sn}__{d2}__CC_taarray.ext'
shutil.move(src, dst)

src = 'temp/csv/' + f'CGINS-OPTAAD-{sn}__{d1}__CC_tcarray.ext'
dst = 'temp/csv/' + f'CGINS-OPTAAD-{sn}__{d2}__CC_tcarray.ext'
shutil.move(src, dst)

# Reload the csv files in order to perform the comparison:

# Now we want to compare dataframe
csv_files = pd.DataFrame(sorted(os.listdir('temp/csv')),columns=['csv'])
csv_files['cal date'] = csv_files['csv'].apply(lambda x: get_file_date(x))
csv_files.set_index('cal date',inplace=True)

# Now we want to compare dataframe
cal_files = pd.DataFrame(sorted(os.listdir('temp/cal')),columns=['cal'])
cal_files.drop([iloc],inplace=True)
cal_files['cal date'] = cal_files['cal'].apply(lambda x: get_file_date(x))
cal_files.set_index('cal date',inplace=True)

df_files = csv_files.join(cal_files,how='outer').fillna(value='-999')

df_files

# **=======================================================================================================================**
# Now, with the csv files correctly named, we can load the info into pandas dataframe which will allow for the direct comparison of calibration coefficients.

dt = '20160826'

a = f'CGINS-OPTAAD-{sn}__{dt}.csv'
b = f'CGINS-OPTAAD-{sn}__{dt}__CC_taarray.ext'
c = f'CGINS-OPTAAD-{sn}__{dt}__CC_tcarray.ext'

# CSV

CSV = pd.read_csv('temp/csv/'+a)
with open('temp/csv/'+b) as file:
    csv_ta = file.read()
    CSV_ta = []
    for line in csv_ta.splitlines():
        line = [float(x) for x in line.split(',')]
        CSV_ta.append(line)
with open('temp/csv/'+c) as file:
    csv_tc = file.read()
    CSV_tc = []
    for line in csv_tc.splitlines():
        line = line.replace('[','').replace(']','')
        line = [float(x) for x in line.split(',')]
        CSV_tc.append(line)

# DEV

DEV = pd.read_csv('temp/cal/'+a)
with open('temp/cal/'+b) as file:
    dev_ta = file.read()
    DEV_ta = []
    for line in dev_ta.splitlines():
        line = [float(x) for x in line.split(',')]
        DEV_ta.append(line)
with open('temp/cal/'+c) as file:
    dev_tc = file.read()
    DEV_tc = []
    for line in dev_tc.splitlines():
        line = [float(x) for x in line.split(',')]
        DEV_tc.append(line)


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


CSV['value'] = CSV['value'].apply(lambda x: reformat_arrays(x))

DEV['value'] = DEV['value'].apply(lambda x: reformat_arrays(x))

# Now compare the results:

CSV = CSV.sort_values(by='name').reset_index().drop(columns='index')
CSV

DEV

DEV['notes'].iloc[0]

np.equal(DEV,CSV)

np.all(np.equal(DEV_ta,CSV_ta))

for j,k in enumerate(DEV_ta):
    check = DEV_ta[j] == CSV_ta[j]
    if not check:
        for m,n in enumerate(DEV_ta[j]):
            check2 = np.equal(DEV_ta[j][m],CSV_ta[j][m])
            if not check2:
                print(str(j)+','+str(m)+': ' + 'DEV - '+str(DEV_ta[j][m]) + ' CSV - '+str(CSV_ta[j][m]) )


np.all(np.equal(DEV_tc,CSV_tc))

for j,k in enumerate(DEV_tc):
    check = DEV_tc[j] == CSV_tc[j]
    if not check:
        for m,n in enumerate(DEV_tc[j]):
            check2 = DEV_tc[j][m] == CSV_tc[j][m]
            if not check2:
                print(str(j)+','+str(m)+': ' + 'DEV - '+str(DEV_tc[j][m]) + ' CSV - '+str(CSV_tc[j][m]) )


DEV_tc 

CSV_tc

optaa.nbins+5

optaa.nbins
