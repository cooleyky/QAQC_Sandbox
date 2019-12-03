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

# # NUTNR METADATA REVIEW
#
# This notebook describes the process for reviewing the calibration coefficients for the NUTNRs, including both the ISUS and SUNA models. The purpose is to check the calibration coefficients contained in the CSVs stored within the asset management repository on GitHub, which are the coefficients utilized by OOI-net for calculating data products, against the different available sources of calibration information to identify when errors were made during entering the calibration csvs. This includes checking the following information:
# 1. The calibration date - this information is stored in the filename of the csv
# 2. Calibration source - identifying all the possible sources of calibration information, and determine which file should supply the calibration info
# 3. Calibration coeffs - checking the accuracy and precision of the numbers stored in the calibration coefficients
#
# The NUTNRs contains 7 different calibration coefficients to check. Two of the calibration coefficients are fixed constants. Four of the coefficients are arrays of 35 values. The possible calibration sources for the NUTNRs are vendor calibration (.cal) files, as well as pre- and post-deployment calibrations (.cal files). A complication is that the calibration documents often contain multiple .cal files. However, if there are multiple .cal files, they are sequentially appended with the alphabet. Consequently, we identify the latest .cal file based on the appended letter to the file.
#
# **========================================================================================================================**

# Import likely important packages, etc.
import sys, os, re
import numpy as np
import pandas as pd
import shutil
from zipfile import ZipFile

from utils import *

# **====================================================================================================================**
# Define the directories where the QCT, Pre, and Post deployment document files are stored, where the vendor documents are stored, where asset tracking is located, and where the calibration csvs are located.

doc_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/NUTNR/NUTNR_Results/'
cal_directory = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/NUTNR/NUTNR_Cal/'
asset_management_directory = '/home/andrew/Documents/OOI-CGSN/ooi-integration/asset-management/calibration/NUTNRB/'

excel_spreadsheet = '/media/andrew/OS/Users/areed/Documents/Project_Files/Documentation/System/System Notebook/WHOI_Asset_Tracking.xlsx'
sheet_name = 'Sensors'

NUTNR = whoi_asset_tracking(spreadsheet=excel_spreadsheet,sheet_name=sheet_name,instrument_class='NUTNR',series='B')
NUTNR

# **======================================================================================================================**
# Now, I want to load all the calibration csvs and group them by UID:

uids = sorted( list( set(NUTNR['UID']) ) )

csv_dict = {}
asset_management = os.listdir(asset_management_directory)
for uid in uids:
    files = [file for file in asset_management if uid in file]
    csv_dict.update({uid: sorted(files)})

# **=======================================================================================================================**
# Get the serial numbers of the instruments and match them to the UIDs:

serial_dict = {}
for uid in uids:
    sn = NUTNR[NUTNR['UID'] == uid]['Supplier\nSerial Number']
    serial_dict.update({uid: str(sn.iloc[0])})    

# **=======================================================================================================================**
# Get the QCT capture files with the following Document Control Numbers (DCNs):
# * ISUS: 3305-00108-XXXXX-A
# * SUNA: 3305-00127-XXXXX-A
#
# For the NUTNRs, the QCT files do not contain any calibration information. Rather, the calibration information is contained in separate **.CAL** files, which are updated each time. 

files = [file for file in os.listdir(doc_directory) if 'A' in file]
qct_files = []
for file in files:
    if '108' in file or '127' in file:
        qct_files.append(file)
    else:
        pass

# **=======================================================================================================================**
# Get the pre-deployment capture files, which should contain **.CAL** files, with the following DCNs:
# * ISUS: 3305-00308-XXXXX-A
# * SUNA: 3305-00327-XXXXX-A

files = [file for file in os.listdir(doc_directory) if 'A' in file]
pre_files = []
for file in files:
    if '308' in file or '327' in file:
        pre_files.append(file)

# Open the Pre-deployment files and get the instrument serial number to match the Pre-deployment DCN to an individual insturment.

pre_paths = []
predeployment = {}
for file in pre_files:
    path = generate_file_path(doc_directory, file, ext=['.zip'])
    with ZipFile(path) as zfile:
        cal_files = [file for file in zfile.namelist() if file.lower().endswith('.cal')]
        if len(cal_files) > 0:
            data = zfile.read(cal_files[0]).decode('ascii')
            lines = data.splitlines()
            _, items, *ignore = lines[0].split(',')
            inst, sn, *ignore = items.split()
            sn = sn.lstrip('0')
            if inst == 'SUNA':
                sn = 'NTR-'+sn
    if predeployment.get(sn) is None:
        predeployment.update({sn: [file]})
    else:
        predeployment[sn].append(file)

predeployment

# Based on the serial numbers, link the instrument uids to their pre-deployment files:

pre_dict = {}
for uid in sorted(serial_dict.keys()):
    sn = serial_dict.get(uid)
    if predeployment.get(sn) is not None:
        pre_dict.update({uid: sorted(predeployment.get(sn))})
    else:
        pre_dict.update({uid: None})

pre_dict

# **=======================================================================================================================**
# Repeat the Pre-deployment process with the post-deployment files. The DCNs are:
# * ISUS: 3305-00508-XXXXX-A
# * SUNA: 3305-00527-XXXXX-A

files = [file for file in os.listdir(doc_directory) if 'A' in file]
post_files = []
for file in files:
    if '508' in file or '527' in file:
        post_files.append(file)

post_paths = []
postdeployment = {}
for file in post_files:
    path = generate_file_path(doc_directory, file, ext=['.zip'])
    with ZipFile(path) as zfile:
        cal_files = [file for file in zfile.namelist() if file.lower().endswith('.cal')]
        if len(cal_files) > 0:
            data = zfile.read(cal_files[0]).decode('ascii')
            lines = data.splitlines()
            _, items, *ignore = lines[0].split(',')
            inst, sn, *ignore = items.split()
            sn = sn.lstrip('0')
            if inst == 'SUNA':
                sn = 'NTR-'+sn
    if postdeployment.get(sn) is None:
        postdeployment.update({sn: [file]})
    else:
        postdeployment[sn].append(file)

post_dict = {}
for uid in sorted(serial_dict.keys()):
    sn = serial_dict.get(uid)
    post_dict.update({uid: postdeployment.get(sn)})

post_dict

# **=======================================================================================================================**
# Now, we need to identify the full paths to the relevant files
#

# Return the filepaths for the csv files:

csv_paths = {}
for uid in sorted(csv_dict.keys()):
    paths = []
    for file in csv_dict.get(uid):
        path = generate_file_path(asset_management_directory, file, ext=['.csv'])
        paths.append(path)
    csv_paths.update({uid: paths})

csv_paths

# Return the filepaths for the predeployment files:

pre_paths = {}
for uid in sorted(pre_dict.keys()):
    paths = []
    if pre_dict.get(uid) is not None:
        for file in pre_dict.get(uid):
            path = generate_file_path(doc_directory, file)
            paths.append(path)
        pre_paths.update({uid: paths})
    else:
        pass

pre_paths;

# Return the filepaths for the post-deployment files:

post_paths = {}
for uid in sorted(post_dict.keys()):
    paths = []
    if post_dict.get(uid) is not None:
        for file in post_dict.get(uid):
            path = generate_file_path(doc_directory, file)
            paths.append(path)
        post_paths.update({uid: paths})
    else:
        post_paths.update({uid: None})

post_paths;

# **=======================================================================================================================** Find and return the calibration files which contain vendor supplied calibration information. This is achieved by searching the calibration directories and matching serial numbers to UIDs:

serial_nums = get_serial_nums(NUTNR, uids)

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
                
                self.coefficients['CC_wl'].append(wl)
                self.coefficients['CC_di'].append(di)
                self.coefficients['CC_eno3'].append(eno3)
                self.coefficients['CC_eswa'].append(eswa)
                
                
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

# **=======================================================================================================================**
# # Source Loading of Calibration Coefficients
# With a NUTNR Calibration object created, we can now begin parsing the different calibration sources for each NUTNR. We will then compare all of the calibration values from each of the sources, checking for any discrepancies between them.

# Below, I plan on going through each of the NUTNR UIDs, and parse the data into csvs. For sources which contain multiple sources, I plan on extracting each of the calibrations to a temporary folder using the following structure:
#
#     <local working directory>/<temp>/<source>/data/<calibration file>
#     
# The separate calibrations will be saved using the standard UFrame naming convention with the following directory structure:
#
#     <local working directory>/<temp>/<source>/<calibration csv>
#     
# The csvs themselves will also be copied to the temporary folder. This allows for the program to be looking into the same temp directory for every NUTNR check.

import shutil

uid = uids[37]
print(uid)

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
        files = [name for name in zfile.namelist() if name.lower().endswith('.cal') and 'Z' not in name]
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
        nutnr = NUTNRCalibration(uid)
        nutnr.load_cal(filepath)
        nutnr.write_csv(savedir)

# **=======================================================================================================================**
# Repeat the above process with the predeployment files:

pre_paths[uid]

try:
    for path in pre_paths[uid]:
        with ZipFile(path) as zfile:
            files = [name for name in zfile.namelist() if name.lower().endswith('.cal') and 'Z' not in name]
            for file in files:
                exdir = path.split('/')[-1].strip('.zip')
                expath = '/'.join((temp_directory,'pre','data',exdir))
                ensure_dir(expath)
                zfile.extract(file,path=expath)
    savedir = '/'.join((temp_directory,'pre'))
    ensure_dir(savedir)
    # Now parse the calibration coefficients
    for dirpath, dirnames, filenames in os.walk('/'.join((temp_directory,'pre','data'))):
        for file in filenames:
            filepath = os.path.join(dirpath, file)
            # With the filepath for the given calibration retrived, I can now start an instance of the NUTNR Calibration
            # object and begin parsing the coefficients
            nutnr = NUTNRCalibration(uid)
            nutnr.load_cal(filepath)
            nutnr.write_csv(savedir)
except KeyError:
    pass


# **=======================================================================================================================**
# Repeat the above process with the post-deployment files:

post_paths[uid]

if post_paths[uid] is not None:
    for path in post_paths[uid]:
        with ZipFile(path) as zfile:
            files = [name for name in zfile.namelist() if name.lower().endswith('.cal') and 'Z' not in name]
            for file in files:
                exdir = path.split('/')[-1].strip('.zip')
                expath = '/'.join((temp_directory,'post','data',exdir))
                ensure_dir(expath)
                zfile.extract(file,path=expath)
    
    savedir = '/'.join((temp_directory,'post'))
    ensure_dir(savedir)
    # Now parse the calibration coefficients
    for dirpath, dirnames, filenames in os.walk('/'.join((temp_directory,'post','data'))):
        for file in filenames:
            filepath = os.path.join(dirpath, file)
            # With the filepath for the given calibration retrived, I can now start an instance of the NUTNR Calibration
            # object and begin parsing the coefficients
            nutnr = NUTNRCalibration(uid)
            nutnr.load_cal(filepath)
            nutnr.write_csv(savedir)


# **=======================================================================================================================**
# # Calibration Coefficient Comparison
# We have now successfully parsed the calibration files from all the possible sources: the vendor calibration files, the pre-deployments files, and the post-deployment files. Furthermore, we have saved csvs in the UFrame format for all of these calibrations. Now, we want to load those csvs into pandas dataframes, which allow for easy element-by-element comparison of calibration coefficients.

# First, load the names of the files into a pandas dataframe to compare between the different calibration dates. This will allow for checking of which calibrations should match up to the csv currently contained in asset management.

def get_file_date(x):
    x = str(x)
    ind1 = x.index('__')
    ind2 = x.index('.')
    return x[ind1+2:ind2]


# CSV from asset management
csv_files = pd.DataFrame(sorted(os.listdir(temp_directory+'/csv')),columns=['csv'])
csv_files['cal date'] = csv_files['csv'].apply(lambda x: get_file_date(x))
csv_files.set_index('cal date',inplace=True)
csv_files

# CSV from vendor calibrations
files = sorted([file for file in os.listdir(temp_directory+'/cal') if not os.path.isdir(temp_directory+'/cal/'+file)])
cal_files = pd.DataFrame(files,columns=['cal'])
cal_files['cal date'] = cal_files['cal'].apply(lambda x: get_file_date(x))
cal_files.set_index('cal date',inplace=True)
cal_files

# CSV from pre-deployment calibrations
files = sorted([file for file in os.listdir(temp_directory+'/pre') if not os.path.isdir(temp_directory+'/pre/'+file)])
pre_files = pd.DataFrame(files,columns=['pre'])
pre_files['cal date'] = pre_files['pre'].apply(lambda x: get_file_date(x))
pre_files.set_index('cal date',inplace=True)
pre_files

# CSV from post-deployment calibrations
files = sorted([file for file in os.listdir(temp_directory+'/post') if not os.path.isdir(temp_directory+'/post/'+file)])
post_files = pd.DataFrame(files,columns=['post'])
post_files['cal date'] = post_files['post'].apply(lambda x: get_file_date(x))
post_files.set_index('cal date',inplace=True)
post_files

# Join the different source file dataframes together for easy visual comparison
df_files = csv_files.join(cal_files,how='outer')
df_files = df_files.join(pre_files,how='outer')
df_files = df_files.join(post_files,how='outer')
df_files = df_files.fillna(value='-999')
df_files

# We can use the above dataframe to assess which files correspond to each other. If any of the csv files need to be renamed, now is the time to go ahead and do so. This will allow for direct comparison.

src = '/'.join((os.getcwd(),'temp','csv','CGINS-NUTNRB-01107__20171128.csv'))
dst = '/'.join((os.getcwd(),'temp','csv','CGINS-NUTNRB-01107__20171129.csv'))
shutil.move(src, dst)

# CSV from asset management
csv_files = pd.DataFrame(sorted(os.listdir(temp_directory+'/csv')),columns=['csv'])
csv_files['cal date'] = csv_files['csv'].apply(lambda x: get_file_date(x))
csv_files.set_index('cal date',inplace=True)
csv_files

# Join the different source file dataframes together for easy visual comparison
df_files = csv_files.join(cal_files,how='outer')
df_files = df_files.join(pre_files,how='outer')
df_files = df_files.join(post_files,how='outer')
df_files = df_files.fillna(value='-999')
df_files

# Now, we have renamed any csv files to their likely calibration source. Our next step is to do the actual coefficient comparisons.

load_directory = '/'.join((temp_directory,'csv'))
fname = 'CGINS-NUTNRB-01107__20181011.csv'
CSV = pd.read_csv(load_directory+'/'+fname)

CSV

load_directory = '/'.join((temp_directory,'pre'))
PRE = pd.read_csv(load_directory+'/'+fname)

PRE

load_directory = '/'.join((temp_directory,'post'))
POST = pd.read_csv(load_directory+'/'+fname)

POST

load_directory = '/'.join((temp_directory,'cal'))
CAL = pd.read_csv(load_directory+'/'+fname)

CAL


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


CSV['value'] = CSV['value'].apply(lambda x: reformat_arrays(x))

PRE['value'] = PRE['value'].apply(lambda x: reformat_arrays(x))

CAL['value'] = CAL['value'].apply(lambda x: reformat_arrays(x))

POST['value'] = POST['value'].apply(lambda x: reformat_arrays(x))

CSV

PRE

POST

CAL

print(PRE['notes'].iloc[0])

np.equal(CSV['value'],PRE['value'])

np.isclose(CSV['value'].iloc[1],PRE['value'].iloc[1],rtol=1e-8,atol=1e-11)



np.isclose(PRE['value'].iloc[1],CSV['value'].iloc[1],rtol=1e-8,atol=1e-11)



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

coeffs_dict.keys()


