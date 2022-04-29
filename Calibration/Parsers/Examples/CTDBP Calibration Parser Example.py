# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # CTDBP Calibration Parser Example
#
# This notebook provides an example of the use of CTDBP Calibration Parser for generating csv files from vendor documentation. The calibration coefficients can be parsed from either the vendor calibration (.cal) file, the vendor xml configuration file (.xmlcon), or from the instrument check-in capture file (either .cap, .log, or .txt). 
# **===================================================================================================================**

# Until the module is set-up, need to 
import sys
sys.path.append('..')
sys.path.append('../Parsers/')

from utils import *

# **===================================================================================================================**
# ### Data Sources and Locations
# Enter the full path to where the calibration and qct files are stored. If running this example as part of the Parsers package, the relatives paths defined below should work.

cal_files = '../Data_Sources/CTDBP/3305-00102-00309-A.cap'
#qct_files = '../Data_Sources/CTDBP/3305-00102-00209-A.txt'

# **===================================================================================================================**
# ### Create the CTDBP Calibration Object
#
# In order to initialize the CTDBP Calibration Object requires entering the UID of the CTD instrument for which the calibration csv is going to be generated. This uid is then checked against the data parsed from the source files as a check that the expected calibration csv matches the source files where the calibration data is being parsed from.
#
# Additionally, we will create a temp directory to output the csv files to during the course of this notebook so that they can be compared with each other at the end, and then erased.

from CTDBPCalibration import CTDBPCalibration

if not os.path.exists('/'.join((os.getcwd(),'temp'))):
    tempdir = '/'.join((os.getcwd(),'temp'))
    os.mkdir(tempdir)
else:
    tempdir = '/'.join((os.getcwd(),'temp'))

# #### Calibration CSV from .cal file
# First, we'll generate a calibration csv from the vendor .cal file (which is stored away in a .zip file). The steps are:
# 1. Initialize a CTDBPCalibration object with the instrument UID
# 2. Call the load_cal method with the path to where the .cal file is stored
# 3. Call the write_csv method with the path to where the csv calibration file should be saved to

ctdbp = CTDBPCalibration(uid='CGINS-CTDBPE-50111')

ctdbp.load_qct(cal_files)

ctdbp.coefficients

ctdbp.write_csv(tempdir)

# Now, lets load the csv file into a dataframe for later comparison with other calibration sources
filename = "CGINS-CTDBPE-50111__20220114.csv"
cal = pd.read_csv(f'temp/{filename}')
cal

# #### Compare with cal files

gitHub_file = f"/home/andrew/Documents/OOI-CGSN/asset-management/calibration/CTDBPE/{filename}"
gitHub = pd.read_csv(gitHub_file)
gitHub

cal == gitHub

# #### Calibration CSV from .xmlcon file
# First, we'll generate a calibration csv from the vendor .xmlcon file (which is stored away in a .zip file). The steps are:
# 1. Initialize a CTDBPCalibration object with the instrument UID
# 2. Call the load_xmlcon method with the path to where the .xmlcon file is stored
# 3. Call the write_csv method with the path to where the csv calibration file should be saved to

ctdbp = CTDBPCalibration(uid='CGINS-CTDBPC-50003')

ctdbp.load_xml(cal_files)

ctdbp.write_csv(tempdir)

# Now, load the csv back into pandas dataframe for later comparison with other methods
xml = pd.read_csv('temp/CGINS-CTDBPC-50003__20190123.csv')
xml

# #### Calibration CSV from QCT file
# First, we'll generate a calibration csv from the vendor .xmlcon file (which is stored away in a .zip file). The steps are:
# 1. Initialize a CTDBPCalibration object with the instrument UID
# 2. Call the load_qct method with the path to where the QCT file is stored
# 3. Call the write_csv method with the path to where the csv calibration file should be saved to

ctdbp = CTDBPCalibration(uid='CGINS-CTDBPC-50003')

ctdbp.load_qct(qct_files)

ctdbp.write_csv(tempdir)

qct = pd.read_csv('temp/CGINS-CTDBPC-50003__20190123.csv')
qct

# **===================================================================================================================**
# #### Comparison of Methods
# Now, we can compare the generated csvs from each of the vendor sources and check that they are in agreement.

cal == xml

qct == cal

qct == xml

# So, we see that although the coefficient serial numbers and names match, that the values don't. Why is that? Probably due to rounding *inherent to the source file.* We can check this by using the numpy function is_close() with a relatively tight difference threshold (0.01%) as a further check: 

import numpy as np

np.isclose(cal['value'],xml['value'])

np.isclose(cal['value'],qct['value'])

np.isclose(xml['value'],qct['value'])

# And delete the temporary folder
shutil.rmtree(tempdir)



