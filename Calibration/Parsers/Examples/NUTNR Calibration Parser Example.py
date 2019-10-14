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

# # NUTNR Calibration Parser Example
#
# This notebook provides an example of the use of NUTNR Calibration Parser for generating csv files from vendor documentation. The calibration coefficients can be parsed from the vendor .cal file. If more than one cal file is present, it will automatically open and parse the latest calibration file. 
# **===================================================================================================================**

# Until the module is set-up, need to 
import sys
sys.path.append('..')
sys.path.append('../Parsers/')

from utils import *

# **===================================================================================================================**
# ### Data Sources and Locations
# Enter the full path to where the calibration and qct files are stored. If running this example as part of the Parsers package, the relatives paths defined below should work.

cal_files = '../Data_Sources/NUTNR/3305-00327-00051-A.zip'

# **===================================================================================================================**
# ### Create the NUTNR Calibration Object
#
# In order to initialize the NUTNR Calibration Object requires entering the UID of the ISUS/SUNA instrument for which the calibration csv is going to be generated. This uid is then checked against the data parsed from the source files as a check that the expected calibration csv matches the source files where the calibration data is being parsed from.
#
# Additionally, we will create a temp directory to output the csv files to during the course of this notebook so that they can be compared with each other at the end, and then erased.

from NUTNRCalibration import NUTNRCalibration

if not os.path.exists('/'.join((os.getcwd(),'temp'))):
    tempdir = '/'.join((os.getcwd(),'temp'))
    os.mkdir(tempdir)
else:
    tempdir = '/'.join((os.getcwd(),'temp'))

# #### Calibration CSV from .cal file
# First, we'll generate a calibration csv from the vendor .cal file (which is stored away in a .zip file). The steps are:
# 1. Initialize a NUTNRCalibration object with the instrument UID
# 2. Call the load_cal method with the path to where the .cal file is stored (may be a compressed zip file)
# 3. Call the write_csv method with the path to where the csv calibration file should be saved to

nutnr = NUTNRCalibration(uid='CGINS-NUTNRB-01102')

nutnr.load_cal(cal_files)

nutnr.write_csv(tempdir)

# **===================================================================================================================**
# Now, load the csv into a pandas dataframe in this notebook as a visual check that all of the data loaded as expected:

cal = pd.read_csv('temp/CGINS-NUTNRB-01102__20181011.csv')
cal

# Looks good!


