# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import csv
import re
import os
import sys
import numpy as np
import pandas as pd

sys.path.append('/home/andrew/Documents/OOI-CGSN/QAQC_Sandbox/Calibration/Parsers/')

from utils import *

from zipfile import ZipFile
import string

from Parsers.CTDBPCalibration import CTDBPCalibration

# **====================================================================================================================**
# Define the directories where the **csv** file to check is stored, and where the **source** file is stored. Make sure to check the following information on your machine via your terminal first:
# 1. The branch of your local asset-management repository matches the location of the SPKIR cals
# 2. Your local asset-management repository has the requisite **csv** file to check
# 3. You have downloaded the **source** of the csv file

github_file = 'calibration/CTDBPE/CGINS-CTDBPE-50004__20210530.csv'

#csv_dir = '/home/andrew/Documents/OOI-CGSN/asset-management/calibration/CTDBPC/'
source_dir = '/home/andrew/Downloads/'
#source_dir = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Instrument_Records/CTDBP/CTDBP_Results/'

for file in os.listdir(source_dir):
    if '3305-00102-00289' in file and not file.endswith('.docx'):
        source_file = file
        print(file)

ctdbp = CTDBPCalibration('CGINS-CTDBPE-50004')

ctdbp.load_qct('/home/andrew/Downloads/' + source_file)

temp_directory = '/'.join((os.getcwd(),'temp'))
# Check if the temp directory exists; if it already does, purge and rewrite
if os.path.exists(temp_directory):
    shutil.rmtree(temp_directory)
    ensure_dir(temp_directory)
else:
    ensure_dir(temp_directory)

ctdbp.date

ctdbp.coefficients

ctdbp.write_csv(temp_directory)

ctdbp.source

ctdbp.serial

ctdbp.serial.lstrip('0')

# **====================================================================================================================**

file = 'CGINS-CTDBPE-50004__20210530.csv'

source_csv = pd.read_csv(temp_directory+'/'+file)

source_csv

am_dir = '/home/andrew/Documents/OOI-CGSN/asset-management/'

am_csv = pd.read_csv(am_dir + 'calibration/CTDBPE/' + file)

am_csv

source_csv == am_csv

(am_csv['name'].iloc[9], source_csv['value'].iloc[9], am_csv['value'].iloc[9])

(am_csv['name'].iloc[13], source_csv['value'].iloc[13], am_csv['value'].iloc[13])


