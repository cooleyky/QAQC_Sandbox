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

import csv
import re
import os
import shutil
import numpy as np
import pandas as pd
from zipfile import ZipFile
import string
import PyPDF2
import textract

from utils import *

basepath = '/media/andrew/OS/Users/areed/Documents/Project_Files/Records/Platform_Records/WFPs/'
source = 'DOFST-K_SBE43F_SN_3523_Calibration_2018'


def extract_pdf(pdf):
    
    ptext = textract.process(pdf, encoding='utf-8')
    ptext = ptext.decode('utf-8')
    
    for line in ptext.splitlines():
        if '(adj)' in line:
            coef, _, val, _ = line.split()
            coef = coef.strip()
            val = val.strip()
            continue
            
        if 'CALIBRATION DATE' in line:
            _, caldate = line.split(':')
            caldate = pd.to_datetime(caldate).strftime('%Y%m%d')
            continue
            
    return coef, val, caldate


for root, dirs, files in os.walk(basepath):
    for name in files:
        if source in name and not name.endswith('.v'):
            filepath = os.path.join(root,name)
            print(filepath)

temp_directory = '/'.join((os.getcwd(),'temp'))
# Check if the temp directory exists; if it already does, purge and rewrite
if os.path.exists(temp_directory):
    shutil.rmtree(temp_directory)
    ensure_dir(temp_directory)
else:
    ensure_dir(temp_directory)

if filepath.endswith('.zip'):
    with ZipFile(filepath) as zfile:
        zfile.extractall(path=temp_directory)

cal = [name for name in os.listdir(temp_directory) if name.endswith('.cal')][0]
pdf = [name for name in os.listdir(temp_directory) if name.endswith('.pdf') and 'SOC' in name][0]

pdf

temp_directory

with open(temp_directory+'/'+cal) as file:
    data = file.read()

for line in data.splitlines():
    print(line)

coefficient_name_map = {
    'FOFFSET':'CC_frequency_offset',
    'Soc':'CC_oxygen_signal_slope',
    'A':'CC_residual_temperature_correction_factor_a',
    'B':'CC_residual_temperature_correction_factor_b',
    'C':'CC_residual_temperature_correction_factor_c',
    'E':'CC_residual_temperature_correction_factor_e'
}
calibration_coeffs = {}

# +
for line in data.splitlines():
    
    
    coef, val = line.split('=')
    coef = coef.strip()
    val = val.strip()
    
    if coef == 'SERIALNO':
        sn = val
        continue
    
    name = coefficient_name_map.get(coef)
    
    if name is not None:
        calibration_coeffs.update({name: float(val)})   
        
coef, val, caldate = extract_pdf(temp_directory+'/'+pdf)
name = coefficient_name_map.get(coef)
if name is not None:
    calibration_coeffs.update({name: float(val)})
# -

pdf

calibration_coeffs

wfp, filename = filepath.split('/')[-2::]
source = f'Date in filename from {pdf}. Source file: {wfp} > {filename} > {cal} and {pdf}.'

# Create a dataframe to write to the csv
data = {
    'serial': ['43' + '-' + sn]*len(calibration_coeffs),
    'name': list(calibration_coeffs.keys()),
    'value': list(calibration_coeffs.values())
}
df = pd.DataFrame().from_dict(data)
# Now merge the coefficients dataframe with the notes
df['notes'] = ''
# Add in the source file
df['notes'].iloc[0] = df['notes'].iloc[0] + ' ' + source
# Sort the data by the coefficient name
df = df.sort_values(by='name')

df

savefile = 'CGINS-DOFSTK-' + str(sn).zfill(5) + '__' + caldate + '.csv'
savefile

df.to_csv(temp_directory+'/'+savefile)

# **====================================================================================================================**
# Now can import the csv file for comparison

csv_name = 'calibration/DOFSTK/' + savefile

csv_file = pd.read_csv('/home/andrew/Documents/OOI-CGSN/asset-management/' + csv_name)

source_file = pd.read_csv(temp_directory + '/' + savefile)

csv_file

source_file.drop(columns='Unnamed: 0', inplace=True)
source_file

csv_file == source_file

csv_file['value'].iloc[5], source_file['value'].iloc[5]

for line in ptext.splitlines():
    print(line)

for line in data.splitlines():
    if '=' in line:
        coef, val, *ignore = line.split('=')
        if coef.strip() in coefficient_name_map.keys():
            calibration_coeffs.update({coefficient_name_map.get(coef.strip()):float(val)})
            
    if 'oxygen S/N' in line:
        sn = line[line.find('=')+len('='):line.rfind(':')].strip()

calibration_coeffs

sn


