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

import pandas as pd
import numpy as np
import os, sys, re


# Import the spreadsheet describing the relevant data
info = pd.read_excel('ADCP_All_Moorings.xlsx')
info.dropna(subset=['ADCP S/N'], inplace=True)

info.head()

np.unique(info['CC_orientation'])

# +
# Now generate the ADCPS/T info necessary
for i in range(len(info)):
    row = info.iloc[i]
    # Now get the relevant info from the row
    values = {
        'CC_scale_factor1': 0.45,
        'CC_scale_factor2': 0.45,
        'CC_scale_factor3': 0.45,
        'CC_scale_factor4': 0.45,
        'CC_bin_size': row['CC_bin_size (cm)'],
        'CC_dist_first_bin': row['CC_dist_first_bin (cm)\n[Depth to 1st bin]'],
        'CC_orientation': int(row['CC_orientation'])
    }
    
    notes = {
        'CC_scale_factor1': 'constant; ' + row['Serial Number'],
        'CC_scale_factor2': 'constant',
        'CC_scale_factor3': 'constant',
        'CC_scale_factor4': 'constant',
        'CC_bin_size': 'in cm',
        'CC_dist_first_bin': 'in cm',
        'CC_orientation': '1 is upward looking; 0 is downward looking'
    }

    serial = int(row['ADCP S/N'])
    date = row['Anchor Launch Date'].strftime('%Y%m%d')
    series = row['ClassSeries']
    
    # Now generate the filename
    filename = 'CGINS-' + series + '-' + str(serial).zfill(5) + '__' + date + '.csv'

    # Now put the relevant data into a data dictionary
    data = {
        'serial': str(serial),
        'name': list(values.keys()),
        'value': list(values.values()),
        'notes': list(notes.values())
    }
    
    
    
    # Generated a pandas dataframe
    df = pd.DataFrame.from_dict(data)
    
    # Save it to a temp folder
    df.to_csv('temp/'+filename, index=False)



# -

row['Ref Des']

# +
values = {
    'CC_scale_factor1': 0.45,
    'CC_scale_factor2': 0.45,
    'CC_scale_factor3': 0.45,
    'CC_scale_factor4': 0.45,
    'CC_bin_size': row['CC_bin_size (cm)'],
    'CC_dist_first_bin': row['CC_dist_first_bin (cm)\n[Depth to 1st bin]'],
    'CC_orientation': row['CC_orientation']    
}

notes = {
    'CC_scale_factor1': 'constant; ' + row['Serial Number'],
    'CC_scale_factor2': 'constant',
    'CC_scale_factor3': 'constant',
    'CC_scale_factor4': 'constant',
    'CC_bin_size': 'in cm',
    'CC_dist_first_bin': 'in cm',
    'CC_orientation': '1 is upward looking; 2 is downward looking'
}

serial = row['ADCP S/N']
date = row['Anchor Launch Date'].strftime('%Y%m%d')
series = row['ClassSeries']
# -

filename = 'CGINS-' + series + '-' + str(int(serial)).zfill(5) + '__' + date + '.csv'
filename

data = {
    'serial': str(int(serial)),
    'name': list(values.keys()),
    'value': list(values.values()),
    'notes': list(notes.values())
}

df = pd.DataFrame.from_dict(data)
df


