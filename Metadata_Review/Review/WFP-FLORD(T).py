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
sn = '2350'
inst = 'FLORD-L'
date = '2012'
source = inst + sn + '_Calibration_' + date

for root, dirs, files in os.walk(basepath):
    for name in files:
        if inst in name and sn in name and date in name and not name.endswith('.v'):
            filepath = os.path.join(root,name)
            print(filepath)

ptext = textract.process(filepath, method='tesseract', language='eng', encoding='utf-8')
ptext = ptext.decode('utf-8')
print(ptext)

ptext = textract.process(filepath, method='tesseract', lanuage='eng', encoding='utf-8')
ptext = ptext.decode('utf-8')
print(ptext)

for line in ptext.splitlines():
    if 'Probe Dark' in line:
        print(line.split())
        
    if 'Wet:' in line:
        print(line.split())
    if 'Calibration Date' in line:
        print(line.split())
    if 'Serial Number' in line:
        print(line.split())


