#!/usr/bin/env python

import datetime
import re
import os
from wcmatch import fnmatch
import pandas as pd
import numpy as np
import string
from zipfile import ZipFile
import csv
import json


class OPTAACalibration():

    def __init__(self, uid):
        self.serial = ''
        self.uid = uid
        self.date = None
        self.cwlngth = []
        self.awlngth = []
        self.tcal = None
        self.tbins = None
        self.ccwo = []
        self.acwo = []
        self.tcarray = []
        self.taarray = []
        self.nbins = None  # number of temperature bins
        self.coefficients = {
                        'CC_taarray': 'SheetRef:CC_taarray',
                        'CC_tcarray': 'SheetRef:CC_tcarray'
                        }

    @property
    def uid(self):
        return self._uid

    @uid.setter
    def uid(self, d):
        r = re.compile('.{5}-.{6}-.{5}')
        if r.match(d) is not None:
            self.serial = 'ACS-' + d.split('-')[2].strip('0')
            self._uid = d
        else:
            raise Exception(f"The instrument uid {d} is not a valid uid. Please check.")

    def load_dev(self, filepath):
        """
        Function loads the dev file for the OPTAA.

        Args:
            filepath - the full path, including the name of the file, to the
                optaa dev file.
        Returns:
            self.date - the date of calibration
            self.tcal - calibration temperature
            self.nbins - number of temperature bins
            self.cwlngth
            self.awlngth
            self.ccwo
            self.acwo
            self.tcarray
            self.taarray
            self.coefficients - a dictionary of the calibration values and
                associated keys following the OOI csv naming convention

        """

        if filepath.endswith('.zip'):
            with ZipFile(filepath) as zfile:
                filename = [name for name in zfile.namelist() if name.endswith('.dev')]
                text = zfile.read(filename[0]).decode('ASCII')

        else:
            with open(filepath) as file:
                text = file.read()

        # Remove extraneous characters from the
        punctuation = ''.join((letter for letter in string.punctuation if letter not in ';/.'))

        for line in text.replace('\t', ' ').splitlines():
            line = ''.join((word for word in line if word not in punctuation))

            if 'tcal' in line:
                data = line.split()
                # Temperature calibration value
                tcal = data.index('tcal')
                self.tcal = data[tcal+1]
                self.coefficients['CC_tcal'] = self.tcal
                # Temperature calibration date
                cal_date = data[-1].strip()
                self.date = pd.to_datetime(cal_date).strftime('%Y%m%d')

            elif ';' in line:
                data, comment = line.split(';')

                if 'temperature bins' in comment:
                    if 'number' in comment:
                        self.nbins = int(data)
                    else:
                        self.tbins = data.split()
                        self.tbins = [float(x) for x in self.tbins]
                        self.coefficients['CC_tbins'] = json.dumps(self.tbins)

                elif 'C and A offset' in comment:
                    data = data.split()
                    self.cwlngth.append(float(data[0][1:]))
                    self.awlngth.append(float(data[1][1:]))
                    self.ccwo.append(float(data[3]))
                    self.acwo.append(float(data[4]))
                    tcrow = [float(x) for x in data[5:self.nbins+5]]
                    tarow = [float(x) for x in data[self.nbins+5:2*self.nbins+5]]
                    self.tcarray.append(tcrow)
                    self.taarray.append(tarow)
                    self.coefficients['CC_cwlngth'] = json.dumps(self.cwlngth)
                    self.coefficients['CC_awlngth'] = json.dumps(self.awlngth)
                    self.coefficients['CC_ccwo'] = json.dumps(self.ccwo)
                    self.coefficients['CC_acwo'] = json.dumps(self.acwo)

                else:
                    pass

            else:
                pass

    def write_csv(self, savepath):
        """
        This function writes the correctly named csv file for the ctd to the
        specified directory.

        Args:
            outpath - directory path of where to write the csv file
        Raises:
            ValueError - raised if the OPTAA's object's coefficient dictionary
                has not been populated
        Returns:
            self.to_csv - a csv of the calibration coefficients which is
                written to the specified directory from the outpath.
        """
        # Now, write to a csv file
        # Create a dataframe to write to the csv
        data = {
            'serial': self.serial,
            'name': list(self.coefficients.keys()),
            'value': list(self.coefficients.values()),
            'notes': ['']*len(self.coefficients)
        }

        df = pd.DataFrame().from_dict(data)

        # Generate the cal csv filename
        filename = self.uid + '__' + self.date + '.csv'
        # Now write to
        check = input(f"Write {filename} to {savepath}? [y/n]: ")
        if check.lower().strip() == 'y':
            df.to_csv(savepath+'/'+filename, index=False)

        # Generate the tc and ta array filename
        tc_name = filename + '__CC_tcarray.ext'
        ta_name = filename + '__CC_taarray.ext'

        def write_array(filename, array):
            with open(filename, 'w') as out:
                array_writer = csv.writer(out)
                array_writer.writerows(array)

        write_array(savepath+'/'+tc_name, self.tcarray)
        write_array(savepath+'/'+ta_name, self.taarray)