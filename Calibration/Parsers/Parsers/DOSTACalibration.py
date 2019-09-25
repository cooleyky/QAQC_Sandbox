#!/usr/bin/env python

import datetime
import re
import os
from wcmatch import fnmatch
import pandas as pd
import numpy as np
import string
from zipfile import ZipFile


class DOSTACalibration():

    def __init__(self, uid, calibration_date):
        self.serial = ''
        self.uid = uid
        self.date = pd.to_datetime(calibration_date).strftime('%Y%m%d')
        self.coefficients = {'CC_conc_coef': None, 'CC_csv': None}
        self.notes = {'CC_conc_coef': None, 'CC_csv': None}

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

    def load_qct(self, filepath):
        """
        Function which parses the output from the QCT check-in and loads them
        into the DOSTA object.

        Args:
            filepath - the full directory path and filename
        Raises:
            ValueError - checks if the serial number parsed from the UID
            matches the serial number stored in the file.
        Returns:
            self.coefficients - populated coefficients dictionary
            self.date - the calibration dates associated with the calibration values
            self.type - the type (i.e. 16+/37-IM) of the CTD
            self.serial - populates the 5-digit serial number of the instrument
        """

        data = {}
        with open(filepath, errors='ignore') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                data.update({reader.line_num: row})

        for key, info in data.items():
            # Find the serial number from the QCT check-in and compare to UID
            if 'serial number' in [x.lower() for x in info]:
                serial_num = info[-1].zfill(5)
                if self.serial != serial_num:
                    raise ValueError(f'Serial number {serial_num.zfill(5)} from the QCT file does not match {self.serial} from the UID.')
                else:
                    pass

            # Find the svu foil coefficients
            if 'svufoilcoef' in [x.lower() for x in info]:
                self.coefficients['CC_csv'] = [float(n) for n in info[3:]]

            # Find the concentration coefficients
            if 'conccoef' in [x.lower() for x in info]:
                self.coefficients['CC_conc_coef'] = [float(n) for n in info[3:]]

    def add_notes(self, notes):
        """
        This function adds notes to the calibration csv based on the
        calibration coefficients.

        Args:
            notes - a dictionary with keys of the calibration coefficients
                which correspond to an entry of desired notes about the
                corresponding coefficients
        Returns:
            self.notes - a dictionary with the entered notes.
        """
        keys = notes.keys()
        for key in keys:
            self.notes[key] = notes[key]

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
        for key in self.coefficients.keys():
            if self.coefficients[key] is None:
                raise ValueError(f'No coefficients for {key} have been loaded.')

        # Create a dataframe to write to the csv
        data = {'serial': [self.serial]*len(self.coefficients),
                'name': list(self.coefficients.keys()),
                'value': list(self.coefficients.values()),
                'notes': list(self.notes.values())
                }
        df = pd.DataFrame().from_dict(data)

        # Generate the csv name
        csv_name = self.uid + '__' + self.date + '.csv'

        # Now write to
        check = input(f"Write {csv_name} to {outpath}? [y/n]: ")
        if check.lower().strip() == 'y':
            df.to_csv(outpath+'/'+csv_name, index=False)
