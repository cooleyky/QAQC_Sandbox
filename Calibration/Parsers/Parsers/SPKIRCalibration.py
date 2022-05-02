import os
import re
import csv
import numpy as np
import pandas as pd
import string
from zipfile import ZipFile


class SPKIRCalibration():
    # Class that stores calibration values for CTDs.

    def __init__(self, uid):
        self.serial = None
        self.uid = uid
        self.date = []
        self.coefficients = {
            'CC_immersion_factor': [],
            'CC_offset': [],
            'CC_scale': []
        }
        self.notes = {
            'CC_immersion_factor': '',
            'CC_offset': '',
            'CC_scale': '',
        }

    @property
    def uid(self):
        return self._uid

    @uid.setter
    def uid(self, d):
        r = re.compile('.{5}-.{6}-.{5}')
        if r.match(d) is not None:
            self._uid = d
            self.serial = d.split('-')[-1].lstrip('0')
        else:
            raise Exception(f"The instrument uid {d} is not a valid uid. Please check.")

    def load_cal(self, filepath):
        """
        Wrapper function to load all of the calibration coefficients from the
        calibration .cal file from the vendor.

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
        Function that opens and reads in .cal file information for a SPKIR.
        Zipfiles are acceptable inputs.

        Args:
            filepath - path to the directory with filename which has the
                calibration coefficients to be parsed and loaded
        Returns:
            data - opened file containing the calibration information read into
                memory but is not parsed
        """

        if filepath.endswith('.zip'):
            with ZipFile(filepath) as zfile:
                # Check if OPTAA has the .dev file
                filename = [name for name in zfile.namelist() if name.lower().endswith('.cal')]

                # Get and open the latest calibration file
                if len(filename) == 1:
                    data = zfile.read(filename[0]).decode('ascii')
                    self.source_file(filepath, filename)

                elif len(filename) > 1:
                    raise FileExistsError(f"Multiple .cal files found in {filepath}.")

                else:
                    raise FileNotFoundError(f"No .cal file found in {filepath}.")

        elif filepath.lower().endswith('.cal'):
            with open(filepath) as file:
                data = file.read()
            self.source_file(filepath, file)

        else:
            raise FileNotFoundError(f"No .cal file found in {filepath}.")

        return data

    def parse_cal(self, data):
        """
        Function which parses the calibration data and loads the calibration
        coefficients into the object structure.

        Args:
            data - calibration data which has been read and loaded into memory
        Raises:
            ValueError - raised if the serial number parsed from the calibration
                data does not match the UID
        Returns:
            self.coefficients - populated dictionary of calibration coefficient
                values
            self.date - all relevant calibration dates parsed into a dictionary
            self.serial - parsed serial data
        """

        flag = False
        for line in data.splitlines():
            if line.startswith('#'):
                parts = line.split('|')
                if len(parts) > 5 and 'Calibration' in parts[-1].strip():
                    cal_date = parts[0].replace('#', '').strip()
                    self.date.append(pd.to_datetime(cal_date).strftime('%Y%m%d'))

            elif line.startswith('SN'):
                parts = line.split()
                _, sn, *ignore = parts
                sn = sn.lstrip('0')
                if self.serial != sn:
                    raise ValueError(f'Instrument serial number {sn} does not match UID {self.uid}')

            elif line.startswith('ED'):
                flag = True

            elif flag:
                offset, scale, immersion_factor = line.split()
                self.coefficients['CC_immersion_factor'].append(float(immersion_factor))
                self.coefficients['CC_offset'].append(float(offset))
                self.coefficients['CC_scale'].append(float(scale))
                flag = False

            else:
                continue

    def source_file(self, filepath, filename):
        """
        Routine which parses out the source file and filename
        where the calibration coefficients are sourced from. Automatically
        stored in the first calibration coefficient "notes" field.

        Args:
            filepath - path to the directory with filename which has the
                calibration coefficients to be parsed and loaded
        Returns:
            self.source - string which contains the parent file and the
                filename of the calibration data source
        """

        if filepath.lower().endswith('.cal'):
            dcn = filepath.split('/')[-2]
            filename = filepath.split('/')[-1]
        else:
            dcn = filepath.split('/')[-1]

        self.source = f'Source file: {dcn} > {filename}'

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
            'name': list(self.notes.keys()),
            'notes': list(self.notes.values())
        })
        df = df.merge(notes, how='outer', left_on='name', right_on='name')

        # Add in the source file
        df['notes'].iloc[0] = df['notes'].iloc[0] + ' ' + self.source

        # Sort the data by the coefficient name
        df = df.sort_values(by='name')

        # Generate the csv names
        csv_name = self.uid + '__' + max(self.date) + '.csv'

        # Write the dataframe to a csv file
        check = input(f"Write {csv_name} to {outpath}? [y/n]: ")
        # check = 'y'
        if check.lower().strip() == 'y':
            df.to_csv(outpath+'/'+csv_name, index=False)


if __name__ == '__main__':
    # Request the input data
    uid = input("Enter the instrument uid: ")
    fpath = input("Enter the full filepath to the calibration file: ")
    outpath = input("Enter the filepath where to save the parsed calibration: ")
    # Initialize the calibration parser
    spkir = SPKIRCalibrationParser(uid)
    spkir.load_cal(fpath)
    spkir.write_csv(outpath)
