import csv
import re
import os
import shutil
import numpy as np
import pandas as pd
from zipfile import ZipFile
import string


class OPTAACalibration():

    def __init__(self, uid):
        self.serial = None
        self.nbins = None
        self.uid = uid
        self.date = []
        self.coefficients = {
            'CC_acwo': [],
            'CC_awlngth': [],
            'CC_ccwo': [],
            'CC_cwlngth': [],
            'CC_taarray': 'SheetRef:CC_taarray',
            'CC_tbins': [],
            'CC_tcal': [],
            'CC_tcarray': 'SheetRef:CC_tcarray'
        }
        self.tcarray = []
        self.taarray = []
        self.notes = {
            'CC_acwo': '',
            'CC_awlngth': '',
            'CC_ccwo': '',
            'CC_cwlngth': '',
            'CC_taarray': '',
            'CC_tbins': '',
            'CC_tcal': '',
            'CC_taarray': ''
        }

    @property
    def uid(self):
        return self._uid

    @uid.setter
    def uid(self, d):
        r = re.compile('.{5}-.{6}-.{5}')
        if r.match(d) is not None:
            self._uid = d
            serial = d.split('-')[-1].lstrip('0')
            self.serial = 'ACS-' + serial
        else:
            raise Exception(f"The instrument uid {d} is not a valid uid. Please check.")

    def load_cal(self, filepath):
        """
        Wrapper function to load all of the calibration coefficients from the
        calibration .dev file from the vendor.

        Args:
            filepath - path to the directory with filename which has the
                calibration coefficients to be parsed and loaded
        Calls:
            open_cal
            parse_cal
        """

        data = self.open_dev(filepath)

        self.parse_dev(data)

    def load_qct(self, filepath):
        """
        Wrapper function to load the calibration coefficients from
        the QCT checkin.

        Args:
            filepath - path to the directory with the filename which has the
                calibration coefficients to be parsed and loaded
        Calls:
            open_cal
            parse_cal
        """

        data = self.open_dev(filepath)

        self.parse_qct(data)

    def open_dev(self, filepath):
        """
        Function that opens and reads in cal file
        information for a OPTAA. Zipfiles are acceptable inputs.

        Args:
            filepath - path to the directory where the calibration files are stored.

        Returns:
            data - opened file containing the calibration information read into memory.
        """

        if filepath.endswith('.zip'):
            with ZipFile(filepath) as zfile:
                # Check if OPTAA has the .dev file
                filename = [name for name in zfile.namelist() if name.lower().endswith('.dev')]

                # Get and open the latest calibration file
                if len(filename) == 1:
                    data = zfile.read(filename[0]).decode('ascii')
                    self.source_file(filepath, filename[0])

                elif len(filename) > 1:
                    raise FileExistsError(f"Multiple .dev files found in {filepath}.")

                else:
                    raise FileNotFoundError(f"No .dev file found in {filepath}.")

        elif filepath.lower().endswith('.dev'):
            with open(filepath) as file:
                data = file.read()
            self.source_file(filepath, file)

        elif filepath.lower().endswith('.dat'):
            with open(filepath) as file:
                data = file.read()
            self.source_file(filepath, file)

        else:
            raise FileNotFoundError(f"No .dev file found in {filepath}.")

        return data

    def source_file(self, filepath, filename):
        """
        Routine which parses out the source file and filename
        where the calibration coefficients are sourced from. Automatically
        stored in the first calibration coefficient "notes" field.
        """

        if filepath.lower().endswith('.dev'):
            dcn = filepath.split('/')[-2]
            filename = filepath.split('/')[-1]
        else:
            dcn = filepath.split('/')[-1]

        self.source = f'Source file: {dcn} > {filename}'

    def parse_dev(self, data):
        """
        Function to parse the .dev file in order to load the
        calibration coefficients for the OPTAA.

        Args:
            data - opened .dev file in ascii-format

        Returns:
            coefficients - a dictionary of the calibration coefficients with
                key:value pairs of name:array_of_values
        """

        for line in data.splitlines():
            # Split the data based on data -> header split
            parts = line.split(';')
            # If the len isn't number 2,
            if len(parts) is not 2:
                # Find the calibration temperature and date
                if 'tcal' in line.lower():
                    line = ''.join((x for x in line if x not in [
                                   y for y in string.punctuation if y is not '/']))
                    parts = line.split()
                    # Calibration temperature
                    tcal = parts[1].replace('C', '')
                    tcal = float(tcal)/10
                    self.coefficients['CC_tcal'] = tcal
                    # Calibration date
                    date = parts[-1].strip(string.punctuation)
                    self.date = pd.to_datetime(date).strftime('%Y%m%d')

            else:
                info, comment = parts

                if comment.strip().startswith('temperature bins'):
                    tbins = [float(x) for x in info.split()]
                    self.coefficients['CC_tbins'] = tbins

                elif comment.strip().startswith('number'):
                    self.nbins = int(float(info.strip()))

                elif comment.strip().startswith('C'):
                    if self.nbins is None:
                        raise AttributeError(f'Failed to load number of temperature bins.')

                    # Parse out the different calibration coefficients
                    parts = info.split()
                    cwlngth = float(parts[0][1:])
                    awlngth = float(parts[1][1:])
                    ccwo = float(parts[3])
                    acwo = float(parts[4])
                    tcrow = [float(x) for x in parts[5:self.nbins+5]]
                    acrow = [float(x) for x in parts[self.nbins+5:2*self.nbins+5]]

                    # Now put the coefficients into the coefficients dictionary
                    self.coefficients['CC_acwo'].append(acwo)
                    self.coefficients['CC_awlngth'].append(awlngth)
                    self.coefficients['CC_ccwo'].append(ccwo)
                    self.coefficients['CC_cwlngth'].append(cwlngth)
                    self.tcarray.append(tcrow)
                    self.taarray.append(acrow)

    def parse_qct(self, data):
        """
        This function is designed to parse the QCT file, which contains the
        calibration data in slightly different format than the .dev file

        Args:
            data - opened qct file loaded and read into memory in ascii-format

        Returns:
            coefficients - a dictionary of the optaa calibration coefficients
        """

        for line in data.splitlines():
            if 'WetView' in line:
                _, _, _, date, time = line.split()
                try:
                    date_time = date + ' ' + time
                    self.date = pd.to_datetime(date_time).strftime('%Y%m%d')
                except:
                    date_time = from_excel_ordinal(float(date) + float(time))
                    self.date = pd.to_datetime(date_time).strftime('%Y%m%d')
                continue

            parts = line.split(';')

            if len(parts) == 2:
                if comment.strip().startswith('temperature bins'):
                    tbins = [float(x) for x in info.split()]
                    self.coefficients['CC_tbins'] = tbins

                elif comment.strip().startswith('number'):
                    self.nbins = int(float(info.strip()))

                elif comment.strip().startswith('C'):
                    if self.nbins is None:
                        raise AttributeError(f'Failed to load number of temperature bins.')
                    # Parse out the different calibration coefficients
                    parts = info.split()
                    cwlngth = float(parts[0][1:])
                    awlngth = float(parts[1][1:])
                    ccwo = float(parts[3])
                    acwo = float(parts[4])
                    tcrow = [float(x) for x in parts[5:self.nbins+5]]
                    acrow = [float(x) for x in parts[self.nbins+5:(2*self.nbins)+5]]

                    # Now put the coefficients into the coefficients dictionary
                    self.coefficients['CC_acwo'].append(acwo)
                    self.coefficients['CC_awlngth'].append(awlngth)
                    self.coefficients['CC_ccwo'].append(ccwo)
                    self.coefficients['CC_cwlngth'].append(cwlngth)
                    self.tcarray.append(tcrow)
                    self.taarray.append(acrow)

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
        csv_name = self.uid + '__' + self.date + '.csv'
        tca_name = self.uid + '__' + self.date + '__' + 'CC_tcarray.ext'
        taa_name = self.uid + '__' + self.date + '__' + 'CC_taarray.ext'

        def write_array(filename, cal_array):
            with open(filename, 'w') as out:
                array_writer = csv.writer(out)
                array_writer.writerows(cal_array)

        # Write the dataframe to a csv file
        check = input(f"Write {csv_name} to {outpath}? [y/n]: ")
        # check = 'y'
        if check.lower().strip() == 'y':
            df.to_csv(outpath+'/'+csv_name, index=False)
            write_array(outpath+'/'+tca_name, self.tcarray)
            write_array(outpath+'/'+taa_name, self.taarray)
