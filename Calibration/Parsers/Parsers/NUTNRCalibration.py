import re
import pandas as pd
import numpy as np
from zipfile import ZipFile


class NUTNRCalibration():
    # Class that stores calibration values for CTDs.

    def __init__(self, uid):
        self.serial = None
        self.uid = uid
        self.coefficients = {
            'CC_cal_temp': [],
            'CC_di': [],
            'CC_eno3': [],
            'CC_eswa': [],
            'CC_lower_wavelength_limit_for_spectra_fit': '217',
            'CC_upper_wavelength_limit_for_spectra_fit': '240',
            'CC_wl': []
        }
        self.date = []
        self.notes = {
            'CC_cal_temp': '',
            'CC_di': '',
            'CC_eno3': '',
            'CC_eswa': '',
            'CC_lower_wavelength_limit_for_spectra_fit': '217',
            'CC_upper_wavelength_limit_for_spectra_fit': '240',
            'CC_wl': ''
        }

    @property
    def uid(self):
        return self._uid

    @uid.setter
    def uid(self, d):
        r = re.compile('.{5}-.{6}-.{5}')
        if r.match(d) is not None:
            self._uid = d
        else:
            raise Exception(f"The instrument uid {d} is not a valid uid. Please check.")

    def load_cal(self, filepath):
        """
        Wrapper function to load all of the calibration coefficients

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
        Function that opens and reads in cal file information for a NUTNR.
        Zipfiles are acceptable inputs.

        Args:
            filepath - full path to the directory and file which contains the
                calibration information
        Calls:
            source_file
        Returns:
            data - opened calibration file read into memory in ascii format
        """

        if filepath.endswith('.zip'):
            with ZipFile(filepath) as zfile:
                # Check if ISUS or SUNA to get the appropriate name
                filename = [name for name in zfile.namelist()
                            if name.lower().endswith('.cal') and 'z' not in name.lower()]

                # Get and open the latest calibration file
                if len(filename) == 1:
                    data = zfile.read(filename[0]).decode('ascii')
                    self.source_file(filepath, filename[0])

                elif len(filename) > 1:
                    filename = [max(filename)]
                    data = zfile.read(filename[0]).decode('ascii')
                    self.source_file(filepath, filename[0])

                else:
                    FileExistsError(f"No .cal file found in {filepath}")

        elif filepath.lower().endswith('.cal'):
            if 'z' not in filepath.lower().split('/')[-1]:
                with open(filepath) as file:
                    data = file.read()
                self.source_file(filepath, file)
        else:
            pass

        return data

    def source_file(self, filepath, filename):
        """
        Routine which parses out the source file and filename
        where the calibration coefficients are sourced from.
        """

        if filepath.lower().endswith('.cal'):
            dcn = filepath.split('/')[-2]
            filename = filepath.split('/')[-1]
        else:
            dcn = filepath.split('/')[-1]

        self.source = f'Source file: {dcn} > {filename}'

    def parse_cal(self, data):
        """
        Parses the NUTNR calibration data. Data should be read into memory and
        in ascii format.

        Args:
            data - ascii-formatted calibration information read into memory
        Returns:
            coefficients - a dictionary of key:value pairs consisting of the
                coefficient name:array of values.
        """

        for k, line in enumerate(data.splitlines()):

            if line.startswith('H'):
                _, info, *ignore = line.split(',')

                # The first line of the cal file contains the serial number
                if k == 0:
                    _, sn, *ignore = info.split()
                    if 'SUNA' in info:
                        self.serial = 'NTR-' + sn
                    else:
                        self.serial = sn

                # File creation time is when the instrument was calibrated.
                # May be multiple times for different cal coeffs
                if 'file creation time' in info.lower():
                    _, _, _, date, time = info.split()
                    date_time = pd.to_datetime(date + ' ' + time).strftime('%Y%m%d')
                    self.date.append(date_time)

                # The temperature at which it was calibrated
                if 't_cal_swa' in info.lower() or 't_cal' in info.lower():
                    _, cal_temp = info.split()
                    self.coefficients['CC_cal_temp'] = cal_temp

            # Now parse the calibration coefficients
            if line.startswith('E'):
                _, wl, eno3, eswa, _, di = line.split(',')

                self.coefficients['CC_wl'].append(wl)
                self.coefficients['CC_di'].append(di)
                self.coefficients['CC_eno3'].append(eno3)
                self.coefficients['CC_eswa'].append(eswa)

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
            to_csv - a csv of the calibration coefficients which is
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

        # Define a function to reformat the notes into an uniform system
        def reformat_notes(x):
            # First, get rid of
            try:
                np.isnan(x)
                x = ''
            except:
                x = str(x).replace('[', '').replace(']', '')
            return x

        # Now merge the coefficients dataframe with the notes
        if len(self.notes) > 0:
            notes = pd.DataFrame().from_dict({
                'name': list(self.notes.keys()),
                'notes': list(self.notes.values())
            })
            df = df.merge(notes, how='outer', left_on='name', right_on='name')
        else:
            df['notes'] = ''

        # Add in the source file
        df['notes'].iloc[0] = df['notes'].iloc[0] + ' ' + self.source

        # Sort the data by the coefficient name
        df = df.sort_values(by='name')

        # Generate the csv name
        cal_date = max(self.date)
        csv_name = self.uid + '__' + cal_date + '.csv'

        # Write the dataframe to a csv file
        check = input(f"Write {csv_name} to {outpath}? [y/n]: ")
        # check = 'y'
        if check.lower().strip() == 'y':
            df.to_csv(outpath+'/'+csv_name, index=False)
