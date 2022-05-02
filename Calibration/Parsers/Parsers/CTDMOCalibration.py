import re
import pandas as pd
import numpy as np
import textract


class CTDMOCalibration():
    """Class which loads, stores, and writes the CTDMO Calibration csvs"""

    def __init__(self, uid):
        self.serial = ''
        self.uid = uid
        self.ctd_type = uid
        self.coefficients = {}
        self.date = {}

        # Name mapping for the MO-type CTDs (when reading from pdfs)
        self.mo_coefficient_name_map = {
            'ptcb1': 'CC_ptcb1',
            'pa2': 'CC_pa2',
            'a3': 'CC_a3',
            'pa0': 'CC_pa0',
            'wbotc': 'CC_wbotc',
            'ptcb0': 'CC_ptcb0',
            'g': 'CC_g',
            'ptempa1': 'CC_ptempa1',
            'ptcb2': 'CC_ptcb2',
            'a0': 'CC_a0',
            'h': 'CC_h',
            'ptca0': 'CC_ptca0',
            'a2': 'CC_a2',
            'cpcor': 'CC_cpcor',
            'pcor': 'CC_cpcor',
            'i': 'CC_i',
            'ptempa0': 'CC_ptempa0',
            'prange': 'CC_p_range',
            'ctcor': 'CC_ctcor',
            'tcor': 'CC_ctcor',
            'a1': 'CC_a1',
            'j': 'CC_j',
            'ptempa2': 'CC_ptempa2',
            'pa1': 'CC_pa1',
            'ptca1': 'CC_ptca1',
            'ptca2': 'CC_ptca2',
        }

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

    @property
    def ctd_type(self):
        return self._ctd_type

    @ctd_type.setter
    def ctd_type(self, d):
        if 'MO' in d:
            self._ctd_type = '37'
        elif 'BP' in d:
            self._ctd_type = '16'
        else:
            self._ctd_type = ''

    def mo_parse_pdf(self, filepath):
        """
        This function extracts the text from a given pdf file.
        Depending on if the text concerns calibration for
        temperature/conductivity or pressure, it calls a further
        function to parse out the individual calibration coeffs.

        Args:
            filepath: the full directory path to the pdf file
                which it to be extracted and parsed.

        Calls:
            mo_parse_p(text, filepath)
            mo_parse_ts(text)

        Attributes:
            coefficients {str:float}: a dictionary containing the calibration
                coefficient names as keys and values as items
        """

        text = textract.process(filepath, encoding='utf-8')
        text = text.decode('utf-8')

        if 'PRESSURE CALIBRATION DATA' in text:
            self.mo_parse_p(filepath)

        elif 'TEMPERATURE CALIBRATION DATA' or 'CONDUCTIVITY CALIBRATION DATA' in text:
            self.mo_parse_ts(text)

        else:
            pass

    def mo_parse_ts(self, text):
        """
        This function parses text from a pdf and loads the appropriate calibration
        coefficients for the temperature and conductivity sensors into the CTDMO
        calibration object.

        Args:
            text: extracted text from a single pdf page

        Attributes:
            coefficients {str:float}: a dictionary containing the calibration
                coefficient names as keys and values as items
            date: calibration dates of the temperature and conductivity

        Raises:
            Exception: if the serial number in the pdf text does not match the
                serial number parsed from the UID
        """

        keys = self.mo_coefficient_name_map.keys()
        for line in text.splitlines():

            if 'CALIBRATION DATE' in line:
                *ignore, cal_date = line.split(':')
                cal_date = pd.to_datetime(cal_date).strftime('%Y%m%d')
                self.date.update({len(self.date): cal_date})

            elif 'SERIAL NUMBER' in line:
                *ignore, serial_num = line.split(':')
                serial_num = serial_num.strip()
                if serial_num != self.serial:
                    raise Exception(
                        f'Instrument serial number {serial_num} does not match UID serial num {self.serial}')

            elif '=' in line:
                key, *ignore, value = line.split()
                name = self.mo_coefficient_name_map.get(key.strip().lower())
                if name is not None:
                    self.coefficients.update({name: value.strip()})
            else:
                continue

    def mo_parse_p(self, filepath):
        """
        Function to parse the pressure calibration information from a pdf. To parse
        the pressure cal info requires re-extracting the text from the pdf file using
        tesseract-ocr rather than the basic pdf2text converter.

        Args:
            text: extracted text from a pdf page using pdf2text
            filepath: full directory path to the pdf file containing the pressure
                calibration info. This is the file which will be re-extracted.

        Attributes:
            coefficients {str:float}: a dictionary containing the calibration
                coefficient names as keys and values as items
            date: calibration dates of the pressure sensor
        """

        # Now, can reprocess using tesseract-ocr rather than pdftotext
        ptext = textract.process(filepath, method='tesseract', encoding='utf-8')
        ptext = ptext.replace(b'\xe2\x80\x94', b'-')
        ptext = ptext.decode('utf-8')
        keys = list(self.mo_coefficient_name_map.keys())

        # Get the calibration date:
        for line in ptext.splitlines():
            if 'CALIBRATION DATE' in line:
                items = line.split()
                ind = items.index('DATE:')
                cal_date = items[ind+1]
                cal_date = pd.to_datetime(cal_date).strftime('%Y%m%d')
                self.date.update({len(self.date): cal_date})

            if 'psia S/N' in line:
                items = line.split()
                ind = items.index('psia')
                prange = items[ind-1]
                name = self.mo_coefficient_name_map.get('prange')
                self.coefficients.update({name: prange})

            # Loop through each line looking for the lines which contain
            # calibration coefficients
            if '=' in line:
                # Tesseract-ocr misreads '0' as O, and 1 as IL
                line = line.replace('O', '0').replace('IL', '1').replace(
                    '=', '').replace(',.', '.').replace(',', '.')
                line = line.replace('L', '1').replace('@', '0').replace('l', '1').replace('--', '-')
                if '11' in line and 'PA2' not in line:
                    line = line.replace('11', '1')
                items = line.split()
                for n, k in enumerate(items):
                    if k.lower() in keys:
                        try:
                            float(items[n+1])
                            name = self.mo_coefficient_name_map.get(k.lower())
                            self.coefficients.update({name: items[n+1]})
                        except:
                            pass
        if 'CC_ptcb2' not in list(self.mo_coefficient_name_map.keys()):
            self.coefficients.update({'CC_ptcb2': '0.000000e+000'})

    def mo_parse_cal(self, filepath):
        """
        Parse and load the CTDMO calibration coefficients from a vendor .cal files

        Args:
            filepath: full directory path to where the .cal file is stored

        Attributes:
            coefficients {str:float}: a dictionary containing the calibration
                coefficient names as keys and values as items
            date: all available calibration dates of the sensors

        Raises:
            Exceptions: if either the serial number or ctd type interpreted from
                the instrument UID does not match the information in the file
        """

        if not filepath.endswith('.cal'):
            raise Exception(f'Not a .cal filetype.')

        with open(filepath) as file:
            data = file.read()

        for line in data.splitlines():
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()

            if 'SERIALNO' in key:
                sn = value
                if self.serial != sn:
                    raise Exception(f'File serial number {sn} does not match UID {self.uid}')

            elif 'CALDATE' in key:
                cal_date = pd.to_datetime(value).strftime('%Y%m%d')
                self.date.update({len(self.date): cal_date})

            elif 'INSTRUMENT_TYPE' in key:
                ctd_type = value[-2:]
                if self.ctd_type != ctd_type:
                    raise Exception(f'CTD type {ctd_type} does not match uid {self.uid}.')

            else:
                if key.startswith('T'):
                    key = key.replace('T', '')
                if key.startswith('C') and len(key) == 2:
                    key = key.replace('C', '')
                name = self.mo_coefficient_name_map.get(key.lower())
                if name is not None:
                    self.coefficients.update({name: value})

        # Now we need to add in the range of the sensor
        name = self.mo_coefficient_name_map.get('prange')
        self.coefficients.update({name: '1450'})

    def mo_parse_qct(self, filepath):
        """
        Parse and load the CTDMO calibration coefficients from QCT checkin

        Args:
            filepath: full directory path and filename of the QCT file

        Attributes:
            coefficients {str:float}: a dictionary containing the calibration
                coefficient names as keys and values as items
            date: all available calibration dates of the sensors

        Raises:
            Exceptions: if either the serial number or ctd type interpreted from
                the instrument UID does not match the information in the file
        """

        with open(filepath, errors='ignore') as file:
            data = file.read()

        data = data.replace('<', ' ').replace('>', ' ')
        keys = self.mo_coefficient_name_map.keys()

        for line in data.splitlines():
            items = line.split()

            # If the line is empty, go to next line
            if len(items) == 0:
                continue

            # Check the serial number from the instrument
            elif 'SERIAL NO' in line:
                ind = items.index('NO.')
                sn = items[ind+1]
                if sn != self.serial:
                    raise Exception(
                        f'Serial number {sn} in QCT document does not match uid serial number {self.serial}')

            # Check if the line contains the calibration date
            elif 'CalDate' in line:
                cal_date = pd.to_datetime(items[1]).strftime('%Y%m%d')
                self.date.update({len(self.date): cal_date})

            # Get the coefficient names and values
            elif items[0].lower() in keys:
                name = self.mo_coefficient_name_map[items[0].lower()]
                self.coefficients.update({name: items[1]})

            else:
                pass

    def write_csv(self, savedir):
        """
        This function writes the correctly named csv file for the ctd to the
        specified directory.

        Args:
            savedir: directory path of where to write the csv file

        Raises:
            ValueError: if the CTDMO object's coefficient dictionary
                has not been populated

        Returns:
            csv: a csv of the calibration coefficients which is written to the
                specified directory from the savedir.
        """

        # Run a check that the coefficients have actually been loaded
        if len(self.coefficients) == 0:
            raise ValueError('No calibration coefficients have been loaded.')

        # Create a dataframe to write to the csv
        data = {'serial': [self.ctd_type + '-' + self.serial]*len(self.coefficients),
                'name': list(self.coefficients.keys()),
                'value': list(self.coefficients.values()),
                'notes': ['']*len(self.coefficients)
                }
        df = pd.DataFrame().from_dict(data)

        # Generate the csv name
        cal_date = max(self.date.values())
        csv_name = self.uid + '__' + cal_date + '.csv'

        # Print out the csv dataframe for visual confirmation
        print(f'Calibration csv for {csv_name}:')
        print(df)

        # Write the dataframe to a csv file
        check = input(f"Write {csv_name} to {savedir}? [y/n]: ")
        check = 'y'
        if check.lower().strip() == 'y':
            df.to_csv(savedir+'/'+csv_name, index=False)
