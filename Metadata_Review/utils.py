#!/usr/bin/env python

import datetime
import re
import os
from wcmatch import fnmatch
import xml.etree.ElementTree as et
import pandas as pd
import numpy as np
import string
from zipfile import ZipFile
import csv
import PyPDF2
from nltk.tokenize import word_tokenize
import json


class CTDCalibration():
    # Class that stores calibration values for CTDs.

    def __init__(self, uid):
        self.serial = ''
        self.uid = uid
        self.ctd_type = uid
        self.coefficients = {}
        self.date = {}

        self.coefficient_name_map = {
            'TA0': 'CC_a0',
            'TA1': 'CC_a1',
            'TA2': 'CC_a2',
            'TA3': 'CC_a3',
            'CPCOR': 'CC_cpcor',
            'CTCOR': 'CC_ctcor',
            'CG': 'CC_g',
            'CH': 'CC_h',
            'CI': 'CC_i',
            'CJ': 'CC_j',
            'G': 'CC_g',
            'H': 'CC_h',
            'I': 'CC_i',
            'J': 'CC_j',
            'PA0': 'CC_pa0',
            'PA1': 'CC_pa1',
            'PA2': 'CC_pa2',
            'PTEMPA0': 'CC_ptempa0',
            'PTEMPA1': 'CC_ptempa1',
            'PTEMPA2': 'CC_ptempa2',
            'PTCA0': 'CC_ptca0',
            'PTCA1': 'CC_ptca1',
            'PTCA2': 'CC_ptca2',
            'PTCB0': 'CC_ptcb0',
            'PTCB1': 'CC_ptcb1',
            'PTCB2': 'CC_ptcb2',
            # additional types for series O
            'C1': 'CC_C1',
            'C2': 'CC_C2',
            'C3': 'CC_C3',
            'D1': 'CC_D1',
            'D2': 'CC_D2',
            'T1': 'CC_T1',
            'T2': 'CC_T2',
            'T3': 'CC_T3',
            'T4': 'CC_T4',
            'T5': 'CC_T5',
        }

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
            'i': 'CC_i',
            'ptempa0': 'CC_ptempa0',
            'prange': 'CC_p_range',
            'ctcor': 'CC_ctcor',
            'a1': 'CC_a1',
            'j': 'CC_j',
            'ptempa2': 'CC_ptempa2',
            'pa1': 'CC_pa1',
            'ptca1': 'CC_ptca1',
            'ptca2': 'CC_ptca2',
        }

        self.o2_coefficients_map = {
            'A': 'CC_residual_temperature_correction_factor_a',
            'B': 'CC_residual_temperature_correction_factor_b',
            'C': 'CC_residual_temperature_correction_factor_c',
            'E': 'CC_residual_temperature_correction_factor_e',
            'SOC': 'CC_oxygen_signal_slope',
            'OFFSET': 'CC_frequency_offset'
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

    def load_pdf(self, filepath):
        """
        This function opens and loads a pdf into a parseable format.

        Args:
            filepath - full directory path with filename
        Raises:
            IOError - error reading or loading text from the pdf object
        Returns:
            text - a dictionary with page numbers as keys and the pdf text as items
        """

        # Open and read the pdf file
        pdfFileObj = open(filepath, 'rb')
        # Create a reader to be parsed
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        # Now, enumerate through the pdf and decode the text
        num_pages = pdfReader.numPages
        count = 0
        text = {}

        while count < num_pages:
            pageObj = pdfReader.getPage(count)
            count = count + 1
            text.update({count: pageObj.extractText()})

        # Run a check that text was actually extracted
        if len(text) == 0:
            raise(IOError(f'No text was parsed from the pdf file {filepath}'))
        else:
            return text

    def read_pdf(self, filepath):
        """
        Function which parses the opened and loaded pdf file into the
        relevant calibration coefficient data. This function works if
        the calibration pdfs have been split based on sensor as well as
        for combined pdfs.

        Args:
            text - the opened and loaded pdf text returned from load_pdf
        Raises:
            Exception - thrown when a relevant calibration information is
                missing from the text
        Returns:
            date - the calibration dates of the temperature, conductivity,
                and pressure sensors of the CTDMO in a dictionary object
            serial - populated serial number of the CTDMO
            coefficients - populated dictionary of the calibration coefficients
                as keys and associated values as items.
        """
        text = self.load_pdf(filepath)

        for page_num in text.keys():
            # Search for the temperature calibration data
            if 'SBE 37 TEMPERATURE CALIBRATION DATA' in text[page_num]:
                tokens = word_tokenize(text[page_num])
                data = [word.lower() for word in tokens if not word in string.punctuation]
                # Now, find and record the calibration date
                if 'calibration' and 'date' in data:
                    cal_ind = data.index('calibration')
                    date_ind = data.index('date')
                    # Run check they are in order
                    if date_ind == cal_ind+1:
                        date = pd.to_datetime(data[date_ind+1]).strftime('%Y%m%d')
                        self.date.update({'TCAL':date})
                    else:
                        raise Exception(f"Can't locate temp calibration date.")
                else:
                    raise Exception(f"Can't locate temp calibration date.")

                # Check for the serial number
                if 'serial' and 'number' in data and len(self.serial) == 0:
                    ser_ind = data.index('serial')
                    num_ind = data.index('number')
                    if num_ind == ser_ind+1:
                        self.serial = data[num_ind+1]
                    else:
                        pass

                # Now, get the calibration coefficients
                for key in self.mo_coefficient_name_map.keys():
                    if key in data:
                        ind = data.index(key)
                        self.coefficients.update({self.mo_coefficient_name_map[key]: data[ind+1]})
                    else:
                        pass

            # Search for the conductivity calibration data
            elif 'SBE 37 CONDUCTIVITY CALIBRATION DATA' in text[page_num]:
                # tokenize the text data and extract only key words
                tokens = word_tokenize(text[page_num])
                data = [word.lower() for word in tokens if not word in string.punctuation]

                # Now, find and record the calibration date
                if 'calibration' and 'date' in data:
                    cal_ind = data.index('calibration')
                    date_ind = data.index('date')
                    # Run check they are in order
                    if date_ind == cal_ind+1:
                        date = pd.to_datetime(data[date_ind+1]).strftime('%Y%m%d')
                        self.date.update({'CCAL': date})
                    else:
                        raise Exception(f"Can't locate conductivity calibration date.")
                else:
                    raise Exception(f"Can't locate conductivity calibration date.")

                # Check for the serial number
                if 'serial' and 'number' in data and len(self.serial) == 0:
                    ser_ind = data.index('serial')
                    num_ind = data.index('number')
                    if num_ind == ser_ind+1:
                        self.serial = data[num_ind+1]
                    else:
                        pass

                # Now, get the calibration coefficients
                for key in self.mo_coefficient_name_map.keys():
                    if key in data:
                        ind = data.index(key)
                        self.coefficients.update({self.mo_coefficient_name_map[key]: data[ind+1]})
                    else:
                        pass

            elif 'SBE 37 PRESSURE CALIBRATION DATA' in text[page_num]:
                # tokenize the text data and extract only key words
                tokens = word_tokenize(text[page_num])
                data = [word.lower() for word in tokens if not word in string.punctuation]

                # Now, find and record the calibration date
                if 'calibration' and 'date' in data:
                    cal_ind = data.index('calibration')
                    date_ind = data.index('date')
                    # Run check they are in order
                    if date_ind == cal_ind+1:
                        date = pd.to_datetime(data[date_ind+1]).strftime('%Y%m%d')
                        self.date.update({'PCAL': date})
                    else:
                        raise Exception(f"Can't locate pressure calibration date.")
                else:
                    raise Exception(f"Can't locate pressure calibration date.")

                # Check for the serial number
                if 'serial' and 'number' in data and len(self.serial) == 0:
                    ser_ind = data.index('serial')
                    num_ind = data.index('number')
                    if num_ind == ser_ind+1:
                        self.serial = data[num_ind+1]
                    else:
                        pass

                # Now, get the calibration coefficients
                for key in self.mo_coefficient_name_map.keys():
                    if key in data:
                        ind = data.index(key)
                        self.coefficients.update({self.mo_coefficient_name_map[key]: data[ind+1]})
                    else:
                        pass

            # Now check for other important information
            else:
                tokens = word_tokenize(text[page_num])
                data = [word.lower() for word in tokens if not word in string.punctuation]

                # Now, find the sensor rating
                if 'sensor' and 'rating' in data:
                    ind = data.index('rating')
                    self.coefficients.update({self.mo_coefficient_name_map['prange']: data[ind+1]})

    def read_cal(self, data):
        """
        Function which reads and parses the CTDBP calibration values stored
        in a .cal file.

        Args:
            filename - the name of the calibration (.cal) file to load. If the
                cal file is not located in the same directory as this script, the
                full filepath also needs to be specified.
        Returns:
            self.coefficients - populated coefficients dictionary
            self.date - the calibration dates associated with the calibration values
            self.type - the type (i.e. 16+/37-IM) of the CTD
            self.serial - populates the 5-digit serial number of the instrument
        """

        for line in data.splitlines():
            key, value = line.replace(" ", "").split('=')

            if key == 'INSTRUMENT_TYPE':
                if value == 'SEACATPLUS':
                    ctd_type = '16'
                elif value == '37SBE':
                    ctd_type = '37'
                else:
                    ctd_type = ''
                if self.ctd_type != ctd_type:
                    raise ValueError(f'CTD type in cal file {ctd_type} does not match the UID type {self.ctd_type}')

            elif key == 'SERIALNO':
                if self.serial != value.zfill(5):
                    raise Exception(f'Serial number {value.zfill(5)} stored in cal file does not match {self.serial} from the UID.')

            elif 'CALDATE' in key:
                self.date.update({key: datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})

            else:
                if self.ctd_type == '16':
                    name = self.coefficient_name_map.get(key)
                elif self.ctd_type == '37':
                    name = self.mo_coefficient_name_map.get(key)
                else:
                    pass

                if not name or name is None:
                    continue
                else:
                    self.coefficients.update({name: value})

    def load_cal(self, filepath):
        """
        Loads all of the calibration coefficients from the vendor cal files for
        a given CTD instrument class.

        Args:
            filepath - directory path to where the zipfiles are stored locally
        Raises:
            FileExistsError - Checks the given filepath that a .cal file exists
        Returns:
            self.coefficients - populated coefficients dictionary
            self.date - the calibration dates associated with the calibration values
            self.type - the type (i.e. 16+/37-IM) of the CTD
            self.serial - populates the 5-digit serial number of the instrument
        """

        if filepath.endswith('.zip'):
            with ZipFile(filepath) as zfile:
                filename = [name for name in zfile.namelist() if '.cal' in name]
                if len(filename) > 0:
                    data = zfile.read(filename[0]).decode('ASCII')
                    self.read_cal(data)
                else:
                    FileExistsError(f"No .cal file found in {filepath}.")

        elif filepath.endswith('.cal'):
            with open(filepath) as filename:
                data = filename.read()
                self.read_cal(data)

        else:
            FileExistsError(f"No .cal file found in {filepath}.")

    def read_xml(self, data):
        """
        Function which reads and parses the CTDBP calibration values stored
        in the xmlcon file.

        Args:
            data - the data string to parse
        Returns:
            self.coefficients - populated coefficients dictionary
            self.date - the calibration dates associated with the calibration values
            self.type - the type (i.e. 16+/37-IM) of the CTD
            self.serial - populates the 5-digit serial number of the instrument
        """

        Tflag  = False
        Cflag  = False
        O2flag = False

        for child in data.iter():
            key = child.tag.upper()
            value = child.text.upper()

            if key == 'NAME':
                if '16PLUS' in value:
                    ctd_type = '16'
                    if self.ctd_type != ctd_type:
                        raise ValueError(f'CTD type in xmlcon file {ctd_type} does not match the UID type {self.ctd_type}')

            # Check if we are processing the calibration values for the temperature sensor
            # If we already have parsed the Temp data, need to turn the flag off
            if key == 'TEMPERATURESENSOR':
                Tflag = True
            elif 'SENSOR' in key and Tflag == True:
                Tflag = False
            else:
                pass

            # Check on if we are now parsing the conductivity data
            if key == 'CONDUCTIVITYSENSOR':
                Cflag = True
            elif 'SENSOR' in key and Cflag == True:
                Cflag = False
            else:
                pass

            # Check if an oxygen sensor has been appended to the CTD configuration
            if key == 'OXYGENSENSOR':
                O2flag = True

            # Check that the serial number in the xmlcon file matches the serial
            # number from the UID
            if key == 'SERIALNUMBER':
                if self.serial != value.zfill(5):
                    raise Exception(f'Serial number {value.zfill(5)} stored in xmlcon file does not match {self.serial} from the UID.')

            # Parse the calibration dates of the different sensors
            if key == 'CALIBRATIONDATE':
                if Tflag:
                    self.date.update({'TCALDATE':datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})
                elif Cflag:
                    self.date.update({'CCALDATE': datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})
                else:
                    self.date.update({'PCALDATE': datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})

            # Now, we get to parse the actual calibration values, but it is necessary to make sure the
            # key names are correct
            if Tflag:
                key = 'T'+key

            name = self.coefficient_name_map.get(key)
            if not name or name is None:
                if O2flag:
                    name = self.o2_coefficients_map.get(key)
                    self.coefficients.update({name: value})
                else:
                    pass
            else:
                self.coefficients.update({name: value})

    def load_xml(self, filepath):
        """
        Loads all of the calibration coefficients from the vendor xmlcon files for
        a given CTD instrument class.

        Args:
            filepath - the name of the xmlcon file to load and parse. If the
                xmlcon file is not located in the same directory as this script,
                the full filepath also needs to be specified. May point to a zipfile.
        Raises:
            FileExistsError - Checks the given filepath that an xmlcon file exists
        Returns:
            self.coefficients - populated coefficients dictionary
            self.date - the calibration dates associated with the calibration values
            self.type - the type (i.e. 16+/37-IM) of the CTD
            self.serial - populates the 5-digit serial number of the instrument
        """

        if filepath.endswith('.zip'):
            with ZipFile(filepath) as zfile:
                filename = [name for name in zfile.namelist() if '.xmlcon' in name]
                if len(filename) > 0:
                    data = et.parse(zfile.open(filename[0]))
                    self.read_xml(data)
                else:
                    FileExistsError(f"No .cal file found in {filepath}.")

        elif filepath.endswith('.xmlcon'):
            with open(filepath) as file:
                data = et.parse(file)
                self.read_xml(data)

        else:
            FileExistsError(f"No .cal file found in {filepath}.")

    def load_qct(self, filepath):
        """
        Function which parses the output from the QCT check-in and loads them into
        the CTD object.

        Args:
            filepath - the full directory path and filename
        Raises:
            ValueError - checks if the serial number parsed from the UID matches the
                the serial number stored in the file.
        Returns:
            self.coefficients - populated coefficients dictionary
            self.date - the calibration dates associated with the calibration values
            self.type - the type (i.e. 16+/37-IM) of the CTD
            self.serial - populates the 5-digit serial number of the instrument
        """

        with open(filepath) as filename:
            data = filename.read()

        if self.ctd_type == '37':
            data = data.replace('<', ' ').replace('>', ' ')
            for line in data.splitlines():
                keys = list(self.mo_coefficient_name_map.keys())
                if any([word for word in line.split() if word.lower() in keys]):
                    name = self.mo_coefficient_name_map.get(line.split()[0])
                    value = line.split()[-1]
                    self.coefficients.update({name: value})

        elif self.ctd_type == '16':
            for line in data.splitlines():
                keys = list(self.coefficient_name_map.keys())
                if any([word for word in line.split() if word in keys]):
                    name = self.coefficient_name_map.get(line.split()[0])
                    value = line.split()[-1]
                    self.coefficients.update({name: value})

                if 'temperature:' in line:
                    self.date.update({'TCAL': pd.to_datetime(line.split()[-1]).strftime('%Y%m%d')})
                elif 'conductivity:' in line:
                    self.date.update({'CCAL': pd.to_datetime(line.split()[-1]).strftime('%Y%m%d')})
                elif 'pressure S/N' in line:
                    self.date.update({'PCAL': pd.to_datetime(line.split()[-1]).strftime('%Y%m%d')})
                else:
                    pass

                if 'SERIAL NO.' in line:
                    ind = line.split().index('NO.')
                    serial_num = line.split()[ind+1]
                    if self.serial != serial_num:
                        raise ValueError(f'UID serial number {self.serial} does not match the QCT serial num {serial_num}')

                if 'SBE 16Plus' in line:
                    if self.ctd_type is not '16':
                        raise TypeError(f'CTD type {self.ctd_type} does not match the qct.')

        else:
            pass

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

        # Now write to
        check = input(f"Write {csv_name} to {outpath}? [y/n]: ")
        if check.lower().strip() == 'y':
            df.to_csv(outpath+'/'+csv_name, index=False)


class DOSTACalibration():
    # Class that stores calibration values for DOSTA's.

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

    def generate_file_path(self, dirpath, filename, ext=['.cap', '.txt', '.log'], exclude=['_V', '_Data_Workshop']):
        """
        Function which searches for the location of the given file and returns
        the full path to the file.

        Args:
            dirpath - parent directory path under which to search
            filename - the name of the file to search for
            ext - file endings to search for
            exclude - optional list which allows for excluding certain
                directories from the search
        Returns:
            fpath - the file path to the filename from the current
                working directory.
        """
        # Check if the input file name has an extension already
        # If it does, parse it for input into the search algo
        if '.' in filename:
            check = filename.split('.')
            filename = check[0]
            ext = ['.'+check[1]]

        for root, dirs, files in os.walk(dirpath):
            dirs[:] = [d for d in dirs if d not in exclude]
            for fname in files:
                if fnmatch.fnmatch(fname, [filename+'*'+x for x in ext]):
                    fpath = os.path.join(root, fname)
                    return fpath

    def load_qct(self, filepath):
        """
        Function which parses the output from the QCT check-in and loads them into
        the DOSTA object.

        Args:
            filepath - the full directory path and filename
        Raises:
            ValueError - checks if the serial number parsed from the UID matches the
                the serial number stored in the file.
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
            if self.coefficients[key] == None:
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


def generate_file_path(dirpath, filename, ext=['.cap', '.txt', '.log'], exclude=['_V', '_Data_Workshop']):
    """
    Function which searches for the location of the given file and returns
    the full path to the file.

    Args:
        dirpath - parent directory path under which to search
        filename - the name of the file to search for
        ext - file endings to search for
        exclude - optional list which allows for excluding certain
            directories from the search
    Returns:
        fpath - the file path to the filename from the current
            working directory.
    """
    # Check if the input file name has an extension already
    # If it does, parse it for input into the search algo
    if '.' in filename:
        check = filename.split('.')
        filename = check[0]
        ext = ['.'+check[1]]

    for root, dirs, files in os.walk(dirpath):
        dirs[:] = [d for d in dirs if d not in exclude]
        for fname in files:
            if fnmatch.fnmatch(fname, [filename+'*'+x for x in ext]):
                fpath = os.path.join(root, fname)
                return fpath


def whoi_asset_tracking(spreadsheet, sheet_name, instrument_class='All', whoi=True, series=None):
    """
    Loads all the individual sensors of a specific instrument class and
    series type. Currently applied only for WHOI deployed instruments.

    Args:
        spreadsheet - directory path and name of the excel spreadsheet with
            the WHOI asset tracking information.
        sheet_name - name of the sheet in the spreadsheet to load
        instrument_class - the type (i.e. CTDBP, CTDMO, PCO2W, etc). Defaults
            to 'All', which will load all of the instruments
        whoi - return only whoi instruments? Defaults to True.
        series - a specified class of the instrument to load. Defaults to None,
            which will load all of the series for a specified instrument class
    """

    all_sensors = pd.read_excel(spreadsheet, sheet_name=sheet_name, header=1)
    # Select a specific class of instruments
    if instrument_class == 'All':
        inst_class = all_sensors
    else:
        inst_class = all_sensors[all_sensors['Instrument\nClass'] == instrument_class]
    # Return only the whoi instruments?
    if whoi:
        whoi_insts = inst_class[inst_class['Deployment History'] != 'EA']
    else:
        whoi_insts = inst_class
    # Slect a specific series of the instrument?
    if series is not None:
        instrument = whoi_insts[whoi_insts['Series'] == series]
    else:
        instrument = whoi_insts

    return instrument


def load_asset_management(instrument, filepath):
    """
    Loads the calibration csv files from a local repository containing
    the asset management information.

    Args:
        instrument - a pandas dataframe with the asset tracking information
            for a specific instrument.
        filepath - the directory path pointing to where the csv files are
            stored.
    Raises:
        TypeError - if the instrument input is not a pandas dataframe
    Returns:
        csv_dict - a dictionary with keys of the UIDs from the instrument dataframe
            which correspond to lists of the relevant calibration csv files

    """

    # Check that the input is a pandas DataFrame
    if type(instrument) != pd.core.frame.DataFrame:
        raise TypeError()

    uids = sorted(list(set(instrument['UID'])))

    csv_dict = {}
    for uid in uids:
        # Get a specified uid from the instrument dataframe
        instrument['UID_match'] = instrument['UID'].apply(lambda x: True if uid in x else False)
        instrument[instrument['UID_match'] == True]

        # Now, get all the csvs from asset management for a particular UID
        csv_files = []
        for file in os.listdir(filepath):
            if fnmatch.fnmatch(file,'*'+uid+'*'):
                csv_files.append(file)
            else:
                pass

        # Update the dictionary storing the asset management files for each UID
        if len(csv_files) > 0:
            csv_dict.update({uid:csv_files})
        else:
            pass

    return csv_dict


def all_the_same(elements):
    """
    This function checks which values in an array are all the same.

    Args:
        elements - an array of values
    Returns:
        error - an array of length (m-1) which checks if

    """
    if len(elements) < 1:
        return True
    el = iter(elements)
    first = next(el, None)
    # check = [element == first for element in el]
    error = [np.isclose(element, first) for element in el]
    return error


def locate_cal_error(array):
    """
    This function locates which source file (e.g. xmlcon vs csv vs cal)
    have calibration values that are different from the others. It does
    NOT identify which is correct, only which is different.

    Args:
        array - A numpy array which contains the values for a specific
                calibration coefficient for a specific date from all of
                the calibration source files
    Returns:
        dataset - a list containing which calibration sources are different
                from the other files
        True - if all of the calibration values are the same
        False - if the first calibration value is different
    """
    # Call the function to check if there are any differences between each of
    # calibration values from the different sheets
    error = all_the_same(array)
    # If they are all the same, return True
    if all(error):
        return True
    # If there is a mixture of True/False, find the false and return them
    elif any(error) == True:
        indices = [i+1 for i, j in enumerate(error) if j is False]
        dataset = list(array.index[indices])
        return dataset
    # Last, if all are false, that means the first value
    else:
        return False


def search_for_errors(df):
    """
    This function is designed to search through a pandas dataframe
    which contains all of the calibration coefficients from all of
    the files, and check for differences.

    Args:
        df - A dataframe which contains all fo the calibration coefficients
        from the asset management csv, qct checkout, and the vendor
        files (.cal and .xmlcon)
    Returns:
        cal_errors - A nested dictionary containing the calibration timestamp,
        the relevant calibration coefficient, and which file(s) have the
        erroneous calibration file.
    """

    cal_errors = {}
    for date in np.unique(df['Cal Date']):
        df2 = df[df['Cal Date'] == date]
        wrong_cals = {}
        for column in df2.columns.values:
            array = df2[column]
            array.sort_index()
            if array.dtype == 'datetime64[ns]':
                pass
            else:
                error = locate_cal_error(array)
                if error is False:
                    wrong_cals.update({column: array.index[0]})
                elif error is True:
                    pass
                else:
                    wrong_cals.update({column: error})

        if len(wrong_cals) < 1:
            cal_errors.update({str(date).split('T')[0]: 'No Errors'})
        else:
            cal_errors.update({str(date).split('T')[0]: wrong_cals})

    return cal_errors


def convert_type(x):
    """
    Converts type from string to float
    """
    if type(x) is str:
        return float(x)
    else:
        return x


def get_instrument_sn(df):
    serial_num = list(df[df['UID_match'] == True]['Supplier\nSerial Number'])
    serial_num = serial_num[0].split('-')[1]
    return serial_num


def get_serial_nums(df, uids):
    """
    Returns the serial numbers of all the instrument uids.

    Args:
        df - dataframe with the asset management information
        uids - list of the uids for the instruments
    Returns:
        serial_nums - a dictionary of uids (key) matched to their
            respective serial numbers

    """
    serial_nums = {}

    for uid in uids:
        df['UID_match'] = df['UID'].apply(lambda x: True if uid in x else False)
        serial_num = list(df[df['UID_match'] == True]['Supplier\nSerial Number'])
        if 'CTD' in uid:
            serial_num = serial_num[0].split('-')[1]
        serial_nums.update({uid: serial_num})

    return serial_nums