#!/usr/bin/env python
import csv
import re
import datetime
import PyPDF2
import string
import pandas as pd
from zipfile import ZipFile
import xml.etree.ElementTree as et
from nltk.tokenize import word_tokenize


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
                        self.date.update({'TCAL': date})
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
                    raise ValueError(
                        f'CTD type in cal file {ctd_type} does not match the UID type {self.ctd_type}')

            elif key == 'SERIALNO':
                if self.serial != value.zfill(5):
                    raise Exception(
                        f'Serial number {value.zfill(5)} stored in cal file does not match {self.serial} from the UID.')

            elif 'CALDATE' in key:
                self.date.update({key: datetime.datetime.strptime(
                    value, '%d-%b-%y').strftime('%Y%m%d')})

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

        Tflag = False
        Cflag = False
        O2flag = False

        for child in data.iter():
            key = child.tag.upper()
            value = child.text.upper()

            if key == 'NAME':
                if '16PLUS' in value:
                    ctd_type = '16'
                    if self.ctd_type != ctd_type:
                        raise ValueError(
                            f'CTD type in xmlcon file {ctd_type} does not match the UID type {self.ctd_type}')

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
                    raise Exception(
                        f'Serial number {value.zfill(5)} stored in xmlcon file does not match {self.serial} from the UID.')

            # Parse the calibration dates of the different sensors
            if key == 'CALIBRATIONDATE':
                if Tflag == True:
                    self.date.update({'TCALDATE': datetime.datetime.strptime(
                        value, '%d-%b-%y').strftime('%Y%m%d')})
                elif Cflag == True:
                    self.date.update({'CCALDATE': datetime.datetime.strptime(
                        value, '%d-%b-%y').strftime('%Y%m%d')})
                else:
                    self.date.update({'PCALDATE': datetime.datetime.strptime(
                        value, '%d-%b-%y').strftime('%Y%m%d')})

            # Now, we get to parse the actual calibration values, but it is necessary to make sure the
            # key names are correct
            if Tflag == True:
                key = 'T'+key

            name = self.coefficient_name_map.get(key)
            if not name or name is None:
                if O2flag == True:
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
        
        if 'CTDBPP' in self.uid:
            with open(filepath, errors='ignore') as filename:
                data = filename.read()
            
            for line in data.splitlines():
                line = line.replace('<',' ').replace('>',' ').replace("'",'')
                
                if 'SerialNumber =' in line:
                    data = line.split()
                    sn = data[-1][3:]
                    if self.serial != sn:
                        raise Exception(f'UID {self.serial} does not match the serial number {sn} in the vendor documents')
                        
                    continue
                
                elif len(line) > 0:
                    data = line.split()
                    if data[0] in self.coefficient_name_map.keys():
                        name = self.coefficient_name_map.get(data[0])
                        self.coefficients.update({name: float(data[1])})
                        continue
                        
                    if data[0] == 'CalDate':
                        self.date.update({len(self.date): pd.to_datetime(data[1]).strftime('%Y%m%d')})
                        continue
                
                else:
                    pass
        
        else:

            with open(filepath, errors='ignore') as filename:
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
                        if self.serial != serial_num.zfill(5):
                            raise ValueError(f'UID serial number {self.serial} does not match the QCT serial num {serial_num}')

                    if 'SBE 16Plus' in line:
                        if self.ctd_type is not '16':
                            raise TypeError(f'CTD type {self.ctd_type} does not match the qct.')

            else:
                pass
        
    def source_file(self, filepath):
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
            self.source = f'Source file: {dcn} > {filename}'
        else:
            dcn = filepath.split('/')[-1]
            self.source = f'Source file: {dcn}'
            

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
        data = {
                'serial': [self.ctd_type + '-' + self.serial.lstrip('0')]*len(self.coefficients),
            'name': list(self.coefficients.keys()),
            'value': list(self.coefficients.values())
        }
        df = pd.DataFrame().from_dict(data)
        # Now merge the coefficients dataframe with the notes
        notes = pd.DataFrame().from_dict({
            'name': list(self.coefficients.keys()),
            'notes': ['']*len(self.coefficients),
        })
        df = df.merge(notes, how='outer', left_on='name', right_on='name')

        # Add in the source file
        df['notes'].iloc[0] = df['notes'].iloc[0] + ' ' + self.source

        # Sort the data by the coefficient name
        df = df.sort_values(by='name')

        # Generate the csv name
        cal_date = max(self.date.values())
        csv_name = self.uid + '__' + cal_date + '.csv'

        # Now write to
        check = input(f"Write {csv_name} to {outpath}? [y/n]: ")
        if check.lower().strip() == 'y':
            df.to_csv(outpath+'/'+csv_name, index=False)

if __name__ == '__main__':
     # Request the input data
     uid = input("Enter the instrument uid: ")
     fpath = input("Enter the full filepath to the calibration file: ")
     outpath = input("Enter the filepath where to save the parsed calibration: ")
     # Initialize the calibration parser
     ctdbp = CTDBPCalibrationParser(uid)
     ctdbp.load_cal(fpath)
     ctdbp.write_csv(outpath)
