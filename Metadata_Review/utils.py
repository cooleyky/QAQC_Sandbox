#!/usr/bin/env python

import datetime
import re
import os
from wcmatch import fnmatch
import xml.etree.ElementTree as et
import pandas as pd
import numpy as np
from zipfile import ZipFile


class CTDCalibration():
    # Class that stores calibration values for CTDs.

    def __init__(self, uid):
        self.serial = ''
        self.uid = uid
        self.coefficients = {}
        self.date = {}
        self.type = ''

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
            key, value = line.replace(" ","").split('=')

            if key == 'INSTRUMENT_TYPE' and value == 'SEACATPLUS':
                self.type = '16'

            elif key == 'SERIALNO':
                if self.serial != value.zfill(5):
                    raise Exception(f'Serial number {value.zfill(5)} stored in cal file does not match {self.serial} from the UID.')

            elif 'CALDATE' in key:
                self.date.update({key: datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})

            else:
                name = self.coefficient_name_map.get(key)
                if not name or name is None:
                    continue
                else:
                    self.coefficients.update({name:value})


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
                    self.type = '16'
    
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
                if Tflag == True:
                    self.date.update({'TCALDATE':datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})
                elif Cflag == True:
                    self.date.update({'CCALDATE':datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})
                else:
                    self.date.update({'PCALDATE':datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})
            
            # Now, we get to parse the actual calibration values, but it is necessary to make sure the
            # key names are correct
            if Tflag is True:
                key = 'T'+key
            
            name = self.coefficient_name_map.get(key)
            if not name or name is None:
                if O2flag == True:
                    name = self.o2_coefficients_map.get(key)
                    self.coefficients.update({name:value})
                else:
                    pass
            else:
                self.coefficients.update({name:value})
                
                
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

        for line in data.splitlines():
    
            if '16plus' in line.lower():
                self.type = '16'
        
            elif 'SERIAL NO.' in line:
                items = line.split()
                ind = items.index('NO.')
                value = items[ind+1].strip().zfill(5)
                if self.serial != value:
                    raise ValueError(f'Serial number {value.zfill(5)} from the QCT file does not match {self.serial} from the UID.')
                else:
                    pass
        
            else:
                items = re.split(': | =', line)
                key = items[0].strip()
                value = items[-1].strip()
        
                if key == 'temperature':
                    self.date.update({'TCALDATE':datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})    
        
                elif key == 'conductivity':
                    self.date.update({'CCALDATE':datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})
                
                elif key == 'pressure S/N':
                    self.date.update({'PCALDATE':datetime.datetime.strptime(value, '%d-%b-%y').strftime('%Y%m%d')})
            
                else:
                    name = self.coefficient_name_map.get(key)
                    if not name or name is None:
                        pass
                    else:
                        self.coefficients.update({name:value})
                
    
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
        data = {'serial':[self.type + '-' + self.serial]*len(self.coefficients),
               'name':list(self.coefficients.keys()),
               'value':list(self.coefficients.values()),
               'notes':['']*len(self.coefficients) }
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
        self.coefficients = {'CC_conc_coef':None,
                                'CC_csv':None}
        self.notes = {'CC_conc_coef':None,
                      'CC_csv':None}
                
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
            
            
    def generate_file_path(self,dirpath,filename,ext=['.cap','.txt','.log'],exclude=['_V','_Data_Workshop']):
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
                data.update({reader.line_num:row})
                
        for key,info in data.items():
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
        data = {'serial':[self.serial]*len(self.coefficients),
               'name':list(self.coefficients.keys()),
               'value':list(self.coefficients.values()),
               'notes':list(self.notes.values()) }
        df = pd.DataFrame().from_dict(data)
        
        # Generate the csv name
        csv_name = self.uid + '__' + self.date + '.csv'
        
        # Now write to 
        check = input(f"Write {csv_name} to {outpath}? [y/n]: ")
        if check.lower().strip() == 'y':
            df.to_csv(outpath+'/'+csv_name, index=False)


def generate_file_path(self,dirpath,filename,ext=['.cap','.txt','.log'],exclude=['_V','_Data_Workshop']):
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


def whoi_asset_tracking(spreadsheet,sheet_name,instrument_class='All',whoi=True,series=None):
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
    
    all_sensors = pd.read_excel(spreadsheet,sheet_name=sheet_name,header=1)
    # Select a specific class of instruments
    if instrument_class == 'All':
        inst_class = all_sensors
    else:
        inst_class  = all_sensors[all_sensors['Instrument\nClass']==instrument_class]
    # Return only the whoi instruments?
    if whoi == True:
        whoi_insts = inst_class[inst_class['Deployment History'] != 'EA']
    else:
        whoi_insts = inst_class
    # Slect a specific series of the instrument?
    if series != None:
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
        
    uids = sorted( list( set( instrument['UID'] ) ) )
    
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
    #check = [element == first for element in el]
    error = [np.isclose(element,first) for element in el]
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
        indices = [i+1 for i, j in enumerate(error) if j == False]
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
        cal_errors - A nested dictionary containing the calibration timestamp, the
        relevant calibration coefficient, and which file(s) have the
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
                if error == False:
                    wrong_cals.update({column:array.index[0]})
                elif error == True:
                    pass
                else:
                    wrong_cals.update({column:error})
        
        if len(wrong_cals) < 1:
            cal_errors.update({str(date).split('T')[0]:'No Errors'})
        else:
            cal_errors.update({str(date).split('T')[0]:wrong_cals})
    
    return cal_errors


def convert_type(x):
	# Converts the input from string to float
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
        serial_nums.update({uid:serial_num})
        
    return serial_nums


def generate_file_path(dirpath,filename,ext=['.cap','.txt','.log'],exclude=['_V','_Data_Workshop']):
    """
    Function which searches for the location of the given file and returns
    the full path to the file.
    
    Args:
        dirpath - parent directory path under which to search
        filename - the name of the file to search for
        ext - 
        exclude - optional list which allows for excluding certain
            directories from the search
    Returns:
        fpath - the file path to the filename from the current
            working directory.
    """
    for root, dirs, files in os.walk(dirpath):
        dirs[:] = [d for d in dirs if d not in exclude]
        for fname in files:
            if fnmatch.fnmatch(fname, [filename+'*'+x for x in ext]):
                fpath = os.path.join(root, fname)
                return fpath


