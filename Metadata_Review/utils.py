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
            if fnmatch.fnmatch(file, '*'+uid+'*'):
                csv_files.append(file)
            else:
                pass

        # Update the dictionary storing the asset management files for each UID
        if len(csv_files) > 0:
            csv_dict.update({uid: csv_files})
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
    elif any(error) is True:
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


def get_calibration_files(serial_nums, dirpath):
    """
    Function which gets all the calibration files associated with the
    instrument serial numbers.

    Args:
        serial_nums - serial numbers of the instruments
        dirpath - path to the directory containing the calibration files
    Returns:
        calibration_files - a dictionary of instrument uids with associated
            calibration files
    """
    calibration_files = {}
    for uid in serial_nums.keys():
        sn = serial_nums.get(uid)
        sn = str(sn[:])
        files = []
        for file in os.listdir(cal_directory):
            if 'Calibration_Files' in file:
                if sn in file:
                    files.append(file)
        calibration_files.update({uid: files})

    return calibration_files


def ensure_dir(file_path):
    """
    Function which checks that the directory where you want
    to save a file exists. If it doesn't, it creates the
    directory.
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path)