import sys
import os
import csv
import re
import numpy as np
import pandas as pd
import shutil
import PyPDF2
from wcmatch import fnmatch


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


def get_qct_files(df, qct_directory):
    """
    Function which gets all the QCT files associated with the
    instrument serial numbers.

    Args:
        serial_nums - serial numbers of the instruments
        dirpath - path to the directory containing the calibration files
    Returns:
        calibration_files - a dictionary of instrument uids with associated
            calibration files
    """

    qct_dict = {}
    uids = list(set(df['UID']))
    for uid in uids:
        df['UID_match'] = df['UID'].apply(lambda x: True if uid in x else False)
        qct_series = df[df['UID_match'] == True]['QCT Testing']
        qct_series = list(qct_series.iloc[0].split('\n'))
        qct_dict.update({uid: qct_series})

    return qct_dict


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
        if type(sn) is list:
            sn = str(sn[0])
        files = []
        for file in os.listdir(cal_directory):
            if 'calibration_file' in file.lower():
                if sn in file:
                    files.append(file)
        calibration_files.update({uid: files})

    return calibration_files


def ensure_dir(filepath):
    """
    Function which checks that the directory where you want
    to save a file exists. If it doesn't, it creates the
    directory.
    """
    if not os.path.exists(filepath):
        os.makedirs(filepath)


def load_csv_info(csv_dict, filepath):
    """
    Loads the calibration coefficient information contained in asset management

    Args:
        csv_dict - a dictionary which associates an instrument UID to the
            calibration csv files in asset management
        filepath - the path to the directory containing the calibration csv files
    Returns:
        csv_cals - a dictionary which associates an instrument UID to a pandas
            dataframe which contains the calibration coefficients. The dataframes
            are indexed by the date of calibration
    """

    # Load the calibration data into pandas dataframes, which are then placed into
    # a dictionary by the UID
    csv_cals = {}
    for uid in csv_dict:
        cals = pd.DataFrame()
        for file in csv_dict[uid]:
            data = pd.read_csv(filepath+file)
            date = file.split('__')[1].split('.')[0]
            data['CAL DATE'] = pd.to_datetime(date)
            cals = cals.append(data)
        csv_cals.update({uid: cals})

    # Pivot the dataframe to be sorted based on calibration date
    for uid in csv_cals:
        csv_cals[uid] = csv_cals[uid].pivot(
            index=csv_cals[uid]['CAL DATE'], columns='name')['value']

    return csv_cals


def splitDataFrameList(df, target_column):
    """
    Args:
        df = dataframe to split
        target_column = the column containing the values to split
    Returns:
        new_rows - a dataframe with each entry for the target column
            separated, with each element moved into a new row. The
            values in the other columns are duplicated across the
            newly divided rows.
    """

    def splitListToRows(row, row_accumulator, target_column):
        split_row = row[target_column]
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)

    new_rows = []
    df.apply(splitListToRows, axis=1, args=(new_rows, target_column))
    new_df = pd.DataFrame(new_rows)
    return new_df


def split_multipage_pdfs(directory, pdfname):
    """
    This function splits a multipage pdf into its
    constituent pages.

    Args:
            directory - the full path to the directory
                    where the pdf file to be split is saved.
            pdfname - the name of the pdf file to be split
    Returns:
            pdfname_page*_.pdf - each of the individual
                    pages of the pdf written to the same
                    directory as the original pdf.

    """

    filepath = '/'.join((directory, pdfname))
    inputpdf = PyPDF2.PdfFileReader(filepath, 'rb')

    for i in range(inputpdf.numPages):
        output = PyPDF2.PdfFileWriter()
        output.addPage(inputpdf.getPage(i))
        filename = '_'.join((filepath.split('.')[0], 'page', str(i)))
        with open(filename+'.pdf', 'wb') as outputStream:
            output.write(outputStream)


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


def get_file_date(x):
    x = str(x)
    ind1 = x.index('__')
    ind2 = x.index('.')
    return x[ind1+2:ind2]


def check_exact_coeffs(coeffs_dict):
    """
    Function to check if the calibration coefficients match exactly. The
    calibration coefficients to be checked should be stored as pandas
    dataframes within a dictionary. The dictionary keys identify which DataFrame
    is associated with which calibration source.
    Args:
        coeffs_dict - a dictionary with the source files (csv, cal, qct) as keys
            with pandas dataframes of the calibration coefficients
    Returns:
        mask - a True/False mask of the calibration coefficient values if they match
    """

    # Part 1: coeff by coeff comparison between each source of coefficients
    keys = list(coeffs_dict.keys())
    comparison = {}
    for i in range(len(keys)):
        names = (keys[i], keys[i - (len(keys)-1)])
        check = len(coeffs_dict.get(keys[i])['value']) == len(
            coeffs_dict.get(keys[i - (len(keys)-1)])['value'])
        if check:
            compare = np.equal(coeffs_dict.get(keys[i])['value'], coeffs_dict.get(
                keys[i - (len(keys)-1)])['value'])
            comparison.update({names: compare})
        else:
            pass

    # Part 2: now do a logical_and comparison between the results from part 1
    keys = list(comparison.keys())
    i = 0
    mask = comparison.get(keys[i])
    while i < len(keys)-1:
        i = i + 1
        mask = np.logical_and(mask, comparison.get(keys[i]))
        print(i)

    return mask


def check_relative_coeffs(coeffs_dict):
    """
    Function to check if the calibration coefficients match exactly. The
    calibration coefficients to be checked should be stored as pandas
    dataframes within a dictionary. The dictionary keys identify which DataFrame
    is associated with which calibration source.
    Args:
        coeffs_dict - a dictionary with the source files (csv, cal, qct) as keys
            with pandas dataframes of the calibration coefficients
    Returns:
        mask - a True/False mask of the calibration coefficient values if they match
            to within a tolerance of 0.001%.
    """

    # Part 1: coeff by coeff comparison between each source of coefficients
    keys = list(coeffs_dict.keys())
    comparison = {}
    for i in range(len(keys)):
        names = (keys[i], keys[i - (len(keys)-1)])
        check = len(coeffs_dict.get(keys[i])['value']) == len(
            coeffs_dict.get(keys[i - (len(keys)-1)])['value'])
        if check:
            compare = np.isclose(coeffs_dict.get(keys[i])['value'], coeffs_dict.get(
                keys[i - (len(keys)-1)])['value'], rtol=1e-5)
            comparison.update({names: compare})
        else:
            pass

    # Part 2: now do a logical_and comparison between the results from part 1
    keys = list(comparison.keys())
    i = 0
    mask = comparison.get(keys[i])
    while i < len(keys)-1:
        i = i + 1
        mask = np.logical_and(mask, comparison.get(keys[i]))
        print(i)

    return mask


def copy_to_local(cal_path):
    """
    Function which copies the files from the cal_path to a locally
    created temp directory
    """

    for filepath in cal_path:
        # Create a folder in which to save extracted data
        folder, *ignore = filepath.split('/')[-1].split('.')
        savedir = '/'.join((os.getcwd(), 'temp', 'cal_data', folder))
        # Now make sure that the save directory exists and can be used
        ensure_dir(savedir)

        if filepath.endswith('.zip'):
            with ZipFile(filepath, 'r') as zfile:
                for file in zfile.namelist():
                    zfile.extract(file, path=savedir)
        else:
            shutil.copy(filepath, savedir)
