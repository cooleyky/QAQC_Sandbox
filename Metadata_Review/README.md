# MetaData Review
This README describes the MetaData Review for OOI-CGSN. The notebooks contained within describe the process for checking the calibration csvs for various deployed instrument classes located in the asset management repository on GitHub. The purpose is to identify when errors were made during the creation of the calibration csvs. This involves checking the following information:
* CSV file name - check that the date in the csv file name matches the date of calibration as well as the UID matches
* Coefficient values - check that accuracy and precision of the numbers stored in the csvs match those reported by the vendor
* Serial Numbers - check that the serial numbers within the associated calibration source files match that in the csvs
* Sources - identify where the calibration coefficients for the  

## Requirements
The CGSN MetaData Review utilizes Python 3.7 and Jupyter Notebooks. The following packages need to be installed:
* Numpy
* Pandas
* Xarray
* Zipfile
* Xml
* Wcmatch
* PyPDF2

These may be installed from the terminal command line with either:

    pip install <package>

or, if you installed Anaconda:

    conda install <package>
    
Additionally, the following package is needed for only the PRESF and CTDMO Metadata Review:
* textract

This package is required to perform Optical Character Recognition (OCR). Installation of textract requires several system updates in order work. The following was performed for Ubuntu 18.04:

    sudo apt-get install python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig
    
Then textract may be installed using pip install textract. If installation failed building wheel for pocketsphinx, then run the following command and retry the pip install textract:

    sudo apt-get install libasound2-dev
    pip install pocketsphinx
    
## Folders
### Bulk_load
This contains several csv files and a Jupyter Notebook for checking the Sensor bulk load. This is a naive implementation that only checks between the WHOI Asset Management Tracking csv and the files in the system.

## Files
### CTDBP Metadata Review
This notebook describes the process for reviewing the calibration coefficients for the CTDBP 16plus V2. 

### CTDMO Metadata Review
This notebook describes the process for reviewing the calibration coefficients for the CTDMO IM-37. The purpose is to check the calibration coefficients contained in the CSVs stored within the asset management repository on GitHub, which are the coefficients utilized by OOI-net for calculating data products, against the different available sources of calibration information to identify when errors were made during entering the calibration csvs. This includes checking the following information:
1. The calibration date - this information is stored in the filename of the csv
2. Calibration source - identifying all the possible sources of calibration information, and determine which file should supply the calibration info
3. Calibration coeffs - checking the accuracy and precision of the numbers stored in the calibration coefficients

The CTDMO contains 24 different calibration coefficients to check. The possible calibration sources for the CTDMOs are vendor PDFs, vendor .cal files, and QCT check-ins. A complication is that the vendor documents are principally available only as PDFs that are copies of images. This requires the use of Optical Character Recognition (OCR) in order to parse the PDFs. Unfortunately, OCR frequently misinterprets certain character combinations, since it utilizes Levenstein-distance to do character matching. 

Furthermore, using OCR to read PDFs requires significant preprocessing of the PDFs to create individual PDFs with uniform metadata and encoding. Without this preprocessing, the OCR will not generate uniformly spaced characters, making parsing not amenable to repeatable automated parsing.

### OPTAA Metadata Review
This notebook describes the process for reviewing the calibration coefficients for the OPTAAs. The purpose is to check the calibration coefficients contained in the CSVs stored within the asset management repository on GitHub, which are the coefficients utilized by OOI-net for calculating data products, against the different available sources of calibration information to identify when errors were made during entering the calibration csvs. This includes checking the following information:
1. The calibration date - this information is stored in the filename of the csv
2. Calibration source - identifying all the possible sources of calibration information, and determine which file should supply the calibration info
3. Calibration coeffs - checking the accuracy and precision of the numbers stored in the calibration coefficients
4. Calibration .ext files - arrays which are referenced by the main csv files and contain more calibration values

The OPTAAs contains 8 different calibration coefficients to check. Five of the coefficients are arrays of varying lengths of values. Additionally, there are two .ext files which are referenced by the calibration csv. These .ext files are separate arrays of values whose name and values also need to be checked. The possible calibration source for the OPTAAs are vendor calibration (.dev) files. The QCT checkin, pre- and post-deployment files do not contain all the necessary calibration information in order to fully check the asset management csvs.


### PRESF Metadata Review
This notebook describes the process for reviewing the calibration coefficients for the PRESF SBE 26plus. The purpose is to check the calibration coefficients contained in the CSVs stored within the asset management repository on GitHub, which are the coefficients utilized by OOI-net for calculating data products, against the different available sources of calibration information to identify when errors were made during entering the calibration csvs. This includes checking the following information:
1. The calibration date - this information is stored in the filename of the csv
2. Calibration source - identifying all the possible sources of calibration information, and determine which file should supply the calibration info
3. Calibration coeffs - checking the accuracy and precision of the numbers stored in the calibration coefficients

The PRESF contains 18 different calibration coefficients to check, two of which are fixed constants. The possible calibration sources for the PRESF are vendor PDFs and QCT check-ins. However, calibrations from the vendor PDFs are split across multiple documents and many are missing either coefficients or PDFs. Consequently, we utilize the QCT check-in as the source of calibration coefficients. The relevant file stored within the QCTs are .hex files.


### NUTNR Metadata Review
This notebook describes the process for reviewing the calibration coefficients for the NUTNRs, including both the ISUS and SUNA models. The purpose is to check the calibration coefficients contained in the CSVs stored within the asset management repository on GitHub, which are the coefficients utilized by OOI-net for calculating data products, against the different available sources of calibration information to identify when errors were made during entering the calibration csvs. This includes checking the following information:
1. The calibration date - this information is stored in the filename of the csv
2. Calibration source - identifying all the possible sources of calibration information, and determine which file should supply the calibration info
3. Calibration coeffs - checking the accuracy and precision of the numbers stored in the calibration coefficients

The NUTNRs contains 7 different calibration coefficients to check. Two of the calibration coefficients are fixed constants. Four of the coefficients are arrays of 35 values. The possible calibration sources for the NUTNRs are vendor calibration (.cal) files, as well as pre- and post-deployment calibrations (.cal files). A complication is that the calibration documents often contain multiple .cal files. However, if there are multiple .cal files, they are sequentially appended with the alphabet. Consequently, we identify the latest .cal file based on the appended letter to the file.

### SPKIR Metadata Review
This notebook describes the process for reviewing the calibration coefficients for the SPKIR. The purpose is to check the calibration coefficients contained in the CSVs stored within the asset management repository on GitHub, which are the coefficients utilized by OOI-net for calculating data products, against the different available sources of calibration information to identify when errors were made during entering the calibration csvs. This includes checking the following information:
1. The calibration date - this information is stored in the filename of the csv
2. Calibration source - identifying all the possible sources of calibration information, and determine which file should supply the calibration info
3. Calibration coeffs - checking the accuracy and precision of the numbers stored in the calibration coefficients

The SPKIRs contains three different calibration coefficients to check. All three of the coefficients are arrays of seven values. The possible calibration sources for the SPKIR are vendor calibration (.cal) files. The QCT, pre- and post-deployment files do not contain the relevant calibration information needed to perform checking.

### utils.py
This python file contains a number of functions utilized by the Jupyter notebooks. This file should be located in the same directory as the Metadata Review notebooks or be added to your path using _sys.path.append(<path_to_file>)_. The functions from the utils.py file are imported into the notebooks as:

    from utils import *
