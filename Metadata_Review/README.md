# MetaData Review
This README describes the MetaData Review for OOI-CGSN. The notebooks contained within describe the process for checking the calibration csvs for various deployed instrument classes located in the asset management repository on GitHub. The purpose is to identify when errors were made during the creation of the calibration csvs. This involves checking the following information:
* CSV file name - check that the date in the csv file name matches the date of calibration as well as the UID matches
* Coefficient values - check that accuracy and precision of the numbers stored in the csvs match those reported by the vendor
* Serial Numbers - check that the 
* 

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
This notebook describes the process for reviewing the calibration coefficients for the CTDMO IM-37. The CTDMO contains 24 different calibration coefficients to check. Additionally, possible calibration sources include vendor documents as well as the QCT check-in.

A complication is that the vendor documents are principally available only as PDFs that are copies of images. This requires the use of Optical Character Recognition (OCR) in order to parse the PDFs. Unfortunately, OCR frequently misinterprets certain character combinations, since it utilizes Levenstein-distance to do character matching.

Furthermore, using OCR to read PDFs requires significant preprocessing of the PDFs to create individual PDFs with uniform metadata and encoding. Without this preprocessing, the OCR will not generate uniformly spaced characters, making parsing significantly more difficult nee impossible.

### OPTAA Metadata Review


### PRESF Metadata Review

### NUTNR Metadata Review

### SPKIR Metadata Review

### utils.py
This python file contains a number of functions utilized by the Jupyter notebooks. This file should be located in the same directory as the Metadata Review notebooks or be added to your path using _sys.path.append(<path_to_file>)_. The functions from the utils.py file are imported into the notebooks as:

    from utils import *
