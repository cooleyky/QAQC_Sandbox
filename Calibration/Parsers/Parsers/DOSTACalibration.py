#!/usr/bin/env python

import datetime
import re
import csv
import pandas as pd
from zipfile import ZipFile
from dateutil.parser import parse
from xml.etree.ElementTree import XML


class DOSTACalibration():

    def __init__(self, uid):
        self.serial = ''
        self.uid = uid
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

    def load_qct(self, filepath):
        """
        Function which parses the output from the QCT check-in and loads them
        into the DOSTA object.

        Args:
            filepath - the full directory path and filename
        Raises:
            ValueError - checks if the serial number parsed from the UID
            matches the serial number stored in the file.
        Returns:
            self.coefficients - populated coefficients dictionary
            self.date - the calibration dates associated with the calibration values
            self.type - the type (i.e. 16+/37-IM) of the CTD
            self.serial - populates the 5-digit serial number of the instrument
        """
        self.source_file(filepath)

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
                    raise ValueError(
                        f'Serial number {serial_num.zfill(5)} from the QCT file does not match {self.serial} from the UID.')
                else:
                    pass

            # Find the svu foil coefficients
            if 'svufoilcoef' in [x.lower() for x in info]:
                self.coefficients['CC_csv'] = [float(n) for n in info[3:]]

            # Find the concentration coefficients
            if 'conccoef' in [x.lower() for x in info]:
                self.coefficients['CC_conc_coef'] = [float(n) for n in info[3:]]
                
    def load_docx(self, filepath):
        """
        Take the path of a docx file as argument, return the text of the file,
        and parse the QCT test date        
        """
        
        WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        PARA = WORD_NAMESPACE + 'p'
        TEXT = WORD_NAMESPACE + 't'
        
        def get_docx_text(filepath):
            """
            Take the path of a docx file as argument, return the text in unicode.
            """
            with ZipFile(filepath) as document:
                xml_content = document.read('word/document.xml')
            tree = XML(xml_content)
            
            paragraphs = []
            for paragraph in tree.getiterator(PARA):
                texts = [node.text for node in paragraph.getiterator(TEXT) if node.text]
                if texts:
                    paragraphs.append(''.join(texts))
            
            return '\n\n'.join(paragraphs)
        
        document = get_docx_text(filepath)
        for line in document.splitlines():
            if 'test date' in line.lower():
                date = parse(line, fuzzy=True)
                self.date = date.strftime('%Y%m%d')
            
    def source_file(self, filepath):
        """
        Routine which parses out the source file and filename
        where the calibration coefficients are sourced from.
        """
        dcn = filepath.split('/')[-2]
        filename = filepath.split('/')[-1]

        self.source = f'Source file: {dcn} > {filename}'

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
            if self.coefficients[key] is None:
                raise ValueError(f'No coefficients for {key} have been loaded.')

        # Create a dataframe to write to the csv
        data = {'serial': [self.serial]*len(self.coefficients),
                'name': list(self.coefficients.keys()),
                'value': list(self.coefficients.values()),
                'notes': list(self.notes.values())
                }
        df = pd.DataFrame().from_dict(data)

        # Add in the source
        df['notes'].iloc[0] = self.source

        # Generate the csv name
        csv_name = self.uid + '__' + self.date + '.csv'

        # Now write to
        check = input(f"Write {csv_name} to {outpath}? [y/n]: ")
        if check.lower().strip() == 'y':
            df.to_csv(outpath+'/'+csv_name, index=False)
