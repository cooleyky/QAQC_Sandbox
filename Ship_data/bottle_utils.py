import os
import sys
import re
import pandas as pd
import numpy as np


class Cast():

    def __init__(self, cast_number):
        self.cast_number = str(cast_number)

    def parse_header(self, header):
        """
        Parse the header of bottle (.btl) files to get critical information
        for the summary spreadsheet.

        Args:
            header - an object containing the header of the bottle file as a list of
                strings, split at the newline.
        Returns:
            self.hdr - a dictionary object containing the start_time, filename, latitude,
                longitude, and cruise id.
        """
        hdr = {}
        for line in header:
            if 'nmea utc' in line.lower():
                start_time = pd.to_datetime(re.split('= |\[', line)[1])
                hdr.update({'Start Time [UTC]': start_time.strftime('%Y-%m-%d %H:%M:%S')})
            elif 'filename' in line.lower():
                hex_name = re.split('=', line)[1].strip()
                hdr.update({'Filename': hex_name})
            elif 'nmea latitude' in line.lower():
                start_lat = re.split('=', line)[1].strip()
                hdr.update({'Start Latitude [degrees]': start_lat})
            elif 'nmea longitude' in line.lower():
                start_lon = re.split('=', line)[1].strip()
                hdr.update({'Start Longitude [degrees]': start_lon})
            elif 'cruise id' in line.lower():
                cruise_id = re.split(':', line)[1].strip()
                hdr.update({'Cruise': cruise_id})
            else:
                pass

        self.header = hdr

    def parse_columns(self, columns):
        """
        Parse the column identifiers
        """

        column_dict = {}
        for line in columns:
            for i, x in enumerate(line.split()):
                try:
                    column_dict[i] = column_dict[i] + ' ' + x
                except:
                    column_dict.update({i: x})

        self.columns = column_dict

    def parse_data(self, data):
        """
        Parse the data in the contents, based on the column identifiers
        """

        # Create a dictionary to store the parsed data
        self.data = {x: [] for x in self.columns.keys()}

        # Parse the data into the dictionary based on the column location
        for line in data:
            if line.endswith('(avg)'):
                values = list(filter(None, re.split('  |\t', line)))
                for i, x in enumerate(values):
                    self.data[i].append(x)
            elif line.endswith('(sdev)'):
                values = list(filter(None, re.split('  |\t', line)))
                self.data[1].append(values[0])
            else:
                pass

        # Join the data and time for each measurement into a single item
        self.data[1] = [' '.join(item) for item in zip(self.data[1][::2], self.data[1][1::2])]

    def add_column_names(self):
        """
        With the parsed data and column identifiers, match up the data and column
        names based on the location
        """

        for key, value in self.columns.items():
            self.data[value] = self.data.pop(key)

    def load_content(self, filepath):
        """
        Load the content of a bottle (.btl) file

        Args:
            filepath - path to the bottle (.btl) file to be loaded
        Returns:
            self.content - a list of each line in the btl file, split at the newline
        """

        with open(filepath) as file:
            content = file.readlines()
        self.content = [x.strip() for x in content]

    def write_csv(self, savepath):
        """
        Write the parsed bottle information to a csv file
        """

        # First, put the data into a pandas dataframe
        df = pd.DataFrame.from_dict(self.data)

        # Now add the parsed header info into the dataframe
        for key, item in self.header.items():
            df[key] = item

        # Add in the cast number to the dataframe
        df = df.insert(0, "Cast", self.cast_number.zfill(3))

        # Write to a csv file
        df.to_csv(savepath + 'Cast' + str(self.cast_number) + '.sum', index=False)

    def parse_cast(self, filepath):
        """
        Function which integrates the separate cast functions to """

        # First, load the file content
        self.load_content(filepath)

        # Second, parse the file content
        header = []
        columns = []
        data = []
        for line in self.content:
            if line.startswith('*') or line.startswith('#'):
                header.append(line)
            else:
                try:
                    float(line[0])
                    data.append(line)
                except:
                    columns.append(line)

        # Third, parse the data, columns, and header info
        self.parse_columns(columns)
        self.parse_header(header)
        self.parse_data(data)

        # Add in the column names
        self.add_column_names()


class Salinity():

    def parse_SAL(self, filepath):
        """
        Reprocess .SAL files into the standardized format.
        
        Parameters
        ----------
        filepath: (str)
            The full path to the .SAL file to be reprocessed.
            
        Returns
        -------
        results: (pandas.DataFrame)
            A dataframe saved to filepath as a csv file.
        """
        
        # Create a dictionary to store the results
        results = {
            "Cruise": [],
            "Station": [],
            "Niskin #": [],
            "Case ID": [],
            "Sample": [],
            "Salinity": [],
            "Unit": []
        }
        
        # Open and read in the .SAL salinity measurement file
        with open(filepath) as f:
            data = f.readlines()
        
        # Parse the .SAL file
        for n, line in enumerate(data):
            # If its the first line, parse the metadata
            if n == 0:
                header = line.replace('"', '').split(',')
                cruise = header[0]
                station = int(header[1])
                cast = int(header[2])
                case = header[8]
            # If only a newline character for the line,
            # want to pass
            elif line == '\n':
                continue
            else:
                values = line.split()
                # If its the end of the file, 
                # ignore the placeholders
                if int(values[0]) == 0:
                    continue
                # Otherwise, parse the sample salinity data
                else:
                    sample = int(values[0])
                    salinity = float(values[2])
                    flag = int(values[3])
                    
                # Next, put the data into the dictionary
                results["Cruise"].append(cruise)
                results["Station"].append(station)
                results["Niskin #"].append(np.nan)
                results["Case ID"].append(case)
                results["Sample"].append(sample)
                results["Salinity"].append(salinity)
                results["Unit"].append("psu")

        # Put the parsed salinity data into a dataframe
        results = pd.DataFrame(results)
        
        # Return the dataframe
        return results
        
        # Write the dataframe to a csv
        # results.to_csv(f.name.replace('.SAL', 'SAL.csv'), index=False)

        
    def process_files(self, directory):
        """
        Processes the salinity files for a cruise and saves a summary
        file which contains all of the salinity data in one csv file
        """
        # First, run check to see if files are held as csv ".SAL" files and if so, preprocess and clean them
        csv_flag = any(files.endswith('.SAL') for files in os.listdir(directory))
        if csv_flag:
            csv_files = [file for file in os.listdir(directory) if file.endswith('.SAL')]
            for file in csv_files:
                self.clean_files(directory + file)

        # Next, generate a pandas dataframe for
        df = pd.DataFrame()

        # Get the unique casts in the salinity data directory
        casts = [file[0:3]
                 for file in os.listdir(directory) if 'SAL' in file and 'Summary' not in file]
        casts = np.unique(casts)
        # Iterate through the casts, and search for if it is an excel or csv file
        for cast in casts:
            files = [file for file in os.listdir(directory) if cast in file and 'SAL' in file]
            excel_flag = any(file for file in files if file.endswith('SAL.xlsx'))
            if excel_flag:
                file = [file for file in files if file.endswith('SAL.xlsx')][0]
                df = df.append(pd.read_excel(directory + file))
            else:
                file = [file for file in files if file.endswith('SAL.csv')][0]
                df = df.append(pd.read_csv(directory + file))

        # Save the processed summary file for salinity
        df.to_csv(directory + 'SAL_Summary.csv', index=False)


class Oxygen():

    def process_oxygen(self, directory):
        # Load the oxygen files into a pandas dataframe
        df = pd.DataFrame()
        files = [file for file in os.listdir(
            directory) if 'oxy' in file.lower() and file.endswith('.xlsx')]
        for file in files:
            df = df.append(pd.read_excel(directory + file))

        # Save the oxygen dataframe to a new summary csv
        df.to_csv(directory + 'OXY_Summary.csv', index=False)
