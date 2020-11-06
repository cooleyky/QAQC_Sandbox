import datetime
import os
import re
import requests
import numpy as np
import pandas as pd
import xarray as xr


def ntp_seconds_to_datetime(ntp_seconds):
    # Set some constants needed for time conversion
    ntp_epoch = datetime.datetime(1900, 1, 1)
    unix_epoch = datetime.datetime(1970, 1, 1)
    ntp_delta = (unix_epoch - ntp_epoch).total_seconds()

    # Convert the datetime
    dt = datetime.datetime.utcfromtimestamp(ntp_seconds - ntp_delta).replace(microseconds=0)
    return dt


def convert_time(ms):
    if ms is None:
        return None
    else:
        return datetime.datetime.utcfromtimestamp(ms/1000)


def get_and_print_api(url, username, token):
    # Request the api output
    r = requests.get(url, auth=(username, token))
    data = r.json()

    # Print the results
    for d in data:
        print(d)

    # Return the data
    return data


def get_api(url, username, token):
    """Request data from the given url with the given credentials."""
    r = requests.get(url, auth=(username, token))
    data = r.json()
    return data


def get_thredds_url(data_request_url, min_time, max_time, username, token):
    """
    Returns the associated thredds url for a desired dataset(s) from the
    OOI api

    Args:
        data_request_url - this is the OOINet url with the platform/node/sensor/
            method/stream information
        min_time - optional to limit the data request to only data after a
            particular date. May be None.
        max_time - optional to limit the data request to only data before a
            particular date. May be None.
        username - your OOINet username
        token - your OOINet authentication token

    Returns:
        thredds_url - a url to the OOI Thredds server which contains the desired
            datasets
    """

    # Ensure proper datetime format for the request
    if min_time is not None:
        min_time = pd.to_datetime(min_time).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        max_time = pd.to_datetime(max_time).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    # Build the query
    params = {
        'beginDT': min_time,
        'endDT': max_time,
    }

    # Request the data
    r = requests.get(data_request_url, params=params, auth=(username, token))
    if r.status_code == 200:
        data_urls = r.json()
    else:
        print(r.reason)
        return None

    # The asynchronous data request is contained in the 'allURLs' key,
    # in which we want to find the url to the thredds server
    for d in data_urls['allURLs']:
        if 'thredds' in d:
            thredds_url = d

    return thredds_url


def get_netcdf_datasets(thredds_url):
    """
    Function which returns the netcdf datasets from a given OOI thredds server.

    Args:
        thredds_url - a url to the OOI Thredds server which contains your desired
            datasets

    Returns:
        datasets - a list of netcdf dataset urls
    """

    import time
    datasets = []
    counter = 0
    tds_url = 'https://opendap.oceanobservatories.org/thredds/dodsC/'

    while not datasets:
        datasets = requests.get(thredds_url).text
        urls = re.findall(r'href=[\'"]?([^\'" >]+)', datasets)
        x = re.findall(r'(ooi/.*?.nc)', datasets)
        for i in x:
            if i.endswith('.nc') == False:
                x.remove(i)
        for i in x:
            try:
                float(i[-4])
            except:
                x.remove(i)
        datasets = [os.path.join(tds_url, i) for i in x]
        if not datasets:
            print(f'Re-requesting data: {counter}')
            counter = counter + 1
            time.sleep(10)
    return datasets


def get_netCDF_datasets(thredds_url):
    """
    Query the asynch url and return the netCDF datasets.
    """
    # This block of code works times the request until fufilled or the request times out (limit=10 minutes)
    start_time = time.time()
    check_complete = thredds_url + '/status.txt'
    r = requests.get(check_complete)
    while r.status_code != requests.codes.ok:
        check_complete = thredds_url + '/status.txt'
        r = requests.get(check_complete)
        elapsed_time = time.time() - start_time
        if elapsed_time > 10*60:
            print('Request time out')
        time.sleep(5)

    # Identify the netCDF urls
    datasets = requests.get(thredds_url).text
    x = re.findall(r'href=["](.*?.nc)', datasets)
    for i in x:
        if i.endswith('.nc') == False:
            x.remove(i)
        for i in x:
            try:
                float(i[-4])
            except:
                x.remove(i)
        datasets = [os.path.join(thredds_url, i) for i in x]
        
    return datasets


def load_netcdf_datasets(datasets):
    """
    Function which opens and loads netcdf files from OOI thredds server
    into an xarray dataset. May accept more than one netcdf file to open, but
    all the netcdf files must be for the same sensor (some sensors automatically
    download companion sensor datasets if they are used in computing a derived
    data product).

    Also recommended to downgrade python package libnetcdf to <= 4.6.1 due to
    strict fill value matching requirements in most recent libnetcdf release
    causing loading errors with OOI netcdf files.

    Args:
        datasets - a list of OOI netcdf datasets

    Returns:
        ds - a sorted xarray dataset with primary dimension of time
    """
    if len(datasets) > 1:
        try:
            ds = xr.open_mfdataset(datasets)
        except Exception as exc:
            if '_FillValue type mismatch' in exc.args[1]:
                # Retry the request with #fillmismatch argument
                ds = xr.open_mfdataset([x+'#fillmismatch' for x in datasets])

        # Need to switch dimensions and sort by the time dimension
        ds = ds.swap_dims({'obs': 'time'})
        ds = ds.sortby('time')

        return ds

    # If a single deployment, will be a single datasets returned
    elif len(datasets) == 1:
        try:
            ds = xr.open_dataset(datasets[0])
        except Exception as exc:
            if '_FillValue type mismatch' in exc.args[1]:
                # Retry the request with #fillmismatch argument
                ds = xr.open_dataset(datasets[0]+'#fillmismatch')

        # Need to switch dimensions and sort by the time dimension
        ds = ds.swap_dims({'obs': 'time'})
        ds = ds.sortby('time')

        return ds

    # If there are no available datasets on the thredds server, let user know
    else:
        print('No datasets to open')
        return None


def request_UFrame_data(url, array, node, sensor, method, stream, min_time, max_time, username, token):
    """
    Function which requests a netCDF dataset(s) from OOI UFrame
    and returns a pandas dataframe with the requested dataset(s)

    Args:
        url - the base OOI API for requesting data
        array - OOI array (e.g. CP01CNSM) from a reference designator
        node - OOI node (e.g. RID27) from a reference designator
        sensor - OOI sensor (e.g. 04-DOSTAD000) from a reference designator
        method - telemetered/host/instrument methods for a given sensor
        stream - a particular stream for a given reference designator
        min_time - optional to limit the data request to only data after a
            particular date. May be None.
        max_time - optional to limit the data request to only data before a
            particular date. May be None.
        username - your OOINet username
        token - your OOINet authentication token

    Calls:
        get_thredds_url
        get_netcdf_datasets
        load_netcdf_datasets

    Returns:
        df - a dataframe with the requested dataset(s) loaded
    """

    # Build the request url
    data_request_url = '/'.join((url, array, node, sensor, method, stream))

    # Query the thredds url
    thredds_url = get_thredds_url(data_request_url, min_time, max_time, username, token)
    if thredds_url is None:
        return None

    # Find and return the netCDF datasets from the thredds url
    datasets = get_netcdf_datasets(thredds_url)

    # Filter to remove companion data streams that aren't the target stream
    datasets = [d for d in datasets if 'ENG' not in d]
    if 'CTD' not in sensor:
        datasets = [d for d in datasets if 'CTD' not in d]

    # Load the netCDF files from the datasets
    if len(datasets) == 0 or datasets == None:
        return None
    else:
        ds = load_netcdf_datasets(datasets)

    # Convert the xarray dataset to a pandas dataframe for ease of use
    df = ds.to_dataframe()

    # Return the dataframe
    return df


# Define a function to return the sensor metadata
def get_sensor_metadata(metadata_url, username, token):
    """
    Function which gets the metadata for a given sensor from OOI Net
    """

    # Request and download the relevant metadata
    r = requests.get(metadata_url, auth=(username, token))
    if r.status_code == 200:
        metadata = r.json()
    else:
        print(r.reason)

    # Put the metadata into a dataframe
    df = pd.DataFrame(columns=metadata.get('parameters')[0].keys())
    for mdata in metadata.get('parameters'):
        df = df.append(mdata, ignore_index=True)

    return df


# Define a function to filter the metadata for the relevant variable info
def get_UFrame_fillValues(data, metadata, stream):
    """
    Function which returns the fill values for a particular data set
    based on that dataset's metadata information

    Args:
        data - a dataframe which contains the data from a given sensor
            stream
        metadata - a dataframe which contains the metadata information
            for the given data
        stream - the particular instrument stream from which the data
            was requested from

    Returns:
        fillValues - a dictionary with key:value pair of variable names
            from the data stream : fill values
    """

    # Filter the metadata down to a particular sensor stream
    mdf = metadata[metadata['stream'] == stream]

    # Based on the variables in data, identify the associated fill values as key:value pair
    fillValues = {}
    for varname in data.columns:
        fv = mdf[mdf['particleKey'] == varname]['fillValue']
        # If nothing, default is NaN
        if len(fv) == 0:
            fv = np.nan
        else:
            fv = list(fv)[0]
            # Empty values in numpy default to NaNs
            if fv == 'empty':
                fv = np.nan
            else:
                fv = float(fv)
        # Update the dictionary
        fillValues.update({
            varname: fv
        })

    return fillValues


# Define a function to calculate the data availability for a day
def calc_UFrame_data_availability(subset_data, fillValues):
    """
    Function which calculates the data availability for a particular dataset.

    Args:
        subset_data - a pandas dataframe with time as the primary index
        fillValues - a dictionary of key:value pairs with keys which correspond
            to the subset_data column headers and values which correspond to the
            associated fill values for that column
    Returns:
        data_availability - a dictionary with keys corresponding to the
            subset_data column headers and values with the percent data available
    """

    # Initialize a dictionary to store results
    data_availability = {}

    for col in subset_data.columns:

        # Check for NaNs in each col
        nans = len(subset_data[subset_data[col].isnull()][col])

        # Check for values with fill values
        fv = fillValues.get(col)
        if fv is not None:
            if np.isnan(fv):
                fills = 0
            else:
                fills = len(subset_data[subset_data[col] == fv][col])
        else:
            fills = 0

        # Get the length of the whole dataframe
        num_data = len(subset_data[col])

        # If there is no data in the time period,
        if num_data == 0:
            data_availability.update({
                col: 0,
            })
        else:
            # Calculate the statistics for the nans, fills, and length
            num_bad = nans + fills
            num_good = num_data - num_bad
            per_good = (num_good/num_data)*100

            # Update the dictionary with the stats for a particular variable
            data_availability.update({
                col: per_good
            })

    return data_availability


# Define a function to bin the time period into midnight-to-midnight days
def time_periods(startDateTime, stopDateTime):
    """
    Generates an array of dates with midnight-to-midnight
    day timespans. The startDateTime and stopDateTime are
    then append to the first and last dates.
    """
    startTime = pd.to_datetime(startDateTime)
    if type(stopDateTime) == float:
        stopDateTime = pd.datetime.now()
    stopTime = pd.to_datetime(stopDateTime)
    days = pd.date_range(start=startTime.ceil('D'), end=stopTime.floor('D'), freq='D')

    # Generate a list of times
    days = [x.strftime('%Y-%m-%dT%H:%M:%SZ') for x in days]
    days.insert(0, startTime.strftime('%Y-%m-%dT%H:%M:%SZ'))
    days.append(stopTime.strftime('%Y-%m-%dT%H:%M:%SZ'))

    return days


def make_dirs(path):
    """
    Function which checks if a path exists,
    and if it doesn't, makes the directory.
    """
    check = os.path.exists(path)
    if not check:
        os.makedirs(path)
    else:
        print(f'"{path}" already exists')


# +
from xml.dom import minidom
from urllib.request import urlopen
from urllib.request import urlretrieve

class OOINet():
    
    def __init__(self, USERNAME, TOKEN):
        
        self.username = USERNAME
        self.token = TOKEN
        self.urls = {
            'data': 'https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv',
            'anno': 'https://ooinet.oceanobservatories.org/api/m2m/12580/anno/find',
            'vocab': 'https://ooinet.oceanobservatories.org/api/m2m/12586/vocab/inv',
            'asset': 'https://ooinet.oceanobservatories.org/api/m2m/12587',
            'deploy': 'https://ooinet.oceanobservatories.org/api/m2m/12587/events/deployment/inv',
            'preload': 'https://ooinet.oceanobservatories.org/api/m2m/12575/parameter',
            'cal': 'https://ooinet.oceanobservatories.org/api/m2m/12587/asset/cal'
        }

        
    def _get_api(self, url):
        """Requests the given url from OOINet."""
        r = requests.get(url, auth=(self.username, self.token))
        data = r.json()
        return data
    
    
    def _ntp_seconds_to_datetime(self, ntp_seconds):
        """Convert OOINet timestamps to unix-convertable timestamps."""
        # Specify some constant needed for timestamp conversions
        ntp_epoch = datetime.datetime(1900, 1, 1)
        unix_epoch = datetime.datetime(1970, 1, 1)
        ntp_delta = (unix_epoch - ntp_epoch).total_seconds()
        
        return datetime.datetime.utcfromtimestamp(ntp_seconds - ntp_delta)
    
    
    def _convert_time(self, ms):
        if ms is None:
            return None
        else:
            return datetime.datetime.utcfromtimestamp(ms/1000)
    
    def get_metadata(self, refdes):
        """
        Get the OOI Metadata for a specific instrument specified by its associated
        reference designator.
        
            Args:
                refdes (str): OOINet standardized reference designator in the form
                    of <array>-<node>-<instrument>.
                    
            Returns:
                results (pandas.DataFrame): A dataframe with the relevant metadata of
                    the given reference designator.
        """
        
        # First, construct the metadata request url
        array, node, instrument = refdes.split("-", 2)
        metadata_request_url = "/".join((self.urls["data"], array, node, instrument, "metadata"))
        
        # Request the metadata
        metadata = self._get_api(metadata_request_url)
        
        # Parse the metadata
        metadata = self.parse_metadata(metadata)
        
        # Add in the reference designator
        metadata["refdes"] = refdes
        
        # Return the metadata
        return metadata
    
    
    def parse_metadata(self, metadata):
        """Parse the metadata dictionary for an instrument returned by OOI into a pandas dataframe."""

        # Put the two keys into separate dataframes
        metadata_times = pd.DataFrame(metadata["times"])
        metadata_parameters = pd.DataFrame(metadata["parameters"])

        # Merge the two into a single dataframe
        results = metadata_parameters.merge(metadata_times, left_on="stream", right_on="stream")
        results.drop_duplicates(inplace=True)

        # Return the results
        return results
    
    
    def get_deployments(self, refdes, deploy_num="-1", results=pd.DataFrame()):
        """
        Get the deployment information for an instrument. Defaults to all deployments
        for a given instrument (reference designator) unless one is supplied.

        Args:
            refdes (str): The reference designator for the instrument for which to request
                deployment information.
            deploy_num (str): Optional to include a specific deployment number. Otherwise
                defaults to -1 which is all deployments.
            results (pandas.DataFrame): Optional. Useful for recursive applications for
                gathering deployment information for multiple instruments.

        Returns:
            results (pandas.DataFrame): A table of the deployment information for the 
                given instrument (reference designator) with deployment number, deployed
                water depth, latitude, longitude, start of deployment, end of deployment,
                and cruise IDs for the deployment and recovery.

        """

        # First, build the request
        array, node, instrument = refdes.split("-",2)
        deploy_url = "/".join((self.urls["deploy"], array, node, instrument, deploy_num))

        # Next, get the deployments from the deploy url. The API returns a list
        # of dictionary objects with the deployment data.
        deployments = self._get_api(deploy_url)

        # Now, iterate over the deployment list and get the associated data for
        # each individual deployment
        while len(deployments) > 0:
            # Get a single deployment
            deployment = deployments.pop()

            # Process the dictionary data
            # Deployment Number
            deploymentNumber = deployment.get("deploymentNumber")

            # Location info
            location = deployment.get("location")
            depth, lat, lon = location["depth"], location["latitude"], location["longitude"]

            # Start and end times of the deployments
            startTime = self._convert_time(deployment.get("eventStartTime"))
            stopTime = self._convert_time(deployment.get("eventStopTime"))

            # Cruise IDs of the deployment and recover cruises
            deployCruiseInfo = deployment.get("deployCruiseInfo")
            recoverCruiseInfo = deployment.get("recoverCruiseInfo")
            sensorUID = deployment.get("sensor")["uid"]
            if deployCruiseInfo is not None:
                deployID = deployCruiseInfo["uniqueCruiseIdentifier"]
            else:
                deployID = None
            if recoverCruiseInfo is not None:
                recoverID = recoverCruiseInfo["uniqueCruiseIdentifier"]
            else:
                recoverID = None

            # Put the data into a pandas dataframe
            data = np.array([[refdes, deploymentNumber, sensorUID, lat, lon, depth, startTime, stopTime, deployID, recoverID]])
            columns = ["refdes","deploymentNumber","sensorUID","latitude","longitude","depth","deployStart","deployEnd","deployCruise","recoverCruise"]
            df = pd.DataFrame(data=data, columns=columns)
            
            # Add the reference desi

            # 
            results = results.append(df)

        return results
    
    
    def get_vocab(self, refdes):
        """
        Return the OOI vocabulary for a given url endpoint. The vocab results contains
        info about the reference designator, names of the 
        
        Args:
            refdes (str): The reference designator for the instrument for which to request
                vocab information.
                
        Returns:
            results (pandas.DataFrame): A table of the vocab information for the given
                reference designator.
        
        """
        # First, construct the vocab request url
        array, node, instrument = refdes.split("-", 2)
        vocab_url = "/".join((self.urls["vocab"], array, node, instrument))
        
        # Next, get the vocab data
        data = self._get_api(vocab_url)
        
        # Put the returned vocab data into a pandas dataframe
        vocab = pd.DataFrame()
        vocab = vocab.append(data)
        
        # Finally, return the results
        return vocab
    
    
    def get_datasets(self, search_url, datasets = pd.DataFrame(), **kwargs):
        """Search OOINet for available datasets for a url"""
        
        # Build the request url
        inst = re.search("[0-9]{2}-[023A-Z]{6}[0-9]{3}", search_url)

        # This means you are at the end-point
        if inst is not None:
            # Get the reference designator info
            array, node, instrument = search_url.split("/")[-3:]
            refdes = "-".join((array, node, instrument))

            # Get the available deployments
            deploy_url = "/".join((self.urls["deploy"], array, node, instrument))
            deployments = self._get_api(deploy_url)

            # Put the data into a dictionary
            info = pd.DataFrame(data=np.array([[array, node, instrument, refdes, search_url, deployments]]),
                               columns=["array","node","instrument","refdes","url","deployments"])
            # add the dictionary to the dataframe
            datasets = datasets.append(info, ignore_index=True)

        else:
            endpoints = self._get_api(search_url)

            while len(endpoints) > 0:

                # Get one endpoint
                new_endpoint = endpoints.pop()

                # Build the new request url
                new_search_url = "/".join((search_url, new_endpoint))

                # Get the datasets for the new given endpoint
                datasets = self.get_datasets(new_search_url, datasets)

        # Once recursion is done, return the datasets
        return datasets
    
    
    def search_datasets(self, array=None, node=None, instrument=None):
        """
        Wrapper around get_datasets to make the construction of the 
        url simpler. Eventual goal is to use this as a search tool.
        """

        # Build the request url
        dataset_url = f'{self.urls["data"]}/{array}/{node}/{instrument}'

        # Truncate the url at the first "none"
        dataset_url = dataset_url[:dataset_url.find("None")-1]

        print(dataset_url)
        # Get the datasets
        datasets = self.get_datasets(dataset_url)

        return datasets
    
    
    def get_datastreams(self, refdes):
        """Function to get the data streams and methods for a specific reference designator."""

        # Build the url
        array, node, instrument = refdes.split("-",2)
        method_url = "/".join((self.urls["data"], array, node, instrument))

        # Build a table linking the reference designators, methods, and data streams
        stream_df = pd.DataFrame(columns=["refdes","method","stream"])
        methods = self._get_api(method_url)
        for method in methods:
            if "bad" in method:
                continue
            stream_url = "/".join((method_url, method))
            streams = self._get_api(stream_url)
            stream_df = stream_df.append({
                "refdes":refdes,
                "method":method,
                "stream":streams
            }, ignore_index=True)

        # Expand so that each row of the dataframe is unique
        stream_df = stream_df.explode('stream').reset_index(drop=True) 

        # Return the results
        return stream_df
    
    
    def get_parameter_data_levels(self, metadata):
        """
        Get the data levels associated with the parameters for a given reference designator.

            Args:
                metadata (pandas.DataFrame): a dataframe which contains the metadata for a
                    given reference designator.

            Returns:
                pid_dict (dict): a dictionary with the data levels for each parameter id (Pid)    
        """
        pdIds = np.unique(metadata["pdId"])
        pid_dict = {}
        for pid in pdIds:
            # Build the preload url
            preload_url = "/".join((self.urls["preload"], pid.strip("PD")))
            # Query the preload data
            preload_data = self._get_api(preload_url)
            data_level = preload_data.get("data_level")
            # Update the results dictionary
            pid_dict.update({pid: data_level})

        return pid_dict
    
    
    def filter_parameter_ids(self, pdId, pid_dict):
        # Check if pdId should be kept
        data_level = pid_dict.get(pdId)
        if data_level == 1:
            return True
        else:
            return False

    
    def get_thredds_url(self, refdes, method, stream, **kwargs):
        """
        Return the url for the THREDDS server for the desired dataset(s).

            Args:
                refdes (str): reference designator for the instrument
                method (str): the method (i.e. telemetered) for the given reference designator
                stream (str): the stream associated with the reference designator and method
            
            Kwargs: optional parameters to pass to OOINet API to limit the results of the query
                beginDT (str): limit the data request to only data after this date.
                endDT (str): limit the data request to only data before this date.
                format (str): e.g. "application/netcdf" (the default)
                include_provenance (str): 'true' returns a text file with the provenance information
                include_annotations (str): 'true' returns a separate text file with annotations for the date range

            Returns:
                thredds_url (str): a url to the OOI Thredds server which contains the desired datasets
        """
        # Build the data request url
        array, node, instrument = refdes.split("-",2)
        data_request_url = "/".join((self.urls["data"], array, node, instrument, method, stream))

        # Ensure proper datetime format for the request    
        if 'beginDT' in kwargs.keys():
            kwargs['beginDT'] = pd.to_datetime(kwargs['beginDT']).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        if 'endDT' in kwargs.keys():
            kwargs['endDT'] = pd.to_datetime(kwargs['endDT']).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            
        # Build the query
        params = kwargs

        # Request the data
        r = requests.get(data_request_url, params=params, auth=(self.username, self.token))
        if r.status_code == 200:
            data_urls = r.json()
        else:
            print(r.reason)
            return None

        # The asynchronous data request is contained in the 'allURLs' key,
        # in which we want to find the url to the thredds server
        for d in data_urls['allURLs']:
            if 'thredds' in d:
                thredds_url = d

        return thredds_url
    
    
    def get_elements(self, url, tag_name, attribute_name):
        """Get elements from an XML file."""
        # usock = urllib2.urlopen(url)
        usock = urlopen(url)
        xmldoc = minidom.parse(usock)
        usock.close()
        tags = xmldoc.getElementsByTagName(tag_name)
        attributes=[]
        for tag in tags:
            attribute = tag.getAttribute(attribute_name)
            attributes.append(attribute)
        return attributes
    
    
    def get_thredds_catalog(self, thredds_url):
        """Get the dataset catalog for the requested data stream."""

        # ==========================================================
        # Parse out the dataset_id from the thredds url
        server_url = 'https://opendap.oceanobservatories.org/thredds/'
        dataset_id = re.findall(r'(ooi/.*)/catalog', thredds_url)[0]

        # ==========================================================
        # This block of code checks the status of the request until
        # the datasets are read; will timeout if longer than 8 mins
        status_url = thredds_url + '?dataset=' + dataset_id + '/status.txt'
        status = requests.get(status_url)
        start_time = time.time()
        while status.status_code != requests.codes.ok:
            elapsed_time = time.time() - start_time
            status = requests.get(status_url)
            if elapsed_time > 10*60:
                print(f'Request time out for {thredds_url}')
                return None
            time.sleep(5)
    
        # ============================================================
        # Parse the datasets from the catalog for the requests url
        catalog_url = server_url + dataset_id + '/catalog.xml'
        catalog = get_elements(catalog_url, 'dataset', 'urlPath')

        return catalog
    
    
    def parse_catalog(self, catalog, exclude=[]):
        """
        Parses the THREDDS catalog for the netCDF files. The exclude
        argument takes in a list of strings to check a given catalog
        item against and, if in the item, not return it.
        """
        datasets = [citem for citem in catalog if citem.endswith('.nc')]
        if type(exclude) is not list:
            raise ValueError(f'arg exclude must be a list')
        for ex in exclude:
            if type(ex) is not str:
                raise ValueError(f'Element {ex} of exclude must be a string.')
            datasets = [dset for dset in datasets if ex not in dset]
        return datasets
    
    
    def download_netCDF_files(self, datasets, save_dir=None):
        """
        Download netCDF files for given netCDF datasets. If no path
        is specified for the save directory, will download the files to
        the current working directory.
        """

        # Specify the server url
        server_url = 'https://opendap.oceanobservatories.org/thredds/'

        # ===========================================================
        # Specify and make the relevant save directory
        if save_dir is not None:
            # Make the save directory if it doesn't exists
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        else:
            save_dir = os.getcwd()

        # ===========================================================
        # Download and save the netCDF files from the HTTPServer
        # to the save directory
        count = 0
        for dset in datasets:
            # Check that the datasets are netCDF
            if not dset.endswith('.nc'):
                raise ValueError(f'Dataset {dset} not netCDF.')
            count += 1
            file_url = server_url + 'fileServer/' + dset
            filename = file_url.split('/')[-1]
            print(f'Downloading file {count} of {len(datasets)}: {dset} \n')
            a = urlretrieve(file_url, '/'.join((save_dir,filename)))
    
    
