# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # OOINet Download Tool
# This is a development notebook for building a tool to download datasets from the 

import os, shutil, sys, time, re, requests, csv, datetime, pytz
import yaml
import pandas as pd
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings("ignore")

# Import user info for accessing UFrame
userinfo = yaml.load(open('../user_info.yaml'))
username = userinfo['apiname']
token = userinfo['apikey']


# +
# Define some useful functions for working with the UFrame api
def get_api(url):
    r = requests.get(url, auth=(username, token))
    data = r.json()
    return data

# Function to make an API request and print the results
def get_and_print_api(url):
    r = requests.get(url, auth=(username, token))
    data = r.json()
    #for d in data:
     #   print(d)
    
    # Return the data
    return data
        
# Specify some functions to convert timestamps
ntp_epoch = datetime.datetime(1900, 1, 1)
unix_epoch = datetime.datetime(1970, 1, 1)
ntp_delta = (unix_epoch - ntp_epoch).total_seconds()

def ntp_seconds_to_datetime(ntp_seconds):
    return datetime.datetime.utcfromtimestamp(ntp_seconds - ntp_delta).replace(microsecond=0)
  
def convert_time(ms):
    if ms is None:
        return None
    else:
        return datetime.datetime.utcfromtimestamp(ms/1000)


# -

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
        
    def get_api(self, url):
        r = requests.get(url, auth=(self.username, self.token))
        data = r.json()
        return data
    
    
    # Get the THREDDS url 
    def get_thredds_url(self, data_request_url, **kwargs):
        """
        Return the url for the THREDDS server for the desired dataset(s).

            Args:
                data_request_url (str): OOINet url with the platform/node/sensor/method/stream information
                username (str): your OOINet username
                token (str): your OOINet authentication token
            
            Kwargs: optional parameters to pass to OOINet API to limit the results of the query
                beginDT (str): limit the data request to only data after this date.
                endDT (str): limit the data request to only data before this date.
                format (str): e.g. "application/netcdf" (the default)
                include_provenance (str): 'true' returns a text file with the provenance information
                include_annotations (str): 'true' returns a separate text file with annotations for the date range

            Returns:
                thredds_url (str): a url to the OOI Thredds server which contains the desired datasets
        """

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
    
    
    def get_instrument_metadata(self, metadata_request_url, results=pd.DataFrame()):
        """Get the OOI Metadata for an instrument(s) for a given url. The url
        can point to any end-point, and will get the metadata for all the 
        instruments

            Args:
                metadata_request_url: OOINet request url for an array/node/instrument
                results: initializes to an empty DataFrame. Can pass in an already
                         existing dataframe with metadata results.

            Returns:
                results: A dataframe with the relevant metadata of all instruments nested
                         below the initial url endpoint.
        """

        # =============
        # First, need to strip off if "metadata" included in url
        if "metadata" in metadata_request_url:
            # Strip out metadata
            metadata_request_url = metadata_request_url[:metadata_request_url.find("metadata")-1]

        #print(metadata_request_url)

        # =============
        # Second, check if the url has the instrument info in it
        x = re.search("[0-9]{2}-[023A-Z]{6}[0-9]{3}", metadata_request_url)

        # If x is not none - that means we're at the instrument url and want to request the metadata
        if x is not None:  
            # Make sure the end-point is the instrument url
            metadata_request_url = metadata_request_url[:x.end()]

            # Build the reference designator
            array, node, sensor = metadata_request_url.split("/")[-3:]
            refdes = "-".join((array, node, sensor))

            # Append "metadata" to the url
            metadata_request_url = "/".join((metadata_request_url, "metadata"))

            # Get the metadata information
            ooi_mdata = self.get_api(metadata_request_url)

            # Parse the metadata
            metadata = self.parse_metadata(ooi_mdata)

            # Add in the reference designator to the metadata
            metadata["refdes"] = refdes

            # Append the instrument specific metadata to the results
            results = results.append(metadata)

        else:
            # Get a list of the available url end-points
            new_endpoints = self.get_api(metadata_request_url)

            while len(new_endpoints) > 0:
                # Get an endpoint from the list
                endpoint = new_endpoints.pop()

                # Recursively iterate until we have
                new_metadata_url = "/".join((metadata_request_url, endpoint))

                # Get the
                results = self.get_instrument_metadata(new_metadata_url, results)

        # Return the metadata results
        return results
    
    
    def get_vocab(self, vocab_url, results):
        """Return the OOI vocabulary for a given url endpoint"""
    
        # First, request the data from the url
        data = self.get_api(vocab_url)

        # Now, we'll move through the list
        while len(data) > 0:

            # Get the last item in the list
            x = data.pop()

            # Now, if the x is a dictionary, we have the vocab data
            if type(x) is dict:
                results = results.append(x, ignore_index=True)

            # However, if x is a str, need to iterate agains
            else:
                new_vocab_url = "/".join((vocab_url, x))

                # And iterate again
                results = get_vocab(new_vocab_url, results)

        # Finally, return the results
        return results

OOINet = OOINet(username, token)

OOINet.username, OOINet.token

# ---
# ## Download a specific NetCDF dataset
# First, I want to be able to request and download a specific netCDF datasets and/or file from OOINet. For development, I'm going to use the Pioneer Array (CP01CNSM) Near-Surface Instrument Frame CTD as the test-sensor for this tool.

data_request_url = "/".join((OOINet.urls["data"], "CP01CNSM", "MFD37", "03-CTDBPD000", "recovered_inst", "ctdbp_cdef_instrument_recovered"))
OOINet.get_api(data_request_url)

# **URL Requests need to be in the form:** < data url >/< array >/< node >/< sensor >/< method >/< stream >

thredds_url = OOINet.get_thredds_url(data_request_url, beginDT="2018-01-01", endDT="2019-01-01")

# **Check for when the request is complete:**

check_complete = thredds_url + "/status"



# ---
# ## Stream Metadata
# For a given stream/data request url, what is the available metadata?

metadata_request_url = "/".join((OOINet.urls["data"], "CP01CNSM", "MFD37"))
metadata = OOINet.get_instrument_metadata(metadata_request_url, results)
metadata

metadata["refdes"].unique()

# ---
# ## Deployment Information
# Develop a function to return the deployment information for a particular reference designator
#

deploy_url = "/".join((OOINet.urls["deploy"], "CP01CNSM", "MFD37", "03-CTDBPD000"))

deployments = OOINet.get_api(deploy_url)
deployments

# +
# Get the associated deployment times
# -

pd.DataFrame().from_dict(results[0])

results[0].get("eventStartTime")

pd.DataFrame(results)

# ---
# ## Vocab Info
# Want to develop a tool which will get the available vocabulary for either all instruments on an array, all instruments on a node, or for a specific instrument.

results = pd.DataFrame()
vocab_url = "/".join((OOINet.urls["vocab"], "CP01CNSM", "MFD37"))
vocab = OOINet.get_vocab(vocab_url, results)

vocab["refdes"].unique()



# Practice recursion
def compoundInterest(principal, compounded, duration, rate):
    totalCompounded = duration * compounded
    for i in range(1, (totalCompounded+1)):
        principal = principal*(1 + (rate/compounded))
    return principal


compoundInterest(15000, 1, 3, 0.03)-15000



# ---
# ## Load Data Tables and Parameters

data_streams = pd.read_csv("Reference Tables/data_streams.csv")
parameters = pd.read_csv("Reference Tables/parameters.csv")
metadata = pd.read_csv("Reference Tables/metadata.csv")

data_streams.head()

data_streams[(data_streams["array"] == "CP01CNSM") & (data_streams["sensor"] == )]

parameters.head()

metadata.head()

cp01cnsm_streams = data_streams[data_streams["array"] == "CP01CNSM"]
cp01cnsm_streams.head()


