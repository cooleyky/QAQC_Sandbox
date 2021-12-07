# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Import libraries
import os, shutil, sys, time, re, requests, csv, datetime, pytz
import yaml
import pandas as pd
import numpy as np
import netCDF4 as nc
import xarray as xr
import warnings
warnings.filterwarnings("ignore")

# Import the OOI M2M tool
sys.path.append("/home/andrew/Documents/OOI-CGSN/ooinet/ooinet/")
from m2m import M2M

# Import user info for connecting to OOINet via M2M
userinfo = yaml.load(open("../../../../QAQC_Sandbox/user_info.yaml"), Loader=yaml.FullLoader)
username = userinfo["apiname"]
token = userinfo["apikey"]

OOINet = M2M(username, token)

OOINet.URLS

# Reset the M2M location to ooinet-dev1-west.intra.oceanobservatories.org
for key in OOINet.URLS:
    url = OOINet.URLS.get(key)
    if "opendap" in url:
        dev1_url = re.sub("opendap", "opendap-dev1-west.intra", url)
    else:
        dev1_url = re.sub("ooinet","ooinet-dev1-west.intra", url)
    OOINet.URLS[key] = dev1_url

OOINet.URLS

OOINet.search_datasets()

refdes = "CP01CNSM-MFD35-05-PCO2WB000"

OOINet.get_datastreams(refdes)

method = "recovered_inst"
stream = "pco2w_abc_instrument"

thredds_url = "https://opendap-dev1-west.intra.oceanobservatories.org/thredds/catalog/ooi/areed@whoi.edu/20211207T183204327Z-CP03ISSM-RID27-03-CTDBPC000-recovered_inst-ctdbp_cdef_instrument_recovered/catalog.html"

thredds_url = OOINet.get_thredds_url(refdes, method, stream)
thredds_url

catalog = OOINet.get_thredds_catalog(thredds_url)
catalog

catalog = sorted(OOINet.parse_catalog(catalog, exclude=["ENG", "blank"]))
catalog

OOINet.REFDES = refdes

data = OOINet.load_netCDF_datasets(catalog)
data

# ### Load netCDF files from local directory

save_dir = f"/home/andrew/Documents/OOI-CGSN/QAQC_Sandbox/QARTOD/Testing/data/testing/{refdes}/"

netCDF_files = [save_dir+f for f in os.listdir(save_dir)]
netCDF_files

from dask.diagnostics import ProgressBar

OOINet.REFDES = refdes

# +
# -------------------------------
# Third, check and remove any files which are malformed
# and remove the bad ones
netCDF_files = OOINet._check_files(netCDF_files)

# Load the datasets into a concatenated xarray DataSet
with ProgressBar():
    print("\n"+f"Loading netCDF_files for {OOINet.REFDES}:")
    ds = xr.open_mfdataset(netCDF_files, preprocess=OOINet._preprocess, parallel=True)

# Add in the English name of the dataset
refdes = "-".join(ds.attrs["id"].split("-")[:4])
vocab = OOINet.get_vocab(refdes)
ds.attrs["Location_name"] = " ".join((vocab["tocL1"].iloc[0],
                                      vocab["tocL2"].iloc[0],
                                      vocab["tocL3"].iloc[0]))


# -



# ## QARTOD Comparison

# +
# First, cut down the dataset size to be more managable
# -

dataVars = ["pco2_seawater", "pco2_seawater_qartod_executed"]
pco2w = data[dataVars]
pco2w

# ### QARTOD values
# Next, load the QARTOD tables from github and parse them into dictionaries.
#
# Changes: None

inst = "pco2w"
param = "pco2_seawater"

import io
import json
def loadQARTOD(refDes,param,sensorType):
    
    (site,node,sensor1,sensor2) = refDes.split('-')
    sensor = sensor1 + '-' + sensor2
    
    ### Load climatology and gross range values
    githubBaseURL = 'https://raw.githubusercontent.com/oceanobservatories/qc-lookup/master/qartod/'
    if 'ph_seawater' in param:
        ClimParam = 'seawater_ph'
    else:
        ClimParam = param
    clim_URL = githubBaseURL + sensorType + '/climatology_tables/' + refDes + '-' + ClimParam + '.csv'
    grossRange_URL = githubBaseURL + sensorType + '/' + sensorType + '_qartod_gross_range_test_values.csv'
    download = requests.get(grossRange_URL)
    if download.status_code == 200:
        df_grossRange = pd.read_csv(io.StringIO(download.content.decode('utf-8')))
        paramString = "{'inp': '" + param + "'}"
        qcConfig = df_grossRange.qcConfig[(df_grossRange.subsite == site) 
                                          & (df_grossRange.node == node) 
                                          & (df_grossRange.sensor == sensor) 
                                          & (df_grossRange.parameters == paramString)]
        qcConfig_json = qcConfig.values[0].replace("'", "\"")
        grossRange_dict = json.loads(qcConfig_json)
    else:
        print('error retriving gross range data')
        grossRange_dict = {}

    download = requests.get(clim_URL)
    if download.status_code == 200:
        df_clim = pd.read_csv(io.StringIO(download.content.decode('utf-8')))
        climRename = {
                'Unnamed: 0':'depth',
                '[1, 1]':'1',
                '[2, 2]':'2',
                '[3, 3]':'3',
                '[4, 4]':'4',
                '[5, 5]':'5',
                '[6, 6]':'6',
                '[7, 7]':'7',
                '[8, 8]':'8',
                '[9, 9]':'9',
                '[10, 10]':'10',
                '[11, 11]':'11',
                '[12, 12]':'12'           
            } 
        
        df_clim.rename(columns=climRename, inplace=True)
        clim_dict = df_clim.set_index('depth').to_dict()
    else:
        print('error retriving climatology data')
        clim_dict = {}
    
    return(grossRange_dict,clim_dict)


grossRange_dict, clim_dict = loadQARTOD(refdes, param, inst)
grossRange_dict, clim_dict

# ### Add Climatology Values
# Next, add the climatology min and max values to the dataset as new data variables, based on the month of the data.
#
# Changes:
# * Renamed "climatologyMin/climatologyMax" to "{parameter name}\_climatologyMin/climatologyMax" in order to allow multiple parameter climatologies to be stored in an given dataset
# * Preallocated the climatology arrays with nans instead of zeros to skip the later step of backfilling nans.
# * Utilize dask to get the months (as integers) in the time variable of the dataset. This avoids loading the data into memory.
# * Utilize direct assignment of the climatologyMin/Max values for each month on the dataset variable arrays. This again keeps the dataset out-of-memory.

# +
import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import ast

def add_climatology_values(ds, param, clim_dict):
    """Adds climatology mins and maxes to the dataset timeseries
    
    Parameters
    ----------
    ds: xarray.Dataset
        Dataset to add climatology values to, with primary dimension "time"
    param: str
        Name of parameter in the passed xarray.Dataset which to add
        climatology values to
    clim_dict: dict
        A dictionary of the climatology values for the given dataset
        loaded from the qartod gitHub repo
        
    Returns
    -------
    ds: xarray.Dataset
        An xarray dataset with climatology mins and maxes added for the given
        parameter (param) to the dataset
        
    Note: Will need to add a pressure function to make this match the original functionality
    """
    
    # First, create a variable name to store the data
    varNameMin = f"{param}_climatologyMin"
    varNameMax = f"{param}_climatologyMax"
    
    # Next, pre-allocate an array with the data
    ds[varNameMin] = ds[param].astype(float) * np.nan
    ds[varNameMax] = ds[param].astype(float) * np.nan
    
    # Get the months
    time = da.from_array(ds.time.dt.month)
    months = np.unique(time).compute()
    
    # Add the climatology min and max based on the month of the measurement
    for month in months:
        climatology = ast.literal_eval(clim_dict[str(month)][str([0, 0])])
        ds[varNameMin][(ds.time.dt.month == month)] = climatology[0]
        ds[varNameMax][(ds.time.dt.month == month)] = climatology[1]
        
    return ds


# -

pco2w = add_climatology_values(pco2w, param, clim_dict)


# ### Add QARTOD flags
# Next, want to calculate the QARTOD flags for the gross range and climatology values and add them to the dataset. 
# Changes:
# * Renamed the "gr_flag/clim_flag" to "{parameter name}\_gr_flag/\_clim_flag" in order to allow multiple parameters to be tested in a single dataset.
# * Utilize direct assignment of the QARTOD flags to avoid loading data into memory.

def create_QARTOD_flags(ds, param, grossRange):
    """Function to add the gross range and climatology flags"""
    
    # Add the gross range flags for a param
    gr_flag = f"{param}_gr_flag"
    ds[gr_flag] = ds[param].astype("int64") * 0 + 1
    gr_suspect = grossRange["qartod"]["gross_range_test"]["suspect_span"]
    gr_fail = grossRange["qartod"]["gross_range_test"]["fail_span"]
    ds[gr_flag][(ds[param] < gr_suspect[0]) | (ds[param] > gr_suspect[1])] = 3
    ds[gr_flag][(ds[param] < gr_fail[0]) | (ds[param] > gr_fail[1])] = 4
     
    # Climatology flags
    clim_flag = f"{param}_clim_flag"
    ds[clim_flag] = ds[param].astype("int64") * 0 + 1
    ds[clim_flag][(ds[f"{param}_climatologyMin"].isnull()) | (ds[f"{param}_climatologyMax"].isnull())] = 2
    ds[clim_flag][(ds[param] < ds[f"{param}_climatologyMin"]) | (ds[param] > ds[f"{param}_climatologyMax"])] = 3
    
    # Check for not evaluated locations
    not_eval = ds[param].isnull()
    ds[gr_flag][not_eval] = 9
    ds[clim_flag][not_eval] = 9
    
    return ds


pco2w = create_QARTOD_flags(pco2w, param, grossRange_dict)
pco2w


# ### Compare test values
# Now, want to compare the values calculated locally with the values returned by OOINet in the "qartod_executed" variables.
#
# Changes:
# * Don't iterate through each data point
# * Change the data type of the {parameter name}\_qartod_executed data array to string to be interperable
# * With the type changed to string, can use the xarray built-in string methods (.str) to parse each value in the "qartod_executed" array
# * Changed the name of "qartod_gr/qartod_clim" to "{parameter name}\_qartod_gr/\_qartod_clim" to allow multiple parameters to be stored in the same dataset
# * Run the test comparison and store as "{parameter name}\_gr_comparison/\_clim_comparison" as a boolean array. This will allow us to quickly count the comparison (using sum) and mask the parameter being tested.

def run_comparison(ds, param):
    
    # First, identify the test order of the qartod tests run
    qartod_name = f"{param}_qartod_executed"
    test_order = ds[qartod_name].attrs["tests_executed"].strip("'").replace(" ", "").split(",")
    
    # Second, identify the index of each test
    clim_index = test_order.index("climatology_test")
    gr_index = test_order.index("gross_range_test")
    
    # Next, convert the OOINet-run QARTOD flags to interperable strings
    ds[qartod_name] = ds[qartod_name].astype(str)
    
    # Parse the qartod flags into the separate test flags
    ds[f"{param}_qartod_gr"] = ds[qartod_name].str.get(gr_index).astype("int")
    ds[f"{param}_qartod_clim"] = ds[qartod_name].str.get(clim_index).astype("int")
    
    # Compare the OOI Qartod with local Qartod
    ds[f"{param}_gr_comparison"] = ds[f"{param}_qartod_gr"] != ds[f"{param}_gr_flag"]
    ds[f"{param}_clim_comparison"] = ds[f"{param}_qartod_clim"] != ds[f"{param}_clim_flag"]
    
    return ds


pco2w = run_comparison(pco2w, param)
pco2w

# ### Execute the comparison
# So far, all the work we've done hasn't actually run any processing. Everything has been done as a set of dask instructions to execute when we call compute().
#
# Below, I first just count the number of missed flags by summing the comparison results, since each "missed" flag is stored as a boolean ```True```, which ```.sum()``` counts as a 1. 

from dask.diagnostics import ProgressBar

with ProgressBar():
    for var in pco2w.variables:
        if "comparison" in var:
            result = pco2w[var].sum().compute()
            print(f"Missed flags for {var}: {result.values}")


def pressureBracket(pressure,clim_dict):
    bracketList = []
    pressBracket = 'notFound'

    for bracket in clim_dict['1'].keys():
        x = re.search(r'\[(.+),(.+)\]', bracket)
        if x:
            bracketList.append([int(x.group(1)),int(x.group(2))])
        else:
            print('bracket parsing error for ' + bracket)
    for bracket in bracketList:
        if (pressure >= bracket[0]) & (pressure < bracket[1]):
            pressBracket = bracket
            break
    
    return pressBracket
