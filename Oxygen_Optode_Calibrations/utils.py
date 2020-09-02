import re
import pandas as pd


def parse_optode(data):
    """Parse the loaded optode data into a dataframe with column headers."""
    columns = ["timestamp", "model", "serial number", "oxygen concentration",
               "oxygen saturation", "temperature", "calibrated phase",
               "temp-compensated calibrated phase", "blue phase", "red phase",
               "blue amplitude", "red amplitude", "raw temperature"]
    df = pd.DataFrame(columns=columns)
    for line in data.splitlines():
        # Now need to parse the lines
        # Get the timestamp of the line
        timeindex = re.search("\[[^\]]*\]", line)
        timestamp = pd.to_datetime(line[timeindex.start()+1:timeindex.end()-1])
        line = line[timeindex.end():].strip()
        # Next, split the data
        model, sn, o2con, o2sat, temp, cal_phase, tcal_phase, blue_phase, red_phase, blue_amp, red_amp, raw_temp = line.split(
            "\t")
        # Put the data into a dataframe
        df = df.append({
            "timestamp": timestamp,
            "model": int(model.strip("!")),
            "serial number": int(sn),
            "oxygen concentration": float(o2con),
            "oxygen saturation": float(o2sat),
            "temperature": float(temp),
            "calibrated phase": float(cal_phase),
            "temp-compensated calibrated phase": float(tcal_phase),
            "blue phase": float(blue_phase),
            "red phase": float(red_phase),
            "blue amplitude": float(blue_amp),
            "red amplitude": float(red_amp),
            "raw temperature": float(raw_temp)
        }, ignore_index=True)

    return df


def load_optode(file):
    """Open, load, and parse a file from the Aanderaa oxygen optode."""
    # Open and load the optode file
    with open(file) as f:
        data = f.read()
        data = data.strip("\n")
        data = data.strip("!")

    # Parse the data into a dataframe with column headers
    df = parse_optode(data)

    return df


def parse_barometer(data):
    """Parse the barometric info into a pandas dataframe."""
    columns = ["timestamp", "temperature", "pressure", "relative humidity"]
    df = pd.DataFrame(columns=columns)

    for line in data:
        # Skip the lines without actual data
        if "+" not in line:
            continue

        # If the line has data, split into its different measurements
        line = line.strip("\x00")
        line = line.replace("+", "")
        rh, temp, pres, time, date = line.split(",")

        # Make a datetime object out of the date and time
        datetime = " ".join((date, time))
        datetime = pd.to_datetime(datetime)

        # Parse the measurements into a dataframe
        df = df.append({
            "timestamp": datetime,
            "temperature": float(temp),
            "pressure": float(pres),
            "relative humidity": float(rh)
        }, ignore_index=True)

    return df


def load_barometer(file):
    """Open, read, and parse barometric data into a dataframe."""
    # Open and read in the barometric data
    with open(file) as f:
        data = f.read()
        data = data.strip("\n")
        data = data.splitlines()

    # Parse the data
    df = parse_barometer(data)

    return df
