% Script for processing the oxygen optode data
% ======================================================================= %

fid = fopen('Data/2020-08-17.log');

% Initialize the data
timestamps = [];
data = [];
i=1;

% Set the variable names for the output table
varNames = ["timestamp", "model", "serialNumber", "oxygenConcentration",...
    "oxygenSaturation", "temperature", "calibratedPhase",...
    "temp-compensatedCalibratedPhase", "bluePhase", "redPhase",...
    "blueAmplitude", "redAmplitude", "rawTemperature"];

% Iterate through the file line-by-line and parse the optode data
while ~feof(fid)
    fline = fgetl(fid);
    if isempty(fline)
        continue
    end
    % Get the timestamp of the measurement
    ind1 = strfind(fline, '[');
    ind2 = strfind(fline, ']');
    timestamp = datenum(fline(ind1+1:ind2-1));
    timestamps(i,1) = timestamp;
    % Read in the other data
    fline = replace(fline, '!', '');
    fline = fline(ind2+1:end);
    data(i,:) = sscanf(fline, '%f')';
    i = i+1;
end

% Parse the data into a table
data = [timestamps, data];
optode = array2table(data);
optode.Properties.VariableNames = varNames;