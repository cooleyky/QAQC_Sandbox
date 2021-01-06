% Function to load the barometer data

fid = fopen("Barometer/F843527A.CSV");

timestamps = [];
relativeHumidity = [];
temperature = [];
pressure = [];

% Initialize the parameters
i = 1;
fline = fgetl(fid);
fline = strtrim(fline);

% Iterate through the headers until reach the actual data
while ~startsWith(fline,"+")
    fline = fgetl(fid);
    fline = strtrim(fline);
end

% Now iterate through the data
while ~feof(fid)
    fline = fgetl(fid);
    fline = strtrim(fline);
    
    % Next, split the data
    fstrs = strsplit(fline,",");
    
    if length(fstrs) == 1
        continue
    end
    
    % Parse and convert the datestring to a MatLab datenum
    timestamp = strcat(fstrs{5}," ",fstrs{4});
    timestamp = datenum(timestamp);
    % Adjust for local vs. UTC time and save the timestamp
    timestamps(i)= datenum(timestamp + hours(4));
    
    % Parse the remaining data
    relativeHumidity(i) = str2num(fstrs{1});
    temperature(i) = str2num(fstrs{2});
    pressure(i) = str2num(fstrs{3});
    
    i = i+1;
end

% Put the data into a nicely formatted table
barometer = table(timestamps', temperature', pressure', relativeHumidity');
barometer.Properties.VariableNames = ["timestamp","temperature","pressure","relativeHumidity"];

