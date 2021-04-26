% Script to process the optode data
clear all; close all;
addpath("..")  

% Location of where the optode and barometer data are stored
optPath = "../data/optode/2021-04-09.log";
barPath = "../data/barometer/2021-04-09.CSV";

% Load the optode and barometer data
optode = f_load_optode(optPath);
barometer = f_load_barometer(barPath);

% Next, interpolate the barometer data to the optode timestamps
new_t = optode.timestamp;
newP = interp1(barometer.timestamp, barometer.pressure, new_t);
newT = interp1(barometer.timestamp, barometer.temperature, new_t);
newRH = interp1(barometer.timestamp, barometer.relativeHumidity, new_t);
barometer = table(new_t, newT, newP, newRH);
barometer.Properties.VariableNames = ["timestamp","temperature","pressure","relativeHumidity"];

% Create a time variable of elapsed minutes
optode.minutes = etime(datevec(optode.timestamp), datevec(optode.timestamp(1)))/60;
barometer.minutes = etime(datevec(barometer.timestamp), datevec(barometer.timestamp(1)))/60;

% Calculate the oxygen saturation value
optode.o2sol = f_O2sol(0,optode.temperature);
optode.o2sat = (optode.oxygenConcentration./optode.o2sol).*100;

% Plot the data
yyaxis left
scatter(optode.timestamp, optode.oxygenConcentration, "b.")
ylabel("Oxygen Concentration [umol/kg]")
yyaxis right
scatter(barometer.timestamp, barometer.temperature, "r.")
ylabel("Air Temperature [\circC]")
datetick("x", "HH:MM:SS")
grid on

% Plot the elapsed time
yyaxis left
scatter(optode.minutes, optode.oxygenConcentration, "b.")
ylabel("Oxygen Concentration [umol/kg]")
yyaxis right
scatter(optode.minutes, optode.temperature, "r.")
ylabel("Optode Temperature [^\circC]")
xlabel("Elapsed Time (minutes)")
grid on

% Plot the oxygen saturation vs time
yyaxis left
scatter(optode.minutes, optode.oxygenSaturation, "b.")
ylabel("Oxygen Saturation [%]")
yyaxis right
scatter(optode.minutes, optode.o2sat, "r.")
ylabel("Optode Saturation [%]")
xlabel("Elapsed Time (minutes)")
grid on

% =========================================================================
% Calculate the 100% saturated oxygen data
tmin = 100;
tmax = 150;
O2optsat = mean(optode.oxygenConcentration(optode.minutes >=100 & optode.minutes <= 150));
O2solsat = mean(optode.o2sol(optode.minutes >= 100 & optode.minutes <= 150));



% =========================================================================
% Determine the zero-saturation value end-point by curve-fitting
% Select the appropriate xdata and ydata
xdata = optode.minutes(optode.minutes > 155 & optode.minutes < 200);
ydata = optode.oxygenConcentration(optode.minutes > 155 & optode.minutes < 200);
% Rescale the xdata to be elapsed minutes
xdata = xdata - xdata(1);

% Fit with the curve-fitting gui
cftool(xdata, ydata)

% Fit using a custom-equation

% Try deconvolving the data
x = optode.minutes;
y = optode.oxygenConcentration;
% Pad the end of the timeseries
y(end:2*length(y)-1) = y(end);
c = exp(-x./0.6729);

% Deconvolve the oxygen data from the optode signal
o2 = deconv(y, c).*sum(c);

% Plot the elapsed time
figure;
scatter(optode.minutes, o2, "k.")
hold on
scatter(optode.minutes, optode.oxygenConcentration, "b.")
ylabel("Oxygen Concentration [umol/kg]")
xlabel("Elapsed Time (minutes)")
ylim([0, 350]);
grid on

