@echo off

rem go.bat, a windows batch file to backup 
rem the raw ctd data and process it.

rem Operation
rem Normally, five replaceable parameters 
rem will be entered in the command line.  
rem like this: 
rem go at18-06 at18-06 003 process batch_sssg.txt

rem First is the name of the configuration file, 
rem Second is the cruise number as found in linus:/home/data/CRUISEID, 
rem Third is the number of the cast.   
rem Fourth is the subdirecty the processed files are to be directed to.  This directory must exist.
rem Fifth is the name of the text file needed by SBEBatch.  

rem This batch file will also back up the data to another 
rem directory 

rem example:
rem go at18-06 at18-06 001 process batch.txt
rem casts must be named in the same way.  The first cast would be
rem at18-06001, the 2nd at18-06002, and so on...

rem There is an optional sixth command line entry that is used for the initial setup
rem of the processing.  Add #w as the sixth entry and the seabird software
rem will pause at the end of each section, allowing the operator to set up the 
rem processing, as well as saving the config files in the correct places.
rem example:
rem go at18-06 at18-06 003 process batch.txt #w

rem BEGIN PROCESSING:
rem set screen to black on white, clear the screen.
color f0
cls

echo    -
echo This program will:
echo -Back up the cast to: C:\data_archives\current
echo -Process the data in a standard method
echo  and put the processed data in the \data\ctd\process\
echo -
echo The cast %3 on cruise %2 is about to be processed.
echo If this is not correct, hit control c now.  Otherwise 
pause
cls

rem First, backup the raw data.  Confirm if overwriting, and verify.
rem copy c:\ctd_data\"cruise directory"\"cruise number+cast 
rem use .hdr, .dat, .con, .bl, .nav or else repeated processing will fill the raw  
rem dir with processed data.  It may be apropriate to comment out some lines 

echo Start Raw Data File Copying-
     copy /-y /v c:\data\ctd\%2%3.hdr   c:\data_archives\current\*.*
     copy /-y /v c:\data\ctd\%2%3.hex   c:\data_archives\current\*.* 
     copy /-y /v c:\data\ctd\%2%3.con   c:\data_archives\current\*.*
     copy /-y /v c:\data\ctd\%2%3.bl    c:\data_archives\current\*.*
     copy /-y /v c:\data\ctd\%2%3.nav   c:\data_archives\current\*.*

echo End File copying
rem Now, set the files to read only.
rem echo change the files to read only
rem attrib +r z:\%2%3.*
rem echo done changing to read only

rem copy the .con file to the process directory for the rosette summary or others.
echo copy the .con file for this cast to the processing directory.
copy  /v c:\data\ctd\%2%3.XMLCON c:\data\ctd\%4\*.*

rem Process the data.  Call SeaBirds data processing software.
echo Start SbeBatch
call sbebatch c:\data\ctd\doc\processing_setup\%5 %1 %2 %3 %4 %6
echo Finished with SbeBatch
 
