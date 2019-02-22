@echo off

rem go.bat, a windows batch file to backup 
rem the raw ctd data and process it.

rem copy the .xmlcon file to the process directory for the rosette summary or others.
echo copy the .xmlcon file for this cast to the processing directory.
copy  /v c:\data\ctd\%2%3.XMLCON c:\data\ctd\%4\*.*

rem Process the data.  Call SeaBirds data processing software.
echo Start SbeBatch
call sbebatch c:\data\ctd\doc\processing_setup\%5 %1 %2 %3 %4 %6
echo Finished with SbeBatch
 
