@echo off

rem gato.bat, a windows batch file to backup 
rem the raw ctd data and process it.

rem Operation
rem   Only two command line entries needed: gato and castname
rem   gato at2630005

rem There is an optional third command line entry that is used for the initial setup
rem   of the processing.  Add #w as the third entry and the seabird software
rem   will pause at the end of each section, allowing the operator to set up the 
rem   processing, as well as saving the config files in the correct places.
rem   example:
rem   gato at2630005 #w


rem This batch file will also back up the data to another directory 

rem The input directory, output directory, con file extension, and the 
rem   txt file to be used by SBEBatch.exe are all defined below.

rem If you want to use a different input or output directory, 
rem   or a different con file extension you must use the #w option, 
rem   or change the constants set below.

rem Define the input directory to look for input hex files in
set indir=C:\DATA\CTD

rem Define the output directory
set outdir=PROCESS\OOI

rem Define the configuration file extension
rem   comment out the one out that is not used 
rem set contype=.CON
set contype=.XMLCON

rem Define the batch text file that SBEBatch will use
set batchtext=prcast.txt


rem BEGIN PROCESSING:
rem set screen to black on white, clear the screen.
color f0
cls



echo            -
echo    This program will:
echo     -Back up the cast to: C:\data_archives\current
echo     -Process the data in a standard method
echo      and put the processed data in the \data\ctd\process\
echo -
echo    Cast %2 is about to be processed.
echo    If this is not correct, hit control c now. 
pause
cls

rem First, backup the raw data.  Confirm if overwriting, and verify.
rem copy c:\ctd_data\"cruise directory"\"cruise number+cast 
rem use .hdr, .dat, .con, .bl, .nav or else repeated processing will fill the raw  
rem dir with processed data.  It may be apropriate to comment out some lines 

echo Start Raw Data File Copying-
     copy /-y /v c:\data\ctd\%1.hdr        c:\data_archives\current\*.*
     copy /-y /v c:\data\ctd\%1.hex        c:\data_archives\current\*.* 
     copy /-y /v c:\data\ctd\%1%contype%   c:\data_archives\current\*.*
     copy /-y /v c:\data\ctd\%1.bl         c:\data_archives\current\*.*
     copy /-y /v c:\data\ctd\%1.nav        c:\data_archives\current\*.*

echo End File copying
rem Now, set the files to read only.
rem echo change the files to read only
rem attrib +r z:\%2%3.*
rem echo done changing to read only

rem copy the .con file to the process directory for the rosette summary or others.
echo copy the .con file for this cast to the processing directory.
copy  /v c:\data\ctd\%1%contype% c:\data\ctd\%outdir%\*.*

rem Process the data.  Call SeaBirds data processing software.
rem   called with:  sbebatch proc_text_file cast_name Con_file_type input_dir output_dir #w_or_not
echo Start SbeBatch
call sbebatch c:\data\ctd\doc\processing_setup\at3001\%batchtext% %1 %contype% %indir% %outdir% %2
echo Finished with SbeBatch
 
