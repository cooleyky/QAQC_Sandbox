Processing CTD data

Overview:
This document explains how to do basic CTD data processing using the tools available from SeaBird Electronics and some basic batch file commands.  This processing is intended to provide data backup and some quickly processed data for science, in the event they don't choose to do it themselves.  Even if science is doing their own data processing, some version of this should still be run in order to back up the raw data.  Also, it has the ability to allow one or more scientists to separately process the data to there own specifications, and place the results in a separate directory.  Note that it is not our intent to provide publishable data, although we have no objections if the data is used for that.  Scientists should study the data processing closely in order to make sure that it is producing what they need and want.

It is assumed here that the operator in charge of this system has a good understanding of how the Seabird data collection and data processing is done.  This method basically follows Seabirds steps for batch processing data.  If you don't understand how to process a raw ctd data file with Seabirds software, learn how to do that first.  The processing explained here uses the Seabird software "SBE Data Processing" and two additional files to process the data.  One is a batch file, here called go.bat.  The other is a text file, here called batch.txt.  Running go.bat backs up the raw data, does some file copying to set up for the Seabird Data Processing software, and then calls the Seabird Data Processing software.    The Seabird Data Processing software uses the file "batch.txt" to control the steps of the processing.  Once the Seabird Data Processing software has completed, control is returned to go.bat which can do further copying etc., if needed.  (At this time, no further processing is done.)  

Computer Setup: 
It is assumed here that the Seabird software "Seasave" and "SBEdataProcessing" are installed on the computer.  In addition, the directories d:\data\ and d:\data\process are required.  (Note, there may be variations in the data storage structure.)  The data directory is shared so that it can mounted on a server, (mike, april, linus, neil, etc.).  A copy of  go.bat and ctdbatch.txt must be in d:\data\doc, as well as copies of all the needed .psa files.  More on this later.

In addition, for the backup process to function properly, you need subdirectory on another computer, (generally not the server). This computer is generally mounted as the ctd computers "Z" drive.  This is intended strictly as a backup, not available to science, so it does not really matter where it is, as long as there is enough room.  The key here is that the data must be backed up to a physically separate hard drive. Note, if you use the same location from one cruise to the next, you will have to either use different file names, or delete the backup data from the previous cruise.  Traditionally, casts are named by the cruise number/name and by a three digit number, such as 18502003.  In this latter case you will just have to remember to occasionally clean out the backup directory.

Processing the data:  
(Note:  Before processing data for the first time, you have to set up the go.bat file, the ctdbatch.txt file and all of the Seabird programs used to process the data.  That will be covered later.)  Open up a command window, and cd to d:\data\doc\processing_setup\.  Normal processing is initiated by running go.bat with five replaceable parameters entered in the command line.  First is the name of the configuration file, second is the first part of the filename, (generally  the number of the cruise), third is the number of the cast.  The fourth item is the name of the subdirecty the processed files are to be directed to.  The fifth item is the text file needed by SBEBatch, normally ctdbatch.txt.  The processing is set up this way to provide a maximum of flexiblity without the need to change the basic setup for each cruise.  More on this later.  Note that you do NOT include the file extensions in the parameters.  (just config, not config.con)

example:
go 172-05 17205 003 process batch.txt
Life will be easier if casts are named in a consistent way.  The first cast would be 17205001, the 2nd 17205002, and so on...  Also, some portions of the ships website may require that the names follow this standard.

Note:  There is an optional sixth command line entry that is used for the initial setup of the processing.  Add #w as the sixth entry and the seabird software will pause at the end of each section, allowing the operator to set up the processing, as well as saving the config files in the correct places.
example:
go 172-05 17205 003 process batch.txt #w
More on this later.


*Later*
Pre cruise setup:
Setting up go.bat
go.bat is set up so that a absolute minimum number of changes are needed for each cruise.  Hopefully none.  It will spit out some error messages if  the files that it expects to see are missing, but this is usually OK.  go.bat is set up to backup the *.hdr, *.dat, *.con, (or *.hex) *.nav, and  *.btl files to the Z drive.  Since not all casts produce all of these types of files, when go.bat tries to copy them, it will produce a (harmless) error message.   Therefore, there is no need to modify the portion of go.bat that does the data backup.  However, you should be sure that go.bat is in fact backing up all of the raw data.  This might not be the case if you had added some unusual type of data collection that creates a file not in the previous list.   Next, go.bat copies the .xmlcon  file specific to the cast being processed to the process directory.  This file is often needed for later steps in the processing.  Finally, go.bat calls the Seabird program "sbebatch".  This is the program that does the actual processing of the CTD data.  go.bat calls this program with 4 or 5 parameters, like this:
sbebatch d:\data\doc\processing_setup\%5   %1   %2   %3  %4  %6
These are the same parameters that are entered when go.bat is run.  %5 is the name of the text file used to control sbebatch, usually, ctdbatch.txt.  Note the inclusion of the entire path for this parameter.   %1 is the name of the .con file for the cruise.  %2 is the first part of the name of the cast,usually the cruise number.  %3 is the cast number, 001, 002, -> 998,999.  (use all three digits.)  %6 is optional, and is used during the initial setup of the processing.  Placing a #w for the sixth entry in go.bat will make the Seabird processing pause at the end of each section, allowing the operator to set up the processing for the cruise and save the .psa files related to each step in the processing.

Setting up ctdbatch.txt.
The ctdbatch.txt file controls the operation of the Seabird program sbebatch.exe.  It is a simple text file which tell sbebatch.exe how to process the CTD data.  Sbebatch.exe reads each line in the file, running the Seabird program listed in each line.  The format for most lines is similar:  First is the name of the Seabird data processing program being called, such as derive, split, asciiout, etc..  This is followed by switches to tell the called program where the input file is, where the output file goes, and where the .psa (config) file is.  These setting must match the setting chosen during the initial setup of the CTD.  See the instructions for SBEdataprocessing for more information.  Setting up ctdbatch.txt consists of insuring that the desired programs are listed, in the correct order.  Comment out any processing steps that are not desired.  (note that the @ symbol is use to mark comments in this file.  Do not use the # symbol)

Final setup of the processing.
The last step of setting for ctd processing is to run a test cast to set up the actually processing of the various Seabird processing files.  It's not possible to setup the processing without an actual data file from the CTD with the actual sensors to be used.   It's also necessary to have at least one bottle fired, at depth, if you plan on processing bottle data.  (see "how to do a fake bottle cast for more on this")  If you are using the same CTD setup as the previous cruise, you can just grab one of those raw data files to do the setup.  (copy it, and change the name to something that matches the new cruise.  Maybe make it ctd number 999)  -So, you have your raw data file.   Open up a command window, and cd to d:\data\doc\processing_setup\.  Run the following command:
go 172-05 17205 999 batch.txt  #w
go.bat will backup the the data and run the first program in ctdbatch.txt.  It will then pause before exiting this first program, which will always be Data Conversion, (aka datcnv).  

Datcnv will present a screen with three tabs on it, "File Setup", Data Setup", and "Header View".  Numerous settings must be saved in order for the processing to work, and others must be saved in order to get the results desired.  The first entry under "File Setup" is "Program setup file"  Use the "Save As" button to save this to d:\data\doc\processing_setup\.  This will save the .psa file under the default name of DatCnv.psa.  This is where you will save your settings when they are complete.  

Under "Instrument configuration file" click on "select", select the .con file that matches the cast (the one with the same name), and click on "open".  Also, click on the box for "match instrument configuration to input file".  For "Input directory", enter d:\data\.  For "input files", select the test cast (*000.dat).  For output directry, enter d:\data\process\ Leave "Name append" and "Output file" blank.  Note:  THIS IS VERY IMPORTANT:  the software has a habit of placing a default entry in the "output file" box.  If you save the setup, ("save", "save as", top entry), while there is an entry in this box, it will scramble the processing.  So always look at the "output file" box and make sure it is blank before saving.

Click on the "Data Setup" tab.  Check the box for "Process scans to the end of file"  For "Output format", select "Binary output".  For "Convert data from", select "Upcast and downcast".  For "Create file types", select "Create both data and bottle file", unless you are not collecting any water.  In that case, select "Create converted data (.CNV) file only".  If you are collecting water, for "source of scan range data" choose "Scans marked with bottle confirm bit".  For "Scan range offset[s]" select 0, and for "Scan range duration[s]" select 2.  Leave "Merge separate header file" blank.

Click on "Select Output Variables..."  In this screen you will select the variables to be processed.  Generally, you will select pressure, temperature, salinity, as the absolute minimum.  You will also need to select any raw data that might be needed for later processing in the program "Derive".  For instance, Derive can  calculate Oxygen with more accuracy, after some other corrections have been done.  To do this, Derive needs the Oxygen Voltage, not Oxygen.   So you would include Oxygen voltage in the selection, to make it available later.  Basically, pick what you think you need.  (If you get to a step in the processing and are missing something, the software will tell you.  Note what is missing, restart the processing of the cast, and fix what needs fixing.)  Click on  "OK" when you have what you want.  Double check the settings under the "Data Setup" tab and click on the "File Setup" tab.  Double check the settings, make sure that the "output file" box is empty, and click on "Save" to save the DatCnv.psa file to d:\data\doc\processing_setup\.  

Click on "Start Process".  Say "Yes" to "is is ok to overwrite .......".  DatCnv will complete it's processing and the next processing program will open in a similar way.  All the settings are similar, though not identical.  Note that d:\data\process is now the input directory, and will be for the rest of the calculations.  After making all the settings, save the .psa file same as before, remembering again to make sure the entry for "output file" is blank.  Click on "Start Process", etc....  Keep going until done.  

Defaults:
DatCnv: 	Minimums is Pressure, Temp, Sal., probably more.
AlignCTD:	All 0, except Salinty, 2 is 0.073
CellTm	Primary- 0.03 for alpha and 7 for 1/beta.  Secondary is identical.
Derive		No defaults, depends on needs.
BinAvg	Usually 1 meter or 1 decibar bins.  (pressure)  
Translate	Binary -> ASCII
Split		Upcast and downcast.
AscOut	"Output header file", "Output data file" "Label columns 'Top of the file' "  "Column seperator 'Space' ", under "Select output variables" pick them all.
Bottlesum	No defaults, depends on needs.  Probably salinity, maybe oxygen.  

@end




Notes:
There is an upperlimit to the number of casts at 999
How to do a fake bottle cast.
How to do multi-cast re-processing.
How to do multi-instrument processing.
How to do multi-group processing (same cast, different processing for different groups.

You must go through in order, else the data wont be there.

Most people should recognize these setting and realize why they are there.  If you don't go read the Seabird manuals.

Use care when editing the ctdbatch.txt file, as the line may wrap with some editors.  Use wordpad.

Save a set of .psa files for future use.

.psa files must exist before running the script.

Some .psa files will hiccup, and you have to use the seabird stuff to set it up manually. Not sure why this is.  Usually it's because it cannot find the con file.  Once that is set, you can set it up as usual with the #w switch.

When matching instrument configuration to input file, note that the .con file with same name must be in same directory as data file.  This in spite of the fact that you are offered a path.

Notes to programmer.  Having the .psa file in the main (data dir) without a switch make is harder to set up multiple type of processing.  Also put the .psa files here, and add % to location of the .psa.  (Done.)    so, go.bat %1 %2 %3 %4 %5 %6 where 1 is confile, 2 is first part of name, 3 is second part of name, 4 is location of file storae (process or special), 5 is ctdtext.txt (also in storage directory) and 6 #w.  (Done.)   
