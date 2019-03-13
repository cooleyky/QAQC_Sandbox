README.md

Author: Andrew Reed, OOI - CGSN Data QA/QC Lead
Date: ver_01 - 20190103

# Cruise: Pioneer-03 Leg 1
## Salts and O2 Data
### Salts Bottle Data
The file names for the salt data are named for the cast number, e.g. 001.SAL corresponds to the Salinity bottles collected for cast #1. However, within each .SAL file, the bottle sample numbers are not clearly related to the "Salts Bottle Number" stored on the CTD Sampling Log. For example, on the CTD Sampling Log, the Salts Bottle Number are named as "B3" or "J7". However, the .SAL numberings all begin at 1 and increase incrementally. Thus, there is no direct mapping of "B9" on the CTD sampling log to a similar sample in a .SAL file.

Consequently, I suggest the following mapping based on the following logic:
1. Match based on the cast number in the CTD sampling log to the .SAL file name
2. Check that the "case number (e.g. the B or J)" in the CTD sampling log match the case number contained in the header of the .SAL file.
3. Match the Niskin bottle to the salinity sample assuming that samples increase incrementally with niskin bottle number.