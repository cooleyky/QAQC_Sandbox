# Notes for Pioneer-04_AT-27
Author: OOI-CGSN @ Woods Hole Oceanographic Institution
Version: 1-02
Date last updated: 2019-07-30

### Processing
Multiple rows for a particular station-cast-niskin pairing is a result of the sample processing routine. When there are duplicate discrete samples for a particular station-cast-niskin, the processing splits the duplicates into separate rows to provide an unique series of associated data for the sample. This allows for separate reporting of duplicates on the sample summary spreadsheet. Consequently, for each row, there will be an unique value for at least one of the columns.


### Niskin
| Cruise | Station | Bottle Position | Comments                                            |
| ------ | ------- | --------------- | --------------------------------------------------- |
| AT-31A | 8       | 1 - 5           | Loose vent seals                                    |
| AT-31A | 8       | 5               | Lost some CHL sample due to spill during filtration |


### Nutrients
All nutrient values are reported as the average of triplicate analysis on a single collected sample. Nitrate+nitrite is necessary for direct comparison with ISUS/SUNA measurements.

### Chlorophyll
All notes with a bit mask of 101 indicates a sample value recorded following dilution due to high chlorophyll signal except for those specifically listed below.

| Cruise | Station | Bottle Position | Comments |
| ------ | ------- | --------------- | -------- |
| AT       |         |                 |          |

(AT27-A, Sta 1, Niskin 5 - 8): over recommended max chlorophyll reading
    (AT27-A, Sta 3, Niskin 5): lab notebook says 5ml extract but very wrong
      chlorophyll result; 10 ml makes sense so 5 ml must be a typo
    (AT27-A, Sta 5, Niskin 7 & 8): over recommended max chlorophyll reading
    (AT27-A, Sta 6, Niskin 5 & 6): over recommended max chlorophyll reading
    (AT27-B, Sta 7, Niskin 5 - 7): over recommended max chlorophyll reading
    (AT27-B, Sta 1, Niskin 6 & 7) - top piece of testtube broke in fridge, high
      reading requiring dilution
    (AT27-B, Sta 4, Niskin 5 & 6) - First dilution still too high, extra drop of extract; 2 dilution
    (AT27-B, Sta 4, Niskin 6) - high reading requiring dilution; 1 dilution still too high; 2nd dilution

| Cruise | Station | Bottle Position | Comments                           |
| ------ | ------- | --------------- | ---------------------------------- |
| AT-31A | 1       | 1 - 3           | Chl reading within noise level     |
| AT-31A | 1       | 4               | Chl reading just above noise level |
| AT-31A | 2       | 2               | Chl reading within noise level     |
| AT-31A | 3       | 1 - 4           | Chl reading within noise level     |
| AT-31A | 4       | 1, 2            | Chl reading within noise level     |
| AT-31A | 4       | 3, 4            | Chl reading just above noise level |
| AT-31B | 3       | 1 - 8           | Very low reading for fluorescence  |





  ### Data Flag Description
  The data flags are presented in the summary sheet as a 16-bit array, read from right-to-left, where a 1 in a particular bit position indicates a particular flag meaning applies. For example, a flag of 0000000000000010 for the column **CTD File Flag** indicates that the cast was a data cast only.

  Additionally, these data flags an assessment of the collection and processing of the relevant data or samples, and are not an assessment of the *accuracy* of the data. For example, a conductivity sensor which has the correct calibration coefficients and functions normally will receive a quality flag of 0000000000000100 (acceptable measurement). However, the calibration coefficients may be out of date and off with respect to the discrete salinity results; this does not affect the assigned flag.

  | Bit Position | Cast Flag                                              | CTD File Flag                                     | CTD Parameter Flag                               | Niskin Flag                      | Discrete Sample Flag                                                | Discrete Duplicate Flag              |
  | ------------ | ------------------------------------------------------ | ------------------------------------------------- | ------------------------------------------------ | -------------------------------- | ------------------------------------------------------------------- | ------------------------------------ |
  | 0            | Notes/Other                                            | Notes/Other                                       | Notes/Other                                      | Notes/Other                      | Notes/Other                                                         | Notes/Other                          |
  | 1            | Delayed start to data collection                       | Data cast only                                    | Not Calibrated                                   | Bottle information unavailable   | Sample for this measurement was drawn but analysis not yet received | Duplicate analysis on same Sample    |
  | 2            | Acceptable; normal cast according to SOP               | Acceptable; file processed according to SOP       | Acceptable measurement                           | No problems noted                | Acceptable; sample processed according to SOP                       | Single Sample                        |
  | 3            | Non-standard winch speed                               | File processed using modified parameters          | Questionable measurement                         | Leaking                          | Questionable measurement                                            | Duplicate analysis from same Niskin  |
  | 4            | Non-standard surface soak time                         | File processed using alternate XMLCON             | Bad measurement                                  | Ran out of water during sampling | Bad measurement                                                     | Triplicate analysis from same Niskin |
  | 5            | Non-standard bottle soak time                          | Missing scans as indicated by module error counts | Not reported                                     | Vent open                        | Not reported                                                        | Unassigned                           |
  | 6            | Sensor issues but cast completed and data collected    | Missing metadata                                  | Calibration coefficients greater than 1 year old | Misfire at wrong depth           | Sample collected out-of-order                                       | Unassigned                           |
  | 7            | Cable issues but cast completed and data collected     | Unassigned                                        | Unassigned                                       | Unknown problem                  | Sample processed using alternative methods                          | Unassigned                           |
  | 8            | Winch issues but cast completed and data collected     | Unassigned                                        | Unassigned                                       | Unassigned                       | Unassigned                                                          | Unassigned                           |
  | 9            | Premature cast end with data and/or data loss          | Unassigned                                        | Unassigned                                       | Unassigned                       | Unassigned                                                          | Unassigned                           |
  | 10           | Significant ship heave                                 | Unassigned                                        | Unassigned                                       | Unassigned                       | Unassigned                                                          | Unassigned                           |
  | 11           | Station position not adequately maintained during cast | Unassigned                                        | Unassigned                                       | Unassigned                       | Unassigned                                                          | Unassigned                           |
  | 12           | Tow-yo, Yo-yo cast                                     | Unassigned                                        | Unassigned                                       | Unassigned                       | Unassigned                                                          | Unassigned                           |
  | 13           | ROV Bottle sample                                      | Unassigned                                        | Unassigned                                       | Unassigned                       | Unassigned                                                          | Unassigned                           |
  | 14           | Unassigned                                             | Unassigned                                        | Unassigned                                       | Unassigned                       | Unassigned                                                          | Unassigned                           |
  | 15           | Unassigned                                             | Unassigned                                        | Unassigned                                       | Unassigned                       | Unassigned                                                          | Unassigned                           |
