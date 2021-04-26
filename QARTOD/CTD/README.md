# OOI-CGSN CTD QARTOD
**Version**: 1-00
**Date**: 2021-02-08
**Author**: Andrew Reed
<br>

Table 1. Revision history <br>

| Version | Description | Release Date |
| ------- | ----------- | ------------ |
| 1-00    | First Draft | 2021-02-08   |


## Introduction
#### Scope
This document outlines the methodology to calculating the parameter thresholds for the QARTOD testing for CTD science parameters.

#### Purpose
The


## Data Sources

#### Platforms
There are three different types of platforms which host CTDs and return temperature and salinity data for OOI-CGSN. The three platforms on which CTDs are deployed include: (1) Fixed-depth plaforms; (2) Fixed-location; (2) Fixed-location vertical Profilers; and (3) Mobile Platforms, such as gliders and AUVs. Each type of platform needs to be handled separately due to varying spatial and temporal coverage.

Fixed-depth platforms includes the surface buoys, near-surface-instrument-frames (NSIFs), and multi-function nodes (MFNs) on the surface mooring arrays as well as the instrument risers on subsurface flanking moorings and hybrid profiler moorings. The instruments mounted on these platforms should vary only slightly in pressure and depth. The main dimension to calculate

Fixed-location vertical profilers are place


## Methodology

#### QC Flags


#### Gross Range Test
The Gross Range Test dep
* fail_span: This is defined as the range outside of which a data point is flagged as bad ("9"). These values are taken from the mechanical and/or calibration limits of the given sensor.
* suspect_span: This is defined as the range outside of which a data point is flagged as suspicious/interesting ("3"). These values are calculated from the available historical data for the given sensor at the given location as the mean +/- 3 standard deviations.

#### Climatology
The climatology test defines a range of values for each month of the year outside of which a data point should be flagged as suspicious/interesting ("3"). The climatology test is calculated by:
1. Taking the monthly mean for  




## References
1. US. Integrated Ocean Observing System, 2015. Manual for Real-Time Quality Control of In-situ Temperature and Salinity Data Version 2.0: A Guide to Quality Control and Quality Assurance of In-situ Temperature and Salinity Observations, 56 pp.
