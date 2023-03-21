# Seismic anisotropy and stress-field variations along the Dead Sea fault zone in northern Israel
Metadata used in the manuscript:
* ## sites.csv - station locations
* events.csv - event information
* sWavePicks.csv - S-wave arrival times

Code used for the manuscript:
* cross_correlation_method.py - the cross-correlation method. produces:
  * SWS_CC.csv
* supplemental_functions.py - functions used in cross_correlation_method.py
* map_of_SWS.py - generates a map with rose diagrams from SWS_CC.csv
* rose_diagrams.py - makes individual ros_diagrams and some time-delay comparisons
* utils.py - functions used in map_of_SWS.py & rose_diagrams.py
