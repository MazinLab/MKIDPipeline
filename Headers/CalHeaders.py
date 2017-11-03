"""
CalHeaders.py

Author: Seth Meeker 2017-04-09
Updated from Original in ARCONS-pipeline for DARKNESS-pipeline

Contains the pytables description classes for certain cal files
"""

import sys
import os
import numpy as np
from tables import *
from Headers import TimeMask
from Headers import pipelineFlags as flags

WaveCalSoln_Description = {
            "resid"     : UInt16Col(),      # unique resonator id
            "pixelrow"  : UInt16Col(),      # physical x location - from beam map
            "pixelcol"  : UInt16Col(),      # physical y location
            "polyfit"   : Float64Col(3),    # polynomial to convert from phase amplitude to wavelength float 64 precision
            "sigma"     : Float64Col(4),     # 1 sigma (Gaussian width) in eV, for blue peak
            "solnrange" : Float32Col(2),    # start and stop wavelengths for the fit in Angstroms
            "wave_flag" : UInt16Col()}      # flag to indicate if pixel is good (0), unallocated (1), dead (2), or failed during wave cal fitting (2+) 


def FlatCalSoln_Description(nWvlBins=13):
    description = {
            "resid"     : UInt16Col(),      # unique resonator id
            "pixelrow"  : UInt16Col(),      # physical x location - from beam map
            "pixelcol"  : UInt16Col(),      # physical y location
            "weights"   : Float64Col(nWvlBins),    #
            "weightUncertainties"     : Float64Col(nWvlBins),     #
            "weightFlags" : UInt16Col(nWvlBins), #
            "flag" : UInt16Col()}      #
    return description


strLength = 100
CalLookup_Description = {
            'obs_run'   : StringCol(strLength),
            'obs_date'   : StringCol(strLength),
            'obs_tstamp'   : StringCol(strLength),
            'waveSoln_run'   : StringCol(strLength),
            'waveSoln_date'   : StringCol(strLength),
            'waveSoln_tstamp'   : StringCol(strLength),
            'waveSoln_isMasterCal'   : BoolCol(),
            'flatSoln_run'   : StringCol(strLength),
            'flatSoln_date'   : StringCol(strLength),
            'flatSoln_tstamp'   : StringCol(strLength),
            'flatSoln_isIllumCal'   : StringCol(strLength),
            'fluxSoln_run'   : StringCol(strLength),
            'fluxSoln_date'   : StringCol(strLength),
            'fluxSoln_tstamp'   : StringCol(strLength),
            'timeMask_run'   : StringCol(strLength),
            'timeMask_date'   : StringCol(strLength),
            'timeMask_tstamp'   : StringCol(strLength),
            'timeAdjustments_run'   : StringCol(strLength),
            'timeAdjustments_date'   : StringCol(strLength),
            'timeAdjustments_tstamp'   : StringCol(strLength),
            'cosmicMask_run'   : StringCol(strLength),
            'cosmicMask_date'   : StringCol(strLength),
            'cosmicMask_tstamp'   : StringCol(strLength),
            'beammap_run'   : StringCol(strLength),
            'beammap_date'   : StringCol(strLength),
            'beammap_tstamp'   : StringCol(strLength),
            'centroidList_run'   : StringCol(strLength),
            'centroidList_date'   : StringCol(strLength),
            'centroidList_tstamp'   : StringCol(strLength),
}
