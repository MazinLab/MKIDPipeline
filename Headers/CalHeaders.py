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


def WaveCalDescription(n_wvl):
    wavecal_description = {"pixel_row": UInt16Col(pos=0),  # beam map row
                           "pixel_col": UInt16Col(pos=1),  # beam map column
                           "resid": UInt32Col(pos=2),  # unique resonator id
                           "wave_flag": UInt16Col(pos=3),  # fit flag
                           "soln_range": Float32Col(2, pos=4),  # wavelength range
                           "polyfit": Float64Col(3, pos=5),  # fit polynomial
                           "sigma": Float64Col(n_wvl, pos=6),  # 1 sigma DelE
                           "R": Float64Col(n_wvl, pos=7)}  # E/DelE
    return wavecal_description


def WaveCalDebugDescription(n_wvl, n_fit, max_hist):
    debug_desc = {"pixel_row": UInt16Col(pos=0),  # beam map row
                  "pixel_col": UInt16Col(pos=1),  # beam map column
                  "resid": UInt32Col(pos=2),  # unique resonator id
                  "hist_flag": UInt16Col(n_wvl, pos=3),  # histogram fit flags
                  "has_data": BoolCol(n_wvl, pos=4),  # bool if there is histogram data
                  "bin_width": Float64Col(n_wvl, pos=5),  # bin widths for histograms
                  "poly_cov": Float64Col(9, pos=6)}  # covariance for poly fit
    for i in range(n_wvl):
        # histogram fits
        debug_desc['hist_fit' + str(i)] = Float64Col(n_fit, pos=7 + 4 * i)
        # histogram covariance
        debug_desc['hist_cov' + str(i)] = Float64Col(n_fit**2, pos=7 + 4 * i + 1)
        # histogram bin centers
        debug_desc['phase_centers' + str(i)] = Float64Col(max_hist, pos=7 + 4 * i + 2)
        # histogram bin counts
        debug_desc['phase_counts' + str(i)] = Float64Col(max_hist, pos=7 + 4 * i + 3)
    return debug_desc


class WaveCalHeader(IsDescription):
    model_name = StringCol(80)


def FlatCalSoln_Description(nWvlBins=13):
    description = {
            "resid"     : UInt32Col(),      # unique resonator id
            "pixelrow"  : UInt16Col(),      # physical x location - from beam map
            "pixelcol"  : UInt16Col(),      # physical y location
            "weights"   : Float64Col(nWvlBins),    #
            "spectrum"   : Float64Col(nWvlBins),    #
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
