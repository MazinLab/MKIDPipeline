from tables import * 
import numpy
import os, sys

class ObsHeader(IsDescription):
    target = StringCol(80)
    dataDir = StringCol(80)
    beammapFile = StringCol(80)
    isWvlCalibrated = BoolCol()
    isFlatCalibrated = BoolCol()
    isSpecCalibrated = BoolCol()
    isLinearityCorrected = BoolCol()
    isPhaseNoiseCorrected = BoolCol()
    isPhotonTailCorrected = BoolCol()
    timeMaskExists = BoolCol()
    startTime = Int32Col()
    expTime = Int32Col()
    wvlBinStart = Float32Col()
    wvlBinEnd = Float32Col()
    energyBinWidth = Float32Col()
    
class ObsFileCols(IsDescription):
    pass
