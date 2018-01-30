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
    wvlCalFile = StringCol(80)
    
class ObsFileCols(IsDescription):
    ResID = UInt32Col(pos=0)
    Time = UInt32Col(pos=1)
    Wavelength = Float32Col(pos=2)
    SpecWeight = Float32Col(pos=3)
    NoiseWeight = Float32Col(pos=4)
