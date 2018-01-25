import tables
import sys, os, struct
import numpy as np
from Headers.ObsFileHeaders import ObsHeader

if __name__=='__main__':
    if len(sys.argv)<3:
        print('Must specify Bin2HDF config file and h5 file path')
        exit()

    cfgFn = str(sys.argv[1])
    cfgFile = open(cfgFn, 'r')
    cfgParamList = cfgFile.read().splitlines()
    cfgFile.close()

    dataDir = cfgParamList[1]
    firstSec = int(cfgParamList[2])
    expTime = int(cfgParamList[3])
    beammapFile = cfgParamList[4]

    filename = str(sys.argv[2])
    hfile = tables.open_file(filename, mode='a')
    
    hfile.create_group('/', 'header', 'Header')
    headerTable = hfile.create_table('/header', 'header', ObsHeader, 'Header')
    
    headerContents = headerTable.row

    headerContents['isWvlCalibrated'] = False
    headerContents['isFlatCalibrated'] = False
    headerContents['isSpecCalibrated'] = False
    headerContents['isLinearityCorrected'] = False
    headerContents['isPhaseNoiseCorrected'] = False
    headerContents['isPhotonTailCorrected'] = False
    headerContents['timeMaskExists'] = False
    headerContents['startTime'] = firstSec
    headerContents['expTime'] = expTime
    headerContents['wvlBinStart'] = 700
    headerContents['wvlBinEnd'] = 1500
    headerContents['energyBinWidth'] = 0.1
    headerContents['target'] = ''
    headerContents['dataDir'] = dataDir
    headerContents['beammapFile'] = beammapFile
    headerContents['wvlCalFile'] = ''

    headerContents.append()
    headerTable.flush()
    hfile.close()

