import tables
import sys, os, struct
import numpy as np
from Headers.ObsFileHeaders import ObsHeader

if __name__=='__main__':
    if len(sys.argv)<3:
        print('Must specify filename, exposure time')
        exit()

    filename = str(sys.argv[1])
    expTime = int(sys.argv[2])
    
    basename = os.path.basename(filename)
    firstSec = int(basename.split('.')[0])

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
    headerContents['dataDir'] = ''
    headerContents['beammapFile'] = ''

    headerContents.append()
    headerTable.flush()
    hfile.close()

