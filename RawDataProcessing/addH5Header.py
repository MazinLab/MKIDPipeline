import tables
import sys, os, struct
import numpy as np

if __name__=='__main__':
    if len(sys.argv)<3:
        print('Must specify filename and exposure time')
        exit()

    filename = str(sys.argv[1])
    expTime = int(sys.argv[2])
    
    basename = os.path.basename(filename)
    firstSec = int(basename.split('.')[0])

    hfile = tables.open_file(filename, mode='a')
    tableData = np.array([(False, False, False, False, False, False, False, firstSec, expTime)], dtype=[('isWvlCalibrated', 'b1'), ('isFlatCalibrated', 'b1'), ('isSpecCalibrated', 'b1'), 
        ('isLinearityCorrected', 'b1'), ('isPhaseNoiseCorrected', 'b1'), ('isPhotonTailCorrected', 'b1'), ('timeMaskExists', 'b1'), ('startTime', 'u8'), ('expTime', 'i4')])
    
    hfile.create_group('/', 'header', 'Header')
    hfile.create_table('/header', 'header', tableData, 'Header')

    hfile.close()

