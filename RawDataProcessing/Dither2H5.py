#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:45:42 2018

@author: bmazin
"""

import sys, os, time, struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from readDict import readDict
import argparse
import datetime
import subprocess

#******************************** Things you must set by hand for this to work!
XPIX = 80
YPIX = 125
binPath = '/home/bmazin/HR8799/rawdata'    # path to raw .bin data
outPath = '/home/bmazin/HR8799/rawdata'    # path to output data
beamFile = '/home/bmazin/HR8799/h5/finalMap_20170924.txt'    # path and filename to beam map file
mapFlag = 1
filePrefix = 'a'   # output h5 file prefix
b2hPath = '/home/bmazin/DarknessPipeline/RawDataProcessing'

#***********************************

parser = argparse.ArgumentParser(description='Process a dither stack into hdf5 files.')
parser.add_argument('ditherName', metavar='file', nargs=1,
                    help='filename of the dither config file')
parser.add_argument('bufferTime', metavar='sec', type=int, nargs=1,
                    help='The amount of time in seconds to exclude from the beginning and end of each h5 file.')

args = parser.parse_args()

try:
    bufferTime = args.bufferTime[0]
except:
    bufferTime = 1
    
print('Starting conversion of '+args.ditherName[0] + ' into HDF5 files, one file per dither position. ')
print('Removing ' + str(args.bufferTime[0]) + ' seconds from the beginning and end of each h5 file.' )

# Read in config file
configData = readDict()
configData.read_from_file(args.ditherName[0])

nPos = configData['nPos']
startTimes = configData['startTimes'] 
stopTimes =  configData['stopTimes'] 

timestamp = int(startTimes[0]) 
value = datetime.datetime.fromtimestamp(timestamp)
print('Data collection started on ' + value.strftime('%Y-%m-%d %H:%M:%S'))

# ********* Bin2HDF config file format is
# Path
# First file number
# nFiles
# beamFile (complete path)
# mapFlag

wd = os.getcwd()
for i in range(nPos):
    print( 'File ' + str(i) + '/'  + str(nPos) + ': ' + str(int(startTimes[i])+bufferTime) + ' to ' + str(int(stopTimes[i])-bufferTime) )
    
    # prepare config file for Bin2HDF
    f = open(wd+'/'+filePrefix+str(i)+'.cfg','w')
    f.write(str(XPIX)+' '+str(YPIX)+'\n')
    f.write(binPath+'\n')
    f.write(str(int(startTimes[i])+bufferTime)+'\n')
    f.write(str(( int(stopTimes[i])-bufferTime) - (int(startTimes[i])+bufferTime) )+'\n')
    #f.write('5\n')   # fast execution for debugging
    f.write(beamFile+'\n')
    f.write(str(mapFlag)+'\n')
    f.write(outPath)
    f.close()
    
    # start Bin2HDF
    #subprocess.run([b2hPath,wd+'/'+filePrefix+str(i)+'.cfg'],stdout=subprocess.PIPE)
    
    # change to RawDataProcessing directory so the python calls in Bin2HDF work OK
    os.chdir(b2hPath)
    process = subprocess.Popen(['./Bin2HDF',wd+'/'+filePrefix+str(i)+'.cfg'], stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        output = (output.decode('UTF-8')).strip()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(str(output))
        
    rc = process.poll()

    # change back to start path
    os.chdir(wd)