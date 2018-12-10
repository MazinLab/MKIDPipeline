#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#TODO TDF. No unique functionality.

"""
Created on Tue Jan  9 09:45:42 2018

@author: bmazin, modification by Isabel
"""

import argparse
import ast
import datetime
import os
import sys
from configparser import ConfigParser
from mkidpipeline.hdf import bin2hdf


#load the parameters from a config file

# define the configuration file path

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

config_directory = sys.argv[1]
config = ConfigParser()
config.read(config_directory)
XPIX = ast.literal_eval(config['Data']['XPIX'])  
YPIX = ast.literal_eval(config['Data']['YPIX'])
binPath = ast.literal_eval(config['Data']['binPath'])  # path to raw .bin data
outPath = ast.literal_eval(config['Data']['outPath'])   # path to output data
beamFile = ast.literal_eval(config['Data']['beamFile'])  # path and filename to beam map file
mapFlag = ast.literal_eval(config['Data']['mapFlag'])
filePrefix = ast.literal_eval(config['Data']['filePrefix'])
b2hPath = ast.literal_eval(config['Data']['b2hPath'])
nPos = ast.literal_eval(config['Data']['nPos'])
intTime =  ast.literal_eval(config['Data']['intTime'])
startTimes = ast.literal_eval(config['Data']['startTimes'])
stopTimes =  ast.literal_eval(config['Data']['stopTimes'])

print('Starting conversion of '+config_directory + ' into HDF5 files, one file per dither position. ')
print('Removing ' + str(args.bufferTime[0]) + ' seconds from the beginning and end of each h5 file.' )

timestamp = int(startTimes[0]) 
value = datetime.datetime.fromtimestamp(timestamp)
print('Data collection started on ' + value.strftime('%Y-%m-%d %H:%M:%S'))

wd = os.getcwd()
for i in range(nPos):
    print( 'Position ' + str(i) + '/'  + str(nPos) + ': ' +
           str(int(startTimes[i])+bufferTime) + ' to ' +
           str(int(stopTimes[i])-bufferTime) )
    b2h_config=bin2hdf.Bin2HdfConfig(datadir=binPath, beamfile=beamFile, outdir=outPath,
                                     starttime=(int(startTimes[i])+bufferTime),
                                     inttime=((int(stopTimes[i])-bufferTime) -
                                              (int(startTimes[i])+bufferTime)),
                                     x=XPIX, y=YPIX, writeto=None)
    bin2hdf.makehdf(b2h_config, maxprocs=1)