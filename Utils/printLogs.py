#!/bin/python
'''
File: printLogs.py
Author: Seth Meeker
Date: Aug 4, 2016

Description: prints all log files for a given DARKNESS target
Usage: from command line -> python printLogs.py targetName [logPath]

INPUTS:
        String targetName - Required. Name of object whose logs you want to view.
                       must match a target name recorded in the log files' names
        String logPath - Optional. Defaults to Pal2016 directory unless provided with alternative.
        
OUTPUTS:
        None. Prints logs to command line.
'''

import sys
import os
import glob

if len(sys.argv) < 2:
    print 'Usage: python ',sys.argv[0],' targetName [logPath]'
    exit(1)
targetName = sys.argv[1]

try:
    logPath=sys.argv[2]
except IndexError:
    logPath = '/mnt/data0/ScienceData/'

for log in sorted(glob.glob(os.path.join(logPath,'*_%s.log')%targetName)):

    print '\n-------------------------------------\n'

    try:
        print 'Full path: ', log
        print 'Filename: ',os.path.basename(log)
        print 'Target: ',targetName
        print 'Timestamp: ',os.path.basename(log).split('_')[0]
        print 'Log contents: '
        f=open(log)
        for line in f:
            print line
        f.close()

    except:
        print 'Failed to read file %s'%log
        
        
        
        
        
