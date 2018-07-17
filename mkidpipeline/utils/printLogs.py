#!/bin/python
"""
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
        
TODO:
        Allow user to specify optional range of timestamps to print logs from
"""

import glob
import os
import sys

if len(sys.argv) < 2:
    #print('Usage: python ',sys.argv[0],' targetName [logPath]')
    #print('To print all log files use targetName = All')
    #exit(1)
    targetName='a'
else:
    targetName = sys.argv[1]

try:
    logPath=sys.argv[2]
except IndexError:
    logPath = '.'

if targetName in ['h', 'help', 'Help', 'HELP']:
    print('Usage: python ',sys.argv[0],' targetName [logPath]')
    print('To print all log files use targetName = ALL')
    exit(1)
if targetName in ['a','all','All','ALL']:
    fileName = '*.log'
else:
    fileName = '*_%s.log'%targetName

for log in sorted(glob.glob(os.path.join(logPath,fileName))):

    print('\n-------------------------------------\n')

    try:
        print('Full path: ', log)
        print('Filename: ',os.path.basename(log))
        print('Target: ',targetName)
        print('Timestamp: ',os.path.basename(log).split('_')[0])
        print('Log contents: ')
        f=open(log)
        for line in f:
            print(line)
        f.close()

    except:
        print('Failed to read file %s'%log)
        
        
        
        
        
