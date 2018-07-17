#!/usr/bin/env python
# encoding: utf-8
# filename: profileBinFile.py

import cProfile
import os
import pstats

import numpy as np

from .readDict import readDict

configFileName = 'profileTest.cfg'
configData = readDict()
configData.read_from_file(configFileName)

# Extract parameters from config file
timeSpan = np.array(configData['timeSpan'], dtype=int)
date = str(configData['date'])
binDir = str(configData['binDir'])
binPath = os.path.join(binDir,date)
dataDir = binPath

cProfile.runctx("binFileC.parseBinFiles(dataDir,np.array([timeSpan[0]]))", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
