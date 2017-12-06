#!/bin/python
"""
Author: Taylor Swift        Date:Dec 4, 2017
She screm.
Opens a twilight flat h5 and makes the spectrum of each pixel.
Then takes the median of each energy over all pixels
A factor is then calculated for each energy in each pixel of its
twilight count rate / median count rate
The factors are written out in an h5 file
"""

import sys,os
import tables
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages


#from Utils.popup import PopUp,plotArray,pop
from RawDataProcessing.darkObsFile import ObsFile
from P3Utils.readDict import readDict
from P3Utils.FileName import FileName
#import HotPix.darkHotPixMask as hp
from Headers.CalHeaders import FlatCalSoln_Description

class FlatCal:
	def __init__(self,paramFile):
        #opens flat file,sets wavelength binnning parameters, and calculates flat factors for the file
		self.params = readDict()
		self.params.read_from_file(paramFile)
		sunsetDate = self.params['sunsetDate']
		flatTstamp = self.params['flatTstamp']
		wvlSunsetDate = self.params['wvlSunsetDate']
		wvlTimestamp = self.params['wvlTimestamp']
		run=self.params['run']
		obsTstamp=self.params['obsTstamp']
		print(sunsetDate)
        #obsFNs = [FileName(run=run,date=sunsetDate,tstamp=obsTstamp)]
	#print(obsFNs)
        #self.obsFileNames = [fn.obs() for fn in obsFNs]
		self.obsFileNames=self.params['ObsFN']
		print(self.obsFileNames)
		self.obsList = [ObsFile(obsFileName) for obsFileName in self.obsFileNames]
		self.flatCalFileName = FileName(run=run,date=sunsetDate,tstamp=flatTstamp).flatSoln()
		wvlCalFileName = FileName(run=run,date=wvlSunsetDate,tstamp=wvlTimestamp).calSoln()
		for iObs,obs in enumerate(self.obsList):
			print('loading beammap',os.environ['MKID_BEAMMAP_PATH'])
			obs.loadBeammapFile(os.environ['MKID_BEAMMAP_PATH'])
			obs.loadWvlCalFile(wvlCalFileName)
		obs.setWvlCutoffs(3000,13000)
		self.beamImage = self.obsList[0].beamImage
		self.wvlFlags = self.obsList[0].wvlFlagTable
		self.nRow = self.obsList[0].nRow
		self.nCol = self.obsList[0].nCol
		print('files opened')
		#self.wvlBinWidth = params['wvlBinWidth'] #angstroms
		self.energyBinWidth = self.params['energyBinWidth'] #eV
		self.wvlStart = self.params['wvlStart'] #angstroms
		self.wvlStop = self.params['wvlStop'] #angstroms
		self.wvlBinEdges = ObsFile.makeWvlBins(self.energyBinWidth,self.wvlStart,self.wvlStop)
		self.intTime = self.params['intTime']
		self.countRateCutoff = self.params['countRateCutoff']
		self.fractionOfChunksToTrim = self.params['fractionOfChunksToTrim']
		#wvlBinEdges includes both lower and upper limits, so number of bins is 1 less than number of edges
		self.nWvlBins = len(self.wvlBinEdges)-1

if __name__ == '__main__':
	paramFile = sys.argv[1]
	print(paramFile)    
	flatcal = FlatCal(paramFile)
