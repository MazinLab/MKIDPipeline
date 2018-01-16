#!/bin/python
"""
Author: Isabel Lipartito        Date:Dec 4, 2017
Opens a twilight flat h5 and makes the spectrum of each pixel.
Then takes the median of each energy over all pixels
A factor is then calculated for each energy in each pixel of its
twilight count rate / median count rate
The factors are written out in an h5 file
"""

import sys,os
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
from configparser import ConfigParser
import tables
from P3Utils.arrayPopup import PopUp,plotArray,pop
from RawDataProcessing.darkObsFile import ObsFile
from P3Utils.readDict import readDict
from P3Utils.FileName import FileName
import HotPix.darkHotPixMask as hp
from Headers.CalHeaders import FlatCalSoln_Description
from Headers import pipelineFlags

class FlatCal:
	'''
	Opens flat file using parameters from the param file, sets wavelength binnning parameters, and calculates flat weights for flat file.  Writes these weights to a h5 file and plots weights both by pixel and in wavelength-sliced images.
	'''
	def __init__(self,config_file='default.cfg'):
		''' 
		Reads in the param file and opens appropriate flat file.  Applies wavelength calibration if it has not already been applied.  Sets wavelength binning parameters.
		'''
		# define the configuration file path
		directory = os.path.dirname(__file__)
		print('directory', directory)
		self.config_directory = os.path.join(directory, 'Params', config_file)

        	# check the configuration file path and read it in
		self.__configCheck(0)
		self.config = ConfigParser()
		self.config.read(self.config_directory)

		# check the configuration file format and load the parameters
		self.__configCheck(1)
		self.run = ast.literal_eval(self.config['Data']['run'])
		self.date = ast.literal_eval(self.config['Data']['date'])
		self.flatCalTstamp = ast.literal_eval(self.config['Data']['flatCalTstamp'])
		self.flatObsTstamps = ast.literal_eval(self.config['Data']['flatObsTstamps'])
		self.wvlDate = ast.literal_eval(self.config['Data']['wvlDate'])
		self.intTime = ast.literal_eval(self.config['Data']['intTime'])
		self.deadtime= ast.literal_eval(self.config['Instrument']['deadtime'])
		self.energyBinWidth = ast.literal_eval(self.config['Instrument']['energyBinWidth'])
		self.wvlStart = ast.literal_eval(self.config['Instrument']['wvlStart'])
		self.wvlStop = ast.literal_eval(self.config['Instrument']['wvlStop'])
		self.countRateCutoff = ast.literal_eval(self.config['Calibration']['countRateCutoff'])
		self.fractionOfChunksToTrim = ast.literal_eval(self.config['Calibration']['fractionOfChunksToTrim'])
		self.timeMaskFileName = ast.literal_eval(self.config['Calibration']['timeMaskFileName'])
		self.timeSpacingCut = ast.literal_eval(self.config['Calibration']['timeSpacingCut'])

		# check the parameter formats
		self.__configCheck(2)

		obsFNs = [FileName(run=self.run,date=self.date,tstamp=obsTstamp) for obsTstamp in self.flatObsTstamps]
		self.obsFileNames = [fn.obs() for fn in obsFNs]
		print(self.obsFileNames)
		self.obsList = [ObsFile(obsFileName) for obsFileName in self.obsFileNames]
		self.flatCalFileName = FileName(run=self.run,date=self.date,tstamp=self.flatCalTstamp).flatSoln()
		if self.wvlDate != '':
			wvlCalFileName = FileName(run=self.run,date=wvlDate,tstamp=wvlTimestamp).calSoln()
		for iObs,obs in enumerate(self.obsList):
			#if self.wvlDate != '':
			#	obs.loadWvlCalFile(wvlCalFileName)
			#else:
                		#obs.loadBestWvlCalFile()
			obs.setWvlCutoffs(3000,13000)
			if self.timeMaskFileName != '':
				if not os.path.exists(self.timeMaskFileName):
					print('Running hotpix for ',obs)
					hp.findHotPixels(obsFile=obs,outputFileName=timeMaskFileName,fwhm=np.inf,useLocalStdDev=True)
					print ('Flux file pixel mask saved to %s"%(timeMaskFileName)')
				obs.loadHotPixCalFile(timeMaskFileName)

		#get beammap from first obs
		self.beamImage = self.obsList[0].beamImage
		self.wvlFlags = self.obsList[0].beamFlagImage 
		self.nXPix = self.obsList[0].nXPix
		self.nYPix = self.obsList[0].nYPix
		self.wvlBinEdges = ObsFile.makeWvlBins(self.energyBinWidth,self.wvlStart,self.wvlStop)
		
		#wvlBinEdges includes both lower and upper limits, so number of bins is 1 less than number of edges
		self.nWvlBins = len(self.wvlBinEdges)-1
		print('files opened')

	def __del__(self):
		pass

	def loadFlatSpectra(self):
		'''
		Reads the flat data into a spectral cube whose dimensions are determined by the number of x and y pixels and the number of wavelength bins.
		'''
		self.spectralCubes = []#each element will be the spectral cube for a time chunk
		self.cubeEffIntTimes = []
		self.frames = []

		for iObs,obs in enumerate(self.obsList):
			for firstSec in range(0,obs.getFromHeader('expTime'),self.intTime):			
				self.obsList[0].info['isWvlCalibrated']=True
				cubeDict = obs.getSpectralCube(firstSec=firstSec,integrationTime=self.intTime,applySpecWeight=False, applyTPFWeight=False,wvlBinEdges = self.wvlBinEdges,energyBinWidth=None,timeSpacingCut = self.timeSpacingCut)
				cube = np.array(cubeDict['cube'],dtype=np.double)
				effIntTime = cubeDict['effIntTime']
				#add third dimension for broadcasting
				effIntTime3d = np.reshape(effIntTime,np.shape(effIntTime)+(1,))
				cube /= effIntTime3d
				cube[np.isnan(cube)]=0 

				#find factors to correct nonlinearity
				rawFrameDict = obs.getPixelCountImage(firstSec=firstSec,integrationTime=self.intTime,scaleByEffInt=True)  
				rawFrame = np.array(rawFrameDict['image'],dtype=np.double)
				rawFrame /= rawFrameDict['effIntTimes']
				nonlinearFactors = 1. / (1. - rawFrame*self.deadtime)  
				nonlinearFactors[np.isnan(nonlinearFactors)]=0.

                
				frame = np.sum(cube,axis=2) #in counts per sec
				#correct nonlinearity due to deadtime in firmware
				frame = frame * nonlinearFactors
                
				nonlinearFactors = np.reshape(nonlinearFactors,np.shape(nonlinearFactors)+(1,))
				cube = cube * nonlinearFactors
                
				self.frames.append(frame)
				self.spectralCubes.append(cube)
				self.cubeEffIntTimes.append(effIntTime3d)
			#obs.file.close()
		self.spectralCubes = np.array(self.spectralCubes)
		self.cubeEffIntTimes = np.array(self.cubeEffIntTimes)
		self.countCubes = self.cubeEffIntTimes * self.spectralCubes

	def checkCountRates(self):
		medianCountRates = np.array([np.median(frame[frame!=0]) for frame in self.frames])
		print(np.shape(medianCountRates))
		boolIncludeFrames = medianCountRates <= self.countRateCutoff
		#mask out frames, or cubes from integration time chunks with count rates too high
		self.spectralCubes = np.array([cube for cube,boolIncludeFrame in zip(self.spectralCubes,boolIncludeFrames) if boolIncludeFrame==True])
		self.frames = [frame for frame,boolIncludeFrame in zip(self.frames,boolIncludeFrames) if boolIncludeFrame==True]
		print('few enough counts in the chunk',zip(medianCountRates,boolIncludeFrames))

	def calculateWeights(self):
		"""
		finds flat cal factors as medians/pixelSpectra for each pixel.  Normalizes these weights at each wavelength bin.
		"""
		cubeWeightsList = []
		self.averageSpectra = []
		deltaWeightsList = []
		for iCube,cube in enumerate(self.spectralCubes):
			effIntTime = self.cubeEffIntTimes[iCube]
			#for each time chunk
			wvlAverages = np.zeros(self.nWvlBins)
			spectra2d = np.reshape(cube,[self.nXPix*self.nYPix,self.nWvlBins ])
			for iWvl in range(self.nWvlBins):
				wvlSlice = spectra2d[:,iWvl]
				goodPixelWvlSlice = np.array(wvlSlice[wvlSlice != 0])#dead pixels need to be taken out before calculating averages
				nGoodPixels = len(goodPixelWvlSlice)
				wvlAverages[iWvl] = np.median(goodPixelWvlSlice)
			weights = np.divide(wvlAverages,cube)
			weights[weights==0] = np.nan
			weights[weights==np.inf] = np.nan
			cubeWeightsList.append(weights)
			deltaWeights = weights/np.sqrt(effIntTime*cube)
			deltaWeightsList.append(deltaWeights)
			self.averageSpectra.append(wvlAverages)
		cubeWeights = np.array(cubeWeightsList)

		deltaCubeWeights = np.array(deltaWeightsList)
		cubeWeightsMask = np.isnan(cubeWeights)
		self.maskedCubeWeights = np.ma.array(cubeWeights,mask=cubeWeightsMask,fill_value=1.)
		print('maskedcubeWeightsshape', np.shape(self.maskedCubeWeights))
		self.maskedCubeDeltaWeights = np.ma.array(deltaCubeWeights,mask=cubeWeightsMask)
		#sort maskedCubeWeights and rearange spectral cubes the same way
		sortedIndices = np.ma.argsort(self.maskedCubeWeights,axis=0)
		identityIndices = np.ma.indices(np.shape(self.maskedCubeWeights))

		sortedWeights = self.maskedCubeWeights[sortedIndices,identityIndices[1],identityIndices[2],identityIndices[3]]
		countCubesReordered = self.countCubes[sortedIndices,identityIndices[1],identityIndices[2],identityIndices[3]]
		cubeDeltaWeightsReordered = self.maskedCubeDeltaWeights[sortedIndices,identityIndices[1],identityIndices[2],identityIndices[3]]

		#trim the beginning and end off the sorted weights for each wvl for each pixel, to exclude extremes from averages
		nCubes = np.shape(self.maskedCubeWeights)[0]
		trimmedWeights = sortedWeights[self.fractionOfChunksToTrim*nCubes:(1-self.fractionOfChunksToTrim)*nCubes,:,:,:]
		trimmedCountCubesReordered = countCubesReordered[self.fractionOfChunksToTrim*nCubes:(1-self.fractionOfChunksToTrim)*nCubes,:,:,:]
		print('trimmed cubes shape',np.shape(trimmedCountCubesReordered))

		self.totalCube = np.ma.sum(trimmedCountCubesReordered,axis=0)
		self.totalFrame = np.ma.sum(self.totalCube,axis=-1)
		plotArray(self.totalFrame)
    

		trimmedCubeDeltaWeightsReordered = cubeDeltaWeightsReordered[self.fractionOfChunksToTrim*nCubes:(1-self.fractionOfChunksToTrim)*nCubes,:,:,:]

		self.flatWeights,summedAveragingWeights = np.ma.average(trimmedWeights,axis=0,weights=trimmedCubeDeltaWeightsReordered**-2.,returned=True)
		self.deltaFlatWeights = np.sqrt(summedAveragingWeights**-1.)#Uncertainty in weighted average is sqrt(1/sum(averagingWeights))
		self.flatFlags = self.flatWeights.mask

		#normalize weights at each wavelength bin
		wvlWeightMedians = np.ma.median(np.reshape(self.flatWeights,(-1,self.nWvlBins)),axis=0)
		self.flatWeights = np.divide(self.flatWeights,wvlWeightMedians)
            
	def plotWeightsWvlSlices(self,verbose=True):
		'''
		Plot weights in images of a single wavelength bin (wavelength-sliced images)
		'''
		flatCalPath,flatCalBasename = os.path.split(self.flatCalFileName)
		pdfBasename = os.path.splitext(flatCalBasename)[0]+'_wvlSlices.pdf'
		pdfFullPath = os.path.join(flatCalPath,pdfBasename)
		pp = PdfPages(pdfFullPath)
		nPlotsPerRow = 2
		nPlotsPerCol = 4 
		nPlotsPerPage = nPlotsPerRow*nPlotsPerCol
		iPlot = 0 
		if verbose:
			print('plotting weights in wavelength sliced images')

		matplotlib.rcParams['font.size'] = 4 
		wvls = self.wvlBinEdges[0:-1]

		for iWvl,wvl in enumerate(wvls):
			if verbose:
				print('wvl ',iWvl)
			if iPlot % nPlotsPerPage == 0:
				fig = plt.figure(figsize=(10,10),dpi=100)

			ax = fig.add_subplot(nPlotsPerCol,nPlotsPerRow,iPlot%nPlotsPerPage+1)
			ax.set_title(r'Weights %.0f $\AA$'%wvl)

			image = self.flatWeights[:,:,iWvl]

			cmap = matplotlib.cm.hot
			cmap.set_bad('#222222')
			handleMatshow = ax.matshow(image,cmap=cmap,origin='lower',vmax=2.,vmin=.5)
			cbar = fig.colorbar(handleMatshow)
        
			if iPlot%nPlotsPerPage == nPlotsPerPage-1:
				pp.savefig(fig)
			iPlot += 1

			ax = fig.add_subplot(nPlotsPerCol,nPlotsPerRow,iPlot%nPlotsPerPage+1)
			ax.set_title(r'Twilight Image %.0f $\AA$'%wvl)

			image = self.totalCube[:,:,iWvl]

			nSdev = 3.
			goodImage = image[np.isfinite(image)]
			vmax = np.mean(goodImage)+nSdev*np.std(goodImage)
			handleMatshow = ax.matshow(image,cmap=cmap,origin='lower',vmax=vmax)
			cbar = fig.colorbar(handleMatshow)

			if iPlot%nPlotsPerPage == nPlotsPerPage-1:
				pp.savefig(fig)
			iPlot += 1

		pp.savefig(fig)
		pp.close()

	def plotMaskWvlSlices(self,verbose=True):
		'''
		Plot mask in images of a single wavelength bin (wavelength-sliced images)
		'''
		flatCalPath,flatCalBasename = os.path.split(self.flatCalFileName)
		pdfBasename = os.path.splitext(flatCalBasename)[0]+'_mask.pdf'
		pdfFullPath = os.path.join(flatCalPath,pdfBasename)
		pp = PdfPages(pdfFullPath)
		nPlotsPerRow = 3 
		nPlotsPerCol = 4 
		nPlotsPerPage = nPlotsPerRow*nPlotsPerCol
		iPlot = 0 
		if verbose:
			print('plotting mask in wavelength sliced images')

		matplotlib.rcParams['font.size'] = 4 
		wvls = self.wvlBinEdges[0:-1]

		for iWvl,wvl in enumerate(wvls):
			if verbose:
				print('wvl ',iWvl)
			if iPlot % nPlotsPerPage == 0:
				fig = plt.figure(figsize=(10,10),dpi=100)

			ax = fig.add_subplot(nPlotsPerCol,nPlotsPerRow,iPlot%nPlotsPerPage+1)
			ax.set_title(r'%.0f $\AA$'%wvl)

			image = self.flatFlags[:,:,iWvl]
			image=image*1
			self.wvlFlags=np.array(self.wvlFlags)
			image += 2*self.wvlFlags  
			image = 3-image

			cmap = matplotlib.cm.gnuplot2
			handleMatshow = ax.matshow(image,cmap=cmap,origin='lower',vmax=2.,vmin=.5)
			cbar = fig.colorbar(handleMatshow)
        
			if iPlot%nPlotsPerPage == nPlotsPerPage-1:
				pp.savefig(fig)
			iPlot += 1
		pp.savefig(fig)
		pp.close()

	def plotWeightsByPixel(self,verbose=True):
		'''
		Plot weights of each wavelength bin for every single pixel
		'''
		flatCalPath,flatCalBasename = os.path.split(self.flatCalFileName)
		pdfBasename = os.path.splitext(flatCalBasename)[0]+'.pdf'
		pdfFullPath = os.path.join(flatCalPath,pdfBasename)
		pp = PdfPages(pdfFullPath)
		nPlotsPerRow = 2
		nPlotsPerCol = 4
		nPlotsPerPage = nPlotsPerRow*nPlotsPerCol
		iPlot = 0 
		if verbose:
			print('plotting weights by pixel at ',pdfFullPath)

		matplotlib.rcParams['font.size'] = 4 
		wvls = self.wvlBinEdges[0:-1]
		nCubes = len(self.maskedCubeWeights)

		for iRow in range(self.nXPix):
			if verbose:
				print('row',iRow)
			for iCol in range(self.nYPix):
				weights = self.flatWeights[iRow,iCol,:]
				deltaWeights = self.deltaFlatWeights[iRow,iCol,:]
				if weights.mask.all() == False:
					if iPlot % nPlotsPerPage == 0:
						fig = plt.figure(figsize=(10,10),dpi=100)

					ax = fig.add_subplot(nPlotsPerCol,nPlotsPerRow,iPlot%nPlotsPerPage+1)
					ax.set_ylim(.5,2.)

					for iCube in range(nCubes):
						cubeWeights = self.maskedCubeWeights[iCube,iRow,iCol]
						ax.plot(wvls,cubeWeights.data,label='weights %d'%iCube,alpha=.7,color=matplotlib.cm.Paired((iCube+1.)/nCubes))
					ax.errorbar(wvls,weights.data,yerr=deltaWeights.data,label='weights',color='k')
                
					ax.set_title('p %d,%d'%(iRow,iCol))
					ax.set_ylabel('weight')
					ax.set_xlabel(r'$\lambda$ ($\AA$)')
					#ax.plot(wvls,flatSpectrum,label='pixel',alpha=.5)

					#ax.legend(loc='lower left')
					#ax2.legend(loc='lower right')
					if iPlot%nPlotsPerPage == nPlotsPerPage-1 or (iRow == self.nXPix-1 and iCol == self.nYPix-1):
						pp.savefig(fig)
					iPlot += 1

					#Put a plot of twilight spectrums for this pixel
					if iPlot % nPlotsPerPage == 0:
						fig = plt.figure(figsize=(10,10),dpi=100)

					ax = fig.add_subplot(nPlotsPerCol,nPlotsPerRow,iPlot%nPlotsPerPage+1)
					for iCube in range(nCubes):
						spectrum = self.spectralCubes[iCube,iRow,iCol]
						ax.plot(wvls,spectrum,label='spectrum %d'%iCube,alpha=.7,color=matplotlib.cm.Paired((iCube+1.)/nCubes))
                
					ax.set_title('p %d,%d'%(iRow,iCol))
					ax.set_xlabel(r'$\lambda$ ($\AA$)')
					ax.set_ylabel('twilight cps')
					#ax.plot(wvls,flatSpectrum,label='pixel',alpha=.5)

					#ax.legend(loc='lower left')
					#ax2.legend(loc='lower right')
					if iPlot%nPlotsPerPage == nPlotsPerPage-1 or (iRow == self.nXPix-1 and iCol == self.nYPix-1):
						pp.savefig(fig)
						#plt.show()
					iPlot += 1
		pp.close()

	def writeWeights(self):
		"""
		Writes an h5 file to put calculated flat cal factors in
		"""
		if os.path.isabs(self.flatCalFileName) == True:
			fullFlatCalFileName = self.flatCalFileName
			print(self.flatCalFileName)
		else:
			scratchDir = os.getenv('MKID_PROC_PATH')
			flatDir = os.path.join(scratchDir,'flatCalSolnFiles')
			fullFlatCalFileName = os.path.join(flatDir,self.flatCalFileName)

		try:
			flatCalFile = tables.open_file(fullFlatCalFileName,mode='w')
		except:
			print('Error: Couldn\'t create flat cal file, ',fullFlatCalFileName)
			return
		print('wrote to',self.flatCalFileName)

		calgroup = flatCalFile.create_group(flatCalFile.root,'flatcal','Table of flat calibration weights by pixel and wavelength')
		calarray = tables.Array(calgroup,'weights',obj=self.flatWeights.data,title='Flat calibration Weights indexed by pixelRow,pixelCol,wavelengthBin')
		flagtable = tables.Array(calgroup,'flags',obj=self.flatFlags,title='Flat cal flags indexed by pixelRow,pixelCol,wavelengthBin. 0 is Good')
		bintable = tables.Array(calgroup,'wavelengthBins',obj=self.wvlBinEdges,title='Wavelength bin edges corresponding to third dimension of weights array')

		descriptionDict = FlatCalSoln_Description(self.nWvlBins)
		caltable = flatCalFile.create_table(calgroup, 'calsoln', descriptionDict,title='Flat Cal Table')
        
		for iRow in range(self.nXPix):
			for iCol in range(self.nYPix):
				weights = self.flatWeights[iRow,iCol,:]
				deltaWeights = self.deltaFlatWeights[iRow,iCol,:]
				flags = self.flatFlags[iRow,iCol,:]
				flag = np.any(self.flatFlags[iRow,iCol,:])
				pixelName = self.beamImage[iRow,iCol]

				entry = caltable.row
				entry['resid'] = pixelName
				entry['pixelrow'] = iRow
				entry['pixelcol'] = iCol
				entry['weights'] = weights
				entry['weightUncertainties'] = deltaWeights
				entry['weightFlags'] = flags
				entry['flag'] = flag
				entry.append()
        
		flatCalFile.flush()
		flatCalFile.close()

		npzFileName = os.path.splitext(fullFlatCalFileName)[0]+'.npz'

	def __configCheck(self, index):
		'''
		Checks the variables loaded in from the configuration file for type and
		consistencey. Run in the '__init__()' method.
		'''
		if index == 0:
			# check for configuration file
			assert os.path.isfile(self.config_directory), \
				self.config_directory + " is not a valid configuration file"
		elif index == 1:
			# check if all sections and parameters exist in the configuration file
			section = "{0} must be a configuration section"
			param = "{0} must be a parameter in the configuration file '{1}' section"

			assert 'Data' in self.config.sections(), section.format('Data')
			assert 'run' in self.config['Data'].keys(), \
				param.format('run', 'Data')
			assert 'date' in self.config['Data'].keys(), \
				param.format('date', 'Data')
			assert 'flatCalTstamp' in self.config['Data'].keys(), \
				param.format('flatCalTstamp', 'Data')
			assert 'flatObsTstamps' in self.config['Data'].keys(), \
				param.format('flatObsTstamps', 'Data')
			assert 'wvlDate' in self.config['Data'].keys(), \
				param.format('wvlDate', 'Data')
			assert 'intTime' in self.config['Data'].keys(), \
				param.format('intTime', 'Data')

			assert 'Instrument' in self.config.sections(), section.format('Instrument')
			assert 'deadtime' in self.config['Instrument'].keys(), \
				param.format('deadtime', 'Instrument')
			assert 'energyBinWidth' in self.config['Instrument'].keys(), \
				param.format('energyBinWidth', 'Instrument')
			assert 'wvlStart' in self.config['Instrument'].keys(), \
				param.format('wvlStart', 'Instrument')
			assert 'wvlStop' in self.config['Instrument'].keys(), \
				param.format('wvlStop', 'Instrument')

			assert 'Calibration' in self.config.sections(), section.format('Calibration')
			assert 'countRateCutoff' in self.config['Calibration'], \
				param.format('countRateCutoff', 'Calibration')
			assert 'fractionOfChunksToTrim' in self.config['Calibration'], \
				param.format('fractionOfChunksToTrim', 'Calibration')
			assert 'timeMaskFileName' in self.config['Calibration'], \
				param.format('timeMaskFileName', 'Calibration')
			assert 'timeSpacingCut' in self.config['Calibration'], \
				param.format('timeSpacingCut', 'Calibration')
		elif index == 2:
			# type check parameters
			assert type(self.run) is str, "Run parameter must be a string."
			assert type(self.date) is str, "Date parameter must be a string."
			assert type(self.flatCalTstamp) is str, "Flat Calibration Timestamp parameter must be a string."
			assert type(self.flatObsTstamps) is list, "Flat Observation Timestamps parameter must be a list"
			assert type(self.wvlDate) is str, "Wavelength Sunset Date parameter must be a string."

			assert type(self.intTime) is int, "integration time parameter must be an integer"
			
			if type(self.deadtime) is int:
				self.deadtime = float(self.deadtime)
			assert type(self.deadtime) is float, "Dead time parameter must be an integer or float"
			
			if type(self.energyBinWidth) is int:
				self.energyBinWidth = float(self.energyBinWidth)
			assert type(self.energyBinWidth) is float, "Energy Bin Width parameter must be an integer or float"

			if type(self.wvlStart) is int:
				self.wvlStart = float(self.wvlStart)
			assert type(self.wvlStart) is float, "Starting Wavelength must be an integer or float"

			if type(self.wvlStop) is int:
				self.wvlStop = float(self.wvlStop)
			assert type(self.wvlStop) is float, "Stopping Wavelength must be an integer or float"

			if type(self.countRateCutoff) is int:
				self.countRateCutoff = float(self.countRateCutoff)
			assert type(self.countRateCutoff) is float, "Count Rate Cutoff must be an integer or float"

			assert type(self.fractionOfChunksToTrim) is int, "Fraction of Chunks to Trim must be an integer"

			for index, flatobsfile in enumerate(self.flatObsTstamps):
				assert type(self.flatObsTstamps[index]) is str, "elements in Flat Observation Timestamps list must be strings"
		else:
			raise ValueError("index must be 0, 1, or 2")

if __name__ == '__main__':
	if len(sys.argv) == 1:
		flatcal = FlatCal()
	else:
		flatcal = FlatCal(config_file=sys.argv[1])
	flatcal.loadFlatSpectra()
	flatcal.checkCountRates()
	flatcal.calculateWeights()
	flatcal.writeWeights()
	flatcal.plotWeightsByPixel()
	flatcal.plotWeightsWvlSlices()
	flatcal.plotMaskWvlSlices()
