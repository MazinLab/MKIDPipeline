#!/bin/python
"""
Author: Taylor Swift        Date:Dec 4, 2017
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
	def __init__(self,paramFile):
        #opens flat file,sets wavelength binnning parameters, and calculates flat factors for the file
        	# check the configuration file path and read it in
		#self.__configCheck(0)
		self.config = ConfigParser()
		self.config.read(paramFile)

		# check the configuration file format and load the parameters
		#self.__configCheck(1)
		self.sunsetDate = ast.literal_eval(self.config['Data']['sunsetDate'])
		self.flatTstamp = ast.literal_eval(self.config['Data']['flatTstamp'])
		self.obsFileName = ast.literal_eval(self.config['Data']['ObsFN'])
		self.run = ast.literal_eval(self.config['Data']['run'])
		self.obsTstamp = ast.literal_eval(self.config['Data']['obsTstamp'])
		self.wvlStart = ast.literal_eval(self.config['Data']['wvlStart'])
		self.wvlStop = ast.literal_eval(self.config['Data']['wvlStop'])
		self.energyBinWidth = ast.literal_eval(self.config['Data']['energyBinWidth'])
		self.countRateCutoff = ast.literal_eval(self.config['Data']['countRateCutoff'])
		self.fractionOfChunksToTrim = ast.literal_eval(self.config['Data']['fractionOfChunksToTrim'])
		self.bLoadBeammap = ast.literal_eval(self.config['Data']['bLoadBeammap'])
		self.timeMaskFileName = ast.literal_eval(self.config['Data']['timeMaskFileName'])
		self.wvlSunsetDate = ast.literal_eval(self.config['Data']['wvlSunsetDate'])
		self.intTime = ast.literal_eval(self.config['Data']['intTime'])
		self.timeSpacingCut = ast.literal_eval(self.config['Data']['timeSpacingCut'])
		self.deadtime= ast.literal_eval(self.config['Data']['deadtime'])

		# check the parameter formats
		#self.__configCheck(2)

		obsFNs = [FileName(run=self.run,date=self.sunsetDate,tstamp=self.obsTstamp)]
		self.obsFileNames = [fn.obs() for fn in obsFNs]
		print(self.obsFileNames)
		self.obsList = [ObsFile(obsFileName) for obsFileName in self.obsFileNames]
		self.flatCalFileName = FileName(run=self.run,date=self.sunsetDate,tstamp=self.flatTstamp).flatSoln()
		if self.wvlSunsetDate != '':
			wvlCalFileName = FileName(run=run,date=wvlSunsetDate,tstamp=wvlTimestamp).calSoln()
		for iObs,obs in enumerate(self.obsList):
			if self.bLoadBeammap == '1':
				print('loading beammap',os.environ['MKID_BEAMMAP_PATH'])
				obs.loadBeammapFile(os.environ['MKID_BEAMMAP_PATH'])
			if self.wvlSunsetDate != '':
				obs.loadWvlCalFile(wvlCalFileName)
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
		self.wvlFlags = self.obsList[0].beamFlagImage #Not sure if we have this or need it
		print(self.wvlFlags)
		self.nXPix = self.obsList[0].nXPix
		self.nYPix = self.obsList[0].nYPix
		self.wvlBinEdges = ObsFile.makeWvlBins(self.energyBinWidth,self.wvlStart,self.wvlStop)
		#wvlBinEdges includes both lower and upper limits, so number of bins is 1 less than number of edges
		self.nWvlBins = len(self.wvlBinEdges)-1
		print(self.nWvlBins)
		print('files opened')

	def __del__(self):
		pass

	def loadFlatSpectra(self):
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
			obs.file.close()
		print('frame',np.shape(self.frames))
		a = np.ma.sum(self.frames,axis=0)
		#plotArray(a)
		self.spectralCubes = np.array(self.spectralCubes)
		self.cubeEffIntTimes = np.array(self.cubeEffIntTimes)
		self.countCubes = self.cubeEffIntTimes * self.spectralCubes
		a = np.ma.sum(self.countCubes,axis=0)
		b = np.ma.sum(a,axis=-1)
		#plotArray(b)

	def checkCountRates(self):
		medianCountRates = np.array([np.median(frame[frame!=0]) for frame in self.frames])
		print(np.shape(medianCountRates))
		boolIncludeFrames = medianCountRates <= self.countRateCutoff
		#mask out frames, or cubes from integration time chunks with count rates too high
		self.spectralCubes = np.array([cube for cube,boolIncludeFrame in zip(self.spectralCubes,boolIncludeFrames) if boolIncludeFrame==True])
		a = np.ma.sum(self.spectralCubes,axis=0)
		b = np.ma.sum(a,axis=-1)
		#plotArray(b)
		self.frames = [frame for frame,boolIncludeFrame in zip(self.frames,boolIncludeFrames) if boolIncludeFrame==True]
		print('few enough counts in the chunk',zip(medianCountRates,boolIncludeFrames))

	def calculateWeights(self):
		"""
		finds flat cal factors as medians/pixelSpectra for each pixel
		"""
		cubeWeightsList = []
		self.averageSpectra = []
		deltaWeightsList = []
		for iCube,cube in enumerate(self.spectralCubes):
			#print('icube', iCube, 'cube', cube)
			effIntTime = self.cubeEffIntTimes[iCube]
			#for each time chunk
			wvlAverages = np.zeros(self.nWvlBins)
			#print('wvlAverages', wvlAverages)
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
		a = np.ma.sum(self.maskedCubeWeights,axis=0)
		b = np.ma.sum(a,axis=-1)
		#plotArray(b)
		self.maskedCubeDeltaWeights = np.ma.array(deltaCubeWeights,mask=cubeWeightsMask)
		#sort maskedCubeWeights and rearange spectral cubes the same way
		sortedIndices = np.ma.argsort(self.maskedCubeWeights,axis=0)
		print('sortedIndicesShape', np.shape(sortedIndices))
		a = np.ma.sum(sortedIndices,axis=0)
		print(np.shape(a))
		b = np.ma.sum(a,axis=-1)
		print(np.shape(b))
		#plotArray(b)
		identityIndices = np.ma.indices(np.shape(self.maskedCubeWeights))

		sortedWeights = self.maskedCubeWeights[sortedIndices,identityIndices[1],identityIndices[2],identityIndices[3]]
		a = np.ma.sum(sortedWeights,axis=0)
		b = np.ma.sum(a,axis=-1)
		plotArray(b)
		countCubesReordered = self.countCubes[sortedIndices,identityIndices[1],identityIndices[2],identityIndices[3]]
		a = np.ma.sum(countCubesReordered,axis=0)
		b = np.ma.sum(a,axis=-1)
		plotArray(b)
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
		b = np.ma.sum(self.flatWeights,axis=0)
		#plotArray(b)
		self.flatWeights = np.divide(self.flatWeights,wvlWeightMedians)
            
	def plotWeightsWvlSlices(self,verbose=True):
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
			#image += 2*self.wvlFlags[:,:]
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
				pixelName=str(pixelName)
				#roach = int(pixelName.split('r')[1].split('/')[0])
				#pixelNum = int(pixelName.split('p')[1].split('/')[0])

				entry = caltable.row
				#entry['roach'] = roach
				#entry['pixelnum'] = pixelNum
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
"""
		#calculate total spectra and medians for programs that expect old format flat cal
		spectra = np.array(np.sum(self.spectralCubes,axis=0))

		wvlAverages = np.zeros(self.nWvlBins)
		spectra2d = np.reshape(spectra,[self.nXPix*self.nYPix,self.nWvlBins ])
		for iWvl in range(self.nWvlBins):
			spectrum = spectra2d[:,iWvl]
			goodSpectrum = spectrum[spectrum != 0]#dead pixels need to be taken out before calculating medians
			wvlAverages[iWvl] = np.median(goodSpectrum)
			np.savez(npzFileName,median=wvlAverages,averageSpectra=np.array(self.averageSpectra),binEdges=self.wvlBinEdges,spectra=spectra,weights=np.array(self.flatWeights.data),deltaWeights=np.array(self.deltaFlatWeights.data),mask=self.flatFlags,totalFrame=self.totalFrame,totalCube=self.totalCube,spectralCubes=self.spectralCubes,countCubes=self.countCubes,cubeEffIntTimes=self.cubeEffIntTimes )
"""

if __name__ == '__main__':
	paramFile = sys.argv[1]    
	flatcal = FlatCal(paramFile)
	flatcal.loadFlatSpectra()
	flatcal.checkCountRates()
	flatcal.calculateWeights()
	flatcal.writeWeights()
	flatcal.plotWeightsByPixel()
	flatcal.plotWeightsWvlSlices()
	flatcal.plotMaskWvlSlices()
