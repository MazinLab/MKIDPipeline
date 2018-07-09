#!/bin/python
"""
Author: Isabel Lipartito        Date:Dec 4, 2017
Opens a twilight flat h5 and breaks it into INTTIME (5 second suggested) blocks.  
For each block, this program makes the spectrum of each pixel.
Then takes the median of each energy over all pixels
A factor is then calculated for each energy in each pixel of its
twilight count rate / median count rate
The factors are written out in an h5 file for each block (You'll get EXPTIME/INTTIME number of files)
Plotting options:  
Entire array: both wavelength slices and masked wavelength slices
Per pixel:  plots of weights vs wavelength next to twilight spectrum OR
            plots of weights vs wavelength, twilight spectrum, next to wavecal solution (has _WavelengthCompare_ in the name)
"""

import sys,os
import ast
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
import matplotlib
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
from configparser import ConfigParser
import tables
from DarknessPipeline.Utils.arrayPopup import PopUp,plotArray,pop
from DarknessPipeline.RawDataProcessing.darkObsFile import ObsFile
from DarknessPipeline.Utils.readDict import readDict
from DarknessPipeline.Utils.FileName import FileName
import DarknessPipeline.Cleaning.HotPix.darkHotPixMask as hp
from DarknessPipeline.Headers.CalHeaders import FlatCalSoln_Description
from DarknessPipeline.Headers import pipelineFlags
from DarknessPipeline.Calibration.WavelengthCal import plotWaveCal as p
from progressbar import ProgressBar, Bar, ETA, Timer, Percentage
from PyPDF2 import PdfFileMerger, PdfFileReader


class FlatCal:
	'''
	Opens flat file using parameters from the param file, sets wavelength binnning parameters, and calculates flat weights for flat file.  Writes these weights to a h5 file and plots weights both by pixel and in wavelength-sliced images.
	'''
	def __init__(self,config_file='default.cfg'):
		''' 
		Reads in the param file and opens appropriate flat file.  Sets wavelength binning parameters.
		'''
		# define the configuration file path
		self.config_file = config_file

        	# check the configuration file path and read it in
		self.__configCheck(0)
		self.config = ConfigParser()
		self.config.read(self.config_file)

		# check the configuration file format and load the parameters
		self.__configCheck(1)
		self.wvlCalFile=ast.literal_eval(self.config['Data']['wvlCalFile'])
		self.flatPath = ast.literal_eval(self.config['Data']['flatPath'])
		self.intTime = ast.literal_eval(self.config['Data']['intTime'])
		self.expTime = ast.literal_eval(self.config['Data']['expTime'])
		self.deadtime= ast.literal_eval(self.config['Instrument']['deadtime'])
		self.energyBinWidth = ast.literal_eval(self.config['Instrument']['energyBinWidth'])
		self.wvlStart = ast.literal_eval(self.config['Instrument']['wvlStart'])
		self.wvlStop = ast.literal_eval(self.config['Instrument']['wvlStop'])
		self.countRateCutoff = ast.literal_eval(self.config['Calibration']['countRateCutoff'])
		self.fractionOfChunksToTrim = ast.literal_eval(self.config['Calibration']['fractionOfChunksToTrim'])
		self.verbose = ast.literal_eval(self.config['Output']['verbose'])
		self.calSolnPath=ast.literal_eval(self.config['Output']['calSolnPath'])
		self.save_plots=ast.literal_eval(self.config['Output']['save_plots'])
		if self.save_plots:
			answer = self.__query("Save Plots flag set to 'yes', this will add ~30 min to the code.  Are you sure you want to save plots?", yes_or_no=True)
			if answer is False:
				self.save_plots=False
				print('Setting save_plots parameter to FALSE')
		self.timeSpacingCut=None

		# check the parameter formats
		self.__configCheck(2)
		
		self.obsList=[ObsFile(self.flatPath)]
		self.flatCalFileName =  self.calSolnPath+'flatcalsoln.h5'
		self.out_directory = self.calSolnPath 

		#get beammap from first obs
		self.beamImage = self.obsList[0].beamImage
		self.wvlFlags = self.obsList[0].beamFlagImage 
		self.nXPix = self.obsList[0].nXPix
		self.nYPix = self.obsList[0].nYPix
		self.wvlBinEdges = ObsFile.makeWvlBins(self.energyBinWidth,self.wvlStart,self.wvlStop)
		
		#wvlBinEdges includes both lower and upper limits, so number of bins is 1 less than number of edges
		self.nWvlBins = len(self.wvlBinEdges)-1
		if self.verbose:
			print('Computing Factors for FlatCal')
			self.pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (',Timer(), ') ', ETA(), ' '], max_value=4*len(range(0,self.expTime,self.intTime))).start()
			self.pbar_iter = 0

	def __del__(self):
		pass

	def loadFlatSpectra(self):
		"""
		Reads the flat data into a spectral cube whose dimensions are determined by the number of x and y pixels and the number of wavelength bins.
		Each element will be the spectral cube for a time chunk
		Find factors to correct nonlinearity due to deadtime in firmware
		"""
		self.spectralCubes = []
		self.cubeEffIntTimes = []
		self.frames = []
		for iObs,obs in enumerate(self.obsList):
			for firstSec in range(0,self.expTime,self.intTime):		
				cubeDict = obs.getSpectralCube(firstSec=firstSec,integrationTime=self.intTime,applySpecWeight=False, applyTPFWeight=False,wvlBinEdges = self.wvlBinEdges,energyBinWidth=None,timeSpacingCut = self.timeSpacingCut)
				cube = np.array(cubeDict['cube'],dtype=np.double)
				if self.verbose:
					self.pbar_iter += 1
					self.pbar.update(self.pbar_iter)

				effIntTime = cubeDict['effIntTime']
				#add third dimension for broadcasting
				effIntTime3d = np.reshape(effIntTime,np.shape(effIntTime)+(1,))
				cube /= effIntTime3d
				cube[np.isnan(cube)]=0 

				rawFrameDict = obs.getPixelCountImage(firstSec=firstSec,integrationTime=self.intTime,scaleByEffInt=True)  
				if self.verbose:
					self.pbar_iter += 1
					self.pbar.update(self.pbar_iter)
				rawFrame = np.array(rawFrameDict['image'],dtype=np.double)
				rawFrame /= rawFrameDict['effIntTimes']
				nonlinearFactors = 1. / (1. - rawFrame*self.deadtime)  
				nonlinearFactors[np.isnan(nonlinearFactors)]=0.

                
				frame = np.sum(cube,axis=2) #in counts per sec
				frame = frame * nonlinearFactors
                
				nonlinearFactors = np.reshape(nonlinearFactors,np.shape(nonlinearFactors)+(1,))
				cube = cube * nonlinearFactors
                
				self.frames.append(frame)
				self.spectralCubes.append(cube)
				self.cubeEffIntTimes.append(effIntTime3d)
			obs.file.close()
		self.spectralCubes = np.array(self.spectralCubes)
		self.cubeEffIntTimes = np.array(self.cubeEffIntTimes)
		self.countCubes = self.cubeEffIntTimes * self.spectralCubes

	def checkCountRates(self):
		'''
		mask out frames, or cubes from integration time chunks with count rates too high
		'''
		medianCountRates = np.array([np.median(frame[frame!=0]) for frame in self.frames])
		boolIncludeFrames = medianCountRates <= self.countRateCutoff
		self.spectralCubes = np.array([cube for cube,boolIncludeFrame in zip(self.spectralCubes,boolIncludeFrames) if boolIncludeFrame==True])
		self.frames = [frame for frame,boolIncludeFrame in zip(self.frames,boolIncludeFrames) if boolIncludeFrame==True]
		if self.verbose:
			self.pbar_iter += 1
			self.pbar.update(self.pbar_iter)

	def calculateWeights(self):
		'''
		finds flat cal factors as medians/pixelSpectra for each pixel.  Normalizes these weights at each wavelength bin.
		Trim the beginning and end off the sorted weights for each wvl for each pixel, to exclude extremes from averages
		'''
		self.flatWeightsList=[]
		for iCube,cube in enumerate(self.spectralCubes):
			cubeWeightsList = []
			self.averageSpectra = []
			deltaWeightsList = []
			effIntTime = self.cubeEffIntTimes[iCube]
			#for each time chunk
			wvlAverages = np.zeros(self.nWvlBins)
			spectra2d = np.reshape(cube,[self.nXPix*self.nYPix,self.nWvlBins ])
			for iWvl in range(self.nWvlBins):
				wvlSlice = spectra2d[:,iWvl]
				goodPixelWvlSlice = np.array(wvlSlice[wvlSlice != 0]) #dead pixels need to be taken out before calculating averages
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
			self.maskedCubeDeltaWeights = np.ma.array(deltaCubeWeights,mask=cubeWeightsMask)

			#sort maskedCubeWeights and rearange spectral cubes the same way
			sortedIndices = np.ma.argsort(self.maskedCubeWeights,axis=0)
			identityIndices = np.ma.indices(np.shape(self.maskedCubeWeights))

			sortedWeights = self.maskedCubeWeights[sortedIndices,identityIndices[1],identityIndices[2],identityIndices[3]]
			countCubesReordered = self.countCubes[sortedIndices,identityIndices[1],identityIndices[2],identityIndices[3]]
			cubeDeltaWeightsReordered = self.maskedCubeDeltaWeights[sortedIndices,identityIndices[1],identityIndices[2],identityIndices[3]]

			nCubes = np.shape(self.maskedCubeWeights)[0]
			trimmedWeights = sortedWeights[self.fractionOfChunksToTrim*nCubes:(1-self.fractionOfChunksToTrim)*nCubes,:,:,:]
			trimmedCountCubesReordered = countCubesReordered[self.fractionOfChunksToTrim*nCubes:(1-self.fractionOfChunksToTrim)*nCubes,:,:,:]

			self.totalCube = np.ma.sum(trimmedCountCubesReordered,axis=0)
			self.totalFrame = np.ma.sum(self.totalCube,axis=-1)
    

			trimmedCubeDeltaWeightsReordered = cubeDeltaWeightsReordered[self.fractionOfChunksToTrim*nCubes:(1-self.fractionOfChunksToTrim)*nCubes,:,:,:]
			'''
			Uncertainty in weighted average is sqrt(1/sum(averagingWeights))
			Normalize weights at each wavelength bin
			'''	
			self.flatWeights,summedAveragingWeights = np.ma.average(trimmedWeights,axis=0,weights=trimmedCubeDeltaWeightsReordered**-2.,returned=True)
			self.countCubesToSave=np.ma.average(trimmedCountCubesReordered,axis=0)
			self.deltaFlatWeights = np.sqrt(summedAveragingWeights**-1.)
			self.flatFlags = self.flatWeights.mask
	
			wvlWeightMedians = np.ma.median(np.reshape(self.flatWeights,(-1,self.nWvlBins)),axis=0)
			self.flatWeights = np.divide(self.flatWeights,wvlWeightMedians)
			self.flatWeightsforplot = np.ma.sum(self.flatWeights,axis=-1)
			self.indexweights=iCube
			flatcal.writeWeights()
			if self.verbose:
				self.pbar_iter += 1
				self.pbar.update(self.pbar_iter)
			if self.save_plots:
				self.indexplot=iCube
				if iCube==0 or iCube==int((self.expTime/self.intTime)/2) or iCube==(int(self.expTime/self.intTime)-1):
					flatcal.plotWeightsWvlSlices()		
					flatcal.plotWeightsByPixelWvlCompare() 
					flatcal.summaryPlot()


	def plotWeightsByPixelWvlCompare(self):
		'''
		Plot weights of each wavelength bin for every single pixel
                Makes a plot of wavelength vs weights, twilight spectrum, and wavecal solution for each pixel
                Essentially does the SAME THING as plotWeightsbyPixel, but this one does a wavecal solution as well
		'''
		if not self.save_plots:
			return
		if self.save_plots:
			self.plotName='WavelengthCompare_'
			self.__setupPlots()
		# path to your wavecal solution file
		file_nameWvlCal = self.wvlCalFile
		if self.verbose:
			print('plotting weights by pixel at ',self.pdfFullPath)
			self.pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (',Timer(), ') ', ETA(), ' '], max_value=self.nXPix).start()
			self.pbar_iter = 0

		matplotlib.rcParams['font.size'] = 4 
		wvls = self.wvlBinEdges[0:-1]
		nCubes = len(self.maskedCubeWeights)

		for iRow in range(self.nXPix):
			for iCol in range(self.nYPix):
				weights = self.flatWeights[iRow,iCol,:]
				deltaWeights = self.deltaFlatWeights[iRow,iCol,:]
				if weights.mask.all() == False:
					if self.iPlot % self.nPlotsPerPage == 0:
						self.fig = plt.figure(figsize=(10,10),dpi=100)

					ax = self.fig.add_subplot(self.nPlotsPerCol,self.nPlotsPerRow,self.iPlot%self.nPlotsPerPage+1)
					ax.set_ylim(.5,2.)

					for iCube in range(nCubes):
						cubeWeights = self.maskedCubeWeights[iCube,iRow,iCol]
						ax.plot(wvls,cubeWeights.data,label='weights %d'%iCube,alpha=.7,color=matplotlib.cm.Paired((iCube+1.)/nCubes))
						ax.errorbar(wvls,weights.data,yerr=deltaWeights.data,label='weights',color='k')
                
					ax.set_title('p %d,%d'%(iRow,iCol))
					ax.set_ylabel('weight')
					ax.set_xlabel(r'$\lambda$ ($\AA$)')
					if self.iPlot%self.nPlotsPerPage == self.nPlotsPerPage-1 or (iRow == self.nXPix-1 and iCol == self.nYPix-1):
						pp.savefig(self.fig)
					self.iPlot += 1

					#Put a plot of twilight spectrums for this pixel
					if self.iPlot % self.nPlotsPerPage == 0:
						self.fig = plt.figure(figsize=(10,10),dpi=100)

					ax = self.fig.add_subplot(self.nPlotsPerCol,self.nPlotsPerRow,self.iPlot%self.nPlotsPerPage+1)
					for iCube in range(nCubes):
						spectrum = self.spectralCubes[iCube,iRow,iCol]
						ax.plot(wvls,spectrum,label='spectrum %d'%iCube,alpha=.7,color=matplotlib.cm.Paired((iCube+1.)/nCubes))
                
					ax.set_title('p %d,%d'%(iRow,iCol))
					ax.set_xlabel(r'$\lambda$ ($\AA$)')

					if self.iPlot%self.nPlotsPerPage == self.nPlotsPerPage-1 or (iRow == self.nXPix-1 and iCol == self.nYPix-1):
						pp.savefig(self.fig)
					self.iPlot += 1
					
					#Plot wavecal solution
					if self.iPlot % self.nPlotsPerPage == 0:
						self.fig = plt.figure(figsize=(10,10),dpi=100)

					ax = self.fig.add_subplot(self.nPlotsPerCol,self.nPlotsPerRow,self.iPlot%self.nPlotsPerPage+1)
					ax.set_ylim(.5,2.)

					for iCube in range(nCubes):
						my_pixel = [iRow, iCol]
						ax=p.plotEnergySolution(file_nameWvlCal, pixel=my_pixel,axis=ax)
                
					ax.set_title('p %d,%d'%(iRow,iCol))
					if self.iPlot%self.nPlotsPerPage == self.nPlotsPerPage-1 or (iRow == self.nXPix-1 and iCol == self.nYPix-1):
						pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
						pdf.savefig(self.fig)
						pdf.close()
						self.__mergePlots()
						self.saved = True
						plt.close('all')
					self.iPlot += 1
			if self.verbose:
				self.pbar_iter += 1
				self.pbar.update(self.pbar_iter)

		self.__closePlots()
		if self.verbose:
			self.pbar.finish()

            
	def plotWeightsWvlSlices(self):
		'''
		Plot weights in images of a single wavelength bin (wavelength-sliced images)
		'''
		self.plotName='WvlSlices_'
		self.__setupPlots()
		matplotlib.rcParams['font.size'] = 4 
		wvls = self.wvlBinEdges[0:-1]
		if self.verbose:
			print('plotting weights in wavelength sliced images')
			self.pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (',Timer(), ') ', ETA(), ' '], max_value=len(wvls)).start()
			self.pbar_iter = 0

		for iWvl,wvl in enumerate(wvls):
			if self.iPlot % self.nPlotsPerPage == 0:
				self.fig = plt.figure(figsize=(10,10),dpi=100)

			ax = self.fig.add_subplot(self.nPlotsPerCol,self.nPlotsPerRow,self.iPlot%self.nPlotsPerPage+1)
			ax.set_title(r'Weights %.0f $\AA$'%wvl)

			image = self.flatWeights[:,:,iWvl]

			cmap = matplotlib.cm.hot
			cmap.set_bad('#222222')
			handleMatshow = ax.matshow(image,cmap=cmap,origin='lower',vmax=2.,vmin=.5)
			cbar = self.fig.colorbar(handleMatshow)
        
			if self.iPlot%self.nPlotsPerPage == self.nPlotsPerPage-1:
				pp.savefig(self.fig)
			self.iPlot += 1

			ax = self.fig.add_subplot(self.nPlotsPerCol,self.nPlotsPerRow,self.iPlot%self.nPlotsPerPage+1)
			ax.set_title(r'Twilight Image %.0f $\AA$'%wvl)

			image = self.totalCube[:,:,iWvl]

			nSdev = 3.
			goodImage = image[np.isfinite(image)]
			vmax = np.mean(goodImage)+nSdev*np.std(goodImage)
			handleMatshow = ax.matshow(image,cmap=cmap,origin='lower',vmax=vmax)
			cbar = self.fig.colorbar(handleMatshow)

			if self.iPlot%self.nPlotsPerPage == self.nPlotsPerPage-1:
				pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
				pdf.savefig(self.fig)
				pdf.close()
				self.__mergePlots()
				self.saved = True
				plt.close('all')
			self.iPlot += 1
			if self.verbose:
				self.pbar_iter += 1
				self.pbar.update(self.pbar_iter)

		self.__closePlots()
		if self.verbose:
			self.pbar.finish()

	def plotMaskWvlSlices(self):
		'''
		Plot mask in images of a single wavelength bin (wavelength-sliced images)
		'''
		if not self.save_plots:
			return
		if self.save_plots:
			self.plotName='MaskWvlSlices_'
			self.__setupPlots()
		matplotlib.rcParams['font.size'] = 4 
		wvls = self.wvlBinEdges[0:-1]
		if self.verbose:
			print(self.pdfFullPath)
			print('plotting mask in wavelength sliced images')
			self.pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (',Timer(), ') ', ETA(), ' '], max_value=len(wvls)).start()
			self.pbar_iter = 0


		for iWvl,wvl in enumerate(wvls):
			if self.iPlot % self.nPlotsPerPage == 0:
				self.fig = plt.figure(figsize=(10,10),dpi=100)

			ax = self.fig.add_subplot(self.nPlotsPerCol,self.nPlotsPerRow,self.iPlot%self.nPlotsPerPage+1)
			ax.set_title(r'%.0f $\AA$'%wvl)

			image = self.flatFlags[:,:,iWvl]
			image=image*1
			self.wvlFlags=np.array(self.wvlFlags)
			#image += 2*self.wvlFlags  
			image = 3-image

			cmap = matplotlib.cm.gnuplot2
			handleMatshow = ax.matshow(image,cmap=cmap,origin='lower',vmax=2.,vmin=.5)
			cbar = self.fig.colorbar(handleMatshow)
        
			if self.iPlot%self.nPlotsPerPage == self.nPlotsPerPage-1: 
				pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
				pdf.savefig(self.fig)
				pdf.close()
				self.__mergePlots()
				self.saved = True
				plt.close('all')
			self.iPlot += 1

		self.__closePlots()
		if self.verbose:
			self.pbar.finish()

	def plotWeightsByPixel(self):
		'''
		Plot weights of each wavelength bin for every single pixel
		'''
		if not self.save_plots:
			return
		if self.save_plots:
			self.plotName='PlotWeightsByPixel_'
			self.__setupPlots()
		pixels=self.nXPix*self.nYPix
		if self.verbose:
			print('plotting weights by pixel at ',self.pdfFullPath)
			self.pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (',Timer(), ') ', ETA(), ' '], max_value=pixels).start()
			self.pbar_iter = 0

		matplotlib.rcParams['font.size'] = 4 
		wvls = self.wvlBinEdges[0:-1]
		nCubes = len(self.maskedCubeWeights)

		for iRow in range(self.nXPix):
			for iCol in range(self.nYPix):
				weights = self.flatWeights[iRow,iCol,:]
				deltaWeights = self.deltaFlatWeights[iRow,iCol,:]
				if weights.mask.all() == False:
					if self.iPlot % self.nPlotsPerPage == 0:
						self.fig = plt.figure(figsize=(10,10),dpi=100)

					ax = self.fig.add_subplot(self.nPlotsPerCol,self.nPlotsPerRow,self.iPlot%self.nPlotsPerPage+1)
					ax.set_ylim(.5,2.)

					for iCube in range(nCubes):
						cubeWeights = self.maskedCubeWeights[iCube,iRow,iCol]
						ax.plot(wvls,cubeWeights.data,label='weights %d'%iCube,alpha=.7,color=matplotlib.cm.Paired((iCube+1.)/nCubes))
						ax.errorbar(wvls,weights.data,yerr=deltaWeights.data,label='weights',color='k')
                
					ax.set_title('p %d,%d'%(iRow,iCol))
					ax.set_ylabel('weight')
					ax.set_xlabel(r'$\lambda$ ($\AA$)')
					if self.iPlot%self.nPlotsPerPage == self.nPlotsPerPage-1 or (iRow == self.nXPix-1 and iCol == self.nYPix-1):
						pp.savefig(self.fig)
					self.iPlot += 1

					#Put a plot of twilight spectrums for this pixel
					if self.iPlot % self.nPlotsPerPage == 0:
						self.fig = plt.figure(figsize=(10,10),dpi=100)

					ax = self.fig.add_subplot(self.nPlotsPerCol,self.nPlotsPerRow,self.iPlot%self.nPlotsPerPage+1)
					for iCube in range(nCubes):
						spectrum = self.spectralCubes[iCube,iRow,iCol]
						ax.plot(wvls,spectrum,label='spectrum %d'%iCube,alpha=.7,color=matplotlib.cm.Paired((iCube+1.)/nCubes))
                
					ax.set_title('p %d,%d'%(iRow,iCol))
					ax.set_xlabel(r'$\lambda$ ($\AA$)')

					if self.iPlot%self.nPlotsPerPage == self.nPlotsPerPage-1 or (iRow == self.nXPix-1 and iCol == self.nYPix-1):
						pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
						pdf.savefig(self.fig)
						pdf.close()
						self.__mergePlots()
						self.saved = True
						plt.close('all')
					self.iPlot += 1
					if self.verbose:
						self.pbar_iter += 1
						self.pbar.update(self.pbar_iter)

		self.__closePlots()
		if self.verbose:
			self.pbar.finish()

	def summaryPlot(self):
		"""
		Writes a summary plot of the Flat Fielding
		"""
		if not self.save_plots:
			return
		if self.save_plots:
			self.plotName='summaryPlot_'
			self.__setupPlots()
		pixels=self.nXPix*self.nYPix
		if self.verbose:
			print('Generating summary plot at ',self.pdfFullPath)
			self.pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (',Timer(), ') ', ETA(), ' '], max_value=pixels).start()
			self.pbar_iter = 0

		matplotlib.rcParams['font.size'] = 4 
		wvls = self.wvlBinEdges[0:-1]
		nCubes = len(self.maskedCubeWeights)

		meanWeightList=np.zeros((self.nXPix, self.nYPix))
		self.fig = plt.figure(figsize=(10,10),dpi=100)
		ax = self.fig.add_subplot(1,1,1)

		for iRow in range(self.nXPix):
			for iCol in range(self.nYPix):
				weights = self.flatWeights[iRow,iCol,:]
				meanWeight=np.nanmean(weights)
				meanWeightList[iRow, iCol]=meanWeight
		plt.imshow(meanWeightList)



				

		

	def writeWeights(self):
		"""
		Writes an h5 file to put calculated flat cal factors in
		"""
		if os.path.isabs(self.flatCalFileName) == True:
			fullFlatCalFileName =self.flatCalFileName
			baseh5path=fullFlatCalFileName.split('.h5')
			fullFlatCalFileName=baseh5path[0]+str(self.indexweights+1)+'.h5'
		else:
			scratchDir = os.getenv('MKID_PROC_PATH')
			flatDir = os.path.join(scratchDir,'flatCalSolnFiles')
			fullFlatCalFileName = os.path.join(flatDir,self.flatCalFileName)
			baseh5path=fullFlatCalFileName.split('.h5')
			fullFlatCalFileName=baseh5path[0]+str(self.indexweights+1)+'.h5'

		if not os.path.exists(fullFlatCalFileName) and self.calSolnPath =='':	
			os.makedirs(fullFlatCalFileName)		

		try:
			flatCalFile = tables.open_file(fullFlatCalFileName,mode='w')
		except:
			print('Error: Couldn\'t create flat cal file, ',fullFlatCalFileName)
			return

		header = flatCalFile.create_group(flatCalFile.root, 'header', 'Calibration information')
		beamImage = tables.Array(header, 'beamMap', obj=self.beamImage) 
		calgroup = flatCalFile.create_group(flatCalFile.root,'flatcal','Table of flat calibration weights by pixel and wavelength')
		calarray = tables.Array(calgroup,'weights',obj=self.flatWeights.data,title='Flat calibration Weights indexed by pixelRow,pixelCol,wavelengthBin')
		specarray = tables.Array(calgroup,'spectrum',obj=self.countCubesToSave.data,title='Twilight spectrum indexed by pixelRow,pixelCol,wavelengthBin')
		flagtable = tables.Array(calgroup,'flags',obj=self.flatFlags,title='Flat cal flags indexed by pixelRow,pixelCol,wavelengthBin. 0 is Good')
		bintable = tables.Array(calgroup,'wavelengthBins',obj=self.wvlBinEdges,title='Wavelength bin edges corresponding to third dimension of weights array')

		descriptionDict = FlatCalSoln_Description(self.nWvlBins)
		caltable = flatCalFile.create_table(calgroup, 'calsoln', descriptionDict,title='Flat Cal Table')
        
		for iRow in range(self.nXPix):
			for iCol in range(self.nYPix):
				weights = self.flatWeights[iRow,iCol,:]
				spectrum=self.countCubesToSave[iRow,iCol,:]
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
				entry['spectrum'] = spectrum
				entry['weightFlags'] = flags
				entry['flag'] = flag
				entry.append()
        
		flatCalFile.flush()
		flatCalFile.close()

		# close progress bar
		if self.verbose:
			self.pbar.finish()
		if self.verbose:
			print('wrote to',fullFlatCalFileName)

	def __setupPlots(self):
		'''
		Initialize plotting variables
		'''
		flatCalPath,flatCalBasename = os.path.split(self.flatCalFileName)
		self.nPlotsPerRow = 3
		self.nPlotsPerCol = 4
		self.nPlotsPerPage = self.nPlotsPerRow*self.nPlotsPerCol
		self.iPlot = 0 
		self.pdfFullPath = self.calSolnPath+self.plotName+str(self.indexplot+1)+'.pdf'

		if os.path.isfile(self.pdfFullPath):
			answer = self.__query("{0} already exists. Overwrite?".format(self.pdfFullPath), yes_or_no=True)
			if answer is False:
				answer = self.__query("Provide a new file name (type exit to quit):")
				if answer == 'exit':
   					raise UserError("User doesn't want to overwrite the plot file " + "... exiting")
				self.pdfFullPath = self.calSolnPath+str(answer)+str(self.indexplot+1)+'.pdf'
				print(self.pdfFullPath)
			else:
				os.remove(self.pdfFullPath)
				print(self.pdfFullPath)

	def __mergePlots(self):
		'''
		Merge recently created temp.pdf with the main file
		'''
		temp_file = os.path.join(self.calSolnPath, 'temp.pdf')
		if os.path.isfile(self.pdfFullPath):
			merger = PdfFileMerger()
			merger.append(PdfFileReader(open(self.pdfFullPath, 'rb')))
			merger.append(PdfFileReader(open(temp_file, 'rb')))
			merger.write(self.pdfFullPath)
			merger.close()
			os.remove(temp_file)
		else:
 			os.rename(temp_file, self.pdfFullPath)

	def __closePlots(self):
		'''
		Safely close plotting variables after plotting since the last page is only saved if it is full.
		'''
		if not self.saved:
			pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
			pdf.savefig(self.fig)
			pdf.close()
			self.__mergePlots()
		plt.close('all')

	def __configCheck(self, index):
		'''
		Checks the variables loaded in from the configuration file for type and
		consistencey. Run in the '__init__()' method.
		'''
		if index == 0:
			# check for configuration file
			assert os.path.isfile(self.config_file), \
				self.config_file + " is not a valid configuration file"
		elif index == 1:
			# check if all sections and parameters exist in the configuration file
			section = "{0} must be a configuration section"
			param = "{0} must be a parameter in the configuration file '{1}' section"

			assert 'Data' in self.config.sections(), section.format('Data')
			assert 'flatPath' in self.config['Data'].keys(), \
				param.format('flatPath', 'Data')                                                
			assert 'wvlCalFile' in self.config['Data'].keys(), \
				param.format('wvlCalFile', 'Data')                                                 
			assert 'intTime' in self.config['Data'].keys(), \
				param.format('intTime', 'Data')
			assert 'expTime' in self.config['Data'].keys(), \
				param.format('expTime', 'Data')

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

			assert 'calSolnPath' in self.config['Output'].keys(), \
				param.format('calSolnPath', 'Output') 
			assert 'verbose' in self.config['Output'].keys(), \
				param.format('verbose', 'Output') 
			assert 'save_plots' in self.config['Output'].keys(), \
				param.format('save_plots', 'Output') 

		elif index == 2:
			# type check parameters
			if self.flatPath != '':
				assert type(self.flatPath) is str, "Flat Path parameter must be a string."
				assert os.path.exists(self.flatPath), "Please confirm the Flat File path provided is correct"
			if self.calSolnPath != '':
				assert type(self.calSolnPath) is str, "Cal Solution Path parameter must be a string."
			if self.wvlCalFile != '':
				assert type(self.wvlCalFile) is str, "WaveCal Solution Path parameter must be a string."
			assert type(self.intTime) is int, "integration time parameter must be an integer"
			assert type(self.expTime) is int, "Exposure time parameter must be an integer"
			
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
			assert type(self.verbose) is bool, "Verbose indicator must be a bool"
			assert type(self.save_plots) is bool, "Save Plots indicator must be a bool"


		else:
			raise ValueError("index must be 0, 1, or 2")

	@staticmethod
	def __query(question, yes_or_no=False, default="no"):
		'''
		Ask a question via raw_input() and return their answer.
		"question" is a string that is presented to the user.
		"yes_or_no" specifies if it is a yes or no question
		"default" is the presumed answer if the user just hits <Enter>.
		It must be "yes" (the default), "no" or None (meaning an answer is required of
		the user). Only used if yes_or_no=True.

		The "answer" return value is the user input for a general question. For a yes or
		no question it is True for "yes" and False for "no".
		'''
		if yes_or_no:
			valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
		if not yes_or_no:
			prompt = ""
			default = None
		elif default is None:
			prompt = " [y/n] "
		elif default == "yes":
			prompt = " [Y/n] "
		elif default == "no":
			prompt = " [y/N] "
		else:
			raise ValueError("invalid default answer: '%s'" % default)

		while True:
			print(question + prompt)
			choice = input().lower()
			if not yes_or_no:
				return choice
			elif default is not None and choice == '':
				return valid[default]
			elif choice in valid:
				return valid[choice]
			else:
				print("Please respond with 'yes' or 'no' (or 'y' or 'n').")


if __name__ == '__main__':
	if len(sys.argv) == 1:
		flatcal = FlatCal()
	else:
		flatcal = FlatCal(config_file=sys.argv[1])
	flatcal.loadFlatSpectra()
	flatcal.checkCountRates()
	flatcal.calculateWeights()
