#!/bin/python
"""
Author: Matt Strader        Date:August 19,2012
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


from Utils.popup import PopUp,plotArray,pop
from Utils.ObsFile import ObsFile
from Utils.readDict import readDict
from Utils.FileName import FileName
import HotPix.darkHotPixMask as hp
from Headers.CalHeaders import FlatCalSoln_Description

class FlatCal:
    def __init__(self,paramFile):
        """
        opens flat file,sets wavelength binnning parameters, and calculates flat factors for the file
        """
        self.params = readDict()
        self.params.read_from_file(paramFile)

        run = self.params['run']
        sunsetDate = self.params['sunsetDate']
        flatTstamp = self.params['flatTstamp']
        wvlSunsetDate = self.params['wvlSunsetDate']
        wvlTimestamp = self.params['wvlTimestamp']
        obsSequence = self.params['obsSequence']
        needTimeAdjust = self.params['needTimeAdjust']
        self.deadtime = self.params['deadtime'] #from firmware pulse detection
        self.timeSpacingCut = self.params['timeSpacingCut']
        bLoadBeammap = self.params.get('bLoadBeammap',False)
            
        obsFNs = [FileName(run=run,date=sunsetDate,tstamp=obsTstamp) for obsTstamp in obsSequence]
        print(obsFNs)
        self.obsFileNames = [fn.obs() for fn in obsFNs]
        self.obsList = [ObsFile(obsFileName) for obsFileName in self.obsFileNames]
        timeMaskFileNames = [fn.timeMask() for fn in obsFNs]
        timeAdjustFileName = FileName(run=run).timeAdjustments()


        print len(self.obsFileNames), 'flat files to co-add'
        self.flatCalFileName = FileName(run=run,date=sunsetDate,tstamp=flatTstamp).flatSoln()
        if wvlSunsetDate != '':
            wvlCalFileName = FileName(run=run,date=wvlSunsetDate,tstamp=wvlTimestamp).calSoln()
        for iObs,obs in enumerate(self.obsList):
            if bLoadBeammap:
                print 'loading beammap',os.environ['MKID_BEAMMAP_PATH']
                obs.loadBeammapFile(os.environ['MKID_BEAMMAP_PATH'])
            if wvlSunsetDate != '':
                obs.loadWvlCalFile(wvlCalFileName)
            else:
                obs.loadBestWvlCalFile()

            if needTimeAdjust:
                obs.loadTimeAdjustmentFile(timeAdjustFileName)
            timeMaskFileName = timeMaskFileNames[iObs]
            print timeMaskFileName
            #Temporary step, remove old hotpix file
            #if os.path.exists(timeMaskFileName):
            #    os.remove(timeMaskFileName)
            obs.setWvlCutoffs(3000,13000)
            if not os.path.exists(timeMaskFileName):
                print 'Running hotpix for ',obs
                hp.findHotPixels(obsFile=obs,outputFileName=timeMaskFileName,fwhm=np.inf,useLocalStdDev=True)
                print "Flux file pixel mask saved to %s"%(timeMaskFileName)
            obs.loadHotPixCalFile(timeMaskFileName)

        #get beammap from first obs
        self.beamImage = self.obsList[0].beamImage
        self.wvlFlags = self.obsList[0].wvlFlagTable

        self.nRow = self.obsList[0].nRow
        self.nCol = self.obsList[0].nCol
        print 'files opened'
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

        #print 'wrote to',self.flatCalFileName

    def __del__(self):
        pass

    def loadFlatSpectra(self):
        self.spectralCubes = []#each element will be the spectral cube for a time chunk
        self.cubeEffIntTimes = []
        self.frames = []
        for iObs,obs in enumerate(self.obsList):
            print 'obs',iObs
            for firstSec in range(0,obs.getFromHeader('exptime'),self.intTime):
                print 'sec',firstSec
                cubeDict = obs.getSpectralCube(firstSec=firstSec,integrationTime=self.intTime,weighted=False,wvlBinEdges = self.wvlBinEdges,timeSpacingCut = self.timeSpacingCut)
                cube = np.array(cubeDict['cube'],dtype=np.double)
                effIntTime = cubeDict['effIntTime']
                #add third dimension for broadcasting
                effIntTime3d = np.reshape(effIntTime,np.shape(effIntTime)+(1,))
                cube /= effIntTime3d
                cube[np.isnan(cube)]=0

                #find factors to correct nonlinearity
                rawFrameDict = obs.getPixelCountImage(firstSec=firstSec,integrationTime=self.intTime,getRawCount=True)
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
        self.spectralCubes = np.array(self.spectralCubes)
        self.cubeEffIntTimes = np.array(self.cubeEffIntTimes)
        self.countCubes = self.cubeEffIntTimes * self.spectralCubes

    def checkCountRates(self):
        medianCountRates = np.array([np.median(frame[frame!=0]) for frame in self.frames])
        boolIncludeFrames = medianCountRates <= self.countRateCutoff
        #boolIncludeFrames = np.logical_and(boolIncludeFrames,medianCountRates >= 200) 
        #mask out frames, or cubes from integration time chunks with count rates too high
        self.spectralCubes = np.array([cube for cube,boolIncludeFrame in zip(self.spectralCubes,boolIncludeFrames) if boolIncludeFrame==True])
        self.frames = [frame for frame,boolIncludeFrame in zip(self.frames,boolIncludeFrames) if boolIncludeFrame==True]
        print 'few enough counts in the chunk',zip(medianCountRates,boolIncludeFrames)

    def calculateWeights(self):
        """
        finds flat cal factors as medians/pixelSpectra for each pixel
        """
        cubeWeightsList = []
        self.averageSpectra = []
        deltaWeightsList = []
        for iCube,cube in enumerate(self.spectralCubes):
            effIntTime = self.cubeEffIntTimes[iCube]
            #for each time chunk
            wvlAverages = np.zeros(self.nWvlBins)
            spectra2d = np.reshape(cube,[self.nRow*self.nCol,self.nWvlBins ])
            for iWvl in xrange(self.nWvlBins):
                wvlSlice = spectra2d[:,iWvl]
                goodPixelWvlSlice = np.array(wvlSlice[wvlSlice != 0])#dead pixels need to be taken out before calculating averages
                nGoodPixels = len(goodPixelWvlSlice)

                #goodPixelWvlSlice = np.sort(goodPixelWvlSlice)
                #trimmedSpectrum = goodPixelWvlSlice[self.fractionOfPixelsToTrim*nGoodPixels:(1-self.fractionOfPixelsToTrim)*nGoodPixels]
                #trimmedPixelWeights = 1/np.sqrt(trimmedSpectrum)
                #histGood,binEdges = np.histogram(self.intTime*goodSpectrum,bins=nBins)
                #histTrim,binEdges = np.histogram(self.intTime*trimmedSpectrum,bins=binEdges)
#                plt.plot(binEdges[0:-1],histGood)
#                def f(fig,axes):
#                    axes.plot(binEdges[0:-1],histGood)
#                    axes.plot(binEdges[0:-1],histTrim)
#                pop(plotFunc=f)
#                plt.plot(binEdges[0:-1],histTrim)

                wvlAverages[iWvl] = np.median(goodPixelWvlSlice)
#                plt.show()
            weights = np.divide(wvlAverages,cube)
            weights[weights==0] = np.nan
            weights[weights==np.inf] = np.nan
            cubeWeightsList.append(weights)

            #Now to get uncertainty in weight:
            #Assuming negligible uncertainty in medians compared to single pixel spectra,
            #then deltaWeight=weight*deltaSpectrum/Spectrum
            #deltaWeight=weight*deltaRawCounts/RawCounts
            # with deltaRawCounts=sqrt(RawCounts)#Assuming Poisson noise
            #deltaWeight=weight/sqrt(RawCounts)
            # but 'cube' is in units cps, not raw counts 
            # so multiply by effIntTime before sqrt
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

        #trim the beginning and end off the sorted weights for each wvl for each pixel, to exclude extremes from averages
        nCubes = np.shape(self.maskedCubeWeights)[0]
        trimmedWeights = sortedWeights[self.fractionOfChunksToTrim*nCubes:(1-self.fractionOfChunksToTrim)*nCubes,:,:,:]
        trimmedCountCubesReordered = countCubesReordered[self.fractionOfChunksToTrim*nCubes:(1-self.fractionOfChunksToTrim)*nCubes,:,:,:]
        print 'trimmed cubes shape',np.shape(trimmedCountCubesReordered)

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
            

        #flagImage = np.shape(self.flatFlags)[2]-np.sum(self.flatFlags,axis=2)
        #plotArray(flagImage)

#        X,Y,Z=np.mgrid[0:self.nRow,0:self.nCol,0:self.nWvlBins]
#        Z=self.wvlBinEdges[Z]
#        fig = plt.figure()
#        ax = Axes3D(fig)
#        handleScatter=ax.scatter(X,Y,Z,c=self.flatWeights,vmax=2,vmin=.5)
#        fig.colorbar(handleScatter)
#        plt.show()
        
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
            print 'plotting weights in wavelength sliced images'

        matplotlib.rcParams['font.size'] = 4 
        wvls = self.wvlBinEdges[0:-1]

        for iWvl,wvl in enumerate(wvls):
            if verbose:
                print 'wvl ',iWvl
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
            print 'plotting mask in wavelength sliced images'

        matplotlib.rcParams['font.size'] = 4 
        wvls = self.wvlBinEdges[0:-1]

        for iWvl,wvl in enumerate(wvls):
            if verbose:
                print 'wvl ',iWvl
            if iPlot % nPlotsPerPage == 0:
                fig = plt.figure(figsize=(10,10),dpi=100)

            ax = fig.add_subplot(nPlotsPerCol,nPlotsPerRow,iPlot%nPlotsPerPage+1)
            ax.set_title(r'%.0f $\AA$'%wvl)

            image = self.flatFlags[:,:,iWvl]
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
        flatCalPath,flatCalBasename = os.path.split(self.flatCalFileName)
        pdfBasename = os.path.splitext(flatCalBasename)[0]+'.pdf'
        pdfFullPath = os.path.join(flatCalPath,pdfBasename)
        pp = PdfPages(pdfFullPath)
        nPlotsPerRow = 2
        nPlotsPerCol = 4
        nPlotsPerPage = nPlotsPerRow*nPlotsPerCol
        iPlot = 0 
        if verbose:
            print 'plotting weights by pixel at ',pdfFullPath

        matplotlib.rcParams['font.size'] = 4 
        wvls = self.wvlBinEdges[0:-1]
        nCubes = len(self.maskedCubeWeights)

        for iRow in xrange(self.nRow):
            if verbose:
                print 'row',iRow
            for iCol in xrange(self.nCol):
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
                    if iPlot%nPlotsPerPage == nPlotsPerPage-1 or (iRow == self.nRow-1 and iCol == self.nCol-1):
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
                    if iPlot%nPlotsPerPage == nPlotsPerPage-1 or (iRow == self.nRow-1 and iCol == self.nCol-1):
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
        else:
            scratchDir = os.getenv('MKID_PROC_PATH')
            flatDir = os.path.join(scratchDir,'flatCalSolnFiles')
            fullFlatCalFileName = os.path.join(flatDir,self.flatCalFileName)

        try:
            flatCalFile = tables.openFile(fullFlatCalFileName,mode='w')
        except:
            print 'Error: Couldn\'t create flat cal file, ',fullFlatCalFileName
            return
        print 'wrote to',self.flatCalFileName

        calgroup = flatCalFile.createGroup(flatCalFile.root,'flatcal','Table of flat calibration weights by pixel and wavelength')
        calarray = tables.Array(calgroup,'weights',object=self.flatWeights.data,title='Flat calibration Weights indexed by pixelRow,pixelCol,wavelengthBin')
        flagtable = tables.Array(calgroup,'flags',object=self.flatFlags,title='Flat cal flags indexed by pixelRow,pixelCol,wavelengthBin. 0 is Good')
        bintable = tables.Array(calgroup,'wavelengthBins',object=self.wvlBinEdges,title='Wavelength bin edges corresponding to third dimension of weights array')

        descriptionDict = FlatCalSoln_Description(self.nWvlBins)
        caltable = flatCalFile.createTable(calgroup, 'calsoln', descriptionDict,title='Flat Cal Table')
        
        for iRow in xrange(self.nRow):
            for iCol in xrange(self.nCol):
                weights = self.flatWeights[iRow,iCol,:]
                deltaWeights = self.deltaFlatWeights[iRow,iCol,:]
                flags = self.flatFlags[iRow,iCol,:]
                flag = np.any(self.flatFlags[iRow,iCol,:])
                pixelName = self.beamImage[iRow,iCol]
                roach = int(pixelName.split('r')[1].split('/')[0])
                pixelNum = int(pixelName.split('p')[1].split('/')[0])

                entry = caltable.row
                entry['roach'] = roach
                entry['pixelnum'] = pixelNum
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

        #calculate total spectra and medians for programs that expect old format flat cal
        spectra = np.array(np.sum(self.spectralCubes,axis=0))

        wvlAverages = np.zeros(self.nWvlBins)
        spectra2d = np.reshape(spectra,[self.nRow*self.nCol,self.nWvlBins ])
        for iWvl in xrange(self.nWvlBins):
            spectrum = spectra2d[:,iWvl]
            goodSpectrum = spectrum[spectrum != 0]#dead pixels need to be taken out before calculating medians
            wvlAverages[iWvl] = np.median(goodSpectrum)
        np.savez(npzFileName,median=wvlAverages,averageSpectra=np.array(self.averageSpectra),binEdges=self.wvlBinEdges,spectra=spectra,weights=np.array(self.flatWeights.data),deltaWeights=np.array(self.deltaFlatWeights.data),mask=self.flatFlags,totalFrame=self.totalFrame,totalCube=self.totalCube,spectralCubes=self.spectralCubes,countCubes=self.countCubes,cubeEffIntTimes=self.cubeEffIntTimes )


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

