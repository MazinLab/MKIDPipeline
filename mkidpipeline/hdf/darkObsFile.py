#!/bin/python
'''
Author: Matt Strader        Date: August 19, 2012
Modified 2017 for Darkness/MEC
Authors: Seth Meeker, Neelay Fruitwala, Alex Walter
Last Updated: April 16, 2018

The class ObsFile is an interface to observation files.  It provides methods 
for typical ways of accessing photon list observation data.  It can also load 
and apply wavelength and flat calibration.  


Class Obsfile:
====Helper functions====
__init__(self,fileName,mode='read',verbose=False)
__del__(self)
loadFile(self, fileName)
getFromHeader(self, name)
pixelIsBad(self, xCoord, yCoord, forceWvl=False, forceWeights=False, forceTPFWeights=False)

====Single pixel access functions====
getPixelPhotonList(self, xCoord, yCoord, firstSec=0, integrationTime= -1, wvlStart=None,wvlStop=None, forceRawPhase=False)
getListOfPixelsPhotonList(self, posList, **kwargs)
getPixelCount(self, *args, applyWeight=True, applyTPFWeight=True, applyTimeMask=False, **kwargs)
getPixelLightCurve(self,*args,lastSec=-1, cadence=1, scaleByEffInt=True, **kwargs)

====Array access functions====
!!!!These functions need some work!!!!
!!!!They aren't robust to edge cases yet!!!!
getPixelCountImage(self, firstSec=0, integrationTime= -1, wvlStart=None,wvlStop=None,applyWeight=True, applyTPFWeight=True, applyTimeMask=False, scaleByEffInt=False, flagToUse=0)
getCircularAperturePhotonList(self, centerXCoord, centerYCoord, radius, firstSec=0, integrationTime=-1, wvlStart=None,wvlStop=None, flagToUse=0)
_makePixelSpectrum(self, photonList, **kwargs)
getSpectralCube(self, firstSec=0, integrationTime=-1, applySpecWeight=False, applyTPFWeight=False, wvlStart=700, wvlStop=1500,wvlBinWidth=None, energyBinWidth=None, wvlBinEdges=None, timeSpacingCut=None, flagToUse=0)
getPixelSpectrum(self, xCoord, yCoord, firstSec=0, integrationTime= -1,applySpecWeight=False, applyTPFWeight=False, wvlStart=None, wvlStop=None,wvlBinWidth=None, energyBinWidth=None, wvlBinEdges=None,timeSpacingCut=None)
makeWvlBins(energyBinWidth=.1, wvlStart=700, wvlStop=1500)

====Data write functions for calibrating====
applyWaveCal(self, file_name)
updateWavelengths(self, xCoord, yCoord, wvlCalArr)
__applyColWeight(self, resID, weightArr, colName)
applySpecWeight(self, resID, weightArr)
applyTPFWeight(self, resID, weightArr)
applyFlatCal(self, calSolnPath,verbose=False)
applyFlag(self, xCoord, yCoord, flag)
undoFlag(self, xCoord, yCoord, flag)
modifyHeaderEntry(self, headerTitle, headerValue)





'''

import glob
import os
import warnings

import astropy.constants
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tables
from interval import interval
from matplotlib.backends.backend_pdf import PdfPages
from regions import CirclePixelRegion, PixCoord

from mkidpipeline.core.pixelflags import h5FileFlags


class ObsFile:
    h = astropy.constants.h.to('eV s').value  #4.135668e-15 #eV s
    c = astropy.constants.c.to('m/s').value   #'2.998e8 #m/s
    nCalCoeffs = 3
    tickDuration = 1e-6    #each integer value is 1 microsecond
    
    def __init__(self, fileName, mode='read', verbose=False):
        """
        Create ObsFile object and load in specified HDF5 file.

        Parameters
        ----------
            fileName: String
                Path to HDF5 File
            mode: String
                'read' or 'write'. File should be opened in 'read' mode 
                unless you are applying a calibration.
            verbose: bool
                Prints debug messages if True
        Returns
        -------
            ObsFile instance
                
        """
        if mode in ['read', 'r']: mode='read'
        elif mode in ['write', 'w','a']: mode='write'
        assert mode=='read' or mode=='write', '"mode" argument must be "read" or "write"'
        self.mode = mode
        self.verbose=verbose
        self.tickDuration = ObsFile.tickDuration
        self.noResIDFlag = 2**32-1      #All pixels should have a unique resID. But if for some reason it doesn't, it'll have this resID
        self.wvlLowerLimit = None
        self.wvlUpperLimit = None
        self.filterIsApplied = False
        #self.timeMaskExists = False
        #self.makeMaskVersion = None
        
        self.loadFile(fileName)
        self.photonTable = self.file.get_node('/Photons/PhotonTable')

    def __del__(self):
        """
        Closes the obs file and any cal files that are open
        """
        try:
            self.file.close()
        except:
            pass
        try:
            self.wvlCalFile.close()
        except:
            pass
        try:
            self.flatCalFile.close()
        except:
            pass
        try:
            self.fluxCalFile.close()
        except:
            pass
        try:
            self.hotPixFile.close()
        except:
            pass


    def loadFile(self, fileName):
        """
        Opens file and loads obs file attributes and beammap
        """
        if self.mode=='read':
            mode = 'r'
        elif self.mode=='write':
            mode = 'a'
            
            
        self.fileName = os.path.basename(fileName)
        self.fullFileName = fileName
        if self.verbose: print("Loading "+self.fullFileName)
        try:
            self.file = tables.open_file(self.fullFileName, mode=mode)
        except IOError:     #check the MKID_RAW_PATH if the full path wasn't given
            self.fileName = fileName
            # make the full file name by joining the input name
            # to the MKID_RAW_PATH (or . if the environment variable
            # is not defined)
            dataDir = os.getenv('MKID_RAW_PATH', '')
            self.fullFileName = os.path.join(dataDir, self.fileName)
            self.file = tables.open_file(self.fullFileName, mode=mode)

        #get the header
        self.header = self.file.root.header.header
        self.titles = self.header.colnames
        self.info = self.header[0] #header is a table with one row

        # get important cal params
        self.defaultWvlBins = ObsFile.makeWvlBins(self.getFromHeader('energyBinWidth'), self.getFromHeader('wvlBinStart'), self.getFromHeader('wvlBinEnd'))
        self.ticksPerSec = int(1.0 / self.tickDuration)
        self.intervalAll = interval[0.0, (1.0 / self.tickDuration) - 1]


        #get the beam image.
        self.beamImage = self.file.get_node('/BeamMap/Map').read()
        self.beamFlagImage = self.file.get_node('/BeamMap/Flag')
        beamShape = self.beamImage.shape
        self.nXPix = beamShape[0]
        self.nYPix = beamShape[1]


    def getFromHeader(self, name):
        """
        Returns a requested entry from the obs file header
        eg. 'expTime'
        """
        entry = self.info[self.titles.index(name)]
        return entry
    
    def pixelIsBad(self, xCoord, yCoord, forceWvl=False, forceWeights=False, forceTPFWeights=False):
        """
        Returns True if the pixel wasn't read out or if a given calibration failed when needed
        
        Parameters
        ----------
        xCoord: int
        yCoord: int
        forceWvl: bool - If true, check that the wvlcalibration was good for this pixel
        forceWeights: bool - If true, check that the flat calibration was good
                           - Will also check if linearitycal, spectralcal worked correctly eventually?
        forceTPFWeights: bool - ignored for now
        
        Returns
        -------
            True if pixel is bad. Otherwise false
        """
        resID = self.beamImage[xCoord][yCoord]
        if resID==self.noResIDFlag: return True     # No resID was given during readout
        pixelFlags = self.beamFlagImage[xCoord, yCoord]
        deadFlags = h5FileFlags['noDacTone']
        if forceWvl and self.getFromHeader('isWvlCalibrated'): deadFlags+=h5FileFlags['waveCalFailed']
        if forceWeights and self.getFromHeader('isFlatCalibrated'): deadFlags+=h5FileFlags['flatCalFailed']
        #if forceWeights and self.getFromHeader('isLinearityCorrected'): deadFlags+=h5FileFlags['linCalFailed']
        #if forceTPFWeights and self.getFromHeader('isPhaseNoiseCorrected'): deadFlags+=h5FileFlags['phaseNoiseCalFailed']
        return (pixelFlags & deadFlags)>0

    def getPixelPhotonList(self, xCoord, yCoord, firstSec=0, integrationTime= -1, wvlStart=None,wvlStop=None, forceRawPhase=False):
        """
        Retrieves a photon list for a single pixel using the attached beammap.

        Parameters
        ----------
        xCoord: int
            x-coordinate of pixel in beammap
        yCoord: int
            y-coordinate index of pixel in beammap
        firstSec: float
            Photon list start time, in seconds relative to beginning of file
        integrationTime: float
            Photon list end time, in seconds relative to firstSec.
            If -1, goes to end of file
        wvlStart: float
            Desired start wavelength range (inclusive). Must satisfy wvlStart <= wvlStop.
            If None, includes all wavelengths less than wvlStop. If file is not wavelength calibrated, this parameter
            specifies the range of desired phase heights.
        wvlStop: float
            Desired stop wavelength range (non-inclusive). Must satisfy wvlStart <= wvlStop.
            If None, includes all wavelengths greater than wvlStart. If file is not wavelength calibrated, this parameter
            specifies the range of desired phase heights.
        forceRawPhase: bool
            If the ObsFile is not wavelength calibrated this flag does nothing.
            If the ObsFile is wavelength calibrated (ObsFile.getFromHeader('isWvlCalibrated') = True) then:
             - forceRawPhase=True will return all the photons in the list (might be phase heights instead of wavelengths)
             - forceRawPhase=False is guarenteed to only return properly wavelength calibrated photons in the photon list

        Returns
        -------
        Structured Numpy Array
            Each row is a photon.
            Columns have the following keys: 'Time', 'Wavelength', 'SpecWeight', 'NoiseWeight'
        
        Time is in microseconds
        Wavelength is in degrees of phase height or nanometers
        SpecWeight is a float in [0,1]
        NoiseWeight is a float in [0,1]
        
        """
        resID = self.beamImage[xCoord][yCoord]
        if (self.pixelIsBad(xCoord, yCoord, not forceRawPhase)
           or firstSec>float(self.getFromHeader('expTime'))                         # Starting time is past total exposure time in file
           or ((wvlStart!=None) and (wvlStop!=None) and (wvlStop<wvlStart))):       # wavelength range invalid
            ##print('BadPixel')
            #print((wvlStop<wvlStart))
            return self.photonTable.read_where('(Time < 0)') #use dummy condition to get empty photon list of correct format
            
        query='(ResID == resID)'
        startTime=0
        if firstSec>0:
            startTime = int(firstSec*self.ticksPerSec) #convert to us
            query+=' & (Time >= startTime)'
        if integrationTime!=-1:
            endTime = startTime + int(integrationTime*self.ticksPerSec)
            query+=' & (Time < endTime)'
        if wvlStart is not None and wvlStart==wvlStop:
            wvl=wvlStart
            query+=' & (Wavelength == wvl)'
        else:
            if wvlStart is not None:
                startWvl=wvlStart
                query+=' & (Wavelength >= startWvl)'
            if wvlStop is not None:
                stopWvl=wvlStop
                query+=' & (Wavelength < stopWvl)'

        return self.photonTable.read_where(query)

    def getListOfPixelsPhotonList(self, posList, **kwargs):
        """
        Retrieves photon lists for a list of pixels.

        Parameters
        ----------
        posList: Nx2 array of ints (or list of 2 element tuples)
            List of (x, y) beammap indices for desired pixels
        **kwargs: list of keywords from getPixelPhotonList()

        Returns
        -------
        List of Structured Numpy Arrays
            The ith element contains a photon list for the ith pixel specified in posList
            Within each structured array:
                Each row is a photon.
                Columns have the following keys: 'Time', 'Wavelength', 'SpecWeight', 'NoiseWeight'

        """

        photonLists = []
        nPix = np.shape(posList)[0]
        for i in range(nPix):
            photonLists.append(self.getPixelPhotonList(posList[i][0], posList[i][1], **kwargs))

        return photonLists



    def getPixelCount(self, *args, applyWeight=True, applyTPFWeight=True, applyTimeMask=False, **kwargs):
        """
        Returns the number of photons received in a single pixel from firstSec to firstSec + integrationTime

        Parameters
        ----------
        *args: args from getPixelPhotonList 
            eg. xCoord, yCoord
        applyWeight: bool
            If True, applies the spectral/flat/linearity weight
        applyTPFWeight: bool
            If True, applies the true positive fraction (noise) weight
        applyTimeMask: bool
            If True, applies the included time mask (if it exists)
        **kwargs: keywords from getPixelPhotonList
            eg. firstSec, wvlStart

        Returns
        -------
        Dictionary with keys:
            'counts':int, number of photon counts
            'effIntTime':float, effective integration time after time-masking is
           `          accounted for.
        """
        try: applyWvl=not kwargs['forceRawPhase']
        except KeyError: applyWvl=False
        try: xCoord = kwargs['xCoord']
        except KeyError: xCoord=args[0]
        try: yCoord = kwargs['yCoord']
        except KeyError: yCoord=args[1]
        if self.pixelIsBad(xCoord, yCoord, forceWvl=applyWvl, forceWeights=applyWeight, forceTPFWeights=applyTPFWeight):
            return 0, 0.0
        
        photonList = self.getPixelPhotonList(*args, **kwargs)
        
        weights = np.ones(len(photonList))
        if applyWeight:
            weights *= photonList['SpecWeight']
        if applyTPFWeight:
            weights *= photonList['NoiseWeight']
        
        try: firstSec = kwargs['firstSec']
        except KeyError: firstSec=0
        try: intTime = kwargs['integrationTime']
        except KeyError: intTime=self.getFromHeader('expTime')
        intTime-=firstSec
        if applyTimeMask:
            raise NotImplementedError
            if self.info['timeMaskExists']:
                pass
                #Apply time mask to photon list
                #update intTime
            else:
                warnings.warn('Time mask does not exist!')

        return {'counts':np.sum(weights), 'effIntTime':intTime}


    def getPixelLightCurve(self,*args,lastSec=-1, cadence=1,scaleByEffInt=True, **kwargs):
        """
        Get a simple light curve for a pixel (basically a wrapper for getPixelCount).

        INPUTS:
            *args: args from getPixelCount
                 xCoord, yCoord
            lastSec: float
                 - end time (sec) within obsFile for the light curve. If -1, returns light curve to end of file.
            cadence: float
                 - cadence (sec) of light curve. i.e., return values integrated every 'cadence' seconds.
            scaleByEffInt: bool
                 - Scale by the pixel's effective integration time
            **kwargs: keywords for getPixelCount, including:
                firstSec
                applyWeight
                wvlStart
                Note: if integrationTime is given as a keyword it will be ignored


        OUTPUTS:
            A single one-dimensional array of flux counts integrated every 'cadence' seconds
            between firstSec and lastSec. Note if step is non-integer may return inconsistent
            number of values depending on rounding of last value in time step sequence (see
            documentation for numpy.arange() ).
        """
        if lastSec==-1:lastSec = self.getFromHeader('expTime')
        try: firstSec=kwargs['firstSec']
        except KeyError: firstSec=0
        if 'integrationTime' in kwargs.keys(): warnings.warn("Integration time is being set to keyword 'cadence'")
        kwargs['integrationTime']=cadence
        data=[self.getPixelCount(*args,**kwargs) for x in np.arange(firstSec,lastSec,cadence)]
        if scaleByEffInt:
            return np.asarray([1.0*x['counts']/x['effIntTime'] for x in data])
        else:
            return np.asarray([x['counts'] for x in data])


    def getPixelCountImage(self, firstSec=0, integrationTime= -1, wvlStart=None,wvlStop=None,
                                 applyWeight=True, applyTPFWeight=True, applyTimeMask=False, 
                                 scaleByEffInt=False, flagToUse=0):
        """
        Returns an image of pixel counts over the entire array between firstSec and firstSec + integrationTime. Can specify calibration weights to apply as
        well as wavelength range.
        
        Does NOT loop over getPixelCount() because it's too slow.
        Instead, grab all the photon data from the H5 file and sort it into pixels with python.

        Parameters
        ----------
        firstSec: float
            Photon list start time, in seconds relative to beginning of file
        integrationTime: float
            Photon list end time, in seconds relative to firstSec.
            If -1, goes to end of file
        wvlRange: (float, float)
            Desired wavelength range of photon list. Must satisfy wvlRange[0] < wvlRange[1].
            If None, includes all wavelengths. If file is not wavelength calibrated, this parameter
            specifies the range of desired phase heights.
        applyWeight: bool
            If True, applies the spectral/flat/linearity weight
        applyTPFWeight: bool
            If True, applies the true positive fraction (noise) weight
        applyTimeMask: bool
            If True, applies the included time mask (if it exists)
        scaleByEffInt: bool
            If True, scales each pixel by (total integration time)/(effective integration time)
        flagToUse: int
            Specifies (bitwise) pixel flags that are suitable to include in image. For
            flag definitions see 'h5FileFlags' in Headers/pipelineFlags.py

        Returns
        -------
        Dictionary with keys:
            'image': 2D numpy array, image of pixel counts
            'effIntTimes':2D numpy array, image effective integration times after time-masking is
           `          accounted for.
        """
        if integrationTime == -1:
            totInt = self.getFromHeader('expTime')-firstSec
        else:
            totInt = integrationTime
        effIntTimes = np.zeros((self.nXPix, self.nYPix), dtype=np.float64)  #default is zero for bad pixel
        #effIntTimes.fill(np.nan)   #Just in case an element doesn't get filled for some reason.
        countImage = np.zeros((self.nXPix, self.nYPix), dtype=np.float64)
        countImage.fill(np.nan)     #default count value is np.nan if it's a bad pixel
        #rawCounts.fill(np.nan)   #Just in case an element doesn't get filled for some reason.
        
        #if integrationTime==-1:
        #    integrationTime = self.getFromHeader('expTime')-firstSec
        #startTs = firstSec*1.e6
        #endTs = startTs + integrationTime*1.e6
        #if wvlRange is None:
        #    photonList = self.photonTable.read_where('((Time >= startTs) & (Time < endTs))')
        #else:
        #    startWvl = wvlRange[0]
        #    endWvl = wvlRange[1]
        #    photonList = self.photonTable.read_where('(Wavelength >= startWvl) & (Wavelength < endWvl) & (Time >= startTs) & (Time < endTs)')
        startTime = int(firstSec*self.ticksPerSec) #convert to us
        query='(Time >= startTime)'
        if integrationTime!=-1:
            endTime = startTime + int(integrationTime*self.ticksPerSec)
            query+=' & (Time < endTime)'
        if wvlStart is not None and wvlStop is not None and wvlStart==wvlStop:
            wvl=startWvl
            query+=' & (Wavelength == wvl)'
        else:
            if wvlStart is not None:
                startWvl=wvlStart
                query+=' & (Wavelength >= startWvl)'
            if wvlStop is not None:
                stopWvl=wvlStop
                query+=' & (Wavelength < stopWvl)'
        photonList = self.photonTable.read_where(query)

        resIDDiffs = np.diff(photonList['ResID'])
        if(np.any(resIDDiffs<0)):
            warnings.warn('Photon list not sorted by ResID! This could take a while...')
            photonList = np.sort(photonList, order='ResID', kind='mergsort') #mergesort is stable, so time order will be preserved
            resIDDiffs = np.diff(photonList['ResID'])
        
        resIDBoundaryInds = np.where(resIDDiffs>0)[0]+1 #indices in photonList where ResID changes; ie marks boundaries between pixel tables
        resIDBoundaryInds = np.insert(resIDBoundaryInds, 0, 0)
        resIDList = photonList['ResID'][resIDBoundaryInds]
        resIDBoundaryInds = np.append(resIDBoundaryInds, len(photonList['ResID']))

        for xCoord in range(self.nXPix): 
            for yCoord in range(self.nYPix): 
                flag = self.beamFlagImage[xCoord, yCoord]
                if(self.beamImage[xCoord, yCoord]!=self.noResIDFlag and (flag|flagToUse)==flagToUse):
                    effIntTimes[xCoord, yCoord] = totInt
                    countImage[xCoord, yCoord] = 0
                    resIDInd = np.where(resIDList==self.beamImage[xCoord, yCoord])[0]
                    if(np.shape(resIDInd)[0]>0):
                        resIDInd = resIDInd[0]
                        if applyWeight==False and applyTPFWeight==False:
                            countImage[xCoord, yCoord] = resIDBoundaryInds[resIDInd+1] - resIDBoundaryInds[resIDInd]
                        else:
                            weights = np.ones(resIDBoundaryInds[resIDInd+1] - resIDBoundaryInds[resIDInd])
                            if applyWeight:
                                weights *= photonList['SpecWeight'][resIDBoundaryInds[resIDInd]:resIDBoundaryInds[resIDInd+1]]
                            if applyTPFWeight:
                                weights *= photonList['NoiseWeight'][resIDBoundaryInds[resIDInd]:resIDBoundaryInds[resIDInd+1]] 
                            countImage[xCoord, yCoord] = np.sum(weights)

        #for i,resID in enumerate(resIDList):
        #    coords = np.where(self.beamFlagImage==resID)
        #    xCoord = coords[0][0]
        #    yCoord = coords[1][0]
        #    flag = self.beamFlagImage[coords]
        #    if (flag|flagToUse)==flag:
        #        if applyWeight==False and applyTPFWeight==False:
                    
                
        if scaleByEffInt:
            countImage *= (totInt / effIntTimes)

        #if getEffInt is True:
        return{'image':countImage, 'effIntTimes':effIntTimes}
        #else:
        #    return secImg

    def getCircularAperturePhotonList(self, centerXCoord, centerYCoord, radius, 
                                      firstSec=0, integrationTime=-1, wvlStart=None,
                                      wvlStop=None, flagToUse=0):
        """
        Retrieves a photon list for the specified circular aperture.
        For pixels that partially overlap with the region, all photons
        are included, and the overlap fraction is multiplied into the
        'NoiseWeight' column.

        Parameters
        ----------
        centerXCoord: float
            x-coordinate of aperture center (pixel units)
        centerYCoord: float
            y-coordinate of aperture center (pixel units)
        radius: float
            radius of aperture
        firstSec: float
            Photon list start time, in seconds relative to beginning of file
        integrationTime: float
            Photon list end time, in seconds relative to firstSec.
            If -1, goes to end of file
        wvlRange: (float, float)
            Desired wavelength range of photon list. Must satisfy wvlRange[0] <= wvlRange[1].
            If None, includes all wavelengths.
        flagToUse: int
            Specifies (bitwise) pixel flags that are suitable to include in photon list. For
            flag definitions see 'h5FileFlags' in Headers/pipelineFlags.py

        Returns
        -------
        Dictionary with keys:
            photonList: numpy structured array
                Time ordered photon list. Adds resID column to keep track
                of individual pixels
            effQE: float
                Fraction of usable pixel area inside aperture
            apertureMask: numpy array
                Image of effective pixel weight inside aperture. "Pixel weight"
                for now is just the area of overlap w/ aperture, with dead
                pixels set to 0.

        """

        center = PixCoord(centerXCoord, centerYCoord)
        apertureRegion = CirclePixelRegion(center, radius)
        exactApertureMask = apertureRegion.to_mask('exact').data
        boolApertureMask = exactApertureMask>0
        apertureMaskCoords = np.transpose(np.array(np.where(boolApertureMask))) #valid coordinates within aperture mask
        photonListCoords = apertureMaskCoords + np.array([apertureRegion.bounding_box.ixmin, apertureRegion.bounding_box.iymin]) #pixel coordinates in image

        # loop through valid coordinates, grab photon lists and store in photonList
        photonList = None
        for i,coords in enumerate(photonListCoords):
            if coords[0]<0 or coords[0]>=self.nXPix or coords[1]<0 or coords[1]>=self.nYPix:
                exactApertureMask[apertureMaskCoords[i,0], apertureMaskCoords[i,1]] = 0
                continue
            flag = self.beamFlagImage[coords[0], coords[1]]
            if (flag | flagToUse) != flagToUse:
                exactApertureMask[apertureMaskCoords[i,0], apertureMaskCoords[i,1]] = 0
                continue
            resID = self.beamImage[coords[0], coords[1]]
            pixPhotonList = self.getPixelPhotonList(coords[0], coords[1], firstSec, integrationTime, wvlStart, wvlStop)
            pixPhotonList['NoiseWeight'] *= exactApertureMask[apertureMaskCoords[i,0], apertureMaskCoords[i,1]]
            if photonList is None:
                photonList = pixPhotonList
            else:
                photonList = np.append(photonList, pixPhotonList)

        photonList = np.sort(photonList, order='Time')
        return photonList, exactApertureMask


    def _makePixelSpectrum(self, photonList, **kwargs):
        """
        Makes a histogram using the provided photon list
        """
        applySpecWeight = kwargs.pop('applySpecWeight', False)
        applyTPFWeight = kwargs.pop('applyTPFWeight', False)
        wvlStart = kwargs.pop('wvlStart', None)
        wvlStop = kwargs.pop('wvlStop', None)
        wvlBinWidth = kwargs.pop('wvlBinWidht', None)
        energyBinWidth = kwargs.pop('energyBinWidth', None)
        wvlBinEdges = kwargs.pop('wvlBinEdges', None)
        timeSpacingCut = kwargs.pop('timeSpacingCut', None)
        
        wvlStart=wvlStart if (wvlStart!=None and wvlStart>0.) else (self.wvlLowerLimit if (self.wvlLowerLimit!=None and self.wvlLowerLimit>0.) else 700)
        wvlStop=wvlStop if (wvlStop!=None and wvlStop>0.) else (self.wvlUpperLimit if (self.wvlUpperLimit!=None and self.wvlUpperLimit>0.) else 1500)


        wvlList = photonList['Wavelength']
        rawCounts = len(wvlList)

        weights = np.ones(len(wvlList))

        if applySpecWeight:
            weights *= photonList['SpecWeight']

        if applyTPFWeight:
            weights *= photonList['NoiseWeight']

        if (wvlBinWidth is None) and (energyBinWidth is None) and (wvlBinEdges is None): #use default/flat cal supplied bins
            spectrum, wvlBinEdges = np.histogram(wvlList, bins=self.defaultWvlBins, weights=weights)

        else: #use specified bins
            if applySpecWeight and self.info['isFlatCalibrated']:
                raise ValueError('Using flat cal, so flat cal bins must be used')
            elif wvlBinEdges is not None:
                assert wvlBinWidth is None and energyBinWidth is None, 'Histogram bins are overspecified!'
                spectrum, wvlBinEdges = np.histogram(wvlList, bins=wvlBinEdges, weights=weights)
            elif energyBinWidth is not None:
                assert wvlBinWidth is None, 'Cannot specify both wavelength and energy bin widths!'
                wvlBinEdges = ObsFile.makeWvlBins(energyBinWidth=energyBinWidth, wvlStart=wvlStart, wvlStop=wvlStop)
                spectrum, wvlBinEdges = np.histogram(wvlList, bins=wvlBinEdges, weights=weights)
            elif wvlBinWidth is not None:
                nWvlBins = int((wvlStop - wvlStart)/wvlBinWidth)
                spectrum, wvlBinEdges = np.histogram(wvlList, bins=nWvlBins, range=(wvlStart, wvlStop), weights=weights)

            else:
                raise Exception('Something is wrong with getPixelSpectrum...')

        if self.filterIsApplied == True:
            if not np.array_equal(self.filterWvlBinEdges, wvlBinEdges):
                raise ValueError("Synthetic filter wvlBinEdges do not match pixel spectrum wvlBinEdges!")
            spectrum*=self.filterTrans
        #if getEffInt is True:
        return {'spectrum':spectrum, 'wvlBinEdges':wvlBinEdges, 'rawCounts':rawCounts}


    def getSpectralCube(self, firstSec=0, integrationTime=-1, applySpecWeight=False, applyTPFWeight=False, wvlStart=700, wvlStop=1500,
                        wvlBinWidth=None, energyBinWidth=None, wvlBinEdges=None, timeSpacingCut=None, flagToUse=0):
        """
        Return a time-flattened spectral cube of the counts integrated from firstSec to firstSec+integrationTime.
        If integration time is -1, all time after firstSec is used.
        If weighted is True, flat cal weights are applied.
        If fluxWeighted is True, spectral shape weights are applied.
        """

        cube = [[[] for yCoord in range(self.nYPix)] for xCoord in range(self.nXPix)]
        effIntTime = np.zeros((self.nXPix,self.nYPix))
        rawCounts = np.zeros((self.nXPix,self.nYPix))
        if integrationTime==-1:
            integrationTime = self.getFromHeader('expTime')
        
        startTime = firstSec*1.e6
        endTime = (firstSec + integrationTime)*1.e6

        masterPhotonList = self.photonTable.read_where('(Time>=startTime)&(Time<endTime)')
        emptyPhotonList = self.photonTable.read_where('Time<0')
        
        resIDDiffs = np.diff(masterPhotonList['ResID'])
        if(np.any(resIDDiffs<0)):
            warnings.warn('Photon list not sorted by ResID! This could take a while...')
            masterPhotonList = np.sort(masterPhotonList, order='ResID', kind='mergsort') #mergesort is stable, so time order will be preserved
            resIDDiffs = np.diff(masterPhotonList['ResID'])
        
        resIDBoundaryInds = np.where(resIDDiffs>0)[0]+1 #indices in masterPhotonList where ResID changes; ie marks boundaries between pixel tables
        resIDBoundaryInds = np.insert(resIDBoundaryInds, 0, 0)
        resIDList = masterPhotonList['ResID'][resIDBoundaryInds]
        resIDBoundaryInds = np.append(resIDBoundaryInds, len(masterPhotonList['ResID']))

        for xCoord in range(self.nXPix):    
            for yCoord in range(self.nYPix):  
                resID = self.beamImage[xCoord, yCoord]
                flag = self.beamFlagImage[xCoord, yCoord]
                resIDInd = np.where(resIDList==resID)[0]
                if(np.shape(resIDInd)[0]>0 and (flag|flagToUse)==flagToUse):
                    resIDInd = resIDInd[0]
                    photonList = masterPhotonList[resIDBoundaryInds[resIDInd]:resIDBoundaryInds[resIDInd+1]]
                else:
                    photonList = emptyPhotonList
                x = self._makePixelSpectrum(photonList, applySpecWeight=applySpecWeight,
                                  applyTPFWeight=applyTPFWeight, wvlStart=wvlStart, wvlStop=wvlStop,
                                  wvlBinWidth=wvlBinWidth, energyBinWidth=energyBinWidth,
                                  wvlBinEdges=wvlBinEdges, timeSpacingCut=timeSpacingCut)
                cube[xCoord][yCoord] = x['spectrum']
                effIntTime[xCoord][yCoord] = integrationTime
                rawCounts[xCoord][yCoord] = x['rawCounts']
                wvlBinEdges = x['wvlBinEdges']
        cube = np.array(cube)
        return {'cube':cube,'wvlBinEdges':wvlBinEdges,'effIntTime':effIntTime, 'rawCounts':rawCounts}

    def getPixelSpectrum(self, xCoord, yCoord, firstSec=0, integrationTime= -1,
                         applySpecWeight=False, applyTPFWeight=False, wvlStart=None, wvlStop=None,
                         wvlBinWidth=None, energyBinWidth=None, wvlBinEdges=None,timeSpacingCut=None):
        """
        returns a spectral histogram of a given pixel integrated from firstSec to firstSec+integrationTime,
        and an array giving the cutoff wavelengths used to bin the wavelength values

        Wavelength Bin Specification:
        Depends on parameters: wvlStart, wvlStop, wvlBinWidth, energyBinWidth, wvlBinEdges.
        Can only specify one of: wvlBinWidth, energyBinWidth, or wvlBinEdges. If none of these are specified,
        default wavelength bins are used. If flat calibration exists and is applied, flat cal wavelength bins
        must be used.

        Parameters
        ----------
        xCoord: int
            x-coordinate of desired pixel.
        yCoord: int
            y-coordinate index of desired pixel.
        firstSec: float
            Start time of integration, in seconds relative to beginning of file
        integrationTime: float
            Total integration time in seconds. If -1, everything after firstSec is used
        applySpecWeight: bool
            If True, weights counts by spectral/flat/linearity weight
        applyTPFWeight: bool
            If True, weights counts by true positive fraction (noise weight)
        wvlStart: float
            Start wavelength of histogram. Only used if wvlBinWidth or energyBinWidth is
            specified (otherwise it's redundant). Defaults to self.wvlLowerLimit or 7000.
        wvlEnd: float
            End wavelength of histogram. Only used if wvlBinWidth or energyBinWidth is
            specified. Defaults to self.wvlUpperLimit or 15000
        wvlBinWidth: float
            Width of histogram wavelength bin. Used for fixed wavelength bin widths.
        energyBinWidth: float
            Width of histogram energy bin. Used for fixed energy bin widths.
        wvlBinEdges: numpy array of floats
            Specifies histogram wavelength bins. wvlStart and wvlEnd are ignored.
        timeSpacingCut: ????
            Legacy; unused


        Returns
        -------
        Dictionary with keys:
            'spectrum' - spectral histogram of given pixel.
            'wvlBinEdges' - edges of wavelength bins
            'effIntTime' - the effective integration time for the given pixel
                           after accounting for hot-pixel time-masking.
            'rawCounts' - The total number of photon triggers (including from
                            the noise tail) during the effective exposure.
        """

        photonList = self.getPixelPhotonList(xCoord, yCoord, firstSec, integrationTime)
        return self._makePixelSpectrum(photonList, applySpecWeight=applySpecWeight,
                                  applyTPFWeight=applyTPFWeight, wvlStart=wvlStart, wvlStop=wvlStop,
                                  wvlBinWidth=wvlBinWidth, energyBinWidth=energyBinWidth,
                                  wvlBinEdges=wvlBinEdges, timeSpacingCut=timeSpacingCut)
        #else:
        #    return spectrum,wvlBinEdges

    
    def loadBeammapFile(self,beammapFileName):
        """
        Load an external beammap file in place of the obsfile's attached beammap
        Can be used to correct pixel location mistakes.
        
        Make sure the beamflag image is correctly transformed to the new beammap
        """
        raise NotImplementedError

    def loadBestWvlCalFile(self,master=True):
        """
        Searchs the waveCalSolnFiles directory tree for the best wavecal to apply to this obsfile.
        if master==True then it first looks for a master wavecal solution
        """
        raise NotImplementedError


    def applyWaveCal(self, file_name):
        """
        loads the wavelength cal coefficients from a given file and applies them to the
        wavelengths table for each pixel. ObsFile must be loaded in write mode.
        """
        # check file_name and status of obsFile
        assert not self.info['isWvlCalibrated'], \
            "the data is already wavelength calibrated"
        assert os.path.exists(file_name), "{0} does not exist".format(file_name)
        wave_cal = tables.open_file(file_name, mode='r')

        self.photonTable.autoindex = False # Don't reindex everytime we change column

        try:
            # appy waveCal
            calsoln = wave_cal.root.wavecal.calsoln.read()
            for (row, column), resID in np.ndenumerate(self.beamImage):
                index = np.where(resID == np.array(calsoln['resid']))
                if len(index[0]) == 1 and (calsoln['wave_flag'][index] == 4 or
                                           calsoln['wave_flag'][index] == 5):
                    poly = calsoln['polyfit'][index]
                    photon_list = self.getPixelPhotonList(row, column)
                    phases = photon_list['Wavelength']
                    poly = np.array(poly)
                    poly = poly.flatten()
                    energies = np.polyval(poly, phases)
                    wavelengths = self.h * self.c / energies * 1e9  # wavelengths in nm
                    self.updateWavelengths(row, column, wavelengths)
                else:
                    self.applyFlag(row, column, 0b00000010)  # failed waveCal
            self.modifyHeaderEntry(headerTitle='isWvlCalibrated', headerValue=True)
            self.modifyHeaderEntry(headerTitle='wvlCalFile',headerValue=str.encode(file_name))
        finally:
            self.photonTable.reindex_dirty() # recompute "dirty" wavelength index
            self.photonTable.autoindex = True # turn on autoindexing 
            wave_cal.close()


    @staticmethod
    def makeWvlBins(energyBinWidth=.1, wvlStart=700, wvlStop=1500):
        """
        returns an array of wavlength bin edges, with a fixed energy bin width
        withing the limits given in wvlStart and wvlStop
        Args:
            energyBinWidth: bin width in eV
            wvlStart: Lower wavelength edge in Angstrom
            wvlStop: Upper wavelength edge in Angstrom
        Returns:
            an array of wavelength bin edges that can be used with numpy.histogram(bins=wvlBinEdges)
        """

        #Calculate upper and lower energy limits from wavelengths
        #Note that start and stop switch when going to energy
        energyStop = ObsFile.h * ObsFile.c * 1.e9 / wvlStart
        energyStart = ObsFile.h * ObsFile.c * 1.e9 / wvlStop
        nWvlBins = int((energyStop - energyStart) / energyBinWidth)
        #Construct energy bin edges
        energyBins = np.linspace(energyStart, energyStop, nWvlBins + 1)
        #Convert back to wavelength and reverse the order to get increasing wavelengths
        wvlBinEdges = np.array(ObsFile.h * ObsFile.c * 1.e9 / energyBins)
        wvlBinEdges = wvlBinEdges[::-1]
        return wvlBinEdges

    def maskTimestamps(self,timestamps,inter=interval(),otherListsToFilter=[]):
        """
        Masks out timestamps that fall in an given interval
        inter is an interval of time values to mask out
        otherListsToFilter is a list of parallel arrays to timestamps that should be masked in the same way
        returns a dict with keys 'timestamps','otherLists'
        """
        # first special case:  inter masks out everything so return zero-length
        # numpy arrays
        if (inter == self.intervalAll):
            filteredTimestamps = np.arange(0)
            otherLists = [np.arange(0) for list in otherListsToFilter]
        else:
            if inter == interval() or len(timestamps) == 0:
                # nothing excluded or nothing to exclude
                # so return all unpacked values
                filteredTimestamps = timestamps
                otherLists = otherListsToFilter
            else:
                # there is a non-trivial set of times to mask.
                slices = calculateSlices(inter, timestamps)
                filteredTimestamps = repackArray(timestamps, slices)
                otherLists = []
                for eachList in otherListsToFilter:
                    filteredList = repackArray(eachList,slices)
                    otherLists.append(filteredList)
        # return the values filled in above
        return {'timestamps':filteredTimestamps,'otherLists':otherLists}


    def setWvlCutoffs(self, wvlLowerLimit=700, wvlUpperLimit=1500):
        """
        Sets wavelength cutoffs so that if convertToWvl(excludeBad=True) or getPixelWvlList(excludeBad=True) is called
        wavelengths outside these limits are excluded.  To remove limits
        set wvlLowerLimit and/or wvlUpperLimit to None.  To use the wavecal limits
        for each individual pixel, set wvlLowerLimit and/or wvlUpperLimit to -1
        NB - changed defaults for lower/upper limits to None (from 3000 and 8000). JvE 2/22/13
        """
        self.wvlLowerLimit = wvlLowerLimit
        self.wvlUpperLimit = wvlUpperLimit

    def switchOffHotPixTimeMask(self):
        """
        Switch off hot pixel time masking - bad pixel times will no longer be
        removed (although the mask remains 'loaded' in ObsFile instance).
        """
        #self.hotPixIsApplied = False
        raise NotImplementedError

    def switchOnHotPixTimeMask(self,reasons=[]):
        """
        Switch on hot pixel time masking. Subsequent calls to getPixelCountImage
        etc. will have bad pixel times removed.
        """
        raise NotImplementedError
        if self.hotPixTimeMask is None:
            raise RuntimeError('No hot pixel file loaded')
        self.hotPixIsApplied = True
        if len(reasons)>0:
            self.hotPixTimeMask.set_mask(reasons)
            #self.hotPixTimeMask.mask = [self.hotPixTimeMask.reasonEnum[reason] for reason in reasons]

    def updateWavelengths(self, xCoord, yCoord, wvlCalArr):
        """
        Changes wavelengths for a single pixel. Overwrites "Wavelength" column w/
        contents of wvlCalArr. NOT reversible unless you have a copy of the original contents.
        ObsFile must be open in "write" mode to use.

        Parameters
        ----------
        resID: int
            resID of pixel to overwrite
        wvlCalArr: array of floats
            Array of calibrated wavelengths. Replaces "Wavelength" column of this pixel's
            photon list.
        """
        resID = self.beamImage[xCoord][yCoord]
        if self.mode!='write':
            raise Exception("Must open file in write mode to do this!")
        if self.info['isWvlCalibrated']:
            warnings.warn("Wavelength calibration already exists!")
        pixelRowInds = self.photonTable.get_where_list('ResID==resID')
        assert len(pixelRowInds) == 0 or (len(pixelRowInds)-1)==(pixelRowInds[-1]-pixelRowInds[0]), 'Table is not sorted by Res ID!'
        assert len(pixelRowInds)==len(wvlCalArr), 'Calibrated wavelength list does not match length of photon list!'
        
        if len(pixelRowInds)>0:
            self.photonTable.modify_column(start=pixelRowInds[0], stop=pixelRowInds[-1]+1, column=wvlCalArr, colname='Wavelength')
            self.photonTable.flush()

    def __applyColWeight(self, resID, weightArr, colName):
        """
        Applies a weight calibration to the column specified by colName.
        Call using applySpecWeight or applyTPFWeight.

        Parameters
        ----------
        resID: int
            resID of desired pixel
        weightArr: array of floats
            Array of cal weights. Multiplied into the "SpecWeight" column.
        colName: string
            Name of weight column. Should be either 'SpecWeight' or 'NoiseWeight'
        """
        if self.mode!='write':
            raise Exception("Must open file in write mode to do this!")
        if self.info['isWvlCalibrated']:
            warnings.warn("Wavelength calibration already exists!")
        pixelRowInds = self.photonTable.get_where_list('ResID==resID')
        assert (len(pixelRowInds)-1)==(pixelRowInds[-1]-pixelRowInds[0]), 'Table is not sorted by Res ID!'
        assert len(pixelRowInds)==len(weightArr), 'Calibrated wavelength list does not match length of photon list!'

        weightArr = np.array(weightArr)
        curWeights = self.photonTable.read_where('resID==ResID')['SpecWeight']
        newWeights = weightArr*curWeights
        self.photonTable.modify_column(start=pixelRowInds[0], stop=pixelRowInds[-1]+1, column=newWeights, colname=colName)
        self.photonTable.flush()

    def applySpecWeight(self, resID, weightArr):
        """
        Applies a weight calibration to the "SpecWeight" column of a single pixel.

        This is where the flat cal, linearity cal, and spectral cal go.
        Weights are multiplied in and replaced; if "weights" are the contents
        of the "SpecWeight" column, weights = weights*weightArr. NOT reversible
        unless the original contents (or weightArr) is saved.

        Parameters
        ----------
        resID: int
            resID of desired pixel
        weightArr: array of floats
            Array of cal weights. Multiplied into the "SpecWeight" column.
        """
        self.__applyColWeight(resID, weightArr, 'SpecWeight')
    
    def applyTPFWeight(self, resID, weightArr):
        """
        Applies a weight calibration to the "SpecWeight" column.

        This is where the TPF (noise weight) goes.
        Weights are multiplied in and replaced; if "weights" are the contents
        of the "NoiseWeight" column, weights = weights*weightArr. NOT reversible
        unless the original contents (or weightArr) is saved.

        Parameters
        ----------
        resID: int
            resID of desired pixel
        weightArr: array of floats
            Array of cal weights. Multiplied into the "NoiseWeight" column.
        """
        self.__applyColWeight(resID, weightArr, 'NoiseWeight')

    def applyFlatCal(self, calSolnPath,verbose=False):
        """
        Applies a flat calibration to the "SpecWeight" column of a single pixel.

        Weights are multiplied in and replaced; if "weights" are the contents
        of the "SpecWeight" column, weights = weights*weightArr. NOT reversible
        unless the original contents (or weightArr) is saved.

        Parameters
        ----------
        calSolnPath: string
            Path to the location of the FlatCal solution files
            should contain the base filename of these solution files
            e.g '/mnt/data0/isabel/DeltaAnd/Flats/DeltaAndFlatSoln.h5')
            Code will grab all files titled DeltaAndFlatSoln#.h5 from that directory
        if verbose: will print resID, row, and column of pixels which have a successful FlatCal
                    will print averaged flat weights of pixels which have a successful FlatCal.
        Will write plots of flatcal solution (5 second increments over a single flat exposure) 
             with average weights overplotted to a pdf for pixels which have a successful FlatCal.
             Written to the calSolnPath+'FlatCalSolnPlotsPerPixel.pdf'
        """
        baseh5path=calSolnPath.split('.h5')
        flatList=glob.glob(baseh5path[0]+'*.h5')     
        assert os.path.exists(flatList[0]), "{0} does not exist".format(flatList[0])  
        assert not self.info['isSpecCalibrated'], \
               "the data is already Flat calibrated"
        pdfFullPath = baseh5path[0]+'FlatCalSolnPlotsPerPixel.pdf'
        pp = PdfPages(pdfFullPath)
        nPlotsPerRow = 2
        nPlotsPerCol = 4
        nPlotsPerPage = nPlotsPerRow*nPlotsPerCol
        iPlot = 0 
        matplotlib.rcParams['font.size'] = 4 
        for (row, column), resID in np.ndenumerate(self.beamImage):
                photon_list = self.getPixelPhotonList(row, column)
                phases = photon_list['Wavelength']
                calarray=[]
                weightarray=[]
                weightarrayUncertainty=[]
                minwavelength=700
                maxwavelength=1500
                heads=np.arange(0,700,100)
                tails=np.arange(1600,2100,100)
                headsweight=np.array([0,0,0,0,0,0,0])+1
                tailsweight=np.array([0,0,0,0,0,0])+1
                for FlatCalFile in flatList:
                      flat_cal = tables.open_file(FlatCalFile, mode='r')
                      calsoln = flat_cal.root.flatcal.calsoln.read()
                      bins=np.array(flat_cal.root.flatcal.wavelengthBins.read())
                      bins=bins.flatten()
                      bins=np.append(heads,bins)
                      bins=np.append(bins,tails)
                      index = np.where(resID == np.array(calsoln['resid']))
                      #if len(index[0]) == 1 and (calsoln['flag'][index] == 0):
                      if len(index[0]) == 1 and not self.pixelIsBad(row, column):
                           print('resID', resID, 'row', row, 'column', column)
                           weights = calsoln['weights'][index]
                           print(weights)
                           weightFlags=calsoln['weightFlags'][index]
                           weightUncertainties=calsoln['weightUncertainties'][index]

                           weights=np.array(weights)
                           weights=weights.flatten()
                           weightsheads=np.zeros(len(heads))+1
                           weightstails=np.zeros(len(tails)+1)+1
                           weights=np.append(weightsheads,weights)
                           weights=np.append(weights,weightstails)
                           weightarray.append(weights)

                           weightUncertainties=np.array(weightUncertainties)
                           weightUncertainties=weightUncertainties.flatten()
                           weightUncertainties=np.append(headsweight,weightUncertainties)
                           weightUncertainties=np.append(weightUncertainties,tailsweight)
                           weightarrayUncertainty.append(weightUncertainties)

                           weightfxncoeffs10=np.polyfit(bins,weights,10)
                           weightfxn10=np.poly1d(weightfxncoeffs10)

                           weightArr=weightfxn10(phases)                    
                           weightArr[np.where(phases < minwavelength)]=0.0
                           weightArr[np.where(phases > maxwavelength)]=0.0
                           calarray.append(weightArr)       
                           
                if calarray and calarray[0].all: 
                     print('hiiiiiiiiiiiiiiiiiiiiii')
                     calweights = np.average(calarray,axis=0)
                     self.applySpecWeight(resID=resID, weightArr=calweights)
                     if verbose:
                        print('CALWEIGHTS', calweights)
                        print('resID', resID, 'row', row, 'column', column)
                        if iPlot % nPlotsPerPage == 0:
                           fig = plt.figure(figsize=(10,10),dpi=100)
                        ax = fig.add_subplot(nPlotsPerCol,nPlotsPerRow,iPlot%nPlotsPerPage+1)
                        ax.set_ylim(0,5)
                        ax.set_xlim(minwavelength,maxwavelength)
                        for i in range(len(weightarray)):
                            ax.plot(bins,weightarray[i],'-',label='weights')
                            ax.errorbar(bins,weightarray[i],yerr=weightarrayUncertainty[i],label='weights')
                        ax.plot(phases, calweights, '.', markersize=5)
                        ax.set_title('p %d,%d'%(row,column))
                        ax.set_ylabel('weight')
                        #ax.set_xlabel(r'$\lambda$ ($\AA$)')

                        if iPlot%nPlotsPerPage == nPlotsPerPage-1 or (row == self.nXPix-1 and column == self.nYPix-1):
                           pp.savefig(fig)
                        iPlot += 1
        pp.close()
        self.modifyHeaderEntry(headerTitle='isSpecCalibrated', headerValue=True)

    def applyFlag(self, xCoord, yCoord, flag):
        """
        Applies a flag to the selected pixel on the BeamFlag array. Flag is a bitmask;
        new flag is bitwise OR between current flag and provided flag. Flag definitions
        can be found in Headers/pipelineFlags.py.

        Parameters
        ----------
        xCoord: int
            x-coordinate of pixel
        yCoord: int
            y-coordinate of pixel
        flag: int
            Flag to apply to pixel
        """
        if self.mode!='write':
            raise Exception("Must open file in write mode to do this!")

        curFlag = self.beamFlagImage[xCoord, yCoord]
        newFlag = curFlag | flag
        self.beamFlagImage[xCoord, yCoord] = newFlag
        self.beamFlagImage.flush()

    def undoFlag(self, xCoord, yCoord, flag):
        """
        Resets the specified flag in the BeamFlag array to 0. Flag is a bitmask;
        only the bit specified by 'flag' is reset. Flag definitions
        can be found in Headers/pipelineFlags.py.

        Parameters
        ----------
        xCoord: int
            x-coordinate of pixel
        yCoord: int
            y-coordinate of pixel
        flag: int
            Flag to undo
        """
        if self.mode!='write':
            raise Exception("Must open file in write mode to do this!")

        curFlag = self.beamFlagImage[xCoord, yCoord]
        notFlag = ~flag
        newFlag = curFlag & notFlag
        self.beamFlagImage[xCoord, yCoord] = newFlag
        self.beamFlagImage.flush()

    def modifyHeaderEntry(self, headerTitle, headerValue):
        """
        Modifies an entry in the header. Useful for indicating whether wavelength cals,
        flat cals, etc are applied

        Parameters
        ----------
        headerTitle: string
            Name of entry to be modified
        headerValue: depends on title
            New value of entry
        """
        if self.mode!='write':
            raise Exception("Must open file in write mode to do this!")
        self.header.modify_column(column=headerValue, colname=headerTitle)
        self.header.flush()
        self.info = self.header[0]


def calculateSlices(inter, timestamps):
    '''
    Hopefully a quicker version of  the original calculateSlices. JvE 3/8/2013

    Returns a list of strings, with format i0:i1 for a python array slice
    inter is the interval of values in timestamps to mask out.
    The resulting list of strings indicate elements that are not masked out

    inter must be a single pyinterval 'interval' object (can be multi-component)
    timestamps is a 1D array of timestamps (MUST be an *ordered* array).

    If inter is a multi-component interval, the components must be unioned and sorted
    (which is the default behaviour when intervals are defined, and is probably
    always the case, so shouldn't be a problem).
    '''
    timerange = interval([timestamps[0],timestamps[-1]])
    slices = []
    slce = "0:"     #Start at the beginning of the timestamps array....
    imax = 0        #Will prevent error if inter is an empty interval
    for eachComponent in inter.components:
        #Check if eachComponent of the interval overlaps the timerange of the
        #timestamps - if not, skip to the next component.

        if eachComponent & timerange == interval(): continue
        #[
        #Possibly a bit faster to do this and avoid interval package, but not fully tested:
        #if eachComponent[0][1] < timestamps[0] or eachComponent[0][0] > timestamps[-1]: continue
        #]

        imin = np.searchsorted(timestamps, eachComponent[0][0], side='left') #Find nearest timestamp to lower bound
        imax = np.searchsorted(timestamps, eachComponent[0][1], side='right') #Nearest timestamp to upper bound
        #As long as we're not about to create a wasteful '0:0' slice, go ahead
        #and finish the new slice and append it to the list
        if imin != 0:
            slce += str(imin)
            slices.append(slce)
        slce = str(imax)+":"
    #Finish the last slice at the end of the timestamps array if we're not already there:
    if imax != len(timestamps):
        slce += str(len(timestamps))
        slices.append(slce)
    return slices


def repackArray(array, slices):
    """
    returns a copy of array that includes only the element defined by slices
    """
    nIncluded = 0
    for slce in slices:
        s0 = int(slce.split(":")[0])
        s1 = int(slce.split(":")[1])
        nIncluded += s1 - s0
    retval = np.zeros(nIncluded)
    iPt = 0;
    for slce in slices:
        s0 = int(slce.split(":")[0])
        s1 = int(slce.split(":")[1])
        iPtNew = iPt + s1 - s0
        retval[iPt:iPtNew] = array[s0:s1]
        iPt = iPtNew
    return retval


#Temporary test
if __name__ == "__main__":
    pass
