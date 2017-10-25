#!/bin/python
'''
Author: Matt Strader        Date: August 19, 2012

The class ObsFile is an interface to observation files.  It provides methods for typical ways of accessing and viewing observation data.  It can also load and apply wavelength and flat calibration.  With calibrations loaded, it can write the obs file out as a photon list

Looks for observation files in $MKID_RAW_PATH and calibration files organized in $MKID_PROC_PATH (intermediate or scratch path)

Class Obsfile:
__init__(self, fileName,verbose=False)
__del__(self)
__iter__(self)
loadFile(self, fileName,verbose=False)
checkIntegrity(self,firstSec=0,integrationTime=-1)
convertToWvl(self, pulseHeights, iRow, iCol, excludeBad=True)
createEmptyPhotonListFile(self)
displaySec(self, firstSec=0, integrationTime= -1, weighted=False,fluxWeighted=False, plotTitle='', nSdevMax=2,scaleByEffInt=False)
getFromHeader(self, name)
getPixel(self, iRow, iCol, firstSec=0, integrationTime= -1)
getPixelWvlList(self,iRow,iCol,firstSec=0,integrationTime=-1,excludeBad=True,dither=True)
getPixelCount(self, iRow, iCol, firstSec=0, integrationTime= -1,weighted=False, fluxWeighted=False, getRawCount=False)
getPixelLightCurve(self, iRow, iCol, firstSec=0, lastSec=-1, cadence=1, **kwargs)
getPixelPacketList(self, iRow, iCol, firstSec=0, integrationTime= -1)
getTimedPacketList_old(self, iRow, iCol, firstSec=0, integrationTime= -1)
getTimedPacketList(self, iRow, iCol, firstSec=0, integrationTime= -1)
getPixelCountImage(self, firstSec=0, integrationTime= -1, weighted=False,fluxWeighted=False, getRawCount=False,scaleByEffInt=False)
getAperturePixelCountImage(self, firstSec=0, integrationTime= -1, y_values=range(46), x_values=range(44), y_sky=[], x_sky=[], apertureMask=np.ones((46,44)), skyMask=np.zeros((46,44)), weighted=False, fluxWeighted=False, getRawCount=False, scaleByEffInt=False)
getSpectralCube(self,firstSec=0,integrationTime=-1,weighted=True,wvlStart=3000,wvlStop=13000,wvlBinWidth=None,energyBinWidth=None,wvlBinEdges=None)
getPixelSpectrum(self, pixelRow, pixelCol, firstSec=0, integrationTime= -1,weighted=False, fluxWeighted=False, wvlStart=3000, wvlStop=13000, wvlBinWidth=None, energyBinWidth=None, wvlBinEdges=None)
getPixelBadTimes(self, pixelRow, pixelCol)
getDeadPixels(self, showMe=False, weighted=True, getRawCount=False)
getNonAllocPixels(self, showMe=False)
getRoachNum(self,iRow,iCol)
getFrame(self, firstSec=0, integrationTime=-1)
loadCentroidListFile(self, centroidListFileName)
loadFlatCalFile(self, flatCalFileName)
loadFluxCalFile(self, fluxCalFileName)
loadHotPixCalFile(self, hotPixCalFileName, switchOnMask=True)
loadTimeAdjustmentFile(self,timeAdjustFileName,verbose=False)
loadWvlCalFile(self, wvlCalFileName)
loadFilter(self, filterName = 'V', wvlBinEdges = None,switchOnFilter = True):
makeWvlBins(energyBinWidth=.1, wvlStart=3000, wvlStop=13000)
parsePhotonPackets(self, packets, inter=interval(),doParabolaFitPeaks=True, doBaselines=True)
plotPixelSpectra(self, pixelRow, pixelCol, firstSec=0, integrationTime= -1,weighted=False, fluxWeighted=False)getApertureSpectrum(self, pixelRow, pixelCol, radius1, radius2, weighted=False, fluxWeighted=False, lowCut=3000, highCut=7000,firstSec=0,integrationTime=-1)
plotPixelLightCurve(self,iRow,iCol,firstSec=0,lastSec=-1,cadence=1,**kwargs)
plotApertureSpectrum(self, pixelRow, pixelCol, radius1, radius2, weighted=False, fluxWeighted=False, lowCut=3000, highCut=7000, firstSec=0,integrationTime=-1)
setWvlCutoffs(self, wvlLowerLimit=3000, wvlUpperLimit=8000)
switchOffHotPixTimeMask(self)
switchOnHotPixTimeMask(self, reasons=[])
switchOffFilter(self)
switchOnFilter(self)
writePhotonList(self)

calculateSlices_old(inter, timestamps)
calculateSlices(inter, timestamps)
repackArray(array, slices)
'''

import sys, os
import warnings
import time

import numpy as np
from numpy import vectorize
from numpy import ma
from scipy import pi
import matplotlib.pyplot as plt
from matplotlib.dates import strpdate2num
from interval import interval, inf
import tables
from tables.nodes import filenode
import astropy.constants

from P3Utils import utils
from P3Utils import MKIDStd
from P3Utils.FileName import FileName
from headers import TimeMask

class ObsFile:
    h = astropy.constants.h.to('eV s').value  #4.135668e-15 #eV s
    c = astropy.constants.c.to('m/s').value   #'2.998e8 #m/s
    angstromPerMeter = 1e10
    nCalCoeffs = 3
    def __init__(self, fileName, verbose=False, makeMaskVersion='v2',repeatable=False):
        """
        load the given file with fileName relative to $MKID_RAW_PATH
        """
        self.makeMaskVersion = makeMaskVersion
        self.loadFile(fileName,verbose=verbose)
        self.beammapFileName = None  #Normally the beammap comes directly from the raw obs file itself, so this is only relevant if a new one is loaded with 'loadBeammapFile'.
        self.wvlCalFile = None #initialize to None for an easy test of whether a cal file has been loaded
        self.wvlCalFileName = None
        self.flatCalFile = None
        self.flatCalFileName = None
        self.fluxCalFile = None
        self.fluxCalFileName = None
        self.filterIsApplied = None
        self.filterTrans = None
        self.timeAdjustFile = None
        self.timeAdjustFileName = None
        self.hotPixFile = None
        self.hotPixFileName = None
        self.hotPixTimeMask = None
        self.hotPixIsApplied = False
        self.cosmicMaskIsApplied = False
        self.cosmicMask = None # interval of times to mask cosmic ray events
        self.cosmicMaskFileName = None
        self.centroidListFile = None
        self.centroidListFileName = None
        self.wvlLowerLimit = None
        self.wvlUpperLimit = None

        self.timeMaskExists = False
        

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
            self.timeAdjustFile.close()
        except:
            pass
        try:
            self.hotPixFile.close()
        except:
            pass
        try:
            self.centroidListFile.close()
        except:
            pass
        self.file.close()


    def __iter__(self):
        """
        Allows easy iteration over pixels in obs file
        use with 'for pixel in obsFileObject:'
        yields a single pixel h5 dataset

        MJS 3/28
        Warning: if timeAdjustFile is loaded, the data from this
        function will not be corrected for roach delays as in getPixel().
        Use getPixel() instead.
        """
        for iRow in range(self.nRow):
            for iCol in range(self.nCol):
                pixelLabel = self.beamImage[iRow][iCol]
                pixelData = self.file.get_node('/' + pixelLabel)
                yield pixelData

    def loadFile(self, fileName,verbose=False):
        """
        Opens file and loads obs file attributes and beammap
        """
        if (os.path.isabs(fileName)):
            self.fileName = os.path.basename(fileName)
            self.fullFileName = fileName
        else:
            self.fileName = fileName
            # make the full file name by joining the input name 
            # to the MKID_RAW_PATH (or . if the environment variable 
            # is not defined)
            dataDir = os.getenv('MKID_RAW_PATH', '/')
            self.fullFileName = os.path.join(dataDir, self.fileName)

        if (not os.path.exists(self.fullFileName)):
            msg='file does not exist: %s'%self.fullFileName
            if verbose:
                print(msg)
            raise Exception(msg)
        
        #open the hdf5 file
        self.file = tables.open_file(self.fullFileName, mode='r')

        #get the header
        #no header yet (20171020)
        # self.header = self.file.root.header.header
        # self.titles = self.header.colnames
        # try:
        #     self.info = self.header[0] #header is a table with one row
        # except IndexError as inst:
        #     if verbose:
        #         print 'Can\'t read header for ',self.fullFileName
        #     raise inst

        # Useful information about data format set here.
        # For now, set all of these as constants.
        # If we get data taken with different parameters, straighten
        # that all out here.

        ## These parameters are for LICK2012 and PAL2012 data
        self.tickDuration = 1e-6 #s
        self.ticksPerSec = int(1.0 / self.tickDuration)
        self.intervalAll = interval[0.0, (1.0 / self.tickDuration) - 1]
        self.nonAllocPixelName = '/r0/p250/'
        #  8 bits - channel
        # 12 bits - Parabola Fit Peak Height
        # 12 bits - Sampled Peak Height
        # 12 bits - Low pass filter baseline
        # 20 bits - Microsecond timestamp

        self.nBitsAfterParabolaPeak = 44
        self.nBitsAfterBaseline = 20
        self.nBitsInPulseHeight = 12
        self.nBitsInTimestamp = 20

        #bitmask of 12 ones
        self.pulseMask = int(self.nBitsInPulseHeight * '1', 2) 
        #bitmask of 20 ones
        self.timestampMask = int(self.nBitsInTimestamp * '1', 2) 

        #get the beam image.
        try:
            self.beamImage = self.file.get_node('/BeamMap/Map').read()
            self.beamFlagImage = self.file.get_node('/BeamMap/Flag').read()
        except Exception as inst:
            if verbose:
                print('Can\'t access beamimage for ',self.fullFileName)
            raise inst

        beamShape = self.beamImage.shape
        self.nRow = beamShape[0]
        self.nCol = beamShape[1]

    def checkIntegrity(self,firstSec=0,integrationTime=-1):
        """
        Checks the obs file for corrupted end-of-seconds
        Corruption is indicated by timestamps greater than 1/tickDuration=1e6
        returns 0 if no corruption found
        """
        corruptedPixels = []
        for iRow in range(self.nRow):
            for iCol in range(self.nCol):
                packetList = self.getPixelPacketList(iRow,iCol,firstSec,integrationTime)
                timestamps,parabolaPeaks,baselines = self.parsePhotonPackets(packetList)
                if np.any(timestamps > 1./self.tickDuration):
                    print('Corruption detected in pixel (',iRow,iCol,')')
                    corruptedPixels.append((iRow,iCol))
        corruptionFound = len(corruptedPixels) != 0
        return corruptionFound
#        exptime = self.getFromHeader('exptime')
#        lastSec = firstSec + integrationTime
#        if integrationTime == -1:
#            lastSec = exptime-1
#            
#        corruptedSecs = []
#        for pixelCoord in corruptedPixels:
#            for sec in xrange(firstSec,lastSec):
#                packetList = self.getPixelPacketList(pixelCoord[0],pixelCoord[1],sec,integrationTime=1)
#                timestamps,parabolaPeaks,baselines = self.parsePhotonPackets(packetList)
#                if np.any(timestamps > 1./self.tickDuration):
#                    pixelLabel = self.beamImage[iRow][iCol]
#                    corruptedSecs.append(sec)
#                    print 'Corruption in pixel',pixelLabel, 'at',sec

    
    
    def createEmptyPhotonListFile(self,*nkwargs,**kwargs):
        """
        creates a photonList h5 file using header in headers.ArconsHeaders
        Shifted functionality to photonlist/photlist.py, JvE May 10 2013.
        See that function for input parameters and outputs.
        """
        import photonlist.photlist      #Here instead of at top to avoid circular imports
        photonlist.photlist.createEmptyPhotonListFile(self,*nkwargs,**kwargs)


#    def createEmptyPhotonListFile(self,fileName=None):
#        """
#        creates a photonList h5 file 
#        using header in headers.ArconsHeaders
#        
#        INPUTS:
#            fileName - string, name of file to write to. If not supplied, default is used
#                       based on name of original obs. file and standard directories etc.
#                       (see usil.FileName). Added 4/29/2013, JvE
#        """
#        
#        if fileName is None:    
#            fileTimestamp = self.fileName.split('_')[1].split('.')[0]
#            fileDate = os.path.basename(os.path.dirname(self.fullFileName))
#            run = os.path.basename(os.path.dirname(os.path.dirname(self.fullFileName)))
#            fn = FileName(run=run, date=fileDate, tstamp=fileTimestamp)
#            fullPhotonListFileName = fn.photonList()
#        else:
#            fullPhotonListFileName = fileName
#        if (os.path.exists(fullPhotonListFileName)):
#            if utils.confirm('Photon list file  %s exists. Overwrite?' % fullPhotonListFileName, defaultResponse=False) == False:
#                exit(0)
#        zlibFilter = tables.Filters(complevel=1, complib='zlib', fletcher32=False)
#        try:
#            plFile = tables.openFile(fullPhotonListFileName, mode='w')
#            plGroup = plFile.createGroup('/', 'photons', 'Group containing photon list')
#            plTable = plFile.createTable(plGroup, 'photons', ArconsHeaders.PhotonList, 'Photon List Data', 
#                                         filters=zlibFilter, 
#                                         expectedrows=300000)  #Temporary fudge to see if it helps!
#        except:
#            plFile.close()
#            raise
#        return plFile
        
    def displaySec(self, firstSec=0, integrationTime= -1, weighted=False,
                   fluxWeighted=False, plotTitle='', nSdevMax=2,
                   scaleByEffInt=False, getRawCount=False, fignum=None, ds9=False,
                   pclip=1.0, **kw):
        """
        plots a time-flattened image of the counts integrated from firstSec to firstSec+integrationTime
        if integrationTime is -1, All time after firstSec is used.  
        if weighted is True, flat cal weights are applied
        If fluxWeighted is True, apply flux cal weights.
        if scaleByEffInt is True, then counts are scaled by effective exposure
        time on a per-pixel basis.
        nSdevMax - max end of stretch scale for display, in # sigmas above the mean.
        getRawCount - if True the raw non-wavelength-calibrated image is
        displayed with no wavelength cutoffs applied (in which case no wavecal
        file need be loaded).
        fignum - as for utils.plotArray (None = new window; False/0 = current window; or 
                 specify target window number).
        ds9 - boolean, if True, display in DS9 instead of regular plot window.
        pclip - set to percentile level (in percent) to set the upper and lower bounds
                of the colour scale.
        **kw - any other keywords passed directly to utils.plotArray()
        
        """
        secImg = self.getPixelCountImage(firstSec, integrationTime, weighted, fluxWeighted,
                                         getRawCount=getRawCount,scaleByEffInt=scaleByEffInt)['image']
        toPlot = np.copy(secImg)
        vmin = np.percentile(toPlot[np.isfinite(toPlot)],pclip)
        vmax = np.percentile(toPlot[np.isfinite(toPlot)],100.-pclip)
        toPlot[np.isnan(toPlot)] = 0    #Just looks nicer when you plot it.
        if ds9 is True:
            utils.ds9Array(secImg)
        else:
            utils.plotArray(secImg, cbar=True, normMax=np.mean(secImg) + nSdevMax * np.std(secImg),
                        plotTitle=plotTitle, fignum=fignum, **kw)

            
    def getFromHeader(self, name):
        """
        Returns a requested entry from the obs file header
        If asked for exptime (exposure time) and some roaches have a timestamp offset
        The returned exposure time will be shortened by the max offset, since ObsFile
        will not retrieve data from seconds in which some roaches do not have data.
        This also affects unixtime (start of observation).
        If asked for jd, the jd is calculated from the (corrected) unixtime
        """
        entry = self.info[self.titles.index(name)]
        if name=='exptime' and self.timeAdjustFile != None:
            #shorten the effective exptime by the number of seconds that 
            #does not have data from all roaches
            maxDelay = np.max(self.roachDelays)
            entry -= maxDelay
        if name=='unixtime' and self.timeAdjustFile != None:
            #the way getPixel retrieves data accounts for individual roach delay,
            #but shifted everything by np.max(self.roachDelays), relabeling sec maxDelay as sec 0
            #so, add maxDelay to the header start time, so all times will be correct relative to it
            entry += np.max(self.roachDelays)
            entry += self.firmwareDelay
        if name=='jd':
            #The jd stored in the raw file header is the jd when the empty file is created
            #but not when the observation starts.  The actual value can be derived from the stored unixtime
            unixEpochJD = 2440587.5
            secsPerDay = 86400
            unixtime = self.getFromHeader('unixtime')
            entry = 1.*unixtime/secsPerDay+unixEpochJD
        return entry
        
    def getPixelPhotonList(self, iRow, iCol, firstSec=0, integrationTime= -1, wvlRange=None):
        """
        Retrieves a pixel using the file's attached beammap.
        If firstSec/integrationTime are provided, only data from the time 
        interval 'firstSec' to firstSec+integrationTime are returned.
        For now firstSec and integrationTime can only be integers.
        If integrationTime is -1, all data after firstSec are returned.

        MJS 3/28
        Updated so if timeAdjustFile is loaded, data retrieved from roaches
        with a delay will be offset to match other roaches.  Also, if some roaches
        have a delay, seconds in which some roaches don't have data are no longer
        retrieved
        """
        resID = self.beamImage[iRow][iCol]
        photonTable = self.file.get_node('/Photons/' + str(resID))
 
        startTime = int(firstSec*self.ticksPerSec) #convert to us
        if integrationTime == -1:
            endTime = photonTable.read(-1)[0][0]
        else:
            endTime = startTime + int(integrationTime*self.ticksPerSec)

        if wvlRange is None:
            photonList = photonTable.read_where('(Time > startTime) & (Time < endTime)')
        
        else:
            startWvl = wvlRange[0]
            endWvl = wvlRange[1]
            photonList = photonTable.read_where('(Time > startTime) & (Time < endTime) & (Wavelength > startWvl) & (Wavelength < endWvl)')

        #return {'pixelData':pixelData,'firstSec':firstSec,'lastSec':lastSec}
        return photonList

    def getListOfPixelsPhotonList(self, posList, firstSec=0, integrationTime=-1, wvlRange=None):
        photonLists = []
        nPix = np.shape(posList)[0]
        for i in range(nPix):
            photonLists.append(self.getPixelPhotonList(posList[i][0], posList[i][1], firstSec, integrationTime, wvlRange))

        return photonLists
 
    def getPixelCount(self, iRow, iCol, firstSec=0, integrationTime= -1, wvlRange=None, applyWeight=True, applyTPFWeight=True, applyTimeMask=True):
        """
        returns the number of photons received in a given pixel from firstSec to firstSec + integrationTime
        - if integrationTime is -1, all time after firstSec is used.  
        - if weighted is True, flat cal weights are applied
        - if fluxWeighted is True, flux weights are applied.
        - if getRawCount is True, the total raw count for all photon event detections
          is returned irrespective of wavelength calibration, and with no wavelength
          cutoffs (in this case, no wavecal file need have been applied, though 
          bad pixel time-masks *will* still be applied if present and switched 'on'.) 
          Otherwise will now always call getPixelSpectrum (which is also capable 
          of handling hot pixel removal) -- JvE 3/1/2013.
        *Note getRawCount overrides weighted and fluxWeighted.
        
        Updated to return effective exp. times; see below. -- JvE 3/2013. 
        Updated to return rawCounts; see below           -- ABW Oct 7, 2014
        
        OUTPUTS:
        Return value is a dictionary with tags:
            'counts':int, number of photon counts
            'effIntTime':float, effective integration time after time-masking is 
                     accounted for.
            'rawCounts': int, total number of photon triggers (including noise)
        """
        photonList = getPixelPhotonList(iRow, iCol, firstSec, integrationTime, wvlRange)
        weights = np.ones(len(photonList))
        if applyWeight:
            weights *= photonList['Spec Weight']
        if applyTPFWeight:
            weights *= photonList['Noise Weight']

        if applyTimeMask:
            if self.timeMaskExists:
                pass
            else
                warnings.warn('Time mask does not exist!')

        return {'counts':np.sum(weights), 'effIntTime':integrationTime}


    def getPixelLightCurve(self,iRow,iCol,firstSec=0,lastSec=-1,cadence=1,
                           **kwargs):
        """
        Get a simple light curve for a pixel (basically a wrapper for getPixelCount).
        
        INPUTS:
            iRow,iCol - Row and column of pixel
            firstSec - start time (sec) within obsFile to begin the light curve
            lastSec - end time (sec) within obsFile for the light curve. If -1, returns light curve to end of file.
            cadence - cadence (sec) of light curve. i.e., return values integrated every 'cadence' seconds.
            **kwargs - any other keywords are passed on to getPixelCount (see above), including:
                weighted
                fluxWeighted  (Note if True, then this should correct the light curve for effective exposure time due to bad pixels)
                getRawCount
                
        OUTPUTS:
            A single one-dimensional array of flux counts integrated every 'cadence' seconds 
            between firstSec and lastSec. Note if step is non-integer may return inconsistent
            number of values depending on rounding of last value in time step sequence (see
            documentation for numpy.arange() ).
            
            If hot pixel masking is turned on, then returns 0 for any time that is masked out.
            (Maybe should update this to NaN at some point in getPixelCount?)
        """
        if lastSec==-1:lSec = self.getFromHeader('exptime')
        else: lSec = lastSec
        return np.array([self.getPixelCount(iRow,iCol,firstSec=x,integrationTime=cadence,**kwargs)['counts']
                       for x in np.arange(firstSec,lSec,cadence)])
    
    
    def plotPixelLightCurve(self,iRow,iCol,firstSec=0,lastSec=-1,cadence=1,**kwargs):
        """
        Plot a simple light curve for a given pixel. Just a wrapper for getPixelLightCurve.
        Also marks intervals flagged as bad with gray shaded regions if a hot pixel mask is
        loaded.
        """
        
        lc = self.getPixelLightCurve(iRow=iRow,iCol=iCol,firstSec=firstSec,lastSec=lastSec,
                                     cadence=cadence,**kwargs)
        if lastSec==-1: realLastSec = self.getFromHeader('exptime')
        else: realLastSec = lastSec
        
        #Plot the lightcurve
        x = np.arange(firstSec+cadence/2.,realLastSec)
        assert len(x)==len(lc)      #In case there are issues with arange being inconsistent on the number of values it returns
        plt.plot(x,lc)
        plt.xlabel('Time since start of file (s)')
        plt.ylabel('Counts')
        plt.title(self.fileName+' - pixel x,y = '+str(iCol)+','+str(iRow))
        
        #Get bad times in time range of interest (hot pixels etc.)
        badTimes = self.getPixelBadTimes(iRow,iCol) & interval([firstSec,realLastSec])   #Returns an 'interval' instance
        lcRange = np.nanmax(lc)-np.nanmin(lc)
        for eachInterval in badTimes:
            plt.fill_betweenx([np.nanmin(lc)-0.5*lcRange,np.nanmax(lc)+0.5*lcRange], eachInterval[0],eachInterval[1],
                              alpha=0.5,color='gray')
        

    def getPixelCountImage(self, firstSec=0, integrationTime= -1, weighted=False,
                           fluxWeighted=False, getRawCount=False,
                           scaleByEffInt=False):
        """
        Return a time-flattened image of the counts integrated from firstSec to firstSec+integrationTime.
        - If integration time is -1, all time after firstSec is used.
        - If weighted is True, flat cal weights are applied. JvE 12/28/12
        - If fluxWeighted is True, flux cal weights are applied. SM 2/7/13
        - If getRawCount is True then the raw non-wavelength-calibrated image is
          returned with no wavelength cutoffs applied (in which case no wavecal
          file need be loaded). *Note getRawCount overrides weighted and fluxWeighted
        - If scaleByEffInt is True, any pixels that have 'bad' times masked out
          will have their counts scaled up to match the equivalent integration 
          time requested.
        RETURNS:
            Dictionary with keys:
                'image' - a 2D array representing the image
                'effIntTimes' - a 2D array containing effective integration 
                                times for each pixel.
                'rawCounts' - a 2D array containing the raw number of counts
                                for each pixel. 
        """
        secImg = np.zeros((self.nRow, self.nCol))
        effIntTimes = np.zeros((self.nRow, self.nCol), dtype=np.float64)
        effIntTimes.fill(np.nan)   #Just in case an element doesn't get filled for some reason.
        rawCounts = np.zeros((self.nRow, self.nCol), dtype=np.float64)
        rawCounts.fill(np.nan)   #Just in case an element doesn't get filled for some reason.
        for iRow in range(self.nRow):
            for iCol in range(self.nCol):
                pcount = self.getPixelCount(iRow, iCol, firstSec, integrationTime,
                                          weighted, fluxWeighted, getRawCount)
                secImg[iRow, iCol] = pcount['counts']
                effIntTimes[iRow, iCol] = pcount['effIntTime']
                rawCounts[iRow,iCol] = pcount['rawCounts']
        if scaleByEffInt is True:
            if integrationTime == -1:
                totInt = self.getFromHeader('exptime')
            else:
                totInt = integrationTime
            secImg *= (totInt / effIntTimes)                    

        #if getEffInt is True:
        return{'image':secImg, 'effIntTimes':effIntTimes, 'rawCounts':rawCounts}
        #else:
        #    return secImg

    def getAperturePixelCountImage(self, firstSec=0, integrationTime= -1, y_values=list(range(46)), x_values=list(range(44)), y_sky=[], x_sky=[], apertureMask=np.ones((46,44)), skyMask=np.zeros((46,44)), weighted=False, fluxWeighted=False, getRawCount=False, scaleByEffInt=False):

        """
        Return a time-flattened image of the counts integrated from firstSec to firstSec+integrationTime 
        This aperture version subtracts out the average sky counts/pixel and includes scaling due to circular apertures. GD 5/27/13
        If integration time is -1, all time after firstSec is used.
        If weighted is True, flat cal weights are applied. JvE 12/28/12
        If fluxWeighted is True, flux cal weights are applied. SM 2/7/13
        If getRawCount is True then the raw non-wavelength-calibrated image is
        returned with no wavelength cutoffs applied (in which case no wavecal
        file need be loaded). JvE 3/1/13
        If scaleByEffInt is True, any pixels that have 'bad' times masked out
        will have their counts scaled up to match the equivalent integration 
        time requested.
        RETURNS:
            Dictionary with keys:
                'image' - a 2D array representing the image
                'effIntTimes' - a 2D array containing effective integration 
                                times for each pixel.
        """
        secImg = np.zeros((self.nRow, self.nCol))
        effIntTimes = np.zeros((self.nRow, self.nCol), dtype=np.float64)
        effIntTimes.fill(np.nan)   #Just in case an element doesn't get filled for some reason.
        skyValues=[]
        objValues=[]
        AreaSky=[]
        AreaObj=[]
        for pix in range(len(y_sky)):
            pcount = self.getPixelCount(y_sky[pix], x_sky[pix], firstSec, integrationTime,weighted, fluxWeighted, getRawCount)
            skyValue=pcount['counts']*skyMask[y_sky[pix]][x_sky[pix]]
            skyValues.append(skyValue)
            AreaSky.append(skyMask[y_sky[pix]][x_sky[pix]])
        skyCountPerPixel = np.sum(skyValues)/(np.sum(AreaSky))
#        print 'sky count per pixel =',skyCountPerPixel
        for pix in range(len(y_values)):
            pcount = self.getPixelCount(y_values[pix], x_values[pix], firstSec, integrationTime,weighted, fluxWeighted, getRawCount)
            secImg[y_values[pix],x_values[pix]] = (pcount['counts']-skyCountPerPixel)*apertureMask[y_values[pix]][x_values[pix]]
            AreaObj.append(apertureMask[y_values[pix]][x_values[pix]])
            effIntTimes[y_values[pix],x_values[pix]] = pcount['effIntTime']
            objValues.append(pcount['counts']*apertureMask[y_values[pix]][x_values[pix]])
        AveObj=np.sum(objValues)/(np.sum(AreaObj))
#        print 'ave obj per pixel (not sub) = ',AveObj
        NumObjPhotons = np.sum(secImg)
#        print 'lightcurve = ',NumObjPhotons
        if scaleByEffInt is True:
            secImg *= (integrationTime / effIntTimes)                    
        #if getEffInt is True:
        return{'image':secImg, 'effIntTimes':effIntTimes, 'SkyCountSubtractedPerPixel':skyCountPerPixel,'lightcurve':NumObjPhotons}
        #else:
        #    return secImg
    
    def getSpectralCube(self,firstSec=0,integrationTime=-1,weighted=True,fluxWeighted=True,wvlStart=3000,wvlStop=13000,wvlBinWidth=None,energyBinWidth=None,wvlBinEdges=None,timeSpacingCut=None):
        """
        Return a time-flattened spectral cube of the counts integrated from firstSec to firstSec+integrationTime.
        If integration time is -1, all time after firstSec is used.
        If weighted is True, flat cal weights are applied.
        If fluxWeighted is True, spectral shape weights are applied.
        """
        cube = [[[] for iCol in range(self.nCol)] for iRow in range(self.nRow)]
        effIntTime = np.zeros((self.nRow,self.nCol))
        rawCounts = np.zeros((self.nRow,self.nCol))

        for iRow in range(self.nRow):
            for iCol in range(self.nCol):
                x = self.getPixelSpectrum(pixelRow=iRow,pixelCol=iCol,
                                  firstSec=firstSec,integrationTime=integrationTime,
                                  weighted=weighted,fluxWeighted=fluxWeighted,wvlStart=wvlStart,wvlStop=wvlStop,
                                  wvlBinWidth=wvlBinWidth,energyBinWidth=energyBinWidth,
                                  wvlBinEdges=wvlBinEdges,timeSpacingCut=timeSpacingCut)
                cube[iRow][iCol] = x['spectrum']
                effIntTime[iRow][iCol] = x['effIntTime']
                rawCounts[iRow][iCol] = x['rawCounts']
                wvlBinEdges = x['wvlBinEdges']
        cube = np.array(cube)
        return {'cube':cube,'wvlBinEdges':wvlBinEdges,'effIntTime':effIntTime, 'rawCounts':rawCounts}

    def getPixelSpectrum(self, pixelRow, pixelCol, firstSec=0, integrationTime= -1,
                         weighted=False, fluxWeighted=False, wvlStart=None, wvlStop=None,
                         wvlBinWidth=None, energyBinWidth=None, wvlBinEdges=None,timeSpacingCut=None):
        """
        returns a spectral histogram of a given pixel integrated from firstSec to firstSec+integrationTime,
        and an array giving the cutoff wavelengths used to bin the wavelength values
        if integrationTime is -1, All time after firstSec is used.  
        if weighted is True, flat cal weights are applied
        if weighted is False, flat cal weights are not applied
        the wavelength bins used depends on the parameters given.
        If energyBinWidth is specified, the wavelength bins use fixed energy bin widths
        If wvlBinWidth is specified, the wavelength bins use fixed wavelength bin widths
        If neither is specified and/or if weighted is True, the flat cal wvlBinEdges is used
        
        wvlStart defaults to self.wvlLowerLimit or 3000
        wvlStop defaults to self.wvlUpperLimit or 13000
        
        ----
        Updated to return effective integration time for the pixel
        Returns dictionary with keys:
            'spectrum' - spectral histogram of given pixel.
            'wvlBinEdges' - edges of wavelength bins
            'effIntTime' - the effective integration time for the given pixel 
                           after accounting for hot-pixel time-masking.
            'rawCounts' - The total number of photon triggers (including from
                            the noise tail) during the effective exposure.
        JvE 3/5/2013
        ABW Oct 7, 2014. Added rawCounts to dictionary
        ----
        """

        wvlStart=wvlStart if (wvlStart!=None and wvlStart>0.) else (self.wvlLowerLimit if (self.wvlLowerLimit!=None and self.wvlLowerLimit>0.) else 3000)
        wvlStop=wvlStop if (wvlStop!=None and wvlStop>0.) else (self.wvlUpperLimit if (self.wvlUpperLimit!=None and self.wvlUpperLimit>0.) else 12000)


        x = self.getPixelWvlList(pixelRow, pixelCol, firstSec, integrationTime,timeSpacingCut=timeSpacingCut)
        wvlList, effIntTime, rawCounts = x['wavelengths'], x['effIntTime'], x['rawCounts']

        if (weighted == False) and (fluxWeighted == True):
            raise ValueError("Cannot apply flux cal without flat cal. Please load flat cal and set weighted=True")

        if (self.flatCalFile is not None) and (((wvlBinEdges is None) and (energyBinWidth is None) and (wvlBinWidth is None)) or weighted == True):
        #We've loaded a flat cal already, which has wvlBinEdges defined, and no other bin edges parameters are specified to override it.
            spectrum, wvlBinEdges = np.histogram(wvlList, bins=self.flatCalWvlBins)
            if weighted == True:#Need to apply flat weights by wavelenth
                spectrum = spectrum * self.flatWeights[pixelRow, pixelCol]
                if fluxWeighted == True:
                    spectrum = spectrum * self.fluxWeights
        else:
            if weighted == True:
                raise ValueError('when weighted=True, flatCal wvl bins are used, so wvlBinEdges,wvlBinWidth,energyBinWidth,wvlStart,wvlStop should not be specified')
            if wvlBinEdges is None:#We need to construct wvlBinEdges array
                if energyBinWidth is not None:#Fixed energy binwidth specified
                    #Construct array with variable wvl binwidths
                    wvlBinEdges = ObsFile.makeWvlBins(energyBinWidth=energyBinWidth, wvlStart=wvlStart, wvlStop=wvlStop)
                    spectrum, wvlBinEdges = np.histogram(wvlList, bins=wvlBinEdges)
                elif wvlBinWidth is not None:#Fixed wvl binwidth specified
                    nWvlBins = int((wvlStop - wvlStart) / wvlBinWidth)
                    spectrum, wvlBinEdges = np.histogram(wvlList, bins=nWvlBins, range=(wvlStart, wvlStop))
                else:
                    raise ValueError('getPixelSpectrum needs either wvlBinWidth,wvlBinEnergy, or wvlBinEdges')
                #else:
                #    nWvlBins = 1
                #    spectrum, wvlBinEdges = np.histogram(wvlList, bins=nWvlBins, range=(wvlStart, wvlStop))
            else:#We are given wvlBinEdges array
                spectrum, wvlBinEdges = np.histogram(wvlList, bins=wvlBinEdges)
       
        if self.filterIsApplied == True:
            if not np.array_equal(self.filterWvlBinEdges, wvlBinEdges):
                raise ValueError("Synthetic filter wvlBinEdges do not match pixel spectrum wvlBinEdges!")
            spectrum*=self.filterTrans

        #if getEffInt is True:
        return {'spectrum':spectrum, 'wvlBinEdges':wvlBinEdges, 'effIntTime':effIntTime, 'rawCounts':rawCounts}
        #else:
        #    return spectrum,wvlBinEdges
    
    def getApertureSpectrum(self, pixelRow, pixelCol, radius1, radius2, weighted=False,
                            fluxWeighted=False, lowCut=3000, highCut=7000,firstSec=0,integrationTime=-1):
        '''
        Creates a spectrum from a group of pixels.  Aperture is defined by pixelRow and pixelCol of
        center, as well as radius.  Wave and flat cals should be loaded before using this
        function.  If no hot pixel mask is applied, taking the median of the sky rather than
        the average to account for high hot pixel counts.
        Will add more options as other pieces of pipeline become more refined.
        (Note - not updated to handle loaded hot pixel time-masks - if applied,
        behaviour may be unpredictable. JvE 3/5/2013).
        '''
        print('Creating dead pixel mask...')
        deadMask = self.getDeadPixels()
        print('Creating wavecal solution mask...')
        bad_solution_mask = np.zeros((self.nRow, self.nCol))
        for y in range(self.nRow):
            for x in range(self.nCol):
                if (self.wvlRangeTable[y][x][0] > lowCut or self.wvlRangeTable[y][x][1] < highCut):
                    bad_solution_mask[y][x] = 1
        print('Creating aperture mask...')
        apertureMask = utils.aperture(pixelCol, pixelRow, radius=radius1)
        print('Creating sky mask...')
        bigMask = utils.aperture(pixelCol, pixelRow, radius=radius2)
        skyMask = bigMask - apertureMask
        #if hotPixMask == None:
        #    y_values, x_values = np.where(np.logical_and(bad_solution_mask == 0, np.logical_and(apertureMask == 0, deadMask == 1)))
        #    y_sky, x_sky = np.where(np.logical_and(bad_solution_mask == 0, np.logical_and(skyMask == 0, deadMask == 1)))
        #else:
        #    y_values, x_values = np.where(np.logical_and(bad_solution_mask == 0, np.logical_and(np.logical_and(apertureMask == 0, deadMask == 1), hotPixMask == 0)))
        #    y_sky, x_sky = np.where(np.logical_and(bad_solution_mask == 0, np.logical_and(np.logical_and(skyMask == 0, deadMask == 1), hotPixMask == 0)))

        y_values, x_values = np.where(np.logical_and(bad_solution_mask == 0, np.logical_and(apertureMask == 0, deadMask == 1)))
        y_sky, x_sky = np.where(np.logical_and(bad_solution_mask == 0, np.logical_and(skyMask == 0, deadMask == 1)))

        #wvlBinEdges = self.getPixelSpectrum(y_values[0], x_values[0], weighted=weighted)['wvlBinEdges']
        print('Creating average sky spectrum...')
        skyspectrum = []
        for i in range(len(x_sky)):
            specDict = self.getPixelSpectrum(y_sky[i],x_sky[i],weighted=weighted, fluxWeighted=fluxWeighted, firstSec=firstSec, integrationTime=integrationTime)
            self.skySpectrumSingle,wvlBinEdges,self.effIntTime = specDict['spectrum'],specDict['wvlBinEdges'],specDict['effIntTime']
            self.scaledSpectrum = self.skySpectrumSingle/self.effIntTime #scaled spectrum by effective integration time
            #print "Sky spectrum"
            #print self.skySpectrumSingle
            #print "Int time"
            #print self.effIntTime
            skyspectrum.append(self.scaledSpectrum)
        sky_array = np.zeros(len(skyspectrum[0]))
        for j in range(len(skyspectrum[0])):
            ispectrum = np.zeros(len(skyspectrum))
            for i in range(len(skyspectrum)):    
                ispectrum[i] = skyspectrum[i][j]
            sky_array[j] = np.median(ispectrum)
            #if hotPixMask == None:
            #    sky_array[j] = np.median(ispectrum)
            #else:
            #    sky_array[j] = np.average(ispectrum)
        print('Creating sky subtracted spectrum...')
        spectrum = []
        for i in range(len(x_values)):
            specDict = self.getPixelSpectrum(y_values[i],x_values[i],weighted=weighted, fluxWeighted=fluxWeighted, firstSec=firstSec, integrationTime=integrationTime)
            self.obsSpectrumSingle,wvlBinEdges,self.effIntTime = specDict['spectrum'],specDict['wvlBinEdges'],specDict['effIntTime']   
            self.scaledSpectrum = self.obsSpectrumSingle/self.effIntTime #scaled spectrum by effective integration time
            spectrum.append(self.scaledSpectrum - sky_array)

            #spectrum.append(self.getPixelSpectrum(y_values[i], x_values[i], weighted=weighted,fluxWeighted=fluxWeighted)['spectrum'] - sky_array)
        summed_array = np.zeros(len(spectrum[0]))
        for j in range(len(spectrum[0])):
            ispectrum = np.zeros(len(spectrum))
            for i in range(len(spectrum)):    
                ispectrum[i] = spectrum[i][j]
            summed_array[j] = np.sum(ispectrum)
        for i in range(len(summed_array)):
            summed_array[i] /= (wvlBinEdges[i + 1] - wvlBinEdges[i])
        return summed_array, wvlBinEdges
    
    def getPixelBadTimes(self, pixelRow, pixelCol, reasons=[]):
        """
        Get the time interval(s) for which a given pixel is bad (hot/cold,
        whatever, from the hot pixel cal file).
        Returns an 'interval' object (see pyinterval) of bad times (in seconds
        from start of obs file).
        """
        if self.hotPixTimeMask is None:
            raise RuntimeError('No hot pixel file loaded')

        return self.hotPixTimeMask.get_intervals(pixelRow,pixelCol,reasons)

    def getDeadPixels(self, showMe=False, weighted=True, getRawCount=False):
        """
        returns a mask indicating which pixels had no counts in this observation file
        1's for pixels with counts, 0's for pixels without counts
        if showMe is True, a plot of the mask pops up
        """
        countArray = np.array([[(self.getPixelCount(iRow, iCol, weighted=weighted,getRawCount=getRawCount))['counts'] for iCol in range(self.nCol)] for iRow in range(self.nRow)])
        deadArray = np.ones((self.nRow, self.nCol))
        deadArray[countArray == 0] = 0
        if showMe == True:
            utils.plotArray(deadArray)
        return deadArray 

    def getNonAllocPixels(self, showMe=False):
        """
        returns a mask indicating which pixels had no beammap locations 
        (set to constant /r0/p250/)
        1's for pixels with locations, 0's for pixels without locations
        if showMe is True, a plot of the mask pops up
        """
        nonAllocArray = np.ones((self.nRow, self.nCol))
        nonAllocArray[np.core.defchararray.startswith(self.beamImage, self.nonAllocPixelName)] = 0
        if showMe == True:
            utils.plotArray(nonAllocArray)
        return nonAllocArray

    def getRoachNum(self,iRow,iCol):
        pixelLabel = self.beamImage[iRow][iCol]
        iRoach = int(pixelLabel.split('r')[1][0])
        return iRoach

    def getFrame(self, firstSec=0, integrationTime=-1):
        """
        return a 2d array of numbers with the integrated flux per pixel,
        suitable for use as a frame in util/utils.py function makeMovie

        firstSec=0 is the starting second to include

        integrationTime=-1 is the number of seconds to include, or -1
        to include all to the end of this file


        output: the frame, in photons per pixel, a 2d numpy array of
        np.unint32

        """
        frame = np.zeros((self.nRow,self.nCol),dtype=np.uint32)
        for iRow in range(self.nRow):
            for iCol in range(self.nCol):
                pl = self.getTimedPacketList(iRow,iCol,
                                             firstSec,integrationTime)
                nphoton = pl['timestamps'].size
                frame[iRow][iCol] += nphoton
        return frame
    
    # a different way to get, with the functionality of getTimedPacketList
    def getPackets(self, iRow, iCol, firstSec, integrationTime, 
                   fields=(),
                   expTailTimescale=None,
                   timeSpacingCut=None,
                   timeMaskLast=True):
        """
        get and parse packets for pixel iRow,iCol starting at firstSec for integrationTime seconds.

        fields is a list of strings to indicate what to parse in
        addition to timestamps: allowed values are 'peakHeights' and
        'baselines'

        expTailTimescale (if not None) subtractes the exponentail tail
        off of one photon from the peakHeight of the next photon.
        This also attempts to counter effects of photon pile-up for
        short (< 100 us) dead times.

        timeSpacingCut (if not None) rejects photons sooner than
        timeSpacingCut seconds after the last photon.

        timeMaskLast -- apply time masks after timeSpacingCut and expTailTimescale.
        set this to "false" to mimic behavior of getTimedPacketList

        return a dictionary containing:
          effectiveIntTime (n seconds)
          timestamps
          other fields requested
        """
        
        warnings.warn('Does anyone use this function?? If not, we should get rid of it')
        
        parse = {'peakHeights': True, 'baselines': True}
        for key in list(parse.keys()):
            try:
                fields.index(key)
            except ValueError:
                parse[key] = False

        lastSec = firstSec+integrationTime
        # Work out inter, the times to mask
        # start with nothing being masked
        inter = interval()
        # mask the hot pixels if requested
        if self.hotPixIsApplied:
            inter = self.getPixelBadTimes(iRow, iCol)
        # mask cosmics if requested
        if self.cosmicMaskIsApplied:
            inter = inter | self.cosmicMask

        # mask the fraction of the first integer second not requested
        firstSecInt = int(np.floor(firstSec))
        if (firstSec > firstSecInt):
            inter = inter | interval([firstSecInt, firstSec])
        # mask the fraction of the last integer second not requested
        lastSecInt = int(np.ceil(firstSec+integrationTime))
        integrationTimeInt = lastSecInt-firstSecInt
        if (lastSec < lastSecInt):
            inter = inter | interval([lastSec, lastSecInt])

        #Calculate the total effective time for the integration after removing
        #any 'intervals':
        integrationInterval = interval([firstSec, lastSec])
        maskedIntervals = inter & integrationInterval  #Intersection of the integration and the bad times for this pixel.
        effectiveIntTime = integrationTime - utils.intervalSize(maskedIntervals)

        pixelData = self.getPixel(iRow, iCol, firstSec=firstSecInt, 
                                  integrationTime=integrationTimeInt)
        # calculate how long a np array needs to be to hold everything
        nPackets = 0
        for packets in pixelData:
            nPackets += len(packets)

        # create empty arrays
        timestamps = np.empty(nPackets, dtype=np.float)
        if parse['peakHeights']: peakHeights=np.empty(nPackets, np.int16)
        if parse['baselines']: baselines=np.empty(nPackets, np.int16)

        # fill in the arrays one second at a time
        ipt = 0
        t = firstSecInt
        for packets in pixelData:
            iptNext = ipt+len(packets)
            timestamps[ipt:iptNext] = \
                t + np.bitwise_and(packets,self.timestampMask)*self.tickDuration
            if parse['peakHeights']:
                peakHeights[ipt:iptNext] = np.bitwise_and(
                    np.right_shift(packets, self.nBitsAfterParabolaPeak), 
                    self.pulseMask)

            if parse['baselines']:
                baselines[ipt:iptNext] = np.bitwise_and(
                    np.right_shift(packets, self.nBitsAfterBaseline), 
                    self.pulseMask)

            ipt = iptNext
            t += 1

        if not timeMaskLast:
            # apply time masks
            # create a mask, "True" mean mask value
            # the call to makeMask dominates the running time
            if self.makeMaskVersion == 'v1':
                mask = ObsFile.makeMaskV1(timestamps, inter)
            else:
                mask = ObsFile.makeMaskV2(timestamps, inter)

            tsMaskedArray = ma.array(timestamps,mask=mask)
            timestamps = ma.compressed(tsMaskedArray)

            if parse['peakHeights']: 
                peakHeights = \
                    ma.compressed(ma.array(peakHeights,mask=mask))
            if parse['baselines']: 
                baselines = \
                    ma.compressed(ma.array(baselines,mask=mask))
        
        #diagnose("getPackets AAA",timestamps,peakHeights,baselines,None)
        if expTailTimescale != None and len(timestamps) > 0:
            #find the time between peaks
            timeSpacing = np.diff(timestamps)
            timeSpacing[timeSpacing < 0] = 1.
            timeSpacing = np.append(1.,timeSpacing)#arbitrarily assume the first photon is 1 sec after the one before it

            # relPeakHeights not used?
            #relPeakHeights = peakHeights-baselines
            
            #assume each peak is riding on the tail of an exponential starting at the peak before it with e-fold time of expTailTimescale
            #print 30*"."," getPackets"
            #print 'dt',timeSpacing[0:10]
            expTails = (1.*peakHeights-baselines)*np.exp(-1.*timeSpacing/expTailTimescale)
            #print 'expTail',expTails[0:10]
            #print 'peak',peakHeights[0:10]
            #print 'peak-baseline',1.*peakHeights[0:10]-baselines[0:10]
            #print 'expT',np.exp(-1.*timeSpacing[0:10]/expTailTimescale)
            #subtract off this exponential tail
            peakHeights = np.array(peakHeights-expTails,dtype=np.int)
            #print 'peak',peakHeights[0:10]


        if timeSpacingCut != None and len(timestamps) > 0:
            timeSpacing = np.diff(timestamps)
            #include first photon and photons after who are at least
            #timeSpacingCut after the previous photon
            timeSpacingMask = np.concatenate([[True],timeSpacing >= timeSpacingCut]) 
            timestamps = timestamps[timeSpacingMask]
            if parse['peakHeights']:
                peakHeights = peakHeights[timeSpacingMask]
            if parse['baselines']:
                baselines = baselines[timeSpacingMask]


        if timeMaskLast:
            # apply time masks
            # create a mask, "True" mean mask value
            # the call to makeMask dominates the running time
            if self.makeMaskVersion == 'v1':
                mask = ObsFile.makeMaskV1(timestamps, inter)
            else:
                mask = ObsFile.makeMaskV2(timestamps, inter)

            tsMaskedArray = ma.array(timestamps,mask=mask)
            timestamps = ma.compressed(tsMaskedArray)

            if parse['peakHeights']: 
                peakHeights = \
                    ma.compressed(ma.array(peakHeights,mask=mask))
            if parse['baselines']: 
                baselines = \
                    ma.compressed(ma.array(baselines,mask=mask))

        # build up the dictionary of values and return it
        retval =  {"effIntTime": effectiveIntTime,
                   "timestamps":timestamps}
        if parse['peakHeights']: 
            retval['peakHeights'] = peakHeights
        if parse['baselines']: 
            retval['baselines'] = baselines
        return retval

    @staticmethod
    def makeMask01(timestamps, inter):
        def myfunc(x): return inter.__contains__(x)
        vecfunc = vectorize(myfunc,otypes=[np.bool])
        return vecfunc(timestamps)

    @staticmethod
    def makeMask(timestamps, inter):
        """
        return an array of booleans, the same length as timestamps,
        with that value inter.__contains__(timestamps[i])
        """
        return ObsFile.makeMaskV2(timestamps, inter)

    @staticmethod
    def makeMaskV1(timestamps, inter):
        """
        return an array of booleans, the same length as timestamps,
        with that value inter.__contains__(timestamps[i])
        """
        retval = np.empty(len(timestamps),dtype=np.bool)
        ainter = np.array(inter)
        t0s = ainter[:,0]
        t1s = ainter[:,1]

        tMin = t0s[0]
        tMax = t1s[-1]
                       
        for i in range(len(timestamps)):
            ts = timestamps[i]
            if ts < tMin:
                retval[i] = False
            elif ts > tMax:
                retval[i] = False
            else:
                tIndex = np.searchsorted(t0s, ts)
                t0 = t0s[tIndex-1]
                t1 = t1s[tIndex-1]
                if ts < t1:
                    retval[i] = True
                else:
                    retval[i] = False
        return retval

    @staticmethod
    def makeMaskV2(timestamps, inter):
        """
        return an array of booleans, the same length as timestamps,
        with that value inter.__contains__(timestamps[i])
        """
        lt = len(timestamps)
        retval = np.zeros(lt,dtype=np.bool)
        for i in inter:
            if len(i) == 2:
                i0 = np.searchsorted(timestamps,i[0])
                if i0 == lt: break # the intervals are later than timestamps
                i1 = np.searchsorted(timestamps,i[1])
                if i1 > 0:
                    i0 = max(i0,0)
                    retval[i0:i1] = True
        return retval

    def loadBeammapFile(self,beammapFileName):
        """
        Load an external beammap file in place of the obsfile's attached beamma
        Can be used to correct pixel location mistakes.
        NB: Do not use after loadFlatCalFile
        """
        #get the beam image.
        scratchDir = os.getenv('MKID_PROC_PATH', '/')
        beammapPath = os.path.join(scratchDir, 'pixRemap')
        fullBeammapFileName = os.path.join(beammapPath, beammapFileName)
        if (not os.path.exists(fullBeammapFileName)):
            print('Beammap file does not exist: ', fullBeammapFileName)
            return
        if (not os.path.exists(beammapFileName)):
            #get the beam image.
            scratchDir = os.getenv('MKID_PROC_PATH', '/')
            beammapPath = os.path.join(scratchDir, 'pixRemap')
            fullBeammapFileName = os.path.join(beammapPath, beammapFileName)
            if (not os.path.exists(fullBeammapFileName)):
                print('Beammap file does not exist: ', fullBeammapFileName)
                return
        else:
            fullBeammapFileName = beammapFileName
        beammapFile = tables.openFile(fullBeammapFileName,'r')
        self.beammapFileName = fullBeammapFileName
        try:
            old_tstamp = self.beamImage[0][0].split('/')[-1]
            self.beamImage = beammapFile.get_node('/beammap/beamimage').read()
            if self.beamImage[0][0].split('/')[-1]=='':
                self.beamImage = np.core.defchararray.add(self.beamImage,old_tstamp)

            self.beamImageRoaches = np.array([[int(s.split('r')[1].split('/')[0]) for s in row] for row in self.beamImage])
            self.beamImagePixelNums = np.array([[int(s.split('p')[1].split('/')[0]) for s in row] for row in self.beamImage])
        except Exception as inst:
            print('Can\'t access beamimage for ',self.fullFileName)

        beamShape = self.beamImage.shape
        self.nRow = beamShape[0]
        self.nCol = beamShape[1]
        
        beammapFile.close()
        
    def loadCentroidListFile(self, centroidListFileName):
        """
        Load an astrometry (centroid list) file into the 
        current obs file instance.
        """
        scratchDir = os.getenv('MKID_PROC_PATH', '/')
        centroidListPath = os.path.join(scratchDir, 'centroidListFiles')
        fullCentroidListFileName = os.path.join(centroidListPath, centroidListFileName)
        if (not os.path.exists(fullCentroidListFileName)):
            print('Astrometry centroid list file does not exist: ', fullCentroidListFileName)
            return
        self.centroidListFile = tables.openFile(fullCentroidListFileName)
        self.centroidListFileName = fullCentroidListFileName
        
    def loadFlatCalFile(self, flatCalFileName):
        """
        loads the flat cal factors from the given file
        NB: if you are going to load a different beammap, call loadBeammapFile before this function
        """
        scratchDir = os.getenv('MKID_PROC_PATH', '/')
        flatCalPath = os.path.join(scratchDir, 'flatCalSolnFiles')
        fullFlatCalFileName = os.path.join(flatCalPath, flatCalFileName)
        if (not os.path.exists(fullFlatCalFileName)):
            print('flat cal file does not exist: ', fullFlatCalFileName)
            raise Exception('flat cal file {} does not exist'.format(fullFlatCalFileName))
        self.flatCalFile = tables.openFile(fullFlatCalFileName, mode='r')
        self.flatCalFileName = fullFlatCalFileName

        self.flatCalWvlBins = self.flatCalFile.root.flatcal.wavelengthBins.read()
        self.nFlatCalWvlBins = len(self.flatCalWvlBins)-1
        self.flatWeights = np.zeros((self.nRow,self.nCol,self.nFlatCalWvlBins),dtype=np.double)
        self.flatFlags = np.zeros((self.nRow,self.nCol,self.nFlatCalWvlBins),dtype=np.uint16)

        try:
            flatCalSoln = self.flatCalFile.root.flatcal.calsoln.read()
            for calEntry in flatCalSoln:
                entryRows,entryCols = np.where((calEntry['roach'] == self.beamImageRoaches) & (calEntry['pixelnum'] == self.beamImagePixelNums))
                try:
                    entryRow = entryRows[0]
                    entryCol = entryCols[0]
                    self.flatWeights[entryRow,entryCol,:] = calEntry['weights']
                    self.flatFlags[entryRow,entryCol,:] = calEntry['weightFlags']
                except IndexError: #entry for an unbeammapped pixel
                    pass 
        
        except tables.exceptions.NoSuchNodeError:
            #loading old (beammap-dependant) flat cal
            print('loading old (beammap-dependant) flat cal')
            self.flatWeights = self.flatCalFile.root.flatcal.weights.read()
            self.flatFlags = self.flatCalFile.root.flatcal.flags.read()
        
    def loadFluxCalFile(self, fluxCalFileName):
        """
        loads the flux cal factors from the given file
        """
        scratchDir = os.getenv('MKID_PROC_PATH', '/')
        fluxCalPath = os.path.join(scratchDir, 'fluxCalSolnFiles')
        fullFluxCalFileName = os.path.join(fluxCalPath, fluxCalFileName)
        if (not os.path.exists(fullFluxCalFileName)):
            print('flux cal file does not exist: ', fullFluxCalFileName)
            raise IOError
        self.fluxCalFile = tables.open_file(fullFluxCalFileName, mode='r')
        self.fluxCalFileName = fullFluxCalFileName
        self.fluxWeights = self.fluxCalFile.root.fluxcal.weights.read()
        self.fluxFlags = self.fluxCalFile.root.fluxcal.flags.read()
        self.fluxCalWvlBins = self.fluxCalFile.root.fluxcal.wavelengthBins.read()
        self.nFluxCalWvlBins = self.nFlatCalWvlBins

    def loadHotPixCalFile(self, hotPixCalFileName, switchOnMask=True,reasons=[]):
        """
        Included for backward compatibility, simply calls loadTimeMask
        """
        self.loadTimeMask(timeMaskFileName=hotPixCalFileName,switchOnMask=switchOnMask,reasons=reasons)

    def loadTimeMask(self, timeMaskFileName, switchOnMask=True,reasons=[]):
        """
        Load a hot pixel time mask from the given file, in a similar way to
        loadWvlCalFile, loadFlatCalFile, etc. Switches on hot pixel
        masking by default.
        Set switchOnMask=False to prevent switching on hot pixel masking.
        """
        import hotpix.hotPixels as hotPixels    #Here instead of at top to prevent circular import problems.

        scratchDir = os.getenv('MKID_PROC_PATH', '/')
        timeMaskPath = os.path.join(scratchDir, 'timeMasks')
        fullTimeMaskFileName = os.path.join(timeMaskPath, timeMaskFileName)
        if (not os.path.exists(fullTimeMaskFileName)):
            print('time mask file does not exist: ', fullTimeMaskFileName)
            raise IOError

        self.hotPixFile = tables.open_file(fullTimeMaskFileName)
        self.hotPixTimeMask = hotPixels.readHotPixels(self.hotPixFile, reasons=reasons)
        self.hotPixFileName = fullTimeMaskFileName
        
        if (os.path.basename(self.hotPixTimeMask.obsFileName)
            != os.path.basename(self.fileName)):
            warnings.warn('Mismatch between hot pixel time mask file and obs file. Not loading/applying mask!')
            self.hotPixTimeMask = None
            raise ValueError
        else:
            if switchOnMask: self.switchOnHotPixTimeMask(reasons=reasons)


    def loadStandardCosmicMask(self, switchOnCosmicMask=True):
        """
        call this method to load the cosmic mask file from the standard location,
        defined in Filename
        """
        fileName = FileName(obsFile=self)
        cfn = fileName.cosmicMask()
        self.loadCosmicMask(cosmicMaskFileName = cfn, switchOnCosmicMask=switchOnCosmicMask)

    def loadCosmicMask(self, cosmicMaskFileName=None, switchOnCosmicMask=True):
        self.cosmicMask = ObsFile.readCosmicIntervalFromFile(cosmicMaskFileName)
        self.cosmicMaskFileName = os.path.abspath(cosmicMaskFileName)
        if switchOnCosmicMask: self.switchOnCosmicTimeMask()

    def setCosmicMask(self, cosmicMask, switchOnCosmicMask=True):
        self.cosmicMask = cosmicMask
        if switchOnCosmicMask: self.switchOnCosmicTimeMask()

    def loadTimeAdjustmentFile(self,timeAdjustFileName,verbose=False):
        """
        loads obsfile specific adjustments to add to all timestamps read
        adjustments are read from timeAdjustFileName
        it is suggested to pass timeAdjustFileName=FileName(run=run).timeAdjustments()
        """

        self.timeAdjustFile = tables.open_file(timeAdjustFileName)
        self.firmwareDelay = self.timeAdjustFile.root.timeAdjust.firmwareDelay.read()[0]['firmwareDelay']
        roachDelayTable = self.timeAdjustFile.root.timeAdjust.roachDelays
        try:
            self.roachDelays = roachDelayTable.readWhere('obsFileName == "%s"'%self.fileName)[0]['roachDelays']
            self.timeAdjustFileName = os.path.abspath(timeAdjustFileName)
        except:
            self.timeAdjustFile.close()
            self.timeAdjustFile=None
            self.timeAdjustFileName=None
            del self.firmwareDelay
            if verbose:
                print('Unable to load time adjustment for '+self.fileName)
            raise

    def loadBestWvlCalFile(self,master=True):
        """
        Searchs the waveCalSolnFiles directory tree for the best wavecal to apply to this obsfile.
        if master==True then it first looks for a master wavecal solution
        """
        #scratchDir = os.getenv('MKID_PROC_PATH', '/')
        #run = FileName(obsFile=self).run
        #wvlDir = scratchDir+"/waveCalSolnFiles/"+run+'/'
        wvlDir = os.path.dirname(os.path.dirname(FileName(obsFile=self).mastercalSoln()))
        #print wvlDir
        obs_t_num = strpdate2num("%Y%m%d-%H%M%S")(FileName(obsFile=self).tstamp)

        wvlCalFileName = None
        wvl_t_num = None
        for root,dirs,files in os.walk(wvlDir):
            for f in files:
                if f.endswith('.h5') and ((master and f.startswith('mastercal_')) or (not master and f.startswith('calsol_'))):
                    tstamp=(f.split('_')[1]).split('.')[0]
                    t_num=strpdate2num("%Y%m%d-%H%M%S")(tstamp)
                    if t_num < obs_t_num and (wvl_t_num == None or t_num > wvl_t_num):
                        wvl_t_num = t_num
                        wvlCalFileName = root+os.sep+f

        if wvlCalFileName==None or not os.path.exists(str(wvlCalFileName)):
            if master:
                print("Could not find master wavecal solutions")
                self.loadBestWvlCalFile(master=False)
            else:
                print("Searched "+wvlDir+" but no appropriate wavecal solution found")
                raise IOError
        else:
            self.loadWvlCalFile(wvlCalFileName)
                
    def loadWvlCalFile(self, wvlCalFileName):
        """
        loads the wavelength cal coefficients from a given file
        """
        if os.path.exists(str(wvlCalFileName)):
            fullWvlCalFileName = str(wvlCalFileName)
        else:
            #scratchDir = os.getenv('MKID_PROC_PATH', '/')
            #wvlDir = os.path.join(scratchDir, 'waveCalSolnFiles')
            wvlDir = os.path.dirname(os.path.dirname(FileName(obsFile=self).mastercalSoln()))
            fullWvlCalFileName = os.path.join(wvlDir, str(wvlCalFileName))
        try:
            # If the file has already been loaded for this ObsFile then just return
            if hasattr(self,"wvlCalFileName") and (self.wvlCalFileName == fullWvlCalFileName):
                return
            self.wvlCalFile = tables.open_file(fullWvlCalFileName, mode='r')
            self.wvlCalFileName = fullWvlCalFileName
            wvlCalData = self.wvlCalFile.root.wavecal.calsoln
            self.wvlCalTable = np.zeros([self.nRow, self.nCol, ObsFile.nCalCoeffs])
            self.wvlErrorTable = np.zeros([self.nRow, self.nCol])
            self.wvlFlagTable = np.zeros([self.nRow, self.nCol])
            self.wvlRangeTable = np.zeros([self.nRow, self.nCol, 2])
            for calPixel in wvlCalData:
                #use the current loaded beammap
                entryRows,entryCols = np.where((calPixel['roach'] == self.beamImageRoaches) & (calPixel['pixelnum'] == self.beamImagePixelNums))
                try:
                    entryRow = entryRows[0]
                    entryCol = entryCols[0]
                    
                    self.wvlFlagTable[entryRow,entryCol] = calPixel['wave_flag']
                    self.wvlErrorTable[entryRow,entryCol] = calPixel['sigma']
                    if calPixel['wave_flag'] == 0:
                        self.wvlCalTable[entryRow,entryCol] = calPixel['polyfit']
                        self.wvlRangeTable[entryRow,entryCol] = calPixel['solnrange']
                except IndexError: #entry for an unbeammapped pixel
                    pass 
#            for calPixel in wvlCalData:
#                #rely on the beammap loaded when the cal was done
#                self.wvlFlagTable[calPixel['pixelrow']][calPixel['pixelcol']] = calPixel['wave_flag']
#                self.wvlErrorTable[calPixel['pixelrow']][calPixel['pixelcol']] = calPixel['sigma']
#                if calPixel['wave_flag'] == 0:
#                    self.wvlCalTable[calPixel['pixelrow']][calPixel['pixelcol']] = calPixel['polyfit']
#                    self.wvlRangeTable[calPixel['pixelrow']][calPixel['pixelcol']] = calPixel['solnrange']
        except IOError:
            print('wavelength cal file does not exist: ', fullWvlCalFileName)
            raise
            
    def loadAllCals(self,calLookupTablePath=None,wvlCalPath=None,flatCalPath=None,
            fluxCalPath=None,timeMaskPath=None,timeAdjustmentPath=None,cosmicMaskPath=None,
            beammapPath=None,centroidListPath=None):
        """
        loads all possible cal files from parameters or a calLookupTable. To avoid loading a particular cal, set the corresponding parameter to the empty string ''
        """

        
        _,_,obsTstamp = FileName(obsFile=self).getComponents()

        if beammapPath is None:
            beammapPath = calLookupTable.beammap(obsTstamp)
        if beammapPath != '':
            self.loadBeammapFile(beammapPath)
            print('loaded beammap',beammapPath)
        else:
            print('did not load new beammap')

        if wvlCalPath is None:
            wvlCalPath = calLookupTable.calSoln(obsTstamp)
        if wvlCalPath != '':
            self.loadWvlCalFile(wvlCalPath)
            print('loaded wavecal',self.wvlCalFileName)
        else:
            print('did not load wavecal')

        if flatCalPath is None:
            flatCalPath = calLookupTable.flatSoln(obsTstamp)
        if flatCalPath != '':
            self.loadFlatCalFile(flatCalPath)
            print('loaded flatcal',self.flatCalFileName)
        else:
            print('did not load flatcal')

        if fluxCalPath is None:
            fluxCalPath = calLookupTable.fluxSoln(obsTstamp)
        if fluxCalPath != '':
            self.loadFluxCalFile(fluxCalPath)
            print('loaded fluxcal',self.fluxCalFileName)
        else:
            print('did not load fluxcal')

        if timeMaskPath is None:
            timeMaskPath = calLookupTable.timeMask(obsTstamp)
        if timeMaskPath != '':
            self.loadTimeMask(timeMaskPath)
            print('loaded time mask',timeMaskPath)
        else:
            print('did not load time mask')

        if timeAdjustmentPath is None:
            timeAdjustmentPath = calLookupTable.timeAdjustments(obsTstamp)
        if timeAdjustmentPath != '':
            self.loadTimeAdjustmentFile(timeAdjustmentPath)
            print('loaded time adjustments',self.timeAdjustFileName)
        else:
            print('did not load time adjustments')

        if cosmicMaskPath is None:
            cosmicMaskPath = calLookupTable.cosmicMask(obsTstamp)
        if cosmicMaskPath != '':
            self.loadCosmicMask(cosmicMaskPath)
            print('loaded cosmic mask',self.cosmicMaskFileName)
        else:
            print('did not load cosmic mask')


        if centroidListPath is None:
            centroidListPath = calLookupTable.centroidList(obsTstamp)
        if centroidListPath != '':
            self.loadCentroidListFile(centroidListPath)
            print('loaded centroid list',self.centroidListFileName)
        else:
            print('did not load centroid list')

    def loadFilter(self, filterName = 'V', wvlBinEdges = None,switchOnFilter = True):
        '''
        '''
        std = MKIDStd.MKIDStd()
        self.rawFilterWvls, self.rawFilterTrans = std._loadFilter(filterName)
        #check to see if wvlBinEdges are provided, and make them if not
        if wvlBinEdges == None:
            if self.flatCalFile is not None:
                print("No wvlBinEdges provided, using bins defined by flatCalFile")
                wvlBinEdges = self.flatCalWvlBins
            else:
                raise ValueError("No wvlBinEdges provided. Please load flatCalFile or make bins with ObsFile.makeWvlBins")
        self.rawFilterTrans/=max(self.rawFilterTrans) #normalize filter to 1
        rebinned = utils.rebin(self.rawFilterWvls, self.rawFilterTrans, wvlBinEdges)
        self.filterWvlBinEdges = wvlBinEdges
        self.filterWvls = rebinned[:,0]
        self.filterTrans = rebinned[:,1]
        self.filterTrans[np.isnan(self.filterTrans)] = 0.0
        if switchOnFilter: self.switchOnFilter()

    def switchOffFilter(self):
        self.filterIsApplied = False
        print("Turned off synthetic filter")

    def switchOnFilter(self):
        if self.filterTrans != None:
            self.filterIsApplied = True
            print("Turned on synthetic filter")
        else:
            print("No filter loaded! Use loadFilter to select a filter first")
            self.filterIsApplied = False

    @staticmethod
    def makeWvlBins(energyBinWidth=.1, wvlStart=3000, wvlStop=13000):
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
        energyStop = ObsFile.h * ObsFile.c * ObsFile.angstromPerMeter / wvlStart
        energyStart = ObsFile.h * ObsFile.c * ObsFile.angstromPerMeter / wvlStop
        nWvlBins = int((energyStop - energyStart) / energyBinWidth)
        #Construct energy bin edges
        energyBins = np.linspace(energyStart, energyStop, nWvlBins + 1)
        #Convert back to wavelength and reverse the order to get increasing wavelengths
        wvlBinEdges = np.array(ObsFile.h * ObsFile.c * ObsFile.angstromPerMeter / energyBins)
        wvlBinEdges = wvlBinEdges[::-1]
        return wvlBinEdges

    def parsePhotonPacketLists(self, packets, doParabolaFitPeaks=True, doBaselines=True):
        """
        Parses an array of uint64 packets with the obs file format
        inter is an interval of time values to mask out
        returns a list of timestamps,parabolaFitPeaks,baselines
        """
        # parse all packets
        packetsAll = [np.array(packetList, dtype='uint64') for packetList in packets] #64 bit photon packet
        timestampsAll = [np.bitwise_and(packetList, self.timestampMask) for packetList in packetsAll]
        outDict = {'timestamps':timestampsAll}

        if doParabolaFitPeaks:
            parabolaFitPeaksAll = [np.bitwise_and(\
                np.right_shift(packetList, self.nBitsAfterParabolaPeak), \
                    self.pulseMask) for packetList in packetsAll]
            outDict['parabolaFitPeaks']=parabolaFitPeaksAll

        if doBaselines:
            baselinesAll = [np.bitwise_and(\
                np.right_shift(packetList, self.nBitsAfterBaseline), \
                    self.pulseMask) for packetList in packetsAll]
            outDict['baselines'] = baselinesAll
            
        return outDict

    def parsePhotonPackets(self, packets, doParabolaFitPeaks=True, doBaselines=True):
        """
        Parses an array of uint64 packets with the obs file format
        inter is an interval of time values to mask out
        returns a list of timestamps,parabolaFitPeaks,baselines
        """
        # parse all packets
        packetsAll = np.array(packets, dtype='uint64') #64 bit photon packet
        timestampsAll = np.bitwise_and(packets, self.timestampMask)

        if doParabolaFitPeaks:
            parabolaFitPeaksAll = np.bitwise_and(\
                np.right_shift(packets, self.nBitsAfterParabolaPeak), \
                    self.pulseMask)
        else:
            parabolaFitPeaksAll = np.arange(0)

        if doBaselines:
            baselinesAll = np.bitwise_and(\
                np.right_shift(packets, self.nBitsAfterBaseline), \
                    self.pulseMask)
        else:
            baselinesAll = np.arange(0)

        timestamps = timestampsAll
        parabolaFitPeaks = parabolaFitPeaksAll
        baselines = baselinesAll
        # return the values filled in above
        return timestamps, parabolaFitPeaks, baselines

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

    def parsePhotonPackets_old(self, packets, inter=interval(),
                           doParabolaFitPeaks=True, doBaselines=True,timestampOffset=0):
        """
        Parses an array of uint64 packets with the obs file format
        inter is an interval of time values to mask out
        returns a list of timestamps,parabolaFitPeaks,baselines
        """

        # first special case:  inter masks out everything so return zero-length
        # numpy arrays
        if (inter == self.intervalAll):
            timestamps = np.arange(0)
            parabolaFitPeaks = np.arange(0)
            baselines = np.arange(0)
        else:
            # parse all packets
            packetsAll = np.array(packets, dtype='uint64') #64 bit photon packet
            timestampsAll = np.bitwise_and(packets, self.timestampMask)

            if doParabolaFitPeaks:
                parabolaFitPeaksAll = np.bitwise_and(\
                    np.right_shift(packets, self.nBitsAfterParabolaPeak), \
                        self.pulseMask)
            else:
                parabolaFitPeaksAll = np.arange(0)

            if doBaselines:
                baselinesAll = np.bitwise_and(\
                    np.right_shift(packets, self.nBitsAfterBaseline), \
                        self.pulseMask)
            else:
                baselinesAll = np.arange(0)

            if inter == interval() or len(timestampsAll) == 0:
                # nothing excluded or nothing to exclude
                # so return all unpacked values
                timestamps = timestampsAll
                parabolaFitPeaks = parabolaFitPeaksAll
                baselines = baselinesAll
            else:
                # there is a non-trivial set of times to mask. 
                slices = calculateSlices(inter, timestampsAll)
                timestamps = repackArray(timestampsAll, slices)
                parabolaFitPeaks = repackArray(parabolaFitPeaksAll, slices)
                baselines = repackArray(baselinesAll, slices)
        # return the values filled in above
        return timestamps, parabolaFitPeaks, baselines

    def plotApertureSpectrum(self, pixelRow, pixelCol, radius1, radius2, weighted=False, fluxWeighted=False, lowCut=3000, highCut=7000, firstSec=0,integrationTime=-1):
        summed_array, bin_edges = self.getApertureSpectrum(pixelCol=pixelCol, pixelRow=pixelRow, radius1=radius1, radius2=radius2, weighted=weighted, fluxWeighted=fluxWeighted, lowCut=lowCut, highCut=highCut, firstSec=firstSec,integrationTime=integrationTime)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(bin_edges[12:-2], summed_array[12:-1])
        plt.xlabel('Wavelength ($\AA$)')
        plt.ylabel('Counts')
        plt.show()

    def plotPixelSpectra(self, pixelRow, pixelCol, firstSec=0, integrationTime= -1,
                         weighted=False, fluxWeighted=False):
        """
        plots the wavelength calibrated spectrum of a given pixel integrated over a given time
        if integrationTime is -1, All time after firstSec is used.  
        if weighted is True, flat cal weights are applied
        """
        spectrum = (self.getPixelSpectrum(pixelRow, pixelCol, firstSec, integrationTime,
                    weighted=weighted, fluxWeighted=fluxWeighted))['spectrum']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.flatCalWvlBins[0:-1], spectrum, label='spectrum for pixel[%d][%d]' % (pixelRow, pixelCol))
        plt.show()

    def setWvlCutoffs(self, wvlLowerLimit=3000, wvlUpperLimit=8000):
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
        self.hotPixIsApplied = False

    def switchOnHotPixTimeMask(self,reasons=[]):
        """
        Switch on hot pixel time masking. Subsequent calls to getPixelCountImage
        etc. will have bad pixel times removed.
        """
        if self.hotPixTimeMask is None:
            raise RuntimeError('No hot pixel file loaded')
        self.hotPixIsApplied = True
        if len(reasons)>0:
            self.hotPixTimeMask.set_mask(reasons)
            #self.hotPixTimeMask.mask = [self.hotPixTimeMask.reasonEnum[reason] for reason in reasons]
        

    def switchOffCosmicTimeMask(self):
        """
        Switch off hot pixel time masking - bad pixel times will no longer be
        removed (although the mask remains 'loaded' in ObsFile instance).
        """
        self.cosmicMaskIsApplied = False

    def switchOnCosmicTimeMask(self):
        """
        Switch on cosmic time masking. Subsequent calls to getPixelCountImage
        etc. will have cosmic times removed.
        """
        if self.cosmicMask is None:
            raise RuntimeError('No cosmic mask file loaded')
        self.cosmicMaskIsApplied = True
    @staticmethod
    def writeCosmicIntervalToFile(intervals, ticksPerSec, fileName,
                                  beginTime, endTime, stride,
                                  threshold, nSigma, populationMax):
        h5f = tables.open_file(fileName, 'w')

        headerGroup = h5f.createGroup("/", 'Header', 'Header')
        headerTable = h5f.createTable(headerGroup,'Header',
                                      cosmicHeaderDescription, 'Header')
        header = headerTable.row
        header['ticksPerSec'] = ticksPerSec
        header['beginTime'] = beginTime
        header['endTime'] = endTime
        header['stride'] = stride
        header['threshold'] = threshold
        header['nSigma'] = nSigma
        header['populationMax'] = populationMax 
        header.append()
        headerTable.flush()
        headerTable.close()
        tbl = h5f.createTable("/", "cosmicMaskData", TimeMask.TimeMask, 
                              "Cosmic Mask")
        for interval in intervals:
            row = tbl.row
            tBegin = max(0,int(np.round(interval[0]*ticksPerSec)))
            row['tBegin'] = tBegin
            tEnd = int(np.round(interval[1]*ticksPerSec))
            row['tEnd'] = tEnd
            row['reason'] = TimeMask.timeMaskReason["cosmic"]
            row.append()
            tbl.flush()
        tbl.close()
        h5f.close()

    @staticmethod
    def readCosmicIntervalFromFile(fileName):
        fid = tables.open_file(fileName, mode='r')
        headerInfo = fid.get_node("/Header","Header")[0]
        ticksPerSec = headerInfo['ticksPerSec']
        table = fid.get_node("/cosmicMaskData")
        enum = table.get_enum('reason')

        retval = interval()
        for i in range(table.nrows):
            temp = (interval[table[i]['tBegin'],table[i]['tEnd']])/ticksPerSec
            retval = retval | temp

        fid.close()
        return retval

    @staticmethod
    def invertInterval(interval0, iMin=float("-inf"), iMax=float("inf")):
        """
        invert the interval

        inputs:
          interval0 -- the interval to invert
          iMin=-inf -- beginning of the new interval
          iMax-inv -- end of the new interval
      
        return:
          the interval between iMin, iMax that is NOT masked by interval0
    """
        if len(interval0) == 0:
            retval = interval[iMin,iMax]
        else:
            retval = interval()
            previous = [iMin,iMin]
            for segment in interval0:
                if previous[1] < segment[0]:
                    temp = interval[previous[1],segment[0]]
                    if len(temp) > 0: 
                        retval = retval | temp
                    previous = segment
            if previous[1] < iMax:
                temp = interval[previous[1],iMax]
                if len(temp) > 0:
                    retval = retval | temp
            return retval

    def writePhotonList(self,*nkwargs,**kwargs): #filename=None, firstSec=0, integrationTime=-1):                       
        """
        Write out the photon list for this obs file.
        See photonlist/photlist.py for input parameters and outputs.
        Shifted over to photonlist/, May 10 2013, JvE. All under construction at the moment.
        """        
        import photonlist.photlist      #Here instead of at top to avoid circular imports
        photonlist.photlist.writePhotonList(self,*nkwargs,**kwargs)
        
        
#        writes out the photon list for this obs file at $MKID_PROC_PATH/photonListFileName
#        currently cuts out photons outside the valid wavelength ranges from the wavecal
#       
#        Currently being updated... JvE 4/26/2013.
#        This version should automatically reject time-masked photons assuming a hot pixel mask is
#        loaded and 'switched on'.
#        
#        INPUTS:
#            filename - string, optionally use to specify non-default output file name
#                       for photon list. If not supplied, default name/path is determined
#                       using original obs. file name and standard directory paths (as per
#                       util.FileName). Added 4/29/2013, JvE.
#            firstSec - Start time within the obs. file from which to begin the
#                       photon list (in seconds, from the beginning of the obs. file).
#            integrationTime - Length of exposure time to extract (in sec, starting from
#                       firstSec). -1 to extract to end of obs. file.
#        
#        """
#        
#        if self.flatCalFile is None: raise RuntimeError, "No flat cal. file loaded"
#        if self.fluxCalFile is None: raise RuntimeError, "No flux cal. file loaded"
#        if self.wvlCalFile is None: raise RuntimeError, "No wavelength cal. file loaded"
#        if self.hotPixFile is None: raise RuntimeError, "No hot pixel file loaded"
#        if self.file is None: raise RuntimeError, "No obs file loaded...?"
#        
#        plFile = self.createEmptyPhotonListFile(filename)
#        #try:
#        plTable = plFile.root.photons.photons
#                
#        try:
#            plFile.copyNode(self.flatCalFile.root.flatcal, newparent=plFile.root, newname='flatcal', recursive=True)
#            plFile.copyNode(self.fluxCalFile.root.fluxcal, newparent=plFile.root, newname='fluxcal', recursive=True)
#            plFile.copyNode(self.wvlCalFile.root.wavecal, newparent=plFile.root, newname='wavecal', recursive=True)
#            plFile.copyNode(self.hotPixFile.root, newparent=plFile.root, newname='timemask', recursive=True)
#            plFile.copyNode(self.file.root.beammap, newparent=plFile.root, newname='beammap', recursive=True)
#            plFile.copyNode(self.file.root.header, newparent=plFile.root, recursive=True)
#        except:
#            plFile.flush()
#            plFile.close()
#            raise
#        
#        plFile.flush()
#
#        fluxWeights = self.fluxWeights      #Flux weights are independent of pixel location.
#        #Extend flux weight/flag arrays as for flat weight/flags.
#        fluxWeights = np.hstack((fluxWeights[0],fluxWeights,fluxWeights[-1]))
#        fluxFlags = np.hstack((pipelineFlags.fluxCal['belowWaveCalRange'], 
#                               self.fluxFlags, 
#                               pipelineFlags.fluxCal['aboveWaveCalRange']))
#
#        for iRow in xrange(self.nRow):
#            for iCol in xrange(self.nCol):
#                flag = self.wvlFlagTable[iRow, iCol]
#                if flag == 0:#only write photons in good pixels  ***NEED TO UPDATE TO USE DICTIONARY***
#                    energyError = self.wvlErrorTable[iRow, iCol] #Note wvlErrorTable is in eV !! Assume constant across all wavelengths. Not the best approximation, but a start....
#                    flatWeights = self.flatWeights[iRow, iCol]
#                    #Extend flat weight and flag arrays at beginning and end to include out-of-wavelength-calibration-range photons.
#                    flatWeights = np.hstack((flatWeights[0],flatWeights,flatWeights[-1]))
#                    flatFlags = np.hstack((pipelineFlags.flatCal['belowWaveCalRange'],
#                                           self.flatFlags[iRow, iCol],
#                                           pipelineFlags.flatCal['aboveWaveCalRange']))
#                    
#                    
#                    #wvlRange = self.wvlRangeTable[iRow, iCol]
#
#                    #---------- Replace with call to getPixelWvlList -----------
#                    #go through the list of seconds in a pixel dataset
#                    #for iSec, secData in enumerate(self.getPixel(iRow, iCol)):
#                        
#                    #timestamps, parabolaPeaks, baselines = self.parsePhotonPackets(secData)
#                    #timestamps = iSec + self.tickDuration * timestamps
#                 
#                    #pulseHeights = np.array(parabolaPeaks, dtype='double') - np.array(baselines, dtype='double')
#                    #wavelengths = self.convertToWvl(pulseHeights, iRow, iCol, excludeBad=False)
#                    #------------------------------------------------------------
#
#                    x = self.getPixelWvlList(iRow,iCol,excludeBad=False,dither=True,firstSec=firstSec,
#                                             integrationTime=integrationTime)
#                    timestamps, wavelengths = x['timestamps'], x['wavelengths']     #Wavelengths in Angstroms
#                    
#                    #Convert errors in eV to errors in Angstroms (see notebook, May 7 2013)
#                    wvlErrors = ((( (energyError*units.eV) * (wavelengths*units.Angstrom)**2 ) /
#                                    (constants.h*constants.c) )
#                                 .to(units.Angstrom).value)
#                        
#                    #Calculate what wavelength bin each photon falls into to see which flat cal factor should be applied
#                    if len(wavelengths) > 0:
#                        flatBinIndices = np.digitize(wavelengths, self.flatCalWvlBins)      #- 1 - 
#                    else:
#                        flatBinIndices = np.array([])
#
#                    #Calculate which wavelength bin each photon falls into for the flux cal weight factors.
#                    if len(wavelengths) > 0:
#                        fluxBinIndices = np.digitize(wavelengths, self.fluxCalWvlBins)
#                    else:
#                        fluxBinIndices = np.array([])
#
#                    for iPhoton in xrange(len(timestamps)):
#                        #if wavelengths[iPhoton] > wvlRange[0] and wavelengths[iPhoton] < wvlRange[1] and binIndices[iPhoton] >= 0 and binIndices[iPhoton] < len(flatWeights):
#                        #create a new row for the photon list
#                        newRow = plTable.row
#                        newRow['Xpix'] = iCol
#                        newRow['Ypix'] = iRow
#                        newRow['ArrivalTime'] = timestamps[iPhoton]
#                        newRow['Wavelength'] = wavelengths[iPhoton]
#                        newRow['WaveError'] = wvlErrors[iPhoton]
#                        newRow['FlatFlag'] = flatFlags[flatBinIndices[iPhoton]]
#                        newRow['FlatWeight'] = flatWeights[flatBinIndices[iPhoton]]
#                        newRow['FluxFlag'] = fluxFlags[fluxBinIndices[iPhoton]]
#                        newRow['FluxWeight'] = fluxWeights[fluxBinIndices[iPhoton]]
#                        newRow.append()
#        #finally:
#        plTable.flush()
#        plFile.close()
#




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

def diagnose(message,timestamps, peakHeights, baseline, expTails):
    print("BEGIN DIAGNOSE message=",message)
    index = np.searchsorted(timestamps,99.000426)
    print("index=",index)
    for i in range(index-1,index+2):
        print("i=%5d timestamp=%11.6f"%(i,timestamps[i]))
    print("ENDED DIAGNOSE message=",message)

class cosmicHeaderDescription(tables.IsDescription):
    ticksPerSec = tables.Float64Col() # number of ticks per second
    beginTime = tables.Float64Col()   # begin time used to find cosmics (seconds)
    endTime = tables.Float64Col()     # end time used to find cosmics (seconds)
    stride = tables.Int32Col()
    threshold = tables.Float64Col()
    nSigma = tables.Int32Col()
    populationMax = tables.Int32Col()


#Temporary test
if __name__ == "__main__":
    obs = ObsFile(FileName(run='PAL2012', date='20121210', tstamp='20121211-051650').obs())
    obs.loadWvlCalFile(FileName(run='PAL2012',date='20121210',tstamp='20121211-052230').calSoln())
    obs.loadFlatCalFile(FileName(obsFile=obs).flatSoln())
    beforeImg = obs.getPixelCountImage(weighted=False,fluxWeighted=False,scaleByEffInt=True)
