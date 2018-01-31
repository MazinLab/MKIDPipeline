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
convertToWvl(self, pulseHeights, xCoord, yCoord, excludeBad=True)
createEmptyPhotonListFile(self)
displaySec(self, firstSec=0, integrationTime= -1, weighted=False,fluxWeighted=False, plotTitle='', nSdevMax=2,scaleByEffInt=False)
getFromHeader(self, name)
getPixel(self, xCoord, yCoord, firstSec=0, integrationTime= -1)
getPixelWvlList(self,xCoord,yCoord,firstSec=0,integrationTime=-1,excludeBad=True,dither=True)
getPixelCount(self, xCoord, yCoord, firstSec=0, integrationTime= -1,weighted=False, fluxWeighted=False, getRawCount=False)
getPixelLightCurve(self, xCoord, yCoord, firstSec=0, lastSec=-1, cadence=1, **kwargs)
getPixelPacketList(self, xCoord, yCoord, firstSec=0, integrationTime= -1)
getTimedPacketList_old(self, xCoord, yCoord, firstSec=0, integrationTime= -1)
getTimedPacketList(self, xCoord, yCoord, firstSec=0, integrationTime= -1)
getPixelCountImage(self, firstSec=0, integrationTime= -1, weighted=False,fluxWeighted=False, getRawCount=False,scaleByEffInt=False)
getAperturePixelCountImage(self, firstSec=0, integrationTime= -1, y_values=range(46), x_values=range(44), y_sky=[], x_sky=[], apertureMask=np.ones((46,44)), skyMask=np.zeros((46,44)), weighted=False, fluxWeighted=False, getRawCount=False, scaleByEffInt=False)
getSpectralCube(self,firstSec=0,integrationTime=-1,weighted=True,wvlStart=3000,wvlStop=13000,wvlBinWidth=None,energyBinWidth=None,wvlBinEdges=None)
getPixelSpectrum(self, pixelRow, pixelCol, firstSec=0, integrationTime= -1,weighted=False, fluxWeighted=False, wvlStart=3000, wvlStop=13000, wvlBinWidth=None, energyBinWidth=None, wvlBinEdges=None)
getPixelBadTimes(self, pixelRow, pixelCol)
getDeadPixels(self, showMe=False, weighted=True, getRawCount=False)
getNonAllocPixels(self, showMe=False)
getRoachNum(self,xCoord,yCoord)
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
plotPixelLightCurve(self,xCoord,yCoord,firstSec=0,lastSec=-1,cadence=1,**kwargs)
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
from interval import interval,inf
import tables
from tables.nodes import filenode
import astropy.constants
from regions import PixCoord, CirclePixelRegion, RectanglePixelRegion
import numpy.lib.recfunctions as nlr

from P3Utils import utils
from P3Utils import MKIDStd
from P3Utils.FileName import FileName
from Headers import TimeMask

class ObsFile:
    h = astropy.constants.h.to('eV s').value  #4.135668e-15 #eV s
    c = astropy.constants.c.to('m/s').value   #'2.998e8 #m/s
    angstromPerMeter = 1e10
    nCalCoeffs = 3
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
        assert mode=='read' or mode=='write', '"mode" argument must be "read" or "write"'
        self.mode = mode
        self.makeMaskVersion = None
        self.loadFile(fileName,verbose=verbose)
        self.photonTable = self.file.get_node('/Photons/PhotonTable')
        self.filterIsApplied = False

        self.filterIsApplied = False
        self.noResIDFlag = 2**32-1
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
        for xCoord in range(self.nXPix):
            for yCoord in range(self.nYPix):
                pixelLabel = self.beamImage[xCoord][yCoord]
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
        if self.mode=='read':
            mode = 'r'
        if self.mode=='write':
            mode = 'a'

        self.file = tables.open_file(self.fullFileName, mode=mode)

        #get the header
        self.header = self.file.root.header.header
        self.titles = self.header.colnames
        try:
            self.info = self.header[0] #header is a table with one row
        except IndexError as inst:
            if verbose:
                print('Can\'t read header for ',self.fullFileName)
            raise inst

        # get important cal params

        self.defaultWvlBins = ObsFile.makeWvlBins(self.getFromHeader('energyBinWidth'), self.getFromHeader('wvlBinStart'), self.getFromHeader('wvlBinEnd'))


        # Useful information about data format set here.
        # For now, set all of these as constants.
        # If we get data taken with different parameters, straighten
        # that all out here.

        ## These parameters are for LICK2012 and PAL2012 data
        self.tickDuration = 1e-6 #s
        self.ticksPerSec = int(1.0 / self.tickDuration)
        self.intervalAll = interval[0.0, (1.0 / self.tickDuration) - 1]
        #  8 bits - channel
        # 12 bits - Parabola Fit Peak Height
        # 12 bits - Sampled Peak Height
        # 12 bits - Low pass filter baseline
        # 20 bits - Microsecond timestamp

        #get the beam image.
        try:
            self.beamImage = self.file.get_node('/BeamMap/Map').read()
            self.beamFlagImage = self.file.get_node('/BeamMap/Flag')
        except Exception as inst:
            if verbose:
                print('Can\'t access beamimage for ',self.fullFileName)
            raise inst

        beamShape = self.beamImage.shape
        self.nXPix = beamShape[0]
        self.nYPix = beamShape[1]

    def checkIntegrity(self,firstSec=0,integrationTime=-1):
        """
        Checks the obs file for corrupted end-of-seconds
        Corruption is indicated by timestamps greater than 1/tickDuration=1e6
        returns 0 if no corruption found
        """
        corruptedPixels = []
        for xCoord in range(self.nXPix):
            for yCoord in range(self.nYPix):
                packetList = self.getPixelPacketList(xCoord,yCoord,firstSec,integrationTime)
                timestamps,parabolaPeaks,baselines = self.parsePhotonPackets(packetList)
                if np.any(timestamps > 1./self.tickDuration):
                    print('Corruption detected in pixel (',xCoord,yCoord,')')
                    corruptedPixels.append((xCoord,yCoord))
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
#                    pixelLabel = self.beamImage[xCoord][yCoord]
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

    def getPixelPhotonList(self, xCoord, yCoord, firstSec=0, integrationTime= -1, wvlRange=None, isWvl=True):
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
        wvlRange: (float, float)
            Desired wavelength range of photon list. Must satisfy wvlRange[0] <= wvlRange[1].
            If None, includes all wavelengths. If file is not wavelength calibrated, this parameter
            specifies the range of desired phase heights.
        isWvl: bool
            If True, wvlRange specifies wavelengths. Else, wvlRange is assumed to specify uncalibrated
            phase heights.

        Returns
        -------
        Structured Numpy Array
            Each row is a photon.
            Columns have the following keys: 'Time', 'Wavelength', 'SpecWeight', 'NoiseWeight'

        """
        resID = self.beamImage[xCoord][yCoord]
        if resID==self.noResIDFlag:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            return self.photonTable.read_where('(Time < 0)') #use dummy condition to get empty photon list of correct format

        startTime = int(firstSec*self.ticksPerSec) #convert to us
        endTime = startTime + int(integrationTime*self.ticksPerSec)
        # if integrationTime == -1:
        #     try:
        #         endTime = startTime + int(self.getFromHeader('expTime'))*self.ticksPerSec
        #     except ValueError:
        #         try:
        #             endTime = startTime + photonTable.read(-1)[0][0]
        #         except IndexError:
        #             endTime = startTime + 1 #Assume table is empty
        # else:
        #     endTime = startTime + int(integrationTime*self.ticksPerSec)

        wvlRange = None   ##IL:  Patch because for some reason wvlRange gets set to false after the getSpectralCube step
        if wvlRange is None and integrationTime==-1:
            photonList = self.photonTable.read_where('ResID==resID')

        elif wvlRange is None:
            photonList = self.photonTable.read_where('(ResID == resID) & (Time >= startTime) & (Time < endTime)')

        else:
            if(isWvl != self.info['isWvlCalibrated']):
                raise Exception('isWvl does not match wavelength cal status! \nisWvlCalibrated = ' + str(self.info['isWvlCalibrated']) + '\nisWvl = ' + str(isWvl))
            startWvl = wvlRange[0]
            endWvl = wvlRange[1]
            assert startWvl <= endWvl, 'wvlRange[0] must be <= wvlRange[1]'
            if integrationTime == -1:
                photonList = photonTable.read_where('(ResID == resID) & (Wavelength >= startWvl) & (Wavelength < endWvl)')
            else:
                photonList = photonTable.read_where('(ResID == resID) & (Time > startTime) & (Time < endTime) & (Wavelength >= startWvl) & (Wavelength < endWvl)')

        #return {'pixelData':pixelData,'firstSec':firstSec,'lastSec':lastSec}
        return photonList

    def getListOfPixelsPhotonList(self, posList, firstSec=0, integrationTime=-1, wvlRange=None):
        """
        Retrieves photon lists for a list of pixels.

        Parameters
        ----------
        posList: Nx2 array of ints (or list of 2 element tuples)
            List of (x, y) beammap indices for desired pixels
        firstSec: float
            Photon list start time, in seconds relative to beginning of file
        integrationTime: float
            Photon list end time, in seconds relative to firstSec.
            If -1, goes to end of file
        wvlRange: (float, float)
            Desired wavelength range of photon list. Must satisfy wvlRange[0] < wvlRange[1].
            If None, includes all wavelengths. If file is not wavelength calibrated, this parameter
            specifies the range of desired phase heights.

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
            photonLists.append(self.getPixelPhotonList(posList[i][0], posList[i][1], firstSec, integrationTime, wvlRange))

        return photonLists

    def getPixelCount(self, xCoord, yCoord, firstSec=0, integrationTime= -1, wvlRange=None, applyWeight=True, applyTPFWeight=True, applyTimeMask=True):
        """
        Returns the number of photons received in a single pixel from firstSec to firstSec + integrationTime

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

        Returns
        -------
        Dictionary with keys:
            'counts':int, number of photon counts
            'effIntTime':float, effective integration time after time-masking is
           `          accounted for.
        """
        photonList = self.getPixelPhotonList(xCoord, yCoord, firstSec, integrationTime, wvlRange)
        weights = np.ones(len(photonList))
        if applyWeight:
            weights *= photonList['SpecWeight']
        if applyTPFWeight:
            weights *= photonList['NoiseWeight']
        if applyTimeMask:
            if self.info['timeMaskExists']:
                pass
            else:
                warnings.warn('Time mask does not exist!')

        return {'counts':np.sum(weights), 'effIntTime':integrationTime}


    def getPixelLightCurve(self,xCoord,yCoord,firstSec=0,lastSec=-1,cadence=1,
                           **kwargs):
        """
        Get a simple light curve for a pixel (basically a wrapper for getPixelCount).

        INPUTS:
            xCoord,yCoord - Row and column of pixel
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
        return np.array([self.getPixelCount(xCoord,yCoord,firstSec=x,integrationTime=cadence,**kwargs)['counts']
                       for x in np.arange(firstSec,lSec,cadence)])


    def plotPixelLightCurve(self,xCoord,yCoord,firstSec=0,lastSec=-1,cadence=1,**kwargs):
        """
        Plot a simple light curve for a given pixel. Just a wrapper for getPixelLightCurve.
        Also marks intervals flagged as bad with gray shaded regions if a hot pixel mask is
        loaded.
        """

        lc = self.getPixelLightCurve(xCoord=xCoord,yCoord=yCoord,firstSec=firstSec,lastSec=lastSec,
                                     cadence=cadence,**kwargs)
        if lastSec==-1: realLastSec = self.getFromHeader('exptime')
        else: realLastSec = lastSec

        #Plot the lightcurve
        x = np.arange(firstSec+cadence/2.,realLastSec)
        assert len(x)==len(lc)      #In case there are issues with arange being inconsistent on the number of values it returns
        plt.plot(x,lc)
        plt.xlabel('Time since start of file (s)')
        plt.ylabel('Counts')
        plt.title(self.fileName+' - pixel x,y = '+str(yCoord)+','+str(xCoord))

        #Get bad times in time range of interest (hot pixels etc.)
        badTimes = self.getPixelBadTimes(xCoord,yCoord) & interval([firstSec,realLastSec])   #Returns an 'interval' instance
        lcRange = np.nanmax(lc)-np.nanmin(lc)
        for eachInterval in badTimes:
            plt.fill_betweenx([np.nanmin(lc)-0.5*lcRange,np.nanmax(lc)+0.5*lcRange], eachInterval[0],eachInterval[1],
                              alpha=0.5,color='gray')


    def getPixelCountImage(self, firstSec=0, integrationTime= -1, wvlRange=None, applyWeight=True, applyTPFWeight=True, applyTimeMask=False, scaleByEffInt=False, flagToUse=0):
        """
        Returns an image of pixel counts over the entire array between firstSec and firstSec + integrationTime. Can specify calibration weights to apply as
        well as wavelength range.

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
            'effIntTime':2D numpy array, image effective integration times after time-masking is
           `          accounted for.
        """
        effIntTimes = np.zeros((self.nXPix, self.nYPix), dtype=np.float64)
        effIntTimes.fill(np.nan)   #Just in case an element doesn't get filled for some reason.
        countImage = np.zeros((self.nXPix, self.nYPix), dtype=np.float64)
        #rawCounts.fill(np.nan)   #Just in case an element doesn't get filled for some reason.
        if integrationTime==-1:
            integrationTime = self.getFromHeader('exptime')-firstSec
        startTs = firstSec*1.e6
        endTs = startTs + integrationTime*1.e6
        if wvlRange is None:
            photonList = self.photonTable.read_where('((Time >= startTs) & (Time < endTs))')
        else:
            startWvl = wvlRange[0]
            endWvl = wvlRange[1]
            photonList = self.photonTable.read_where('(Wavelength >= startWvl) & (Wavelength < endWvl) & (Time >= startTs) & (Time < endTs)')

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
                if(self.beamImage[xCoord, yCoord]!=self.noResIDFlag and (flag|flagToUse)==flag):
                    effIntTimes[xCoord, yCoord] = integrationTime
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
                    
                
        if scaleByEffInt is True:
            if integrationTime == -1:
                totInt = self.getFromHeader('exptime')
            else:
                totInt = integrationTime
            countImage *= (totInt / effIntTimes)

        #if getEffInt is True:
        return{'image':countImage, 'effIntTimes':effIntTimes}
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
        secImg = np.zeros((self.nXPix, self.nYPix))
        effIntTimes = np.zeros((self.nXPix, self.nYPix), dtype=np.float64)
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

    def getCircularAperturePhotonList(self, centerXCoord, centerYCoord, radius, firstSec=0, integrationTime=-1, wvlRange=None, flagToUse=0):
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
            pixPhotonList = self.getPixelPhotonList(coords[0], coords[1], firstSec, integrationTime, wvlRange)
            pixPhotonList['NoiseWeight'] *= exactApertureMask[apertureMaskCoords[i,0], apertureMaskCoords[i,1]]
            if photonList is None:
                photonList = pixPhotonList
            else:
                photonList = np.append(photonList, pixPhotonList)

        photonList = np.sort(photonList, order='Time')
        return {'photonList':photonList, 'effQE':np.sum(exactApertureMask)/(np.pi*radius**2), 'apertureMask':exactApertureMask}



    def getSpectralCube(self, firstSec=0, integrationTime=-1, applySpecWeight=False, applyTPFWeight=False, wvlStart=700, wvlStop=1500,
                        wvlBinWidth=None, energyBinWidth=None, wvlBinEdges=None, timeSpacingCut=None):
        """
        Return a time-flattened spectral cube of the counts integrated from firstSec to firstSec+integrationTime.
        If integration time is -1, all time after firstSec is used.
        If weighted is True, flat cal weights are applied.
        If fluxWeighted is True, spectral shape weights are applied.
        """

        cube = [[[] for yCoord in range(self.nYPix)] for xCoord in range(self.nXPix)]
        effIntTime = np.zeros((self.nXPix,self.nYPix))
        rawCounts = np.zeros((self.nXPix,self.nYPix))

        for xCoord in range(self.nXPix):
            for yCoord in range(self.nYPix):
                x = self.getPixelSpectrum(xCoord=xCoord,yCoord=yCoord,
                                  firstSec=firstSec, applySpecWeight=applySpecWeight,
                                  applyTPFWeight=applyTPFWeight, wvlStart=wvlStart, wvlStop=wvlStop,
                                  wvlBinWidth=wvlBinWidth, energyBinWidth=energyBinWidth,
                                  wvlBinEdges=wvlBinEdges, timeSpacingCut=timeSpacingCut)
                cube[xCoord][yCoord] = x['spectrum']
                effIntTime[xCoord][yCoord] = x['effIntTime']
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

        wvlStart=wvlStart if (wvlStart!=None and wvlStart>0.) else (self.wvlLowerLimit if (self.wvlLowerLimit!=None and self.wvlLowerLimit>0.) else 700)
        wvlStop=wvlStop if (wvlStop!=None and wvlStop>0.) else (self.wvlUpperLimit if (self.wvlUpperLimit!=None and self.wvlUpperLimit>0.) else 1500)


        photonList = self.getPixelPhotonList(xCoord, yCoord, firstSec, integrationTime)
        wvlList = photonList['Wavelength']
        rawCounts = len(wvlList)

        if integrationTime==-1:
            effIntTime = self.getFromHeader('expTime')
        else:
            effIntTime = integrationTime
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
        return {'spectrum':spectrum, 'wvlBinEdges':wvlBinEdges, 'effIntTime':effIntTime, 'rawCounts':rawCounts}
        print('again', spectrum)
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
        bad_solution_mask = np.zeros((self.nXPix, self.nYPix))
        for y in range(self.nXPix):
            for x in range(self.nYPix):
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
        countArray = np.array([[(self.getPixelCount(xCoord, yCoord, weighted=weighted,getRawCount=getRawCount))['counts'] for yCoord in range(self.nYPix)] for xCoord in range(self.nXPix)])
        deadArray = np.ones((self.nXPix, self.nYPix))
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
        nonAllocArray = np.ones((self.nXPix, self.nYPix))
        nonAllocArray[np.core.defchararray.startswith(self.beamImage, self.nonAllocPixelName)] = 0
        if showMe == True:
            utils.plotArray(nonAllocArray)
        return nonAllocArray

    def getRoachNum(self,xCoord,yCoord):
        pixelLabel = self.beamImage[xCoord][yCoord]
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
        frame = np.zeros((self.nXPix,self.nYPix),dtype=np.uint32)
        for xCoord in range(self.nXPix):
            for yCoord in range(self.nYPix):
                pl = self.getTimedPacketList(xCoord,yCoord,
                                             firstSec,integrationTime)
                nphoton = pl['timestamps'].size
                frame[xCoord][yCoord] += nphoton
        return frame

    # a different way to get, with the functionality of getTimedPacketList
    def getPackets(self, xCoord, yCoord, firstSec, integrationTime,
                   fields=(),
                   expTailTimescale=None,
                   timeSpacingCut=None,
                   timeMaskLast=True):
        """
        get and parse packets for pixel xCoord,yCoord starting at firstSec for integrationTime seconds.

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
            inter = self.getPixelBadTimes(xCoord, yCoord)
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

        pixelData = self.getPixel(xCoord, yCoord, firstSec=firstSecInt,
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
        self.nXPix = beamShape[0]
        self.nYPix = beamShape[1]

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
        self.flatWeights = np.zeros((self.nXPix,self.nYPix,self.nFlatCalWvlBins),dtype=np.double)
        self.flatFlags = np.zeros((self.nXPix,self.nYPix,self.nFlatCalWvlBins),dtype=np.uint16)

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
        raise NotImplementedError
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
            obs.applyWaveCal(wvlCalFileName)

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

        # appy waveCal
        calsoln = wave_cal.root.wavecal.calsoln.read()
        for (row, column), resID in np.ndenumerate(self.beamImage):
            print(resID)
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
        wave_cal.close()

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            photonTable = self.file.get_node('/Photons/' + str(resID))
        assert len(photonTable)==len(wvlCalArr), 'Calibrated wavelength list does not match length of photon list!'

        photonTable.modify_column(column=wvlCalArr, colname='Wavelength')
        photonTable.flush()

    def applySpecWeight(self, resID, weightArr):
        """
        Applies a weight calibration to the "SpecWeight" column.

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
        if self.mode!='write':
            raise Exception("Must open file in write mode to do this!")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            photonTable = self.file.get_node('/Photons/' + str(resID))
        assert len(photonTable)==len(weightArr), 'Calibrated wavelength list does not match length of photon list!'

        weightArr = np.array(weightArr)
        curWeights = photonTable.col('SpecWeight')
        newWeights = weightArr*curWeights
        photonTable.modify_column(column=newWeights, colname='SpecWeight')
        photonTable.flush()

    def applyFlatCal(self, FlatCalFile):
        assert not self.info['isSpecCalibrated'], \
                "the data is already Flat calibrated"
        assert os.path.exists(FlatCalFile), "{0} does not exist".format(FlatCalFile)
        flat_cal = tables.open_file(FlatCalFile, mode='r')
        calsoln = flat_cal.root.flatcal.calsoln.read()
        bins=np.array(flat_cal.root.flatcal.wavelengthBins.read())
        bins=bins.flatten()
        minwavelength=700
        maxwavelength=1500
        heads=np.array([0,100,200,300,400,500,600])
        tails=np.array([1600,1700,1800,1900,2000])
        bins=np.append(heads,bins)
        bins=np.append(bins,tails)
        for (row, column), resID in np.ndenumerate(self.beamImage):
            index = np.where(resID == np.array(calsoln['resid']))
            if len(index[0]) == 1 and (calsoln['flag'][index] == 0):
                print('resID', resID, 'row', row, 'column', column)
                weights = calsoln['weights'][index]
                photon_list = self.getPixelPhotonList(row, column)
                phases = photon_list['Wavelength']
                weights=np.array(weights)
                weights=weights.flatten()
                weightsheads=np.zeros(len(heads))+weights[0]
                weightstails=np.zeros(len(tails)+1)+weights[len(weights)-1]
                weights=np.append(weightsheads,weights)
                weights=np.append(weights,weightstails)
                weightfxncoeffs10=np.polyfit(bins,weights,10)
                weightfxn10=np.poly1d(weightfxncoeffs10)
                photon_list = self.getPixelPhotonList(row, column)
                phases = photon_list['Wavelength']
                weightArr=weightfxn10(phases)
                weightArr[np.where(phases < minwavelength)]=1.0
                weightArr[np.where(phases > maxwavelength)]=1.0
                obs.applySpecWeight(obsfile, resID=resID, weightArr=weightArr)
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
#        for xCoord in xrange(self.nXPix):
#            for yCoord in xrange(self.nYPix):
#                flag = self.wvlFlagTable[xCoord, yCoord]
#                if flag == 0:#only write photons in good pixels  ***NEED TO UPDATE TO USE DICTIONARY***
#                    energyError = self.wvlErrorTable[xCoord, yCoord] #Note wvlErrorTable is in eV !! Assume constant across all wavelengths. Not the best approximation, but a start....
#                    flatWeights = self.flatWeights[xCoord, yCoord]
#                    #Extend flat weight and flag arrays at beginning and end to include out-of-wavelength-calibration-range photons.
#                    flatWeights = np.hstack((flatWeights[0],flatWeights,flatWeights[-1]))
#                    flatFlags = np.hstack((pipelineFlags.flatCal['belowWaveCalRange'],
#                                           self.flatFlags[xCoord, yCoord],
#                                           pipelineFlags.flatCal['aboveWaveCalRange']))
#
#
#                    #wvlRange = self.wvlRangeTable[xCoord, yCoord]
#
#                    #---------- Replace with call to getPixelWvlList -----------
#                    #go through the list of seconds in a pixel dataset
#                    #for iSec, secData in enumerate(self.getPixel(xCoord, yCoord)):
#
#                    #timestamps, parabolaPeaks, baselines = self.parsePhotonPackets(secData)
#                    #timestamps = iSec + self.tickDuration * timestamps
#
#                    #pulseHeights = np.array(parabolaPeaks, dtype='double') - np.array(baselines, dtype='double')
#                    #wavelengths = self.convertToWvl(pulseHeights, xCoord, yCoord, excludeBad=False)
#                    #------------------------------------------------------------
#
#                    x = self.getPixelWvlList(xCoord,yCoord,excludeBad=False,dither=True,firstSec=firstSec,
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
#                        newRow['Xpix'] = yCoord
#                        newRow['Ypix'] = xCoord
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
