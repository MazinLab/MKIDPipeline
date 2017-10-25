import os
import glob
import matplotlib.pyplot as plt
import numpy
import types
import string
import astropy.io.fits as pyfits #changed from pyfits 20171023
from P3Utils import smooth
import sys
from scipy.constants import *
import math
class MKIDStd:
    """
    This class contains the spectra of several standard stars and other
    objects. These spectra may be plotted and used to compare with data 
    from the MKID detector.

    Wavelength and flux values and text files describing each object are
    saved in the data directory. Each object is described in a .txt file.
    This file lists the file name that contains the data, the units, a 
    citation, and a brief description of the object. 
    """

    def __init__(self, referenceWavelength=5500):
        """
        Loads up the list of objects we know about, filters, and
        Balmer wavelengths.
        referenceWavelength is used in plot() to normalize spectra
        """
        self.referenceWavelength=referenceWavelength
        self.objects = {}
        self.filters = {}
        self.filterList = ['U','B','V','R','I','u','g','r','i','z']
        self.this_dir, this_filename = os.path.split(__file__)
        pattern = os.path.join(self.this_dir,"data","*.txt")
        for file in glob.glob(pattern):
            name,ext = os.path.splitext(os.path.basename(file))
            dictionary = self._loadDictionary(file)
            self.objects[name] = dictionary
        self.balmerwavelengths = [6563,4861,4341,4102,3970,3889,3835,3646]
        self.lymanwavelengths = [1216,1026,973,950,938,931,926,923,921,919]
        self._loadUBVRIFilters()
        self._loadSDSSFilters()
        # h is in Joules/sec and c is in meters/sec. 
        # This k value is used in conversions between counts and ergs
        self.k = ((1.0*10**-10)/(1.0*10**7))/h/c

        self.vegaInCounts = "not loaded yet"

    def _loadFilter(self, filterName):
        if filterName in ['U','B','V','R','I']:
            print("Loading Johnson %s filter"%filterName)
            self._loadUBVRIFilters()
        elif filterName in ['u','g','r','i','z']:
            print("Loading SDSS %s filter"%filterName)
            self._loadSDSSFilters()
        else:
            raise ValueError("INVALID FILTER. Currently supported filters:", self.filterList)
        wvls = self.filters[filterName][0]
        trans = self.filters[filterName][1]
        return wvls, trans

    def _loadUBVRIFilters(self):
            
        filterFileName = os.path.join(self.this_dir,"data","ph08_UBVRI.mht")
        f = open(filterFileName,'r')
        nFilter = -1
        nToRead = -1
        iFilter = -1
        iRead = 0
        for line in f:
            if (nFilter == -1) :
                nFilter = int(line)
            elif (nToRead <= 0):
                nToRead = int(line)
                iFilter += 1
                filter = self.filterList[iFilter]
                self.filters[filter] = numpy.zeros((2,nToRead))
                iRead = 0
            else:
                nToRead -= 1
                vals = line.split()
                self.filters[filter][0,iRead] = vals[0]
                self.filters[filter][1,iRead] = vals[1]
                iRead += 1    

    def _loadSDSSFilters(self):
        for filter in ['u','g','i','r','z']:
            filterFileName = os.path.join(self.this_dir,"data",filter+'.mht')
            temp = numpy.loadtxt(filterFileName)
            npts = temp.shape[0]
            self.filters[filter] = numpy.zeros((2,npts))
            for i in range(npts):
                self.filters[filter][0,i] = temp[i,0]
                self.filters[filter][1,i] = temp[i,3]
            

    def _loadDictionary(self,file):
        retval = {}
        for line in open(file):
            vals = line.strip().split(" = ");
            retval[vals[0]] = vals[1:]
        return retval

    def load(self,name):
        """
        Returns a two dimensional numpy array where a[:,0] is
        wavelength in Angstroms and a[:,1] is flux in 
        counts/sec/angstrom/cm^2
        
        Noisy spectra are smoothed with window_len in the .txt file.
        Ergs and AB Mag units are automatically converted to counts.
        """
        fname = self.objects[name]['dataFile']
        fullFileName = os.path.join(self.this_dir,"data",fname[0])
        if (string.count(fullFileName,"fit")):
            a = self.loadSdssSpecFits(fullFileName)
        else:
            a = numpy.loadtxt(fullFileName)
            
        len = int(self.objects[name]['window_len'][0])
        if len > 1:
            #temp = smooth.smooth(a[:,1], window_len=len)[len/2:-(len/2)]
            temp = smooth.smooth(a[:,1], window_len=len)
            a[:,1] = temp[1:]
        try:
            fluxUnit = self.objects[name]['fluxUnit'][0]
            scale = float(fluxUnit.split()[0])
            a[:,1] *= scale
        except ValueError:
            scale = 1

        ergs = string.count(self.objects[name]['fluxUnit'][0],"ergs")
        if ergs:
            a[:,1] *= (a[:,0] * self.k)
        mag = string.count(self.objects[name]['fluxUnit'][0],"mag")
        if mag:
            a[:,1] = \
                (10**(-2.406/2.5))*(10**(-0.4*a[:,1]))/(a[:,0]**2) * \
                (a[:,0] * self.k)
        return a
    
    def normalizeFlux(self,a):
        """
        This function normalizes the flux at self.referenceWavelength
        """
        referenceFlux = self.getFluxAtReferenceWavelength(a)
        a[:,1] /= referenceFlux
        return a

    def countsToErgs(self,a):
        """
        This function changes the units of the spectra from counts to 
        ergs. 
        """
        a[:,1] /= (a[:,0] * self.k)
        return a

    def ergsToCounts(self,a):
        """
        This function changes the units of the spectra from ergs to 
        counts. 
        """
        a[:,1] *= (a[:,0] * self.k)
        return a
    
    def measureBandPassFlux(self,aFlux,aFilter):
        """
        This function measures the band pass flux of the object in the
        filter.
        """
        sum = 0
        sumd = 0
        filter = numpy.interp(aFlux[:,0], aFilter[0,:], aFilter[1,:], 0, 0)
        for i in range(aFlux[:,0].size-1):
            dw = aFlux[i+1,0] - aFlux[i,0]
            flux = aFlux[i,1]*filter[i]/aFlux[i,0]
            sum += flux*dw
            sumd += filter[i]*dw
        sum /= self.k
        sum /= sumd
        return sum

    def _getVegaMag(self, aFlux, aFilter):
        #if self.vegaInCounts == "not loaded yet":
        self.vegaInCounts = self.load("vega")
        sumNumerator = 0.0
        sumDenominator = 0.0
        filter = numpy.interp(aFlux[:,0], aFilter[0,:], aFilter[1,:], 0, 0)
        vFlux = numpy.interp(
            aFlux[:,0], self.vegaInCounts[:,0], self.vegaInCounts[:,1], 0, 0)
        for i in range(aFlux[:,0].size-1):
            dw = aFlux[i+1,0] - aFlux[i,0]
            sumNumerator += aFlux[i,1]*filter[i]*dw
            sumDenominator += vFlux[i]*filter[i]*dw
            #print "i=%4d filter=%5.3f flux=%f" % (i,filter[i], aFlux[i,1])
        #print "    sumNumerator=",sumNumerator
        #print "  sumDenominator=",sumDenominator
        mag = -2.5*math.log10(sumNumerator/sumDenominator) + 0.03
        return mag

    def getVegaMag(self,name,Filter):
        """
        Returns the magnitude of the desired object at the desired filter.
        """
        aFlux = self.load(name)
        aFilter = self.filters[Filter]
        a = self._getVegaMag(aFlux, aFilter)
        return a

    def plot(self,name="all",xlog=False,ylog=True,xlim=[3000,13000],normalizeFlux=True,countsToErgs=False):
        """
        Makes a png file that plots the arrays a[:,0] (wavelength) and
        a[:,1] (flux) with balmer wavelengths indicated. Individual
        spectra are labeled and indicated by a legend.
        
        plot() plots the spectrum of all standard stars in the program.
        plot(['vega','bd17']) returns only the spectrum for those two
        stars.
        plot('vega') plots the spectrum for only that star.

        Whether the axes are plotted logaritmically is controlled by the
        option parameter xlog and ylog.  
        
        The optional parameter xlim sets the wavelength limits of the plot.
        The plot y limits are from flux values for wavelengths included
        in xlim.

        The flux values are in counts/sec/cm^2/A by default, but they can be 
        changed to ergs by setting countsToErgs=True when calling the 
        function.
        
        By default fluxes are normalized to 1 at self.refernceWavelength
        and setting normalizeFlux=False disables normalization
        
        The filename of the plot in the current working director is returned.
        """
        if (name == "all"):
            listofobjects = list(self.objects.keys())
            listofobjects.sort()
            plotName = "all"
        elif (isinstance(name, list)):
            listofobjects = name
            plotName = name[0]+"_group"
        else:
            plotName = name
            listofobjects = [name]
        plt.clf()
        plotYMin = -1
        plotYMax = -1
        for tname in listofobjects:
            a = self.load(tname)
            if (countsToErgs):
                a = self.countsToErgs(a)
            if (normalizeFlux):
                a = self.normalizeFlux(a)
            a.shape
            x = a[:,0]
            y = a[:,1]
            if (not xlog and ylog):
                plt.semilogy(x,y, label=tname)
            if (not ylog and xlog):
                plt.semilogx(x,y, label=tname)
            if (not xlog and not ylog):
                plt.plot(x,y, label=tname)
            if (xlog and ylog):
                plt.loglog(x,y, label=tname)
            imin = numpy.searchsorted(x,xlim[0])
            imax = numpy.searchsorted(x,xlim[1])
            ytemp = y[imin:imax]
            ymin = abs(ytemp).min()
            ymax = ytemp.max()
            if (plotYMin == -1):
                plotYMin = ymin
                plotYMax = ymax
            else:
                plotYMin = min(plotYMin,ymin)
                plotYMax = max(plotYMax,ymax)
        for x in self.balmerwavelengths:
            plt.plot([x,x],[plotYMin,plotYMax], 'r--')
        plt.xlabel('wavelength(Angstroms)')
        if (countsToErgs):
            ylabel = 'flux(ergs/sec/cm2/A)'
        else:
            ylabel = 'flux(counts/sec/cm2/A)'
        if (normalizeFlux):
            ylabel += '['+str(self.referenceWavelength)+']'
        plt.ylabel(ylabel)
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(bbox_to_anchor=(1.05,1), loc=2, prop={'size':10}, borderaxespad=0.)
        plt.xlim(xlim)
        plt.ylim([plotYMin,plotYMax])
        fullPlotName = plotName+'.png'
        plt.savefig(fullPlotName)
        return fullPlotName

    def plotFilters(self):
        """
        Plots all filters.  This includes both the UBVRI and the SDSS 
        filters. 

        Note that the array is reversed, compared to that used 
        by the plot() function, so wavelength values are in
        self.filters[filterName][0,:] and the relative transmission is
        self.filters[filterName][1,:]

        The plot made by the filters is saved as filters.png
        """
        plt.clf()
        listoffilters = self.filterList
        for filter in listoffilters:
            a = self.filters[filter]
            y = a[1,:]
            x = a[0,:]
            plt.plot(x,y, label=filter)
            plt.legend()
            #plt.show()
            plt.savefig('filters'+'.png')

    def getFluxAtReferenceWavelength(self, a):
        """
        returns the flux value at self.referenceWavelength
        """
        x = a[:,0]
        y = a[:,1]
        index = numpy.searchsorted(x, self.referenceWavelength);
        if index < 0:
            index = 0
        if index > x.size - 1:
            index = x.size - 1

        return y[index]

    def showUnits(self):
        """
        Returns flux units from the original data files for the spectra 
        of all objects.
        """
        for name in list(self.objects.keys()):
            fluxUnit = self.objects[name]['fluxUnit']
            print(name, " ", fluxUnit)

    def loadSdssSpecFits(self, fullFileName):
        """
        Allows spectral data from a SDSS fits file to be read into the 
        program
        """
        f = pyfits.open(fullFileName)
        coeff0 = f[0].header['COEFF0']
        coeff1 = f[0].header['COEFF1']
        n = len(f[1].data)
        retval = numpy.zeros([n,2])
        retval[:,0] = numpy.arange(n)
        retval[:,0] = 10**(coeff0+coeff1*retval[:,0])
        for i in range(n):
            retval[i][1] = f[1].data[i][0]
        return retval

    def rep2(self):
        names = list(self.objects.keys())
        names.sort()
        for name in names:
            print("name=",name)
            vMag = self.getVegaMag(name,'V')
            print("name=%15s   vMag=%+f" % (name, vMag))
    def report(self):
        """
        Creates a text document called Report.log that reports the units,
        citation, magnitude, and description of each object.
        """
        old_stdout = sys.stdout
        log_file = open("Report.log","w")
        print("sys.stdout=",sys.stdout)
        sys.stdout = log_file
        names = list(self.objects.keys())
        names.sort()
        for name in names:
            fluxUnit = self.objects[name]['fluxUnit'][0]
            wavelengthUnit = self.objects[name]['wavlengthUnit'][0]
            citation = self.objects[name]['citation'][0]
            description = self.objects[name]['description'][0]
            a = self.load(name)
            points = a[:,1].size
            x = a[:,0]
            y = a[:,1]
            xmin = x.min()
            xmax = x.max()
            bMag = self.getVegaMag(name,'B')
            vMag = self.getVegaMag(name,'V')
            bmv = bMag - vMag

            print("----------------------------------------------------------")
            print("Name: %s" %name)
            print("Units: Flux: %s Wavelength: %s " %(fluxUnit, wavelengthUnit))
            print("Citation: %s" %citation)
            print("Description: %s." %description)
            print("Calculated V=%.2f  B-V=%f" % (vMag, bmv))
            print("Number of Points: %d Wavelength: Max =%9.3f Min = %10.3f" \
                %(points, xmin, xmax))
        sys.stdout = old_stdout
        log_file.close()
