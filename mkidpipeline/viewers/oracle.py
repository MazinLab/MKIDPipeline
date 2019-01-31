#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:28:30 2018

@author: clint


GO TO THE ENCLOSING DIRECTORY AND RUN IT FROM THE TERMINAL WITH THE FOLLOWING COMMAND:
python oracle.py

"""

import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# from matplotlib.widgets import Cursor
import sys,os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSignal
from mkidpipeline.hdf.photontable import ObsFile
from scipy.optimize import curve_fit
import os.path
from mkidpipeline.utils import pdfs
from mkidpipeline.speckle import binned_rician as binnedRE
from scipy.special import factorial
import time



class subWindow(QMainWindow):
    
    #set up a signal so that the window closes when the main window closes
    #closeWindow = QtCore.pyqtSignal()
    
    #replot = pyqtsignal()
    
    def __init__(self,parent=None):
        super(QMainWindow, self).__init__(parent)
        self.parent=parent
        
        self.effExpTime = .010  #units are s
        
        self.create_main_frame()
        self.activePixel = parent.activePixel
        self.parent.updateActivePix.connect(self.setActivePixel)
        self.a = parent.a
        self.spinbox_startTime = parent.spinbox_startTime
        self.spinbox_integrationTime = parent.spinbox_integrationTime
        self.spinbox_startLambda = parent.spinbox_startLambda
        self.spinbox_stopLambda = parent.spinbox_stopLambda
        self.image = parent.image
        self.beamFlagMask = parent.beamFlagMask
        self.apertureRadius = 2.27/2   #Taken from Seth's paper (not yet published in Jan 2018)
        self.apertureOn = False
        self.lineColor = 'blue'
        self.minLambda = parent.minLambda
        self.maxLambda = parent.maxLambda
        

    def getAxes(self):
        return self.ax
        
    def create_main_frame(self):
        """
        Makes GUI elements on the window
        """
        self.main_frame = QWidget()
        
        # Figure
        self.dpi = 100
        self.fig = Figure((3.0, 2.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.ax = self.fig.add_subplot(111)
        
        #create a navigation toolbar for the plot window
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        #create a vertical box for the plot to go in.
        vbox_plot = QVBoxLayout()
        vbox_plot.addWidget(self.canvas)
        
        #check if we need effective exposure time controls in the window, and add them if we do. 
        try:
            self.spinbox_effExpTime
        except:
            pass
        else:
            label_expTime = QLabel('effective exposure time [ms]')
            button_plot = QPushButton("Plot")
            
            hbox_expTimeControl = QHBoxLayout()
            hbox_expTimeControl.addWidget(label_expTime)
            hbox_expTimeControl.addWidget(self.spinbox_effExpTime)
            hbox_expTimeControl.addWidget(button_plot)
            vbox_plot.addLayout(hbox_expTimeControl)
            
            self.spinbox_effExpTime.setMinimum(1)
            self.spinbox_effExpTime.setMaximum(200)
            self.spinbox_effExpTime.setValue(1000*self.effExpTime)
            button_plot.clicked.connect(self.plotData)
            
        vbox_plot.addWidget(self.toolbar)

        
        #combine everything into another vbox
        vbox_combined = QVBoxLayout()
        vbox_combined.addLayout(vbox_plot)
        
        #Set the main_frame's layout to be vbox_combined
        self.main_frame.setLayout(vbox_combined)

        #Set the overall QWidget to have the layout of the main_frame.
        self.setCentralWidget(self.main_frame)
                
        
    def draw(self):
        #The plot window calls this function
        self.canvas.draw()
        self.canvas.flush_events()     
        
        
    def setActivePixel(self):
        self.activePixel = self.parent.activePixel
#        if self.parent.image[self.activePixel[0],self.activePixel[1]] ==0: #only plot data from good pixels
#            self.lineColor = 'red'
#        else:
#            self.lineColor = 'blue'
        try:
            self.plotData() #put this in a try statement in case it doesn't work. This way it won't kill the whole gui. 
        except:
            pass


    def plotData(self):
        return #just create a dummy function that we'll redefine in the child classes
                # this way the signal to update the plots is handled entirely 
                # by this subWindow base class
                
    def getPhotonList(self):
        # use this function to make the call to the correct obsfile method
        if self.apertureOn == True:
            photonList,aperture = self.a.getCircularAperturePhotonList(self.activePixel[0], self.activePixel[1], radius = self.apertureRadius, firstSec = self.spinbox_startTime.value(), integrationTime=self.spinbox_integrationTime.value(), wvlStart = self.spinbox_startLambda.value(), wvlStop=self.spinbox_stopLambda.value(), flagToUse=0)
        
        else:
            wvlStart=self.spinbox_startLambda.value()
            wvlStop=self.spinbox_stopLambda.value()
            t1 = time.time()
            
            # it's WAY faster to not specify start/stop wavelengths. If that cut isn't
            # necessary, don't specify those keywords.
            if wvlStart==self.minLambda and wvlStop==self.maxLambda:
                photonList = self.a.getPixelPhotonList(self.activePixel[0], self.activePixel[1], firstSec = self.spinbox_startTime.value(), integrationTime=self.spinbox_integrationTime.value())
                cmd = 'photonList = self.a.getPixelPhotonList(self.activePixel[0], self.activePixel[1], firstSec = self.spinbox_startTime.value(), integrationTime=self.spinbox_integrationTime.value())'
            elif wvlStart==self.minLambda:
                photonList = self.a.getPixelPhotonList(self.activePixel[0], self.activePixel[1], firstSec = self.spinbox_startTime.value(), integrationTime=self.spinbox_integrationTime.value(), wvlStart=self.spinbox_startLambda.value())
                cmd = 'photonList = self.a.getPixelPhotonList(self.activePixel[0], self.activePixel[1], firstSec = self.spinbox_startTime.value(), integrationTime=self.spinbox_integrationTime.value(), wvlStart=self.spinbox_startLambda.value())'
            elif wvlStop==self.maxLambda:
                photonList = self.a.getPixelPhotonList(self.activePixel[0], self.activePixel[1], firstSec = self.spinbox_startTime.value(), integrationTime=self.spinbox_integrationTime.value(), wvlStop=self.spinbox_stopLambda.value())
                cmd = 'photonList = self.a.getPixelPhotonList(self.activePixel[0], self.activePixel[1], firstSec = self.spinbox_startTime.value(), integrationTime=self.spinbox_integrationTime.value(), wvlStop=self.spinbox_stopLambda.value())'
            else:
                photonList = self.a.getPixelPhotonList(self.activePixel[0], self.activePixel[1], firstSec = self.spinbox_startTime.value(), integrationTime=self.spinbox_integrationTime.value(), wvlStart=self.spinbox_startLambda.value(),wvlStop=self.spinbox_stopLambda.value())
                cmd = 'photonList = self.a.getPixelPhotonList(self.activePixel[0], self.activePixel[1], firstSec = self.spinbox_startTime.value(), integrationTime=self.spinbox_integrationTime.value(), wvlStart=self.spinbox_startLambda.value(),wvlStop=self.spinbox_stopLambda.value())'
            t2 = time.time()

            print('\ncmd = ' + cmd)
            print('\nTime to getPixelPhotonList(): ', t2 - t1)
        
        return photonList
            


class timeStream(subWindow):
    # this class inherits from the subWindow class.
    def __init__(self,parent):
        
        self.spinbox_effExpTime = QDoubleSpinBox()  # make sure that the parent class will know that we need an effExpTime control
        
        # call the init function from the superclass 'subWindow'.
        super(timeStream, self).__init__(parent)
        self.setWindowTitle("Light Curve")
        self.plotData()
        self.draw()
        

        

        
    def plotData(self):
        self.ax.clear()
     
        t1 = time.time()
        self.photonList = self.getPhotonList()
        t2 = time.time()
        print('timer: ', t2 - t1)
        
        self.effExpTime = self.spinbox_effExpTime.value()/1000
        self.lightCurveIntensityCounts, self.lightCurveIntensity, self.lightCurveTimes = binnedRE.getLightCurve(self.photonList['Time']/1e6,self.spinbox_startTime.value(),self.spinbox_startTime.value()+self.spinbox_integrationTime.value(),self.effExpTime)

        self.ax.plot(self.lightCurveTimes,self.lightCurveIntensity,color = self.lineColor)
        self.ax.set_xlabel('time [seconds]')
        self.ax.set_ylabel('intensity [cps]')
        self.ax.set_title('pixel ({},{})' .format(self.activePixel[0],self.activePixel[1]))
        self.draw()


        
        
class intensityHistogram(subWindow):
    # this class inherits from the subWindow class.
    def __init__(self,parent):
        
        self.spinbox_effExpTime = QDoubleSpinBox()  #make sure that the parent class will know that we need an effExpTime control
        
        #call the init function from the superclass 'subWindow'. 
        super(intensityHistogram, self).__init__(parent)
        self.setWindowTitle("Intensity Histogram")
        self.plotData()
        self.draw()
        
   
    
    
    """
    def fitBlurredMR(self,bins,intensityHist):   #this needs some work
        sigma = np.sqrt(intensityHist)
        sigma[np.where(sigma==0)] = 1
        try:
            popt2,pcov2 = curve_fit(binnedRE.blurredMR,bins,intensityHist,p0=[1,1],sigma=sigma,bounds=(0,np.inf))
            
            Ic = popt2[0]
            Is = popt2[1]
        except RuntimeError:
            Ic, Is = 1,0.1
        
        return Ic,Is
    """
            

    
        
        
    def plotData(self): 
        
        sstep = 1
        
        self.ax.clear()
        
        self.photonList = self.getPhotonList()
        
        self.effExpTime = self.spinbox_effExpTime.value()/1000
        
        #self.lightCurveIntensityCounts, self.lightCurveIntensity, self.lightCurveTimes = self.getLightCurve()
        
        self.lightCurveIntensityCounts, self.lightCurveIntensity, self.lightCurveTimes = binnedRE.getLightCurve(self.photonList['Time']/1e6,self.spinbox_startTime.value(),self.spinbox_startTime.value()+self.spinbox_integrationTime.value(),self.effExpTime)
        
#        self.intensityHist, self.bins = self.histogramLC(self.lightCurveIntensityCounts)
        
        self.intensityHist, self.bins = binnedRE.histogramLC(self.lightCurveIntensityCounts)
        # [self.intensityHist] = counts
        
        Nbins = max(30,len(self.bins))
        
        self.ax.bar(self.bins,self.intensityHist)
        self.ax.set_xlabel('intensity, counts per {:.3f} sec'.format(self.effExpTime))
        self.ax.set_ylabel('frequency')
        self.ax.set_title('pixel ({},{})' .format(self.activePixel[0],self.activePixel[1]))
        
        if np.sum(self.lightCurveIntensityCounts) > 0:
#            try:
#                popt, pcov = curve_fit(binnedRE.modifiedRician,self.bins,self.intensityHist,p0=[1,1])
#            
#                Ic = popt[0]
#                Is = popt[1]
#            except RuntimeError:
#                Ic, Is, =1,0.1
            
#            self.ax.plot(np.arange(self.Nbins, step=sstep),binnedRE.modifiedRician(np.arange(self.Nbins, step=sstep),Ic,Is),'.-r',label = 'MR from numpy.curve_fit')
            
#            self.ax.set_title('pixel ({},{})  Ic = {:.2f}, Is = {:.2f}, Ic/Is = {:.2f}' .format(self.activePixel[0],self.activePixel[1],Ic,Is,Ic/Is))
            
            mu = np.mean(self.lightCurveIntensityCounts)
            var = np.var(self.lightCurveIntensityCounts)  
            
            k = np.arange(Nbins)
            poisson= np.exp(-mu)*np.power(mu,k)/factorial(k)
            self.ax.plot(np.arange(len(poisson)),poisson,'.-c',label = 'Poisson')
            
            Ic_final,Is_final,covMatrix = binnedRE.fitBlurredMR(self.bins,self.intensityHist,self.effExpTime)
            
            self.ax.plot(np.arange(Nbins, step=sstep),binnedRE.blurredMR(np.arange(Nbins, step=sstep),Ic_final,Is_final),'.-k',label = 'blurred MR from curve_fit. Ic,Is = {:.2f}, {:.2f}'.format(Ic_final/self.effExpTime,Is_final/self.effExpTime))
            
#            self.ax.set_title('pixel ({},{})  Ic = {:.2f}, Is = {:.2f}, Ic/Is = {:.2f}' .format(self.activePixel[0],self.activePixel[1],Ic_final,Is_final,Ic_final/Is_final))
#            self.ax.set_title('pixel ({},{})  Ic,Ic_f = {:.2f},{:.2f}, Is,Is_f = {:.2f},{:.2f}, Ic/Is, Ic_f/Is_f = {:.2f},{:.2f}' .format(self.activePixel[0],self.activePixel[1],Ic,Ic_final,Is,Is_final,Ic/Is,Ic_final/Is_final))
            

            try:
                IIc = np.sqrt(mu**2 - var + mu)
            except:
                pass
            else:
                IIs = mu - IIc

        
                self.ax.plot(np.arange(Nbins, step=sstep),binnedRE.blurredMR(np.arange(Nbins, step=sstep),IIc,IIs),'.-b',label = r'blurred MR from $\sigma$ and $\mu$. Ic,Is = {:.2f}, {:.2f}'.format(IIc/self.effExpTime,IIs/self.effExpTime))
                
#                self.ax.set_title('pixel ({},{})  Ic,IIc = {:.2f},{:.2f}, Is,IIs = {:.2f},{:.2f}, Ic/Is, IIc/IIs = {:.2f},{:.2f}' .format(self.activePixel[0],self.activePixel[1],Ic_final,IIc,Is_final,IIs,Ic_final/Is_final,IIc/IIs))
            
            self.ax.set_title('pixel ({},{})' .format(self.activePixel[0],self.activePixel[1]))
                
            self.ax.legend()
        
        self.draw()

        
        
        
        
        
class spectrum(subWindow):
    #this class inherits from the subWindow class. 
    def __init__(self,parent):
        #call the init function from the superclass 'subWindow'. 
        super(spectrum, self).__init__(parent)
        self.setWindowTitle("Spectrum")
        self.plotData()
        self.draw()
        
        
        
    def plotData(self): 
        self.ax.clear()
        temp = self.a.getPixelSpectrum(self.activePixel[0], self.activePixel[1], firstSec = self.spinbox_startTime.value(), integrationTime=self.spinbox_integrationTime.value())
        
        self.spectrum = temp['spectrum']
        self.wvlBinEdges = temp['wvlBinEdges']
        #self.effIntTime = temp['effIntTime']
        self.rawCounts = temp['rawCounts']
        
        self.wvlBinCenters = np.diff(self.wvlBinEdges)/2 + self.wvlBinEdges[:-1]
        
        
        self.ax.plot(self.wvlBinCenters,self.spectrum,'-o')
        self.ax.set_xlabel('wavelength [nm]')
        self.ax.set_ylabel('intensity [counts]')
        self.ax.set_title('pixel ({},{})' .format(self.activePixel[0],self.activePixel[1]))
        self.draw()
        
        
        
        
class pulseHeightHistogram(subWindow):
    #this class inherits from the subWindow class. 
    def __init__(self,parent):
        #call the init function from the superclass 'subWindow'. 
        super(pulseHeightHistogram, self).__init__(parent)
        self.setWindowTitle("Pulse Heights")
        self.plotData()
        self.draw()
        
        
        
    def plotData(self): 
        self.ax.clear()
        
        pulseHeights = self.a.getPixelPhotonList(self.activePixel[0], self.activePixel[1])['Wavelength']
        
        hist,binEdges = np.histogram(pulseHeights,bins=100)
        
        binCenters = np.diff(binEdges)/2 + binEdges[:-1]
        
        
        self.ax.bar(binCenters,hist)
        self.ax.set_xlabel('raw phase [uncalibrated degrees]')
        self.ax.set_ylabel('[counts]')
        self.ax.set_title('pixel ({},{})' .format(self.activePixel[0],self.activePixel[1]))
        self.draw()
        

        
        
      
        
        
        
        
        
        
        
        
        
#####################################################################################        
        
        
        
        


class mainWindow(QMainWindow):
    
    
    updateActivePix = pyqtSignal()

    def __init__(self,parent=None):
        QMainWindow.__init__(self,parent=parent)
        self.initializeEmptyArrays()
        self.setWindowTitle('oracle')
        self.resize(600,850)  #(600,850 works for clint's laptop screen. Units are pixels I think.)
        self.create_main_frame()
        self.create_status_bar()
        self.createMenu()
        self.plotNoise()
        
        
    def initializeEmptyArrays(self,nCol = 10,nRow = 10):
        self.nCol = nCol
        self.nRow = nRow
        self.IcMap = np.zeros(self.nRow*self.nCol).reshape((self.nRow,self.nCol))
        self.IsMap = np.zeros(self.nRow*self.nCol).reshape((self.nRow,self.nCol))
        self.IcIsMap = np.zeros(self.nRow*self.nCol).reshape((self.nRow,self.nCol))
        self.rawCountsImage = np.zeros(self.nRow*self.nCol).reshape((self.nRow,self.nCol))
        self.hotPixMask = np.zeros(self.nRow*self.nCol).reshape((self.nRow,self.nCol))
        self.hotPixCut = 2300
        self.image = np.zeros(self.nRow*self.nCol).reshape((self.nRow,self.nCol))
        self.activePixel = [0,0]
        self.sWindowList = []
        
        
    def loadDataFromH5(self,*args):
        #a = ObsFile('/Users/clint/Documents/mazinlab/ScienceData/PAL2017b/20171004/1507175503.h5')
        if os.path.isfile(self.filename):
            try:
                self.a = ObsFile(self.filename)
            except:
                print('darkObsFile failed to load file. Check filename.\n',self.filename)
            else:
                print('data loaded from .h5 file')
                self.h5_filename_label.setText(self.filename)
                self.initializeEmptyArrays(len(self.a.beamImage),len(self.a.beamImage[0]))
                self.beamFlagImage = np.transpose(self.a.beamFlagImage.read())
                self.beamFlagMask = self.beamFlagImage==0  #make a mask. 0 for good beam map
                self.makeHotPixMask()
                self.radio_button_beamFlagImage.setChecked(True)
                self.callPlotMethod()
                #set the max integration time to the h5 exp time in the header
                self.expTime = self.a.getFromHeader('expTime')
                self.wvlBinStart = self.a.getFromHeader('wvlBinStart')
                self.wvlBinEnd = self.a.getFromHeader('wvlBinEnd')
                
                #set the max and min values for the lambda spinboxes
#                self.spinbox_startLambda.setMinimum(self.wvlBinStart)
                #check if the data is wavecaled and set the limits on the spinboxes accordingly
                if self.a.getFromHeader('isWvlCalibrated'):
                    self.minLambda = self.wvlBinStart
                    self.maxLambda = self.wvlBinEnd
                else:
                    self.minLambda = -200
                    self.maxLambda = 200
                    self.label_startLambda.setText('start phase [uncal degrees]')
                    self.label_stopLambda.setText('stop phase [uncal degrees]')
                    
                    
                self.spinbox_stopLambda.setMinimum(self.minLambda)
                self.spinbox_startLambda.setMaximum(self.maxLambda)
                self.spinbox_stopLambda.setMaximum(self.maxLambda)
                self.spinbox_startLambda.setMinimum(self.minLambda)
                self.spinbox_startLambda.setValue(self.minLambda)
                self.spinbox_stopLambda.setValue(self.maxLambda)
                    

                
                #set the max value of the integration time spinbox
                self.spinbox_startTime.setMinimum(0)
                self.spinbox_startTime.setMaximum(self.expTime)
                self.spinbox_integrationTime.setMinimum(0)
                self.spinbox_integrationTime.setMaximum(self.expTime)
                self.spinbox_integrationTime.setValue(self.expTime)

        
        
        
        
    def plotBeamImage(self):
        #check if obsfile object exists
        try:
            self.a
        except:
            print('\nNo obsfile object defined. Select H5 file to load.\n')
            return
        else:
        
            
            #clear the axes
            self.ax1.clear()  
            
            self.image = self.beamFlagImage
            
            self.cbarLimits = np.array([np.amin(self.image),np.amax(self.image)])
            
            self.ax1.imshow(self.image,interpolation='none')
            self.fig.cbar.set_clim(np.amin(self.image),np.amax(self.image))
            self.fig.cbar.draw_all()

            self.ax1.set_title('beam flag image')
            
            self.ax1.axis('off')
            
            # self.cursor = Cursor(self.ax1, useblit=True, color='red', linewidth=.5)
            
            self.draw()
        
        
        
    def plotImage(self,*args):        
        # check if obsfile object exists
        try:
            self.a
        except:
            print('\nNo obsfile object defined. Select H5 file to load.\n')
            return
        else:
            # clear the axes
            self.ax1.clear()  
            
            if self.spinbox_integrationTime.value()==self.spinbox_integrationTime.maximum():
                integrationTime = -1
            else:
                integrationTime = self.spinbox_integrationTime.value()

            t1 = time.time()
            temp = self.a.getPixelCountImage(firstSec = self.spinbox_startTime.value(), integrationTime=integrationTime,applyWeight=False,flagToUse = 0,wvlStart=self.spinbox_startLambda.value(), wvlStop=self.spinbox_stopLambda.value())
            print('\nTime for getPixelCountImage = ', time.time() - t1)

            self.rawCountsImage = np.transpose(temp['image'])
            
            self.image = self.rawCountsImage
            self.image[np.where(np.logical_not(np.isfinite(self.image)))]=0
            # self.image = self.rawCountsImage*self.beamFlagMask
            # self.image = self.rawCountsImage*self.beamFlagMask*self.hotPixMask
            self.image = 1.0*self.image/self.spinbox_integrationTime.value()
            
            self.cbarLimits = np.array([np.amin(self.image),np.amax(self.image)])
            
            self.ax1.imshow(self.image,interpolation='none',vmin = self.cbarLimits[0],vmax = self.cbarLimits[1])
            
            self.fig.cbar.set_clim(self.cbarLimits[0],self.cbarLimits[1])
            self.fig.cbar.draw_all()

            self.ax1.set_title('Raw counts')
            
            self.ax1.axis('off')

            # TODO: fix the cursor. It was running super slow, so I turned it off.
            # self.cursor = Cursor(self.ax1, useblit=True, color='red', linewidth=.5)
            
            # self.ax1.plot(np.arange(10),np.arange(10)**2)
            
            
            self.draw()
            
            
            


        
        
        
        
        
    def plotIcIs(self):
        #check if obsfile object exists
        try:
            self.a
        except:
            print('\nNo obsfile object defined. Select H5 file to load.\n')
            return
        else:
            self.ax1.clear() #clear the axes
            
            for col in range(self.nCol):
                print(col,'/80')
                for row in range(self.nRow):
                    photonList = self.a.getPixelPhotonList(col, row, firstSec = self.spinbox_startTime.value(), integrationTime=self.spinbox_integrationTime.value(), wvlStart=self.spinbox_startLambda.value(),wvlStop=self.spinbox_stopLambda.value())
                    
                    
                    
                    
                    effExpTime = .00001 #10 ms/1000

                    
                    lightCurveIntensityCounts, lightCurveIntensity, lightCurveTimes = binnedRE.getLightCurve(photonList['Time']/1e6,self.spinbox_startTime.value(),self.spinbox_startTime.value()+self.spinbox_integrationTime.value(),effExpTime)
                    
                    intensityHist, bins = binnedRE.histogramLC(lightCurveIntensityCounts)
                    # [self.intensityHist] = counts
                    
                    Nbins = max(30,len(bins))
                    
                    if np.sum(lightCurveIntensityCounts) > 0:

                        
                        Ic_final,Is_final,covMatrix = binnedRE.fitBlurredMR(bins,intensityHist,effExpTime)
                    
                        self.IcMap[row][col] = Ic_final
                        self.IsMap[row][col] = Is_final
                
            self.image = self.IsMap
            self.image[np.where(np.logical_not(np.isfinite(self.image)))]=0
#            self.image = self.rawCountsImage*self.beamFlagMask
            #self.image = self.rawCountsImage*self.beamFlagMask*self.hotPixMask
#            self.image = 1.0*self.image/self.spinbox_integrationTime.value()
            
            self.cbarLimits = np.array([np.amin(self.image),np.amax(self.image)])
            
            self.ax1.imshow(self.image,interpolation='none',vmin = self.cbarLimits[0],vmax = self.cbarLimits[1])
            
            self.fig.cbar.set_clim(self.cbarLimits[0],self.cbarLimits[1])
            self.fig.cbar.draw_all()

            self.ax1.set_title('Raw counts')
            
            self.ax1.axis('off')
            
            # self.cursor = Cursor(self.ax1, useblit=True, color='red', linewidth=.5)
            
            
            self.draw()






        
    def plotNoise(self,*args):
        #clear the axes
        self.ax1.clear()  
        
        #debugging- generate some noise to plot
        self.image = np.random.randn(self.nRow,self.nCol)
                    
        self.foo = self.ax1.imshow(self.image,interpolation='none')
        self.cbarLimits = np.array([np.amin(self.image),np.amax(self.image)])
        self.fig.cbar.set_clim(self.cbarLimits[0],self.cbarLimits[1])
        self.fig.cbar.draw_all()
        
        self.ax1.set_title('some generated noise...')
        
        self.ax1.axis('off')
        
        # self.cursor = Cursor(self.ax1, useblit=True, color='red', linewidth=.5)

        self.draw()
        
        
        
    def callPlotMethod(self):
        if self.radio_button_img.isChecked() == True:
            self.plotNoise()
        elif self.radio_button_ic_is.isChecked() == True:
            self.plotIcIs()
        elif self.radio_button_beamFlagImage.isChecked() == True:
            self.plotBeamImage()
        elif self.radio_button_rawCounts.isChecked() == True:
            self.plotImage()
        else:
            self.plotNoise()
            
            
            
            
    def makeHotPixMask(self):
        #if self.hotPixMask[row][col] = 0, it's a hot pixel. If 1, it's good. 
        temp = self.a.getPixelCountImage(firstSec = 0, integrationTime=1,applyWeight=False,flagToUse = 0)
        rawCountsImage = np.transpose(temp['image'])
        for col in range(self.nCol):
            for row in range(self.nRow):
                if rawCountsImage[row][col] < self.hotPixCut:
                    self.hotPixMask[row][col] = 1

        

    
    def create_main_frame(self):
        """
        Makes GUI elements on the window
        """
        #Define the plot window. 
        self.main_frame = QWidget()
        self.dpi = 100
        self.fig = Figure((5.0, 10.0), dpi=self.dpi, tight_layout=True) #define the figure, set the size and resolution
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.ax1 = self.fig.add_subplot(111)
        self.foo = self.ax1.imshow(self.image,interpolation='none')
        self.fig.cbar = self.fig.colorbar(self.foo)
        
        
        button_plot = QPushButton("Plot image")
        button_plot.setEnabled(True)
        button_plot.setToolTip('Click to update image.')
        button_plot.clicked.connect(self.callPlotMethod)
        
        
        button_quickLoad = QPushButton("Quick Load H5")
        button_quickLoad.setEnabled(True)
        button_quickLoad.setToolTip('Will change functionality later.')
        button_quickLoad.clicked.connect(self.quickLoadH5)
        
        #spinboxes for the start & stop times
        self.spinbox_startTime = QSpinBox()
        self.spinbox_integrationTime = QSpinBox()
        
        #labels for the start/stop time spinboxes
        label_startTime = QLabel('start time')
        label_integrationTime = QLabel('integration time')
        
        #spinboxes for the start & stop wavelengths
        self.spinbox_startLambda = QDoubleSpinBox()
        self.spinbox_stopLambda = QDoubleSpinBox()
        
        #labels for the start/stop time spinboxes
        self.label_startLambda = QLabel('start wavelength [nm]')
        self.label_stopLambda = QLabel('stop wavelength [nm]')      
        
        #label for the filenames
        self.h5_filename_label = QLabel('no file loaded')
        
        #label for the active pixel
        self.activePixel_label = QLabel('Active Pixel ({},{}) {}'.format(self.activePixel[0],self.activePixel[1],self.image[self.activePixel[1],self.activePixel[0]]))
        
        #make the radio buttons
        #self.radio_button_noise = QRadioButton("Noise")
        self.radio_button_img = QRadioButton(".IMG")
        self.radio_button_ic_is = QRadioButton("Ic/Is")
        #self.radio_button_bin = QRadioButton(".bin")
        #self.radio_button_decorrelationTime = QRadioButton("Decorrelation Time")
        self.radio_button_beamFlagImage = QRadioButton("Beam Flag Image")
        self.radio_button_rawCounts = QRadioButton("Raw Counts")
        self.radio_button_img.setChecked(True)
        
        #create a vertical box for the plot to go in.
        vbox_plot = QVBoxLayout()
        vbox_plot.addWidget(self.canvas)
        
        #create a v box for the timespan spinboxes
        vbox_timespan = QVBoxLayout()
        vbox_timespan.addWidget(label_startTime)
        vbox_timespan.addWidget(self.spinbox_startTime)
        vbox_timespan.addWidget(label_integrationTime)
        vbox_timespan.addWidget(self.spinbox_integrationTime)
        
        #create a v box for the wavelength spinboxes
        vbox_lambda = QVBoxLayout()
        vbox_lambda.addWidget(self.label_startLambda)
        vbox_lambda.addWidget(self.spinbox_startLambda)
        vbox_lambda.addWidget(self.label_stopLambda)
        vbox_lambda.addWidget(self.spinbox_stopLambda)
        
        #create an h box for the buttons
        hbox_buttons = QHBoxLayout()
        hbox_buttons.addWidget(button_plot)
        hbox_buttons.addWidget(button_quickLoad) #################################################
        
        #create an h box for the time and lambda v boxes
        hbox_time_lambda = QHBoxLayout()
        hbox_time_lambda.addLayout(vbox_timespan)
        hbox_time_lambda.addLayout(vbox_lambda)
        
        #create a v box combining spinboxes and buttons
        vbox_time_lambda_buttons = QVBoxLayout()
        vbox_time_lambda_buttons.addLayout(hbox_time_lambda)
        vbox_time_lambda_buttons.addLayout(hbox_buttons)
        
        #create a v box for the radio buttons
        vbox_radio_buttons = QVBoxLayout()
        #vbox_radio_buttons.addWidget(self.radio_button_noise)
        vbox_radio_buttons.addWidget(self.radio_button_img)
        vbox_radio_buttons.addWidget(self.radio_button_ic_is)
        #vbox_radio_buttons.addWidget(self.radio_button_bin)
        #vbox_radio_buttons.addWidget(self.radio_button_decorrelationTime)
        vbox_radio_buttons.addWidget(self.radio_button_beamFlagImage)
        vbox_radio_buttons.addWidget(self.radio_button_rawCounts)
        
        #create a h box combining the spinboxes, buttons, and radio buttons
        hbox_controls = QHBoxLayout()
        hbox_controls.addLayout(vbox_time_lambda_buttons)
        hbox_controls.addLayout(vbox_radio_buttons)
        
        #create a v box for showing the files that are loaded in memory
        vbox_filenames = QVBoxLayout()
        vbox_filenames.addWidget(self.h5_filename_label)
        vbox_filenames.addWidget(self.activePixel_label)

        
        #Now create another vbox, and add the plot vbox and the button's hbox to the new vbox.
        vbox_combined = QVBoxLayout()
        vbox_combined.addLayout(vbox_plot)
        vbox_combined.addLayout(hbox_controls)
        vbox_combined.addLayout(vbox_filenames)
        
        #Set the main_frame's layout to be vbox_combined
        self.main_frame.setLayout(vbox_combined)

        #Set the overall QWidget to have the layout of the main_frame.
        self.setCentralWidget(self.main_frame)
        
        #set up the pyqt5 events
        cid = self.fig.canvas.mpl_connect('motion_notify_event', self.hoverCanvas)
        cid2 = self.fig.canvas.mpl_connect('button_press_event', self.mousePressed)
        cid3 = self.fig.canvas.mpl_connect('scroll_event', self.scroll_ColorBar)
        

        
        
    def quickLoadH5(self):
#        self.filename = '/Users/clint/Documents/mazinlab/ScienceData/PAL2017b/20171004/1507175503_old.h5'
        self.filename = '/Users/clint/Documents/mazinlab/ScienceData/PAL2017b/20171004/1507173006.h5'
        self.loadDataFromH5()  
        


        
        
    def draw(self):
        #The plot window calls this function
        self.canvas.draw()
        self.canvas.flush_events()
        
        
    def hoverCanvas(self,event):
        if event.inaxes is self.ax1:
            col = int(round(event.xdata))
            row = int(round(event.ydata))
            if row < self.nRow and col < self.nCol:
                self.status_text.setText('({:d},{:d}) {}'.format(col,row,self.image[row,col]))
                
                
    def scroll_ColorBar(self,event):
        if event.inaxes is self.fig.cbar.ax:
            stepSize = 0.1  #fractional change in the colorbar scale
            if event.button == 'up':
                self.cbarLimits[1] *= (1 + stepSize)   #increment by step size
                self.fig.cbar.set_clim(self.cbarLimits[0],self.cbarLimits[1])
                self.fig.cbar.draw_all()
                self.ax1.imshow(self.image,interpolation='none',vmin = self.cbarLimits[0],vmax = self.cbarLimits[1])
            elif event.button == 'down':
                self.cbarLimits[1] *= (1 - stepSize)   #increment by step size
                self.fig.cbar.set_clim(self.cbarLimits[0],self.cbarLimits[1])
                self.fig.cbar.draw_all()
                self.ax1.imshow(self.image,interpolation='none',vmin = self.cbarLimits[0],vmax = self.cbarLimits[1])
                
            else:
                pass
                
        self.draw()
        
        
                
                
    def mousePressed(self,event):
#        print('\nclick event registered!\n')
        if event.inaxes is self.ax1:  #check if the mouse-click was within the axes. 
            #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %('double' if event.dblclick else 'single', event.button,event.x, event.y, event.xdata, event.ydata))
            
            if event.button == 1:
                #print('\nit was the left button that was pressed!\n')
                col = int(round(event.xdata))
                row = int(round(event.ydata))
                self.activePixel = [col,row]
                self.activePixel_label.setText('Active Pixel ({},{}) {}'.format(self.activePixel[0],self.activePixel[1],self.image[self.activePixel[1],self.activePixel[0]]))
                
                self.updateActivePix.emit()  #emit a signal for other plots to update
                
            elif event.button == 3:
                print('\nit was the right button that was pressed!\n')
                
                
        elif event.inaxes is self.fig.cbar.ax:   #reset the scale bar       
            if event.button == 1:
                self.cbarLimits = np.array([np.amin(self.image),np.amax(self.image)])
                self.fig.cbar.set_clim(self.cbarLimits[0],self.cbarLimits[1])
                self.fig.cbar.draw_all()
                self.ax1.imshow(self.image,interpolation='none',vmin = self.cbarLimits[0],vmax = self.cbarLimits[1])
                self.draw()
        else:
            pass
        

                
                
        
    def create_status_bar(self):
        #Using code from ARCONS-pipeline as an example:
        #ARCONS-pipeline/util/popup.py
        self.status_text = QLabel("")
        self.statusBar().addWidget(self.status_text, 1)
        
        
    def createMenu(self):   
        #Using code from ARCONS-pipeline as an example:
        #ARCONS-pipeline/util/quicklook.py
        self.menubar = self.menuBar()
        self.fileMenu = self.menubar.addMenu("&File")
        
        openFileButton = QAction(QIcon('exit24.png'), 'Open H5 File', self)
        openFileButton.setShortcut('Ctrl+O')
        openFileButton.setStatusTip('Open an H5 File')
        openFileButton.triggered.connect(self.getFileNameFromUser)
        self.fileMenu.addAction(openFileButton)
        
        
        exitButton = QAction(QIcon('exit24.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)
        self.fileMenu.addAction(exitButton)
        
        
        #make a menu for plotting
        self.plotMenu = self.menubar.addMenu("&Plot")
        plotLightCurveButton = QAction('Light Curve', self)
        plotLightCurveButton.triggered.connect(self.makeTimestreamPlot)
        plotIntensityHistogramButton = QAction('Intensity Histogram',self)
        plotIntensityHistogramButton.triggered.connect(self.makeIntensityHistogramPlot)
        plotSpectrumButton = QAction('Spectrum',self)
        plotSpectrumButton.triggered.connect(self.makeSpectrumPlot)
        plotPhaseHistButton = QAction('Phase Height Histogram',self)
        plotPhaseHistButton.triggered.connect(self.makePulseHeightHistogramPlot)
        self.plotMenu.addAction(plotLightCurveButton)
        self.plotMenu.addAction(plotIntensityHistogramButton)
        self.plotMenu.addAction(plotSpectrumButton)
        self.plotMenu.addAction(plotPhaseHistButton)

        
        self.menubar.setNativeMenuBar(False) #This is for MAC OS


        
        
    def getFileNameFromUser(self):
        # look at this website for useful examples
        # https://pythonspot.com/pyqt5-file-dialog/
        try:def_loc = os.environ['MKID_DATA_DIR']
        except KeyError:def_loc='.'
        filename, _ = QFileDialog.getOpenFileName(self, 'Select One File', def_loc,filter = '*.h5')

        self.filename = filename
        self.loadDataFromH5(self.filename)
        
        
    def makeTimestreamPlot(self):
        sWindow = timeStream(self)
        sWindow.show()
        self.sWindowList.append(sWindow)
        
        
    def makeIntensityHistogramPlot(self):
        sWindow = intensityHistogram(self)
        sWindow.show()
        self.sWindowList.append(sWindow)
        
        
    def makeSpectrumPlot(self):
        sWindow = spectrum(self)
        sWindow.show()
        self.sWindowList.append(sWindow)
        
        
    def makePulseHeightHistogramPlot(self):
        sWindow = pulseHeightHistogram(self)
        sWindow.show()
        self.sWindowList.append(sWindow)
        
        
        
if __name__ == "__main__":
    a = QApplication(sys.argv)
    foo = mainWindow()
    foo.show()
    sys.exit(a.exec_())
