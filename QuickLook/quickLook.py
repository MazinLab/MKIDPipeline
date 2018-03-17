#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:28:30 2018

@author: clint


DON'T RUN THIS FROM INSIDE SPYDER. GO TO THE ENCLOSING DIRECTORY AND RUN IT 
FROM THE TERMINAL.

laptop:
-----------------
cd /Users/clint/Documents/mazinlab/DarknessPipeline/P3Utils
python clint_quickLook.py


dark:
---------------------------
cd /mnt/data0/clint/DarknessPipeline/P3Utils
python clint_quickLook.py

"""

import numpy as np
#import time, warnings, traceback
#import ConfigParser
#from functools import partial
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import Cursor
import sys,os
#import shutil
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QObject, pyqtSignal
import darkObsFile
from scipy.optimize import curve_fit
import os.path
import lightCurves as lc
import pdfs
from scipy.special import factorial



class subWindow(QMainWindow):
    
    #set up a signal so that the window closes when the main window closes
    #closeWindow = QtCore.pyqtSignal()
    
    #replot = pyqtsignal()
    
    def __init__(self,parent=None):
        super(QMainWindow, self).__init__(parent)
        self.parent=parent
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
        self.exposureTime = 0.01
        self.lineColor = 'blue'
        

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
        
        
        #create a vertical box for the plot to go in.
        vbox_plot = QVBoxLayout()
        vbox_plot.addWidget(self.canvas)
        
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
        #use this function to make the call to the correct obsfile method
        if self.apertureOn == True:
            self.photonList,aperture = self.a.getCircularAperturePhotonList(self.activePixel[0], self.activePixel[1], radius = self.apertureRadius, firstSec = self.spinbox_startTime.value(), integrationTime=self.spinbox_integrationTime.value(), wvlRange = (self.spinbox_startLambda.value(),self.spinbox_stopLambda.value()), flagToUse=0)
        else:
            self.photonList = self.a.getPixelPhotonList(self.activePixel[0], self.activePixel[1], firstSec = self.spinbox_startTime.value(), integrationTime=self.spinbox_integrationTime.value(), wvlRange = (self.spinbox_startLambda.value(),self.spinbox_stopLambda.value()))
            
            
            
        
    def lightCurve(self):
        #take a time stream and bin it up into a lightcurve
        #in other words, take a list of photon time stamps and figure out the 
        #intensity during each exposureTime, which is ~.01 sec
        self.histBinEdges = np.arange(self.spinbox_startTime.value(),self.spinbox_startTime.value()+self.spinbox_integrationTime.value(),self.exposureTime)
        self.hist,_ = np.histogram(self.photonList['Time']/10**6,bins=self.histBinEdges) #if histBinEdges has N elements, hist has N-1
        self.lightCurveIntensityCounts = 1.*self.hist  #units are photon counts
        self.lightCurveIntensity = 1.*self.hist/self.exposureTime  #units are counts/sec
        self.lightCurveTimes = self.histBinEdges[:-1] + 1.0*self.exposureTime/2
        




class timeStream(subWindow):
    #this class inherits from the subWindow class. 
    def __init__(self,parent):
        #call the init function from the superclass 'subWindow'. 
        super(timeStream, self).__init__(parent)
        self.setWindowTitle("Light Curve")
        self.plotData()
        self.draw()
        

        
    def plotData(self): 
        self.ax.clear()
     
        self.getPhotonList()
        self.lightCurve()

        self.ax.plot(self.lightCurveTimes,self.lightCurveIntensity,color = self.lineColor)
        self.ax.set_xlabel('time [seconds]')
        self.ax.set_ylabel('intensity [cps I think]')
        self.ax.set_title('pixel ({},{})' .format(self.activePixel[0],self.activePixel[1]))
        self.draw()
        
        
        
        
        
        
class intensityHistogram(subWindow):
    #this class inherits from the subWindow class. 
    def __init__(self,parent):
        #call the init function from the superclass 'subWindow'. 
        super(intensityHistogram, self).__init__(parent)
        self.setWindowTitle("Intensity Histogram")
        self.plotData()
        self.draw()
        
        
    
    def histogramLC(self):
        #makes a histogram of the light curve intensities
        self.Nbins=30
        norm=True
        centers = True
        self.intHist, binEdges = np.histogram(self.lightCurveIntensityCounts,bins=self.Nbins,range=[0,self.Nbins])

        if norm==True:
            self.intHist = self.intHist/float(len(self.lightCurveIntensityCounts))
            
        if centers==True:
            bws = np.diff(binEdges)
            cents = binEdges[:-1]+bws[0]/2.0
            self.bins = cents
        else:
            self.bins = binEdges

    
        
        
    def plotData(self): 
        self.ax.clear()
        
        self.getPhotonList()
        self.lightCurve()
        self.histogramLC()
        
        popt, pcov = curve_fit(pdfs.modifiedRician,range(0,self.Nbins),self.intHist,p0=[1,1])
        Ic = popt[0]
        Is = popt[1]
        
        mu = np.mean(self.lightCurveIntensityCounts)
        var = np.var(self.lightCurveIntensityCounts)  #add fudge factor here :P
        print('\nmu and var are {:.2f} and {:.2f}\n'.format(mu,var))
        IIc = np.sqrt(mu**2 - var + mu)
        IIs = mu - IIc
        
        k = np.arange(10)
        poisson= np.exp(-mu)*np.power(mu,k)/factorial(k)
        convolvedMR = np.convolve(pdfs.modifiedRician(np.arange(self.Nbins),popt[0],popt[1]),poisson,mode = 'same')
        
        
        
#        print('\nIIs and IIc are {:.2f} and {:.2f}\n'.format(IIs,IIc))
        
        self.ax.bar(self.bins - .5,self.intHist)
        self.ax.plot(np.arange(self.Nbins, step=0.1),pdfs.modifiedRician(np.arange(self.Nbins, step=0.1),popt[0],popt[1]),'r')
        self.ax.plot(np.arange(self.Nbins, step=0.1),pdfs.modifiedRician(np.arange(self.Nbins, step=0.1),IIc,IIs),'b')
        
#        self.ax.plot(np.arange(self.Nbins),pdfs.modifiedRician(np.arange(self.Nbins),popt[0],popt[1]) - self.intHist,'r')
        self.ax.plot(np.arange(len(convolvedMR)),convolvedMR,'k')


        self.ax.set_xlabel('intensity, counts per {:.2f} sec'.format(self.exposureTime))
        self.ax.set_ylabel('frequency')
#        self.ax.set_title('pixel ({},{})  Ic = {:.1f}, Is = {:.1f}, Ic/Is = {:.1f}' .format(self.activePixel[0],self.activePixel[1],Ic,Is,Ic/Is))
        self.ax.set_title('pixel ({},{})  Ic,IIc = {:.2f},{:.2f}, Is,IIs = {:.2f},{:.2f}, Ic/Is, IIc/IIs = {:.2f},{:.2f}' .format(self.activePixel[0],self.activePixel[1],Ic,IIc,Is,IIs,Ic/Is,IIc/IIs))
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
        

        
        
        
        
#####################################################################################        
        
        
        
        


class mainWindow(QMainWindow):
    
    
    updateActivePix = pyqtSignal()

    def __init__(self,parent=None):
        QMainWindow.__init__(self,parent=parent)
        self.initializeEmptyArrays()
        self.setWindowTitle('quickLook.py')
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
        #self.a = darkObsFile.ObsFile('/Users/clint/Documents/mazinlab/ScienceData/PAL2017b/20171004/1507175503.h5')
        #a = darkObsFile.ObsFile('/Users/clint/Documents/mazinlab/ScienceData/PAL2017b/20171004/1507175503.h5')
        if os.path.isfile(self.filename):
            try:
                self.a = darkObsFile.ObsFile(self.filename)
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
                self.spinbox_stopLambda.setMinimum(self.wvlBinStart)
                self.spinbox_startLambda.setMaximum(self.wvlBinEnd)
                self.spinbox_stopLambda.setMaximum(self.wvlBinEnd)
                self.spinbox_startLambda.setValue(self.wvlBinStart)
                self.spinbox_stopLambda.setValue(self.wvlBinEnd)
                
                #set the max value of the integration time spinbox
                self.spinbox_startTime.setMinimum(0)
                self.spinbox_startTime.setMaximum(self.expTime)
                self.spinbox_integrationTime.setMinimum(0)
                self.spinbox_integrationTime.setMaximum(self.expTime)
                self.spinbox_integrationTime.setValue(self.expTime)
                
#        self.a = darkObsFile.ObsFile(os.path.join(os.environ['MKID_DATA_DIR'],'1507175503.h5'))
        
        
        
        
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
            
            self.ax1.imshow(self.image,interpolation='none')
            self.cbar.set_clim(np.amin(self.image),np.amax(self.image))
            self.cbar.draw_all()

            self.ax1.set_title('beam flag image')
            
            self.ax1.axis('off')
            
            self.cursor = Cursor(self.ax1, useblit=True, color='red', linewidth=2)
            
            self.draw()
        
        
        
    def plotImage(self,*args):
        
        #self.a = darkObsFile.ObsFile(os.path.join(os.environ['MKID_DATA_DIR'],'1507175503.h5'))
        
        #check if obsfile object exists
        try:
            self.a
        except:
            print('\nNo obsfile object defined. Select H5 file to load.\n')
            return
        else:
        
        
            #clear the axes
            self.ax1.clear()  
            
            temp = self.a.getPixelCountImage(firstSec = self.spinbox_startTime.value(), integrationTime=self.spinbox_integrationTime.value(),applyWeight=False,flagToUse = 0,wvlRange = (self.spinbox_startLambda.value(),self.spinbox_stopLambda.value()))
            self.rawCountsImage = np.transpose(temp['image'])
            
#            self.image = self.rawCountsImage
#            self.image = self.rawCountsImage*self.beamFlagMask
            self.image = self.rawCountsImage*self.beamFlagMask*self.hotPixMask
            self.image = self.image/self.spinbox_integrationTime.value()
            
            self.ax1.imshow(self.image,interpolation='none')
            self.cbar.set_clim(np.amin(self.image),np.amax(self.image))
            self.cbar.draw_all()

            self.ax1.set_title('Raw counts')
            
            self.ax1.axis('off')
            
            self.cursor = Cursor(self.ax1, useblit=True, color='red', linewidth=2)
            
            #self.ax1.plot(np.arange(10),np.arange(10)**2)
            
            cid = self.fig.canvas.mpl_connect('motion_notify_event', self.hoverCanvas)
            
            self.draw()
            


        
        
        
        
        
    def plotIcIs(self):
        self.ax.clear()






        
    def plotNoise(self,*args):
        #clear the axes
        self.ax1.clear()  
        
        #debugging- generate some noise to plot
        self.image = np.random.randn(self.nRow,self.nCol)
                    
        self.foo = self.ax1.imshow(self.image,interpolation='none')
        self.cbar.set_clim(np.amin(self.image),np.amax(self.image))
        self.cbar.draw_all()
        
        self.ax1.set_title('some image to plot...')
        
        self.ax1.axis('off')
        
        self.cursor = Cursor(self.ax1, useblit=True, color='red', linewidth=2)
        
        #self.ax1.plot(np.arange(10),np.arange(10)**2)
        
        cid = self.fig.canvas.mpl_connect('motion_notify_event', self.hoverCanvas)
        cid2 = self.fig.canvas.mpl_connect('button_press_event', self.mousePressed)
        
        self.draw()
        
        
        
    def callPlotMethod(self):
        if self.radio_button_img.isChecked() == True:
            self.plotNoise()
        elif self.radio_button_ic_is.isChecked() == True:
            self.plotNoise()
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
        
        #let's flag some pixels by hand
        self.hotPixMask[51][31] = 0
        self.hotPixMask[55][46] = 0
        self.hotPixMask[66][77] = 0

        

    
    def create_main_frame(self):
        """
        Makes GUI elements on the window
        """
        #Define the plot window. 
        self.main_frame = QWidget()
        self.dpi = 100
        self.fig = Figure((1.0, 20.0), dpi=self.dpi) #define the figure, set the size and resolution
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.ax1 = self.fig.add_subplot(111)
        self.foo = self.ax1.imshow(self.image,interpolation='none')
        self.cbar = self.fig.colorbar(self.foo)
        
        
        button_plot = QPushButton("Plot image")
        button_plot.setEnabled(True)
        button_plot.setToolTip('Click to update image.')
#        button_plot.clicked.connect(self.plotImage)
        button_plot.clicked.connect(self.callPlotMethod)
        
        
        button_subwindow = QPushButton("Quick Load H5")
        button_subwindow.setEnabled(True)
        button_subwindow.setToolTip('Will change functionality later.')
        button_subwindow.clicked.connect(self.quickLoadH5)
        
        #spinboxes for the start & stop times
        self.spinbox_startTime = QSpinBox()
        self.spinbox_integrationTime = QSpinBox()
        
        #labels for the start/stop time spinboxes
        label_startTime = QLabel('start time')
        label_integrationTime = QLabel('integration time')
        
        #spinboxes for the start & stop wavelengths
        self.spinbox_startLambda = QSpinBox()
        self.spinbox_stopLambda = QSpinBox()
        
        #labels for the start/stop time spinboxes
        label_startLambda = QLabel('start wavelength [nm]')
        label_stopLambda = QLabel('stop wavelength [nm]')      
        
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
        vbox_lambda.addWidget(label_startLambda)
        vbox_lambda.addWidget(self.spinbox_startLambda)
        vbox_lambda.addWidget(label_stopLambda)
        vbox_lambda.addWidget(self.spinbox_stopLambda)
        
        #create an h box for the buttons
        hbox_buttons = QHBoxLayout()
        hbox_buttons.addWidget(button_plot)
        hbox_buttons.addWidget(button_subwindow)
        
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
        

        
        
    def quickLoadH5(self):
        self.filename = '/Users/clint/Documents/mazinlab/ScienceData/PAL2017b/20171004/1507175503.h5'
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
                
        else:
            return

                
                
        
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
        self.plotMenu.addAction(plotLightCurveButton)
        self.plotMenu.addAction(plotIntensityHistogramButton)
        self.plotMenu.addAction(plotSpectrumButton)
        
        
        
        
        
        self.menubar.setNativeMenuBar(False) #This is for MAC OS

#        loadObsAction = createAction(self,"Load obs file...",slot=self.loadObsFileWindow)
#        loadCalsAction = createAction(self,"Load cal files...",slot=self.loadCalFiles)
#        
#        quitAction = createAction(self,"&Quit", slot=self.close, 
#            shortcut="Ctrl+Q", tip="Close the application")
#        
#        addActions(self.fileMenu, 
#            (loadObsAction, loadCalsAction, None, quitAction))
#
#        self.windowsMenu = self.menuBar().addMenu("&Windows")
#        headerAction = createAction(self,"Header Info",slot=self.headerWindow.show)
#        imgParamsAction = createAction(self,"Image Plot Parameters",slot=self.imageParamsWindow.show)
#        newPlotWindowAction = createAction(self,'New Plot Window',slot=self.newPlotWindow)
#
#        addActions(self.windowsMenu,(newPlotWindowAction,None,headerAction,imgParamsAction))
#        
#        self.helpMenu = self.menuBar().addMenu("&Help")
#        aboutAction = createAction(self,"&About", 
#            shortcut='F1', slot=self.aboutMessage)
#        
#        addActions(self.helpMenu, (aboutAction,))
        
        
    def getFileNameFromUser(self):
        # look at this website for useful examples
        # https://pythonspot.com/pyqt5-file-dialog/
        
        #filename, _ = QFileDialog.getOpenFileName(self, 'Select One File', '/Users/clint/Documents/mazinlab/ScienceData/PAL2017b/20171004/',filter = '*.h5')
        
        filename, _ = QFileDialog.getOpenFileName(self, 'Select One File', os.environ['MKID_DATA_DIR'],filter = '*.h5')
        

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
        
        
        
if __name__ == "__main__":
    a = QApplication(sys.argv)
    foo = mainWindow()
    foo.show()
    sys.exit(a.exec_())