#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:28:30 2018

@author: clint


1507175503

GO TO THE ENCLOSING DIRECTORY AND RUN IT FROM THE TERMINAL WITH THE FOLLOWING COMMAND:
python quickLook_img.py

set the system variable $MKID_IMG_DIR to the place where you want to look for img files

"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import Cursor
import sys,os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QObject, pyqtSignal
#import darkObsFile
#from scipy.optimize import curve_fit
import os.path
#import lightCurves as lc
#import pdfs
#from scipy.special import factorial
import datetime



class mainWindow(QMainWindow):
    
    
    updateActivePix = pyqtSignal()

    def __init__(self,parent=None):
        QMainWindow.__init__(self,parent=parent)
        self.initializeEmptyArrays()
        self.setWindowTitle('quickLook_img.py')
        self.create_main_frame()
        self.create_status_bar()
        self.createMenu()
        
        
    def initializeEmptyArrays(self,nCol = 80,nRow = 125):
        self.nCol = nCol
        self.nRow = nRow

        self.rawCountsImage = np.zeros(self.nRow*self.nCol).reshape((self.nRow,self.nCol))
        self.hotPixMask = np.zeros(self.nRow*self.nCol).reshape((self.nRow,self.nCol))
        self.hotPixCut = 2400
        self.image = np.zeros(self.nRow*self.nCol).reshape((self.nRow,self.nCol))

        
        

    def loadFilenames(self,filename):
        
        
        print('\nloading img filenames')
        
        
        self.imgPath = os.path.dirname(filename)
        
#        print(self.imgPath)
        
        fileListRaw = []
        timeStampList = np.zeros(len(os.listdir(self.imgPath)))
        ii = 0
        for file in os.listdir(self.imgPath):
            if file.endswith(".img"):
                fileListRaw = fileListRaw + [os.path.join(self.imgPath, file)]
                timeStampList[ii] = np.fromstring(file[:-4],dtype=int, sep=' ')[0]
            else:
                continue
            ii+=1
        self.fileListRaw = fileListRaw
        self.timeStampList = timeStampList

        


        #load the log filenames
        
        print('\nloading log filenames')
        logFilenameList_all = os.listdir(os.environ['MKID_DATA_DIR'])
        logFilenameList = []
        logTimestampList = []
        
        for logFilename in logFilenameList_all:
            
            if logFilename.endswith("telescope.log"):
                
                continue
            elif logFilename.endswith(".log"):
                logFilenameList.append(logFilename)
#                print(logFilename)
#                print(logFilename[:10])
                logTimestampList.append(np.fromstring(logFilename[:10],dtype=int, sep=' ')[0])
                
        
        self.logTimestampList = np.asarray(logTimestampList)
        #print('\n\nself.logTimestampList is ')
        self.logFilenameList = logFilenameList

        
        #set up the spinbox limits and start value, which will be the file you selected
        self.spinbox_imgTimestamp.setMinimum(timeStampList[0])
        self.spinbox_imgTimestamp.setMaximum(timeStampList[-1])
        self.spinbox_imgTimestamp.setValue(np.fromstring(os.path.basename(filename)[:-4],dtype=int, sep=' ')[0])
        
        self.spinbox_darkStart.setMinimum(timeStampList[0])
        self.spinbox_darkStart.setMaximum(timeStampList[-10])
        self.spinbox_darkStart.setValue(np.fromstring(os.path.basename(filename)[:-4],dtype=int, sep=' ')[0])
        


        
        
        
        

        
        
    def plotImage(self,filename = None):        
        
        if filename == None:
            filename = self.fileListRaw[np.where(self.timeStampList==self.spinbox_imgTimestamp.value())[0][0]]
        

        self.ax1.clear()  
        
        
        self.rawImage = np.transpose(np.reshape(np.fromfile(open(filename, mode='rb'),dtype=np.uint16), (self.nCol,self.nRow)))
#        self.cleanedImage = self.image
#        self.cleanedImage[np.where(self.cleanedImage>self.hotPixCut)] = 0
#        self.ax1.imshow(self.cleanedImage)
        
        
        if self.checkbox_darkSubtract.isChecked():
            self.cleanedImage = self.rawImage - self.darkFrame
            self.cleanedImage[np.where(self.cleanedImage<0)] = 0
            
        else:
            self.cleanedImage = self.rawImage
        
        #colorbar auto
        if self.checkbox_colorbar_auto.isChecked():
            self.cbarLimits = np.array([0,np.amax(self.image)])
            self.fig.cbar.set_clim(self.cbarLimits[0],self.cbarLimits[1])
            self.fig.cbar.draw_all()
        else:
            self.cbarLimits = np.array([0,self.spinbox_colorBarMax.value()])
            self.fig.cbar.set_clim(self.cbarLimits[0],self.cbarLimits[1])
            self.fig.cbar.draw_all()
                  
        self.cleanedImage[np.where(self.cleanedImage>self.hotPixCut)] = 0
        self.image = self.cleanedImage
        self.ax1.imshow(self.image,vmin = self.cbarLimits[0],vmax = self.cbarLimits[1])
        
        
        self.draw()
        
        

    def getDarkFrame(self):
        #get an average dark from darkStart to darkStart + darkIntTime
        darkIntTime = self.spinbox_darkIntTime.value()
        darkFrame = np.zeros(darkIntTime*self.nRow*self.nCol).reshape((darkIntTime,self.nRow,self.nCol))
        
        
#        darkFrame = np.zeros(self.nRow*self.nCol).reshape((self.nRow,self.nCol))
#        for ii in range(darkIntTime):
#            darkFrameFilename = self.fileListRaw[np.where(self.timeStampList==(self.spinbox_darkStart.value()+ii))[0][0]]
#            darkFrame += np.transpose(np.reshape(np.fromfile(open(darkFrameFilename, mode='rb'),dtype=np.uint16), (self.nCol,self.nRow)))
#
#        self.darkFrame = darkFrame/ii
#        print(self.darkFrame)
        
        
        for ii in range(darkIntTime):
            darkFrameFilename = self.fileListRaw[np.where(self.timeStampList==(self.spinbox_darkStart.value()+ii))[0][0]]
            darkFrame[ii] = np.transpose(np.reshape(np.fromfile(open(darkFrameFilename, mode='rb'),dtype=np.uint16), (self.nCol,self.nRow)))

        self.darkFrame = np.median(darkFrame,axis=0)

        
        
        
    def plotBlank(self):
        self.ax1.imshow(np.zeros(self.nRow*self.nCol).reshape((self.nRow,self.nCol)))
        
        
        
    def updateLogLabel(self,fileExist = True):
        
        timestamp = self.spinbox_imgTimestamp.value()
        
        #check if the img exists, if not then return
        if fileExist==False:
            text = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S\n\n') + 'no .img file found'
            self.label_log.setText(text)
            return
        
        
        
        diffs = timestamp - self.logTimestampList
        diffs[np.where(diffs<0)] = np.amax(diffs)
        logLabelTimestamp = self.logTimestampList[np.argmin(diffs)]

        labelFilename = self.logFilenameList[np.where(self.logTimestampList==logLabelTimestamp)[0][0]]
        
        
        #print('labelFilename is ', os.path.join(os.environ['MKID_DATA_DIR'],labelFilename))
        fin=open(os.path.join(os.environ['MKID_DATA_DIR'],labelFilename),'r')
        text = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S\n') + labelFilename[:-4] + '\n' + fin.read()
        self.label_log.setText(text)
        fin.close()



    
    def create_main_frame(self):
        """
        Makes GUI elements on the window
        """
        #Define the plot window. 
        self.main_frame = QWidget()
        self.dpi = 100
        self.fig = Figure(figsize = (4.0, 5.0), dpi=self.dpi) #define the figure, set the size and resolution
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.ax1 = self.fig.add_subplot(111)
        self.foo = self.ax1.imshow(self.image,interpolation='none')
        self.fig.cbar = self.fig.colorbar(self.foo)
        
        
        #spinboxes for the img timestamp
        self.spinbox_imgTimestamp = QSpinBox()
        self.spinbox_imgTimestamp.valueChanged.connect(self.spinBoxValueChange)
        
        #spinboxes for specifying dark frames
        self.spinbox_darkStart = QSpinBox()
        self.spinbox_darkStart.valueChanged.connect(self.getDarkFrame)
        self.spinbox_darkIntTime = QSpinBox()
                #set up the limits and initial value of the darkIntTime
        self.spinbox_darkIntTime.setMinimum(1)
        self.spinbox_darkIntTime.setMaximum(1000)
        self.spinbox_darkIntTime.setValue(10)
        self.spinbox_darkIntTime.valueChanged.connect(self.getDarkFrame)
        

        
        
        #labels for the start/stop time spinboxes
        label_imgTimestamp = QLabel('IMG timestamp')
        label_darkStart = QLabel('dark Start')
        label_darkIntTime = QLabel('dark int time [s]')
        
        
        #make a checkbox for the colorbar autoscale
        self.checkbox_colorbar_auto = QCheckBox()
        self.checkbox_colorbar_auto.setChecked(False)
        self.checkbox_colorbar_auto.stateChanged.connect(self.spinBoxValueChange)
        
        label_checkbox_colorbar_auto = QLabel('Auto colorbar')
        
        self.spinbox_colorBarMax = QSpinBox()
        self.spinbox_colorBarMax.setRange(1,2500)
        self.spinbox_colorBarMax.setValue(2000)
        self.spinbox_colorBarMax.valueChanged.connect(self.spinBoxValueChange)
        
        
        #make a checkbox for the dark subtract
        self.checkbox_darkSubtract = QCheckBox()
        self.checkbox_darkSubtract.setChecked(False)
        self.checkbox_darkSubtract.stateChanged.connect(self.spinBoxValueChange)
        
        #make a label for the dark subtract checkbox
        label_darkSubtract = QLabel('dark subtract')
        
        
        #make a label for the logs
        self.label_log = QLabel('')

    
        #create a vertical box for the plot to go in.
        vbox_plot = QVBoxLayout()
        vbox_plot.addWidget(self.canvas)
        
        #create a v box for the timestamp spinbox
        vbox_imgTimestamp = QVBoxLayout()
        vbox_imgTimestamp.addWidget(label_imgTimestamp)
        vbox_imgTimestamp.addWidget(self.spinbox_imgTimestamp)

        #make an hbox for the dark start
        hbox_darkStart = QHBoxLayout()
        hbox_darkStart.addWidget(label_darkStart)
        hbox_darkStart.addWidget(self.spinbox_darkStart)
        
        #make an hbox for the dark integration time
        hbox_darkIntTime = QHBoxLayout()
        hbox_darkIntTime.addWidget(label_darkIntTime)
        hbox_darkIntTime.addWidget(self.spinbox_darkIntTime)
        
        #make an hbox for the dark subtract checkbox
        hbox_darkSubtract = QHBoxLayout()
        hbox_darkSubtract.addWidget(label_darkSubtract)
        hbox_darkSubtract.addWidget(self.checkbox_darkSubtract)
        
        #make a vbox for the autoscale colorbar
        hbox_autoscale = QHBoxLayout()
        hbox_autoscale.addWidget(label_checkbox_colorbar_auto)
        hbox_autoscale.addWidget(self.checkbox_colorbar_auto)
        hbox_autoscale.addWidget(self.spinbox_colorBarMax)
        
        #make a vbox for dark times
        vbox_darkTimes = QVBoxLayout()
        vbox_darkTimes.addLayout(hbox_darkStart)
        vbox_darkTimes.addLayout(hbox_darkIntTime)
        vbox_darkTimes.addLayout(hbox_darkSubtract)
        vbox_darkTimes.addLayout(hbox_autoscale)
        
        
        
        #hbox_imgTimestamp.addWidget(self.checkbox_colorbar_auto)    #we can add these later
        #hbox_imgTimestamp.addWidget(label_checkbox_colorbar_auto)
#        hbox_imgTimestamp.addWidget(self.label_log)

#        hbox_imgTimestamp.addWidget(button_quickLoad) #################################################
        
        hbox_controls = QHBoxLayout()
        hbox_controls.addLayout(vbox_imgTimestamp)
        hbox_controls.addLayout(vbox_darkTimes)
        hbox_controls.addWidget(self.label_log)


        
        #Now create another vbox, and add the plot vbox and the button's hbox to the new vbox.
        vbox_combined = QVBoxLayout()
        vbox_combined.addLayout(vbox_plot)
#        vbox_combined.addLayout(hbox_imgTimestamp)
        vbox_combined.addLayout(hbox_controls)
        
        #Set the main_frame's layout to be vbox_combined
        self.main_frame.setLayout(vbox_combined)

        #Set the overall QWidget to have the layout of the main_frame.
        self.setCentralWidget(self.main_frame)
        
        #set up the pyqt5 events
        cid = self.fig.canvas.mpl_connect('motion_notify_event', self.hoverCanvas)
        cid3 = self.fig.canvas.mpl_connect('scroll_event', self.scroll_ColorBar)
        

        
        
    def spinBoxValueChange(self):     
        try:
            filename = self.fileListRaw[np.where(self.timeStampList==self.spinbox_imgTimestamp.value())[0][0]]
        except:
            self.plotBlank()
            self.updateLogLabel(fileExist = False)
        else:
            self.plotImage(filename)
            self.updateLogLabel()
            
        self.draw()
        


        
        
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
        
        openFileButton = QAction(QIcon('exit24.png'), 'Open img File', self)
        openFileButton.setShortcut('Ctrl+O')
        openFileButton.setStatusTip('Open an img File')
        openFileButton.triggered.connect(self.getFileNameFromUser)
        self.fileMenu.addAction(openFileButton)
        
        
        exitButton = QAction(QIcon('exit24.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)
        self.fileMenu.addAction(exitButton)
        
      
        self.menubar.setNativeMenuBar(False) #This is for MAC OS


        
        
    def getFileNameFromUser(self):
        # look at this website for useful examples
        # https://pythonspot.com/pyqt5-file-dialog/
        
        filename, _ = QFileDialog.getOpenFileName(self, 'Select One File', os.environ['MKID_IMG_DIR'],filter = '*.img')
        
#        filename, _ = QFileDialog.getOpenFileName(self, 'Select One File', '/Users/clint/Documents/mazinlab/ScienceDataIMGs/PAL2017b/20171004',filter = '*.img')

        self.filename = filename
        self.loadFilenames(self.filename)
        
        
        
        
        
if __name__ == "__main__":
    a = QApplication(sys.argv)
    foo = mainWindow()
    foo.show()
    sys.exit(a.exec_())