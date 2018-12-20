#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:28:30 2018

@author: clint

GO TO THE ENCLOSING DIRECTORY AND RUN IT FROM THE TERMINAL WITH THE FOLLOWING COMMAND:
python quickLook_img.py

set the system variable $MKID_IMG_DIR to the place where you want to look for img files
set the system variable $MKID_RAW_PATH to the place where you want to look for log files

"""

import datetime
import os
import os.path
import sys

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class mainWindow(QMainWindow):
    
    
    updateActivePix = pyqtSignal()

    def __init__(self,parent=None):
        QMainWindow.__init__(self,parent=parent)
        self.initializeEmptyArrays()
        self.setWindowTitle('quickLook_img.py')
        self.resize(600,850)  #(600,850 works for clint's laptop screen. Units are pixels I think.)
        self.create_main_frame()
        self.create_status_bar()
        self.createMenu()
        #self.load_beam_map()
        
        
    def initializeEmptyArrays(self,nCol = 80,nRow = 125):
        self.nCol = nCol
        self.nRow = nRow

        self.rawCountsImage = np.zeros(self.nRow*self.nCol).reshape((self.nRow,self.nCol))
        self.hotPixMask = np.zeros(self.nRow*self.nCol).reshape((self.nRow,self.nCol))
        self.hotPixCut = 2400
        self.image = np.zeros(self.nRow*self.nCol).reshape((self.nRow,self.nCol))
        self.beamFlagMask = np.zeros(self.nRow*self.nCol).reshape((self.nRow,self.nCol))

        
        

    def get_nPixels(self,filename):	
        #140 x 146 for MEC
        #80 x 125 for darkness	
        	
        npixels = len(np.fromfile(open(filename, mode='rb'),dtype=np.uint16))	
        print('npixels = ', npixels, '\n')	
        	
        if npixels == 10000: #darkness	
            nCol = 80	
            nRow = 125	
            print('\n\ncamera is DARKNESS/PICTURE-C\n\n')	
        elif npixels == 20440:  #mec
            nCol = 140	
            nRow = 146
            print('\n\ncamera is MEC\n\n')	
        else:	
            raise ValueError('img does not have 10000 or 20440 pixels')
           	
        return nCol, nRow


    def load_IMG_filenames(self,filename):
        print('\nloading img filenames')
               
        self.imgPath = os.path.dirname(filename)        
        fileListRaw = []
        timeStampList = np.array([])
        ii = 0
        for file in os.listdir(self.imgPath):
            if file.endswith(".img"):
                fileListRaw = fileListRaw + [os.path.join(self.imgPath, file)]
                timeStampList = np.append(timeStampList,np.fromstring(file[:-4],dtype=int, sep=' ')[0])
            else:
                continue
            ii+=1


        #the files may not be in chronological order, so let's enforce it
        fileListRaw = np.asarray(fileListRaw)
        fileListRaw = fileListRaw[np.argsort(timeStampList)]
        timeStampList = np.sort(np.asarray(timeStampList))

        self.fileListRaw = fileListRaw
        self.timeStampList = timeStampList

        print('\nfound {:d} .img files\n'.format(len(self.timeStampList)))
        print('first timestamp: ',self.timeStampList[0])
        print('last timestamp:  ',self.timeStampList[-1],'\n')

        

    def load_log_filenames(self):
        #check if directory exists
        if not os.path.exists(self.logPath):
            text = 'log file path not found.\n Check log file path.'
            self.label_log.setText(text)

            self.logTimestampList = np.asarray([])
            self.logFilenameList = np.asarray([])

            return



        #load the log filenames
        print('\nloading log filenames\n')
        logFilenameList = []
        logTimestampList = []
        
        for logFilename in os.listdir(self.logPath):
            
            if logFilename.endswith("telescope.log"):
                continue
            elif logFilename.endswith(".log"):
                logFilenameList.append(logFilename)
                logTimestampList.append(np.fromstring(logFilename[:10],dtype=int, sep=' ')[0])
                

        #the files may not be in chronological order, so let's enforce it
        logFilenameList = np.asarray(logFilenameList)
        logFilenameList = logFilenameList[np.argsort(logTimestampList)]
        logTimestampList = np.sort(np.asarray(logTimestampList))
        
        self.logTimestampList = np.asarray(logTimestampList)
        self.logFilenameList = logFilenameList


    def load_beam_map(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Select One File', '/mnt/data0/Darkness/20180522/Beammap/',filter = '*.txt')
        resID, flag, xPos, yPos = np.loadtxt(filename, unpack=True,dtype = int)

        #resID, flag, xPos, yPos = np.loadtxt('/mnt/data0/Darkness/20180522/Beammap/finalMap_20180524.txt', unpack=True,dtype = int)

        temp = np.nonzero(flag) #get the indices of the nonzero elements. 

        self.beamFlagMask[yPos[temp]][xPos[temp]]=1 #beamFlagMask is 1 when the pixel is not beam mapped
        #self.beamFlagMask = beamFlagMask



    def initialize_spinbox_values(self,filename):
        #set up the spinbox limits and start value, which will be the file you selected
        self.spinbox_imgTimestamp.setMinimum(self.timeStampList[0])
        self.spinbox_imgTimestamp.setMaximum(self.timeStampList[-1])
        self.spinbox_imgTimestamp.setValue(np.fromstring(os.path.basename(filename)[:-4],dtype=int, sep=' ')[0])
        
        self.spinbox_darkStart.setMinimum(self.timeStampList[0])
        self.spinbox_darkStart.setMaximum(self.timeStampList[-10])
        self.spinbox_darkStart.setValue(np.fromstring(os.path.basename(filename)[:-4],dtype=int, sep=' ')[0])
        

        
        
    def plotImage(self,filename = None):        
        
        if filename == None:
            filename = self.fileListRaw[np.where(self.timeStampList==self.spinbox_imgTimestamp.value())[0][0]]      

        self.ax1.clear()         
        
        self.rawImage = np.transpose(np.reshape(np.fromfile(open(filename, mode='rb'),dtype=np.uint16), (self.nCol,self.nRow)))        
        
        
#        image=np.fromfile(open(fn, mode='rb'),dtype=np.uint16)
#        image = np.transpose(np.reshape(image, (self.nCols, self.nRows)))
        
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
        self.cleanedImage = self.cleanedImage*np.logical_not(self.beamFlagMask)
        self.image = self.cleanedImage
        self.ax1.imshow(self.image,vmin = self.cbarLimits[0],vmax = self.cbarLimits[1])
        self.ax1.axis('off')
        
        
        self.draw()
        
        

    def getDarkFrame(self):
        #get an average dark from darkStart to darkStart + darkIntTime
        darkIntTime = self.spinbox_darkIntTime.value()
        darkFrame = np.zeros(darkIntTime*self.nRow*self.nCol).reshape((darkIntTime,self.nRow,self.nCol))
        
        for ii in range(darkIntTime):
            try:
                darkFrameFilename = self.fileListRaw[np.where(self.timeStampList==(self.spinbox_darkStart.value()+ii))[0][0]]
            except:
                pass
            else:
                darkFrame[ii] = np.transpose(np.reshape(np.fromfile(open(darkFrameFilename, mode='rb'),dtype=np.uint16), (self.nCol,self.nRow)))

        self.darkFrame = np.median(darkFrame,axis=0)

        
        
        
    def plotBlank(self):
        self.ax1.imshow(np.zeros(self.nRow*self.nCol).reshape((self.nRow,self.nCol)))
        
        
        
    def updateLogLabel(self,IMG_fileExists = True):

        timestamp = self.spinbox_imgTimestamp.value()

        #check if self.logTimestampList has more than zero entries. If not, return.
        if len(self.logTimestampList)==0:
            text = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S\n\n') + 'no log file found.\n Check log file path.'
            self.label_log.setText(text)
            return

        
        #check if the img exists, if not then return
        if IMG_fileExists==False:
            text = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S\n\n') + 'no .img file found'
            self.label_log.setText(text)
            return

        #check if a nearby log file exists, then pick the closest one
        diffs = timestamp - self.logTimestampList
        if np.sum(np.abs(diffs)<3600)==0: #nearby means within 1 hour. 
            text = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S\n\n') + 'nearest log is ' + str(np.amin(diffs)) + '\nseconds away from img'
            self.label_log.setText(text)
            return

        diffs[np.where(diffs<0)] = np.amax(diffs)

        logLabelTimestamp = self.logTimestampList[np.argmin(diffs)]

        labelFilename = self.logFilenameList[np.where(self.logTimestampList==logLabelTimestamp)[0][0]]
        
        
        #print('labelFilename is ', os.path.join(os.environ['MKID_RAW_PATH'],labelFilename))
        #fin=open(os.path.join(os.environ['MKID_RAW_PATH'],labelFilename),'r')
        fin=open(os.path.join(self.logPath,labelFilename),'r')
        text = 'img timestamp:\n' + datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') + '\n\nLogfile time:\n' + datetime.datetime.fromtimestamp(logLabelTimestamp).strftime('%Y-%m-%d %H:%M:%S\n') + '\n' + labelFilename[:-4] + '\n' + fin.read()
        self.label_log.setText(text)
        fin.close()



    
    def create_main_frame(self):
        """
        Makes GUI elements on the window
        """
        #Define the plot window. 
        self.main_frame = QWidget()
        self.dpi = 100
        self.fig = Figure(figsize = (5.0, 10.0), dpi=self.dpi, tight_layout=True) #define the figure, set the max size (inches) and resolution. Overall window size is set with QMainWindow parameter. 
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.axis('off')
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
        
        
        #make a label to display the IMG path and the MKID_RAW_PATH. Also set up log path variable
        try:
            os.environ['MKID_IMG_DIR']
        except:
            labelText = 'MKID_IMG_DIR:      could not find MKID_IMG_DIR'
            self.imgPath = '/'
        else:
            labelText = 'MKID_IMG_DIR:      ' + os.environ['MKID_IMG_DIR']
            self.imgPath = os.environ['MKID_IMG_DIR']
        
        self.label_IMG_path = QLabel(labelText)
        self.label_IMG_path.setToolTip('Look for img files in this directory. To change, go to File>Open img file')  

        try:
            os.environ['MKID_RAW_PATH']
        except:
            labelText = 'MKID_RAW_PATH:  could not find MKID_RAW_PATH'
            self.logPath = '/'
        else:
            labelText = 'MKID_RAW_PATH:  ' + os.environ['MKID_RAW_PATH']
            self.logPath = os.environ['MKID_RAW_PATH']

        self.label_log_path = QLabel(labelText)
        self.label_log_path.setToolTip('Look for log files in this directory. To change, go to File>Change log path.') 



    
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
        
        hbox_controls = QHBoxLayout()
        hbox_controls.addLayout(vbox_imgTimestamp)
        hbox_controls.addLayout(vbox_darkTimes)
        hbox_controls.addWidget(self.label_log)


        
        #Now create another vbox, and add the plot vbox and the button's hbox to the new vbox.
        vbox_combined = QVBoxLayout()
        vbox_combined.addLayout(vbox_plot)
#        vbox_combined.addLayout(hbox_imgTimestamp)
        vbox_combined.addLayout(hbox_controls)
        vbox_combined.addWidget(self.label_IMG_path)
        vbox_combined.addWidget(self.label_log_path)
        
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
            self.updateLogLabel(IMG_fileExists = False)
        else:
            self.plotImage(filename)
            self.updateLogLabel()
            
        


        
        
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

        openFileButton = QAction('Open img File', self)
        openFileButton.setShortcut('Ctrl+O')
        openFileButton.setStatusTip('Open an img File')
        openFileButton.triggered.connect(lambda x: self.getFileNameFromUser(fileType = 'img'))
        self.fileMenu.addAction(openFileButton)


        changeLogDirectory_Button = QAction('Change log directory', self)
        changeLogDirectory_Button.setShortcut('Ctrl+l')
        changeLogDirectory_Button.setStatusTip('Opens a dialog box so user can select log file manually.')
        changeLogDirectory_Button.triggered.connect(lambda x: self.getFileNameFromUser(fileType = 'log'))
        self.fileMenu.addAction(changeLogDirectory_Button)

        self.fileMenu.addSeparator()
        
        
        exitButton = QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)
        self.fileMenu.addAction(exitButton)
        
      
        self.menubar.setNativeMenuBar(False) #This is for MAC OS


        
        
    def getFileNameFromUser(self,fileType):
        # look at this website for useful examples
        # https://pythonspot.com/pyqt5-file-dialog/
        if fileType == 'img':
            try:
                filename, _ = QFileDialog.getOpenFileName(self, 'Select One File', self.imgPath,filter = '*.img')
            except:
                filename, _ = QFileDialog.getOpenFileName(self, 'Select One File', '/',filter = '*.img')
                    
            
            if filename=='':
                print('\nno file selected\n')
                return

            self.imgPath = os.path.dirname(filename)
            self.label_IMG_path.setText('img path:  ' + self.imgPath)

            self.filename = filename
            self.load_IMG_filenames(self.filename)
            self.load_log_filenames()
            self.nCol, self.nRow = self.get_nPixels(self.filename)
            self.initializeEmptyArrays(self.nCol,self.nRow)
            self.initialize_spinbox_values(self.filename)

        elif fileType == 'log':
            try:
                filename, _ = QFileDialog.getOpenFileName(self, 'Select One File', self.logPath,filter = '*.log')
            except:
                filename, _ = QFileDialog.getOpenFileName(self, 'Select One File', '/',filter = '*.log')
                    
            
            if filename=='':
                print('\nno file selected\n')
                return
            
            self.logPath = os.path.dirname(filename)
            self.label_log_path.setText('log path:  ' + self.logPath)
            self.load_log_filenames()
            self.updateLogLabel()

        else:
            return
        
        
        
if __name__ == "__main__":
    a = QApplication(sys.argv)
    foo = mainWindow()
    foo.show()
    sys.exit(a.exec_())
