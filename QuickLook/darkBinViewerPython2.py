'''
Author: Seth Meeker        Date: August 5, 2016
Based mostly on darkQuickViewer.py
TODO:
        Right now images are loaded and diplayed properly from the parsed dictionary
        Need to implement appending of phases and other parsed data into full photon lists
        Speed up parsing of data packets. VERY SLOW NOW!
'''

import sys, os, time, struct
import numpy as np
from PyQt4 import QtCore
from PyQt4 import QtGui
import matplotlib
matplotlib.rcParams['backend.qt4']='PyQt4'
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from functools import partial
from DarknessPipeline.Utils.parsePacketDump2Python2 import parsePacketData
import DarknessPipeline.Utils.parsePacketDump2Python2
import DarknessPipeline.Cleaning.HotPix.darkHotPixMaskPython2 as dhpm


basePath = '/mnt/data0/ScienceData/'
imageShape = {'nRows':125,'nCols':80}


class DarkQuick(QtGui.QMainWindow):
    def __init__(self, run, date,startTstamp, endTstamp):
        
        self.dataPath = basePath+str(run)+'/'+str(date)+'/'
        self.startTstamp = startTstamp
        self.endTstamp = endTstamp
        self.imageStack = []
        self.currentImageIndex = 0
        
        self.darkLoaded = False
        self.subtractDark = False
        #temporary kludge to select dark frame time range
        #self.darkStart = 1469342778
        #self.darkEnd = 1469342798
        
        self.darkStart = 0
        self.darkEnd = 0
        
        self.app = QtGui.QApplication([])
        self.app.setStyle('plastique')
        super(DarkQuick,self).__init__()
        self.setWindowTitle('Darkness Binary Viewer')
        self.createWidgets()
        self.initWindows()
        self.createMenu()
        self.createStatusBar()
        self.arrangeMainFrame()
        self.connectControls()

        self.badPixMask = None
        self.beammap=None
        #self.applyBeammap()
        self.loadBadPixMask()

        #self.applyBeammap()
        if self.subtractDark:
            self.generateDarkFrame()        
        self.loadImageStack()
        self.getObsImage()

    def show(self):
        super(DarkQuick,self).show()
        self.app.exec_()

    def initWindows(self):
        self.imageParamsWindow = ImageParamsWindow(self)
        self.plotWindows = []

    def newPlotWindow(self):
        newPlotId = len(self.plotWindows)
        plotWindow = PlotWindow(parent=self,plotId=newPlotId,selectedPixels=self.arrayImageWidget.selectedPixels)
        self.plotWindows.append(plotWindow)
        self.connect(self.arrayImageWidget,QtCore.SIGNAL('newPixelSelection(PyQt_PyObject)'), plotWindow.newPixelSelection)
        
        plotWindow.show()

    def createWidgets(self):
        self.mainFrame = QtGui.QWidget()
        
        self.label_startTstamp = QtGui.QLabel(str(self.startTstamp))
        self.label_endTstamp = QtGui.QLabel(str(self.endTstamp))
        self.lineEdit_currentTstamp = QtGui.QLineEdit(str(self.startTstamp+self.currentImageIndex))
        
        self.checkbox_applyDark = QtGui.QCheckBox('Subtract Dark')
        self.button_generateDark = QtGui.QPushButton('Generate Dark')
        self.checkbox_applyDark.setChecked(False)
        self.lineEdit_darkStart = QtGui.QLineEdit(str(self.darkStart))
        self.lineEdit_darkEnd = QtGui.QLineEdit(str(self.darkEnd))
        
        self.arrayImageWidget = ArrayImageWidget(parent=self,hoverCall=self.hoverCanvas)
        self.button_jumpToBeginning = QtGui.QPushButton('|<')
        self.button_jumpToEnd = QtGui.QPushButton('>|')
        self.button_incrementBack = QtGui.QPushButton('<')
        self.button_incrementForward = QtGui.QPushButton('>')

        self.button_jumpToBeginning.setMaximumWidth(30)
        self.button_jumpToEnd.setMaximumWidth(30)
        self.button_incrementBack.setMaximumWidth(30)
        self.button_incrementForward.setMaximumWidth(30)


        # Create the navigation toolbar, tied to the canvas
        #self.canvasToolbar = NavigationToolbar(self.canvas, self.mainFrame)

    def arrangeMainFrame(self):

        canvasBox = layoutBox('V',[self.arrayImageWidget])
        incrementControlsBox = layoutBox('H',[1,self.label_startTstamp,self.button_jumpToBeginning,
                self.button_incrementBack,self.lineEdit_currentTstamp,
                self.button_incrementForward,self.button_jumpToEnd,self.label_endTstamp,1])
        darkControlsBox = layoutBox('H',[1, self.checkbox_applyDark, 'Timestamps ', self.lineEdit_darkStart,
                ' to ',self.lineEdit_darkEnd,self.button_generateDark,1])


        mainBox = layoutBox('V',[canvasBox,incrementControlsBox,darkControlsBox])

        self.mainFrame.setLayout(mainBox)
        self.setCentralWidget(self.mainFrame)
  
    def createStatusBar(self):
        self.statusText = QtGui.QLabel("Click Pixels")
        self.statusBar().addWidget(self.statusText, 1)
        
    def createMenu(self):        
        self.fileMenu = self.menuBar().addMenu("&File")
        
        
        quitAction = createAction(self,"&Quit", slot=self.close, 
            shortcut="Ctrl+Q", tip="Close the application")
        
        self.windowsMenu = self.menuBar().addMenu("&Windows")
        imgParamsAction = createAction(self,"Image Plot Parameters",slot=self.imageParamsWindow.show)
        newPlotWindowAction = createAction(self,'New Plot Window',slot=self.newPlotWindow)

        addActions(self.windowsMenu,(newPlotWindowAction,None,imgParamsAction))
        
        self.helpMenu = self.menuBar().addMenu("&Help")
        aboutAction = createAction(self,"&About", 
            shortcut='F1', slot=self.aboutMessage)
        
        addActions(self.helpMenu, (aboutAction,))

    def connectControls(self):
        self.connect(self.button_jumpToBeginning,QtCore.SIGNAL('clicked()'), self.jumpToBeginning)
        self.connect(self.button_jumpToEnd,QtCore.SIGNAL('clicked()'), self.jumpToEnd)
        self.connect(self.button_incrementForward,QtCore.SIGNAL('clicked()'), self.incrementForward)
        self.connect(self.button_incrementBack,QtCore.SIGNAL('clicked()'), self.incrementBack)
        self.connect(self.lineEdit_currentTstamp,QtCore.SIGNAL('editingFinished()'),self.jumpToTstamp)
        self.connect(self.checkbox_applyDark,QtCore.SIGNAL('stateChanged(int)'),self.applyDark)
        self.connect(self.button_generateDark,QtCore.SIGNAL('clicked()'), self.generateDarkFrame)

    def addClickFunc(self,clickFunc):
        self.arrayImageWidget.addClickFunc(clickFunc)

    def applyDark(self):
        if not self.checkbox_applyDark.isChecked():
            self.subtractDark = False
        else:
            if self.darkLoaded==False:
                self.generateDarkFrame()
            self.subtractDark = True

    def generateDarkFrame(self):
        self.darkStart = int(self.lineEdit_darkStart.text())
        self.darkEnd = int(self.lineEdit_darkEnd.text())
        self.darkTimes = np.arange(self.darkStart, self.darkEnd+1)
        darkFrames = []
        for iTs,ts in enumerate(self.darkTimes):
            try:
                imagePath = os.path.join(self.dataPath,str(ts)+'.bin')
                print imagePath
                with open(imagePath,'rb') as dumpFile:
                    data = dumpFile.read()

                nBytes = len(data)
                nWords = nBytes/8 #64 bit words
                
                #break into 64 bit words
                words = np.array(struct.unpack('>{:d}Q'.format(nWords), data),dtype=object)
                parseDict = parsePacketData(words,verbose=False)
                image = parseDict['image']
                
                if self.beammap is not None:
                    newImage = np.zeros(image.shape)
                    for y in range(len(newImage)):
                        for x in range(len(newImage[0])):
                            newX=int(self.beammap[y,x][0])
                            newY=int(self.beammap[y,x][1])
                            if newX >0 and newY>0: 
                                newImage[newY,newX] = image[y,x]
                    image = newImage

            except (IOError, ValueError):
                image = np.zeros((imageShape['nRows'], imageShape['nCols']),dtype=np.uint16)
                print "Failed to load dark frame..."
            darkFrames.append(image)
            
        self.darkStack = np.array(darkFrames)
        self.darkFrame = np.median(self.darkStack, axis=0)
        print "Generated median dark frame from timestamps %i to %i"%(self.darkStart, self.darkEnd)
        self.darkLoaded = True

    def loadBadPixMask(self):
        #hard coded for now. Should supply bad pix mask in gui menu and load it there

        #20170410 WL data bad pixel mask
        #hpmPath = '/mnt/data0/CalibrationFiles/darkHotPixMasks/20170410/1491870115.npz'

        #20170409 pi Her data bad pixel mask
        #hpmPath = '/mnt/data0/CalibrationFiles/darkHotPixMasks/20170409/1491826154.npz'

        #20170408 tau Boo data bad pixel mask
        #hpmPath = '/mnt/data0/CalibrationFiles/darkHotPixMasks/20170408/1491732000.npz'

        #20170410 HD91782 bad pixel mask
        hpmPath = '/mnt/data0/CalibrationFiles/darkHotPixMasks/20170410/1491894755.npz'
        self.badPixMask = dhpm.loadMask(hpmPath)

    def loadImageStack(self):
    
        self.timestampList = np.arange(self.startTstamp,self.endTstamp+1)
        
        images = []
        self.photonTstamps = np.array([])
        self.photonPhases = np.array([])
        self.photonBases = np.array([])
        self.photonXs = np.array([])
        self.photonYs = np.array([])
        self.photonPixelIDs = np.array([])
        
        
        for iTs,ts in enumerate(self.timestampList):
            try:
                imagePath = os.path.join(self.dataPath,str(ts)+'.bin')
                print imagePath
                with open(imagePath,'rb') as dumpFile:
                    data = dumpFile.read()

                nBytes = len(data)
                nWords = nBytes/8 #64 bit words
                
                #break into 64 bit words
                words = np.array(struct.unpack('>{:d}Q'.format(nWords), data),dtype=object)

                parseDict = parsePacketData(words,verbose=False)

                photonTimes = np.array(parseDict['photonTimestamps'])
                phasesDeg = np.array(parseDict['phasesDeg'])
                basesDeg = np.array(parseDict['basesDeg'])
                xCoords = np.array(parseDict['xCoords'])
                yCoords = np.array(parseDict['yCoords'])
                pixelIds = np.array(parseDict['pixelIds'])
                image = parseDict['image']

           
                if self.beammap is not None:
                    newImage = np.zeros(image.shape)
                    for y in range(len(newImage)):
                        for x in range(len(newImage[0])):
                            newX=int(self.beammap[y,x][0])
                            newY=int(self.beammap[y,x][1])
                            if newX >0 and newY>0: 
                                newImage[newY,newX] = image[y,x]
                                #print '('+str(x)+', '+str(y)+') --> ('+str(newX)+', '+str(newY)+')'
                            #else: 
                            #    print '('+str(x)+', '+str(y)+') --> 0'
                            #    newImage[y,x]=0
                    image = newImage
                
                
            except (IOError, ValueError):
                image = np.zeros((imageShape['nRows'], imageShape['nCols']),dtype=np.uint16)  
                
            self.photonTstamps = np.append(self.photonTstamps,photonTimes)
            self.photonPhases = np.append(self.photonPhases,phasesDeg)
            self.photonBases = np.append(self.photonBases,basesDeg)
            self.photonXs = np.append(self.photonXs,xCoords)
            self.photonYs = np.append(self.photonYs,yCoords)
            self.photonPixelIDs = np.append(self.photonPixelIDs, pixelIds)
            images.append(image)
            
        self.imageStack = np.array(images)

    def applyBeammap(self):
        #from addPixID import getPixelIdentificationInfo
        pixInfo = getPixelIdentificationInfo()
        beammapFN = '/mnt/data0/Darkness/20160722/beammap20160722.txt'
        beammapData = np.loadtxt(beammapFN)
        #self.beammap=[0,0]*(80*125)
        #self.beammap=np.reshape(self.beammap,(125,80,-1))
        self.beammap = np.zeros((125,80,2))
        for x in range(80):
            for y in range(125):
                newX = -1
                newY=-1
                try:
                    indx = int(np.where((pixInfo['initX']==x) & (pixInfo['initY']==y))[0][0])
                    resID = pixInfo['resID'][indx]
                    #print resID
                    newIndx = int(np.where(beammapData[:,0]==resID)[0][0])
                    newX = int(beammapData[newIndx, 2])
                    newY = int(beammapData[newIndx, 3])
                    #print '('+str(x)+', '+str(y)+') --> ('+str(newX)+', '+str(newY)+')'
                except IndexError: pass    
                self.beammap[y,x] = [newX, newY]
                

    def getObsImage(self):
        self.lineEdit_currentTstamp.setText(str(self.startTstamp+self.currentImageIndex))
        paramsDict = self.imageParamsWindow.getParams()
        image = self.imageStack[self.currentImageIndex]
        if self.subtractDark:
            if not self.darkLoaded:
                print "Warning: no dark frame loaded"
            else:
                zeroes = np.where(self.darkFrame>image)
                self.image=image-self.darkFrame
                self.image[zeroes] = 0.
        else:
            self.image = image
        
        if self.badPixMask is not None:
            self.image[np.where(self.badPixMask==1)]=0

        self.plotArray(self.image,**paramsDict['plotParams'])
        print self.currentImageIndex

    def jumpToBeginning(self):
        self.currentImageIndex = 0
        self.getObsImage()

    def jumpToEnd(self):
        self.currentImageIndex = len(self.imageStack)-1
        self.getObsImage()

    def incrementForward(self):
        if self.currentImageIndex < len(self.imageStack)-1:
            self.currentImageIndex = self.currentImageIndex + 1
        else:
            print 'Warning: can\'t increment any more'
        self.getObsImage()

    def incrementBack(self):
        if self.currentImageIndex > 0:
            self.currentImageIndex = self.currentImageIndex - 1
        else:
            print 'Warning: can\'t decrement any more'
        self.getObsImage()

    def jumpToTstamp(self):
        desiredTstamp = int(self.lineEdit_currentTstamp.text())
        if (desiredTstamp > self.endTstamp) or (desiredTstamp < self.startTstamp):
            print 'Warning: requested time stamp is outside available range'
        else:
            self.currentImageIndex = desiredTstamp-self.startTstamp
            self.getObsImage()

    def savePlot(self):
        file_choices = "PNG (*.png)|*.png"
        
        path = unicode(QFileDialog.getSaveFileName(self, 
                        'Save file', '', 
                        file_choices))
        if path:
            self.canvas.print_figure(path, dpi=self.dpi)
            self.statusBar().showMessage('Saved to %s' % path, 2000)
    
    def aboutMessage(self):
        msg = """ Use to open and view DARKNESS .img files
        """
        QtGui.QMessageBox.about(self, "Dark Quick File Viewer", msg.strip())

    def plotArray(self,*args,**kwargs):
        self.arrayImageWidget.plotArray(*args,**kwargs)

    def hoverCanvas(self,event):
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if row < self.arrayImageWidget.nRow and col < self.arrayImageWidget.nCol:
            self.statusText.setText('(x,y,z) = ({:d},{:d},{})'.format(col,row,self.arrayImageWidget.image[row,col]))

            

class ModelessWindow(QtGui.QDialog):
    def __init__(self,parent=None):
        super(ModelessWindow,self).__init__(parent=parent)
        self.parent=parent
        self.initUI()
        self._want_to_close = False

    def closeEvent(self, evt):
        if self._want_to_close:
            super(ModelessWindow, self).closeEvent(evt)
        else:
            evt.ignore()
            self.setVisible(False)

    def initUI(self):
        pass

class PlotWindow(QtGui.QDialog):
    def __init__(self,parent=None,plotId=0,selectedPixels=[]):
        super(PlotWindow,self).__init__(parent=parent)
        self.parent=parent
        self.id = plotId
        self.selectedPixels = selectedPixels
        self.initUI()

    def closeEvent(self,evt):
        super(PlotWindow,self).closeEvent(evt)

    def draw(self):
        self.fig.canvas.draw()

    def initUI(self):
        #first gui controls that apply to all modes
        self.checkbox_trackSelection = QtGui.QCheckBox('Plot selected pixel(s)',self)
        self.checkbox_trackSelection.setChecked(True)

        self.checkbox_trackTimes = QtGui.QCheckBox('Use main window times',self)
        self.checkbox_trackTimes.setChecked(True)
        self.connect(self.checkbox_trackTimes,QtCore.SIGNAL('stateChanged(int)'),self.changeTrackTimes)

        self.checkbox_clearPlot = QtGui.QCheckBox('Clear axes before plotting',self)
        self.checkbox_clearPlot.setChecked(True)

        self.button_drawPlot = QtGui.QPushButton('Plot',self)
        self.connect(self.button_drawPlot,QtCore.SIGNAL('clicked()'), self.updatePlot)
        self.dpi = 100
        self.fig = Figure((10.0, 3.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        #self.canvasToolbar = NavigationToolbar(self.canvas, self)
        self.axes = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.07,right=.93,top=.93,bottom=0.15)
        self.axes.tick_params(axis='both', which='major', labelsize=8)

        cid = self.fig.canvas.mpl_connect('button_press_event', self.buttonPressCanvas)
        cid = self.fig.canvas.mpl_connect('button_release_event', self.buttonReleaseCanvas)
        cid = self.fig.canvas.mpl_connect('motion_notify_event', self.motionNotifyCanvas)
        self.selectFirstPoint = None
        self.selectSecondPoint = None
        self.selectedRange = None
        self.selecting = False
        self.lastPlotType = None

        self.combobox_plotType = QtGui.QComboBox(self)
        self.plotTypeStrs = ['Light Curve', 'Phase Histogram', 'Intensity Histogram']
        self.combobox_plotType.addItems(self.plotTypeStrs)
        self.connect(self.combobox_plotType,QtCore.SIGNAL('activated(QString)'), self.changePlotType)


        #light curve controls
        self.textbox_intTime = QtGui.QLineEdit('1')
        self.textbox_intTime.setFixedWidth(50)

        lightCurveControlsBox = layoutBox('H',['Int Time',self.textbox_intTime,'s',1.])
        self.lightCurveControlsGroup = QtGui.QGroupBox('Light Curve Controls',parent=self)
        self.lightCurveControlsGroup.setLayout(lightCurveControlsBox)

        #intensity hist controls
        self.textbox_exposureTime = QtGui.QLineEdit('1')
        self.textbox_exposureTime.setFixedWidth(50)
        
        self.checkbox_restrictIntensityRange = QtGui.QCheckBox('Restrict intensity range')
        self.connect(self.checkbox_restrictIntensityRange,QtCore.SIGNAL('stateChanged(int)'),self.changeRestrictIntensityRange)
        self.checkbox_restrictIntensityRange.setChecked(False)

        self.textbox_intensityHistStart = QtGui.QLineEdit('')
        self.textbox_intensityHistEnd = QtGui.QLineEdit('')
        self.changeRestrictIntensityRange()

        self.textbox_intensityHistNBins = QtGui.QLineEdit('1')

        intensityHistControlsBoxRow0 = layoutBox('H',[self.checkbox_restrictIntensityRange,5.,'Start intensity',self.textbox_intensityHistStart,1.,'End intensity',self.textbox_intensityHistEnd,10.])
        intensityHistControlsBoxRow1 = layoutBox('H',['Intensity bin resolution', self.textbox_intensityHistNBins,5.,'Short Exposure Time',self.textbox_exposureTime,'s', 10.])
        intensityHistControlsBox = layoutBox('V',[intensityHistControlsBoxRow0,intensityHistControlsBoxRow1])

        self.intensityHistControlsGroup = QtGui.QGroupBox('Intensity Histogram Controls',parent=self)
        self.intensityHistControlsGroup.setLayout(intensityHistControlsBox)
        self.intensityHistControlsGroup.setVisible(False)

        #phase hist controls
        self.checkbox_restrictPhaseRange = QtGui.QCheckBox('Restrict phase range')
        self.connect(self.checkbox_restrictPhaseRange,QtCore.SIGNAL('stateChanged(int)'),self.changeRestrictPhaseRange)
        self.checkbox_restrictPhaseRange.setChecked(False)

        self.checkbox_keepRawPhase = QtGui.QCheckBox('Raw units')

        self.textbox_phaseHistStart = QtGui.QLineEdit('')
        self.textbox_phaseHistEnd = QtGui.QLineEdit('')
        self.changeRestrictPhaseRange()

        self.textbox_phaseHistNBins = QtGui.QLineEdit('1')
        self.combobox_phaseHistType = QtGui.QComboBox(self)
        self.phaseHistTypeStrs = ['Peaks','Baselines','Peaks - Baselines']
        self.combobox_phaseHistType.addItems(self.phaseHistTypeStrs)
        self.combobox_phaseHistType.setCurrentIndex(2)

        phaseHistControlsBoxRow0 = layoutBox('H',[self.combobox_phaseHistType,1.,self.checkbox_keepRawPhase,1.,self.checkbox_restrictPhaseRange,10.])
        phaseHistControlsBoxRow1 = layoutBox('H',['Start phase',self.textbox_phaseHistStart,1.,'End phase',self.textbox_phaseHistEnd,1.,'Number of ADC units per bin',self.textbox_phaseHistNBins,10.])
        phaseHistControlsBox = layoutBox('V',[phaseHistControlsBoxRow0,phaseHistControlsBoxRow1])

        self.phaseHistControlsGroup = QtGui.QGroupBox('Phase Histogram Controls',parent=self)
        self.phaseHistControlsGroup.setLayout(phaseHistControlsBox)
        self.phaseHistControlsGroup.setVisible(False)

        #time controls
        self.textbox_startTime = QtGui.QLineEdit(str(self.parent.startTstamp))
        self.textbox_startTime.setFixedWidth(120)
        self.textbox_endTime = QtGui.QLineEdit(str(self.parent.endTstamp))
        self.textbox_endTime.setFixedWidth(120)
        self.timesGroup = QtGui.QGroupBox('',parent=self)
        timesBox = layoutBox('H',['Start Time',self.textbox_startTime,'s',1.,'End Time',self.textbox_endTime,'s',10.])
        self.timesGroup.setLayout(timesBox)
        self.timesGroup.setVisible(False)
        timesChoiceBox = layoutBox('H',[self.checkbox_trackTimes,self.timesGroup])

        checkboxBox = layoutBox('H',[self.checkbox_trackSelection,self.combobox_plotType,2.,self.button_drawPlot])
        controlsBox = layoutBox('H',[self.lightCurveControlsGroup,self.phaseHistControlsGroup, self.intensityHistControlsGroup])

        mainBox = layoutBox('V',[checkboxBox,timesChoiceBox,self.checkbox_clearPlot,controlsBox,self.canvas])
        self.setLayout(mainBox)

    def buttonPressCanvas(self,event):
        if event.inaxes is self.axes:# and self.canvasToolbar.mode == '':
            x = event.xdata
            y = event.ydata
            self.selectFirstPoint = (x,y)
            self.selecting = True

    def buttonReleaseCanvas(self,event):
        self.selecting = False

    def motionNotifyCanvas(self,event):
        if self.selecting and event.inaxes is self.axes:# and self.canvasToolbar.mode == '' and not self.selectFirstPoint is None:
            x = round(event.xdata)
            y = round(event.ydata)
            if self.selectSecondPoint is None or np.round(self.selectSecondPoint[0]) != np.round(x):
                self.selectSecondPoint = (x,y)
                xClicks = np.array([self.selectSecondPoint[0],self.selectFirstPoint[0]])
                self.selectedRange = (int(np.floor(np.min(xClicks))),int(np.ceil(np.max(xClicks))))
            try:
                self.selectedRangeShading.remove()
            except:
                pass
            self.selectedRangeShading = self.axes.axvspan(self.selectedRange[0],self.selectedRange[1],facecolor='b',alpha=0.3)
            self.draw()
        pass

    def changePlotType(self,plotType):
        if plotType == 'Light Curve':
            self.lightCurveControlsGroup.setVisible(True)
            self.phaseHistControlsGroup.setVisible(False)
            self.intensityHistControlsGroup.setVisible(False)
        elif plotType == 'Phase Histogram':
            self.lightCurveControlsGroup.setVisible(False)
            self.phaseHistControlsGroup.setVisible(True)
            self.intensityHistControlsGroup.setVisible(False)
        elif plotType == 'Intensity Histogram':
            self.lightCurveControlsGroup.setVisible(False)
            self.phaseHistControlsGroup.setVisible(False)
            self.intensityHistControlsGroup.setVisible(True)

    def newPixelSelection(self,selectedPixels):
        self.selectedPixels = selectedPixels
        self.updatePlot()

    def updatePlot(self):
        plotType = self.combobox_plotType.currentText()
        if self.checkbox_clearPlot.isChecked() or plotType != self.lastPlotType:
            self.axes.cla()
        if plotType == 'Light Curve':
            self.plotLightCurve()
        elif plotType == 'Phase Histogram':
            self.plotPhaseHist()
        elif plotType == 'Intensity Histogram':
            self.plotIntensityHist()
        self.lastPlotType = plotType
        self.draw()
        print 'plot updated'
        
    def changeTrackTimes(self):
        if self.checkbox_trackTimes.isChecked():
            self.timesGroup.setVisible(False)
        else:
            self.timesGroup.setVisible(True)
        
    def changeRestrictPhaseRange(self):
        if self.checkbox_restrictPhaseRange.isChecked():
            self.textbox_phaseHistStart.setEnabled(True)
            self.textbox_phaseHistEnd.setEnabled(True)
        else:
            self.textbox_phaseHistStart.setEnabled(False)
            self.textbox_phaseHistEnd.setEnabled(False)
            
    def changeRestrictIntensityRange(self):
        if self.checkbox_restrictIntensityRange.isChecked():
            self.textbox_intensityHistStart.setEnabled(True)
            self.textbox_intensityHistEnd.setEnabled(True)
        else:
            self.textbox_intensityHistStart.setEnabled(False)
            self.textbox_intensityHistEnd.setEnabled(False)
    
    def plotLightCurve(self,getRaw=False):
        if self.checkbox_trackTimes.isChecked():
            startTime = float(self.parent.label_startTstamp.text())
            endTime = float(self.parent.label_endTstamp.text())
        else:
            startTime = float(self.textbox_startTime.text())
            endTime = float(self.textbox_endTime.text())
        histIntTime = float(self.textbox_intTime.text())
        firstSec = startTime
        duration = endTime-startTime
        histBinEdges = np.arange(startTime,endTime,histIntTime)
        hists = []
        if histBinEdges[-1]+histIntTime == endTime:
            histBinEdges = np.append(histBinEdges,endTime)
        for col,row in self.selectedPixels:
            selPixelId = self.parent.photonPixelIDs[np.where((self.parent.photonXs==col) & (self.parent.photonYs==row))][0]
            msTimestamps = self.parent.photonTstamps[np.where(self.parent.photonPixelIDs==selPixelId)]
            timestamps = 1.0E-3*(msTimestamps-msTimestamps[0])+self.parent.startTstamp
            hist,_ = np.histogram(timestamps,bins=histBinEdges)
            hists.append(hist)

        hists = np.array(hists)
        hist = np.sum(hists,axis=0)

        binWidths = np.diff(histBinEdges)
        self.lightCurve = 1.*hist#/binWidths
        plotHist(self.axes,histBinEdges,self.lightCurve)
        x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        x_formatter.set_scientific(False)
        self.axes.xaxis.set_major_formatter(x_formatter)
        self.axes.set_xlabel('time (s)')
        self.axes.set_ylabel('counts')
        
        ''' # Old version of lightcurve plotting from IMG files
        for col,row in self.selectedPixels:
            self.lightCurve = self.parent.imageStack[:,row,col]
        self.axes.plot(self.parent.timestampList, self.lightCurve)
        x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
        x_formatter.set_scientific(False)
        self.axes.xaxis.set_major_formatter(x_formatter)
        self.axes.set_xlabel('time (s)')
        self.axes.set_ylabel('counts per sec')
        '''
    
    def plotPhaseHist(self):
        phaseHistType = self.combobox_phaseHistType.currentText()
        if self.checkbox_trackTimes.isChecked():
            startTime = float(self.parent.label_startTstamp.text())
            endTime = float(self.parent.label_endTstamp.text())
        else:
            startTime = float(self.textbox_startTime.text())
            endTime = float(self.textbox_endTime.text())
        histIntTime = float(self.textbox_intTime.text())
        firstSec = startTime
        duration = endTime-startTime
        hists = []
        nBinsPerUnit = int(self.textbox_phaseHistNBins.text())
        histBinEdges = None
        scaleToDegrees = 1. #180./np.pi*1./2**9
        zeroOffset = 0 #2**11 # phase is +/-4 radians, where 0 radians is represented by bin value 2048
        if self.checkbox_restrictPhaseRange.isChecked():
            range = np.array([float(self.textbox_phaseHistStart.text()),float(self.textbox_phaseHistEnd.text())])
            if not self.checkbox_keepRawPhase.isChecked():
                range = range / scaleToDegrees
        else:
            range = None
        pixCutOffPhases=[]
        for col,row in self.selectedPixels:
            #if not self.checkbox_keepRawPhase.isChecked():
            #    cutOffWvl = self.parent.obs.wvlRangeTable[row, col, 0]
            #    pixCutOffPhases.append(self.parent.obs.convertWvlToPhase(cutOffWvl,row,col))
            
            selPixelId = self.parent.photonPixelIDs[np.where((self.parent.photonXs==col) & (self.parent.photonYs==row))][0]

            timestamps = self.parent.photonTstamps[np.where(self.parent.photonPixelIDs==selPixelId)]
            peakHeights = self.parent.photonPhases[np.where(self.parent.photonPixelIDs==selPixelId)]
            baselines = self.parent.photonBases[np.where(self.parent.photonPixelIDs==selPixelId)]
            if phaseHistType == 'Peaks':
                listToHist = peakHeights+baselines
            elif phaseHistType == 'Baselines':
                listToHist = baselines
            elif phaseHistType == 'Peaks - Baselines':
                listToHist = peakHeights

            if histBinEdges == None:
                if not self.checkbox_restrictPhaseRange.isChecked():
                    range = np.array([np.min(listToHist),np.max(listToHist)])
                bins = (range[1]-range[0])//nBinsPerUnit
            else:
                bins = histBinEdges

            hist,histBinEdges = np.histogram(listToHist,bins=bins,range=range)
            hists.append(hist)

        hists = np.array(hists)
        self.hist = np.sum(hists,axis=0)

        binWidths = np.diff(histBinEdges)
            
        if self.checkbox_keepRawPhase.isChecked():
            xlabel = 'phase'
        else:
            histBinEdges = histBinEdges * scaleToDegrees
            xlabel = 'phase (${}^\circ$)'
            pixCutOffPhases=np.asarray(pixCutOffPhases) * scaleToDegrees
        plotHist(self.axes,histBinEdges,self.hist)
        for pixCutOffPhase in pixCutOffPhases:
            self.axes.axvline(x=pixCutOffPhase,color='k')
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel('counts')

    def plotIntensityHist(self):
        if self.checkbox_trackTimes.isChecked():
            startTime = float(self.parent.label_startTstamp.text())
            endTime = float(self.parent.label_endTstamp.text())
        else:
            startTime = float(self.textbox_startTime.text())
            endTime = float(self.textbox_endTime.text())
        histIntTime = float(self.textbox_exposureTime.text())

        #generate light curve same as plotLightcurve routine
        histBinEdges = np.arange(startTime,endTime,histIntTime)
        hists = []
        if histBinEdges[-1]+histIntTime == endTime:
            histBinEdges = np.append(histBinEdges,endTime)
        for col,row in self.selectedPixels:
            selPixelId = self.parent.photonPixelIDs[np.where((self.parent.photonXs==col) & (self.parent.photonYs==row))][0]
            msTimestamps = self.parent.photonTstamps[np.where(self.parent.photonPixelIDs==selPixelId)]
            timestamps = 1.0E-3*(msTimestamps-msTimestamps[0])+self.parent.startTstamp
            hist,_ = np.histogram(timestamps,bins=histBinEdges)
            hists.append(hist)

        hists = np.array(hists)
        lightCurveHist = np.sum(hists,axis=0)
        binWidths = np.diff(histBinEdges)
        
        #take light curve and bin into histogram of intensities
        intHistBinEdges = None
        nBinsPerUnitInt = int(self.textbox_intensityHistNBins.text())
        
        pixCutOffIntensities=[]
        listToHist = lightCurveHist
        
        if self.checkbox_restrictIntensityRange.isChecked():
            range = np.array([float(self.textbox_intensityHistStart.text()),float(self.textbox_intensityHistEnd.text())])
        else:
            range = np.array([np.min(listToHist),np.max(listToHist)])
            
        intBins = (range[1]-range[0])//nBinsPerUnitInt
        intHist,intHistBinEdges = np.histogram(listToHist,bins=intBins,range=range)
        self.intHist = np.array(intHist)

        binWidths = np.diff(intHistBinEdges)
            
        xlabel = 'Counts'
        plotHist(self.axes,intHistBinEdges,self.intHist)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel('N_frames')

class ImageParamsWindow(ModelessWindow):
    def initUI(self):
        self.combobox_cmap = QtGui.QComboBox(self)
        self.cmapStrs = ['hot','gray','jet','gnuplot2','Paired']
        self.combobox_cmap.addItems(self.cmapStrs)

        plotArrayBox = layoutBox('V',[self.combobox_cmap])
        plotArrayGroup = QtGui.QGroupBox('plotArray parameters',self)
        plotArrayGroup.setLayout(plotArrayBox)

        mainBox = layoutBox('V',[plotArrayGroup])

        self.setLayout(mainBox)
            
    def getParams(self):
        plotParamsDict = {}
        cmapStr = str(self.combobox_cmap.currentText())
        cmap = getattr(matplotlib.cm,cmapStr)
        if cmapStr != 'gray':
            cmap.set_bad('0.15')
        plotParamsDict['cmap']=cmap
        #plotParamsDict['vmax']=200
        outDict = {}
        outDict['plotParams'] = plotParamsDict
        return outDict
    
class ArrayImageWidget(QtGui.QWidget):
    def __init__(self,parent=None,hoverCall=None):
        super(ArrayImageWidget,self).__init__(parent=parent)
        self.parent=parent
        # Create the mpl Figure and FigCanvas objects. 
        self.hoverCall = hoverCall
        self.selectPixelsMode = 'singlePixel'
        self.selectionPatches = []
        self.selectedPixels = []
        self.overlayImage = None
        self.initUI()
        

    def initUI(self):
        self.dpi = 100
        self.fig = Figure((5.0, 8.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.axes = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.07,right=.93,top=.93,bottom=0.07)
        self.plotArray(np.arange(9).reshape((3,3)))

        self.clickFuncs = []
        cid = self.fig.canvas.mpl_connect('scroll_event', self.scrollColorBar)
        cid = self.fig.canvas.mpl_connect('button_press_event', self.clickColorBar)
        cid = self.fig.canvas.mpl_connect('motion_notify_event', self.hoverCanvas)
        cid = self.fig.canvas.mpl_connect('button_press_event', self.clickCanvas)
        canvasBox = layoutBox('V',[self.canvas,])
        self.setLayout(canvasBox)

    def plotOverlayImage(self,image,color='green',**kwargs):
        self.overlayImage = image
        self.overlayImageKwargs = kwargs
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',[color,color],256)
        cmap._init() # create the _lut array, with rgba values
        alphas = np.linspace(0, 1., cmap.N+3)
        cmap._lut[:,-1] = alphas
        self.overlayCmap = cmap
        self.drawOverlayImage()

    def removeOverlayImage(self):
        self.overlayImage = None
        try:
            self.handleOverlayMatshow.remove()
        except:
            pass
        self.draw()

    def drawOverlayImage(self):
        try:
            self.handleOverlayMatshow.remove()
        except:
            pass
        self.handleOverlayMatshow = self.axes.matshow(self.overlayImage,cmap=self.overlayCmap,origin='upper',**self.overlayImageKwargs)
        self.draw()

    def plotArray(self,image,normNSigma=3,title='',**kwargs):
        self.image = image
        self.imageShape = np.shape(image)
        self.nRow = self.imageShape[0]
        self.nCol = self.imageShape[1]
        if not 'vmax' in kwargs:
            goodImage = image[np.isfinite(image)]
            kwargs['vmax'] = np.mean(goodImage)+normNSigma*np.std(goodImage)
        if not 'cmap' in kwargs:
            defaultCmap=matplotlib.cm.hot
            defaultCmap.set_bad('0.15')
            kwargs['cmap'] = defaultCmap
        if not 'origin' in kwargs:
            kwargs['origin'] = 'upper'

        self.fig.clf()
        self.selectionPatches = []
        self.axes = self.fig.add_subplot(111)
        self.axes.set_title(title)

        self.matshowKwargs = kwargs
        self.handleMatshow = self.axes.matshow(image,**kwargs)
        self.fig.cbar = self.fig.colorbar(self.handleMatshow)
        if not self.overlayImage is None:
            self.drawOverlayImage()
        else:
            self.draw()
        print 'image drawn'

    def drawSelections(self):
        for patch in self.selectionPatches:
            patch.remove()
        self.selectionPatches = []
        
        for pixelCoord in self.selectedPixels:
            lowerLeftCorner = tuple(np.subtract(pixelCoord, (0.5,0.5)))
            patch = matplotlib.patches.Rectangle(xy=lowerLeftCorner,width=1.,
                    height=1.,edgecolor='blue',facecolor='none')
            self.selectionPatches.append(patch)
            self.axes.add_patch(patch)

    def addClickFunc(self,clickFunc):
        self.clickFuncs.append(clickFunc)

    def emitNewSelection(self):
        self.emit(QtCore.SIGNAL('newPixelSelection(PyQt_PyObject)'),self.selectedPixels)

    def clickCanvas(self,event):
        if event.inaxes is self.axes:
            col = round(event.xdata)
            row = round(event.ydata)
            if self.selectPixelsMode == 'singlePixel':
                self.selectedPixels = [(col,row)]
                self.draw()
                self.emitNewSelection()
            for func in self.clickFuncs:
                func(row=row,col=col)


    def hoverCanvas(self,event):
        if event.inaxes is self.axes:
            self.hoverCall(event)

    def scrollColorBar(self, event):
        if event.inaxes is self.fig.cbar.ax:
            increment=0.05
            currentClim = self.fig.cbar.mappable.get_clim()
            currentRange = currentClim[1]-currentClim[0]
            if event.button == 'up':
                if QtGui.QApplication.keyboardModifiers()==QtCore.Qt.ControlModifier:
                    newClim = (currentClim[0]+increment*currentRange,currentClim[1])
                elif QtGui.QApplication.keyboardModifiers()==QtCore.Qt.NoModifier:
                    newClim = (currentClim[0],currentClim[1]+increment*currentRange)
            if event.button == 'down':
                if QtGui.QApplication.keyboardModifiers()==QtCore.Qt.ControlModifier:
                    newClim = (currentClim[0]-increment*currentRange,currentClim[1])
                elif QtGui.QApplication.keyboardModifiers()==QtCore.Qt.NoModifier:
                    newClim = (currentClim[0],currentClim[1]-increment*currentRange)
            self.fig.cbar.mappable.set_clim(newClim)
            self.fig.canvas.draw()

    def clickColorBar(self,event):
        if event.inaxes is self.fig.cbar.ax:
            self.fig.currentClim = self.fig.cbar.mappable.get_clim()
            lower = self.fig.currentClim[0]
            upper = self.fig.currentClim[1]
            fraction = event.ydata
            currentRange = upper-lower
            clickedValue = lower+fraction*currentRange
            extrapolatedValue = lower+event.ydata*currentRange
            if event.button == 1:
                if QtGui.QApplication.keyboardModifiers()==QtCore.Qt.ControlModifier:
                    newClim = (clickedValue,upper)
                elif QtGui.QApplication.keyboardModifiers()==QtCore.Qt.NoModifier:
                    newClim = (lower,clickedValue)
            if event.button == 3:
                if QtGui.QApplication.keyboardModifiers()==QtCore.Qt.ControlModifier:
                    newClim = ((lower-fraction*upper)/(1.-fraction),upper)
                elif QtGui.QApplication.keyboardModifiers()==QtCore.Qt.NoModifier:
                    newClim = (lower,lower+currentRange/fraction)
            self.fig.cbar.mappable.set_clim(newClim)
            self.fig.canvas.draw()

    def draw(self):
        self.drawSelections()
        self.fig.canvas.draw()

    def addClickFunc(self,clickFunc):
        self.clickFuncs.append(clickFunc)



#gui functions
def addActions(target, actions):
    for action in actions:
        if action is None:
            target.addSeparator()
        else:
            target.addAction(action)

def createAction(  gui, text, slot=None, shortcut=None, 
                    icon=None, tip=None, checkable=False, 
                    signal="triggered()"):
    action = QtGui.QAction(text, gui)
    if icon is not None:
        action.setIcon(QIcon(":/%s.png" % icon))
    if shortcut is not None:
        action.setShortcut(shortcut)
    if tip is not None:
        action.setToolTip(tip)
        action.setStatusTip(tip)
    if slot is not None:
        gui.connect(action, QtCore.SIGNAL(signal), slot)
    if checkable:
        action.setCheckable(True)
    return action

def layoutBox(type,elements):
    if type == 'vertical' or type == 'V':
        box = QtGui.QVBoxLayout()
    elif type == 'horizontal' or type == 'H':
        box = QtGui.QHBoxLayout()
    else:
        raise TypeError('type should be one of [\'vertical\',\'horizontal\',\'V\',\'H\']')

    for element in elements:
        try:
            box.addWidget(element)
        except:
            try:
                box.addLayout(element)
            except:
                try:
                    box.addStretch(element)
                except:
                    try:
                        label = QtGui.QLabel(element)
                        box.addWidget(label)
                        #label.adjustSize()
                    except:
                        print 'could\'t add {} to layout box'.format(element)
    return box

def plotHist(ax,histBinEdges,hist,**kwargs):
    ax.plot(histBinEdges,np.append(hist,hist[-1]),drawstyle='steps-post',**kwargs)


if __name__ == "__main__":
    kwargs = {}
    if len(sys.argv) != 5:
        print 'Usage: {} run date tstampStart tstampEnd'.format(sys.argv[0])
        exit(0)
    else:
        kwargs['run'] = str(sys.argv[1])
        kwargs['date'] = int(sys.argv[2])
        kwargs['startTstamp'] = int(sys.argv[3])
        kwargs['endTstamp'] = int(sys.argv[4])

    form = DarkQuick(**kwargs)
    form.show()



