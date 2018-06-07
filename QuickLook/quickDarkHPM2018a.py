import glob
import sys, os, time, struct
import numpy as np
import tables
from scipy import ndimage
from scipy import signal
import astropy
import _pickle as pickle
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from functools import partial
from DarknessPipeline.P3Utils.parsePacketDump2 import parsePacketData
from DarknessPipeline.P3Utils.arrayPopup import plotArray
from DarknessPipeline.P3Utils.readDict import readDict

import DarknessPipeline.Cleaning.HotPix.darkHotPixMask2018a as dhpm

import image_registration as ir
from DarknessPipeline.ImageReg.loadStack import loadIMGStack, loadBINStack
import DarknessPipeline.ImageReg.irUtils as irUtils
from DarknessPipeline.P3Utils.plottingTools import DraggableColorbar
from DarknessPipeline.P3Utils.plottingTools import MyNormalize

darkSpanImg=[1527762695,1527762715]
target='WL05302018d'
outputDir = '/mnt/data0/ProcessedData/2018a/imageStacks/'
imgDir = '/mnt/ramdisk/'
maxCut=1500
sigma=3
numCols=80
numRows=125

dataDir=imgDir

print("Making HPM")
darkHPMImg = dhpm.makeMask(basePath=imgDir,startTimeStamp=darkSpanImg[0],     
stopTimeStamp=darkSpanImg[1], coldCut=True, maxCut=maxCut,sigma=sigma,manualArray=None)
hpFN='darkHPMImg_'+target+'.npz'  #Save the hot pixel mask into the output directory
hpPath = os.path.join(outputDir,hpFN)
np.savez(hpPath,darkHPMImg = darkHPMImg)
plotArray(darkHPMImg, title='Hot Pixel Mask Image', origin='upper')


print("Loading dark frame to make Dark and Saving It")
darkStackImg = loadIMGStack(dataDir, darkSpanImg[0], darkSpanImg[1], nCols=numCols, nRows=numRows, verbose=False)
darkImg = irUtils.medianStack(darkStackImg)
plotArray(darkImg,title='DarkImg',origin='upper')
darkImg[np.where(darkHPMImg==1)]=np.nan
plotArray(darkImg,title='DarkImg HP Masked',origin='upper')
darkFN='darkImg_'+target+'.npz'  #Save the dark into the output directory
darkPath = os.path.join(outputDir,darkFN)
np.savez(darkPath,darkImg = darkImg)


