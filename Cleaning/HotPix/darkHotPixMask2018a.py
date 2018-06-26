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
from DarknessPipeline.P3Utils.arrayPopup import plotArray
import DarknessPipeline.P3Utils
import warnings
from DarknessPipeline.ImageReg.loadStack import loadIMGStack
import image_registration as ir
import DarknessPipeline.ImageReg.irUtils as irUtils


#ultimately want the path to be flexible: use .bin files if they exist,
#and fall back on .img files if necessary (typically we didn't record
#.bin files for dark data in 2017a)
#basePath = '/mnt/data0/ScienceDataIMGs/'
nCols=80
nRows=125

def makeMask(basePath=None, startTimeStamp=None, stopTimeStamp=None, verbose=False, sigma=3, maxCut=1500, coldCut=False, manualArray=None):
    '''
    MaxCut sets level for initial hot pixel cut. Everything with cps>maxCut -> np.nan
    Sigma determines level for second cut. Everything with cps>mean+sigma*stdev -> np.nan
    If coldCut=True, third cut where everything with cps<mean-sigma*stdev -> np.nan
    manualArray is a way to give a list of bad pixels manually, in format [[row,col],[row,col],...]


    '''

    dataPath = basePath
    stack = loadIMGStack(dataPath, startTimeStamp, stopTimeStamp, nCols=nCols, nRows=nRows)

    medStack = irUtils.medianStack(stack) #####

    if verbose:
        try:
            plotArray(medStack,title='Median Dark Stack')
        except:
            plt.matshow(medStack)
            plt.show()

    #initial masking, take out anything with cps > maxCut
    mask = np.zeros(np.shape(medStack),dtype=int)
    mask[np.where(medStack>=maxCut)]=1

    if verbose:
        try:
            plotArray(mask, title='cps>%i == 1'%maxCut)
        except:
            plt.matshow(mask)
            plt.show()

    medStack[np.where(mask==1)]=np.nan
    medStack[np.where(medStack==0)]=np.nan
    if verbose:
        try:
            plotArray(medStack, title='Median Stack with mask 1')
        except:
            plt.matshow(medStack)
            plt.show()

    #second round of masking, where cps > mean+sigma*std
    mask2 = np.zeros(np.shape(medStack),dtype=int)
    mask2[np.where(medStack>=np.nanmean(medStack)+sigma*np.nanstd(medStack))]=1
    #plotArray(mask2, title='ColdCutNotYetApplied')
    print('Standard Deviation ',np.nanstd(medStack))
    print('Mean ', np.nanmean(medStack))
    #if coldCut is true, also mask cps < mean-sigma*std
    if coldCut==True:
        mask2[np.where(medStack<=np.nanmean(medStack)-sigma*np.nanstd(medStack))]=1
        #plotArray(mask2, title='ColdCutApplied')

    if verbose:
        try:
            plotArray(mask2, title='cps>mean+%i-sigma == 1'%sigma)
        except:
            plt.matshow(mask2)
            plt.show()

    medStack[np.where(mask2==1)]=np.nan
    if verbose:
        try:
            plotArray(medStack, title='Median Stack with mask 2')
        except:
            plt.matshow(medStack)
            plt.show()

    finalMask = mask+mask2

    # provide an easy means to pipe in an array of manually identified pixels
    if manualArray is not None:
        for pix in manualArray:
            finalMask[pix[0],pix[1]]=1


    if verbose:
        try:
            plotArray(finalMask, title='Final mask')
        except:
            plt.matshow(finalMask)
            plt.show()
        print("Number masked pixels = ", len(np.array(np.where(finalMask==1)).flatten()))

    return finalMask

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("No arguments provided, running on test files")
        startTS = 1527203919
        stopTS = 1527203924
        verb=True
        coldCut=True
        basePath = '/mnt/ramdisk/'

    mask = makeMask(basePath=basePath, startTimeStamp=startTS, stopTimeStamp=stopTS, verbose=verb,
                    coldCut=coldCut,manualArray=None)


