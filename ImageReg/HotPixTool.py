'''
Author Isabel Lipartito 2018-05-02

A quick routine to remove hot pixels before image registration, adapted from SR Meeker's hot pixel routine (which used dark img files)

We first apply a hard cut on anything over 2450 cps, since the firmware has a soft
2500 cps limit to stop buffers from overflowing with hot pixels' data. Anything over ~2450
cps in the dark is cut outright. We then take the mean cps on the remaining pixels,
and further cut anything with cps > 5-sigma over the mean.

Accepts .np files, makes and saves a masked version of the npz file and the hot pixel mask.  Does not overrwite original image, just makes a new one with HPM appended to the name.
Assumes .np files are in cps (i.e. those that come from H5Stacker
'''

import glob
import sys, os, time, struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import DarknessPipeline.Utils
from DarknessPipeline.Utils.arrayPopup import plotArray
import warnings
from DarknessPipeline.ImageReg.loadStack import loadIMGStack
import image_registration as ir
import DarknessPipeline.ImageReg.irUtils

nCols=80
nRows=125

def makeMaskedImage(imagePath=None, verbose=False, sigma=None, maxCut=2450, coldCut=False):
    '''
    MaxCut sets level for initial hot pixel cut. Everything with cps>maxCut -> np.nan
    Sigma determines level for second cut. Everything with cps>mean+sigma*stdev -> np.nan
    If coldCut=True, third cut where everything with cps<mean-sigma*stdev -> np.nan

    imagePath:  The full image path to where your .np file is
    verbose:  Do you want it to make the plots or not
    sigma:  How many standard deviations away from the mean is an acceptable pixel value
    maxCut:  Initial cut, 2450 as per SR Meeker
    coldCut:  Should pixels that are too low in value be cut (mean-sigma*st.dev)

    '''

    array=np.load(imagePath)
    imageDirPath,imageBasename = os.path.split(imagePath)

    if verbose:
        try:
            plotArray(array,title='Unmasked Image')
        except:
            plt.matshow(array)
            plt.show()

    #initial masking, take out anything with cps > maxCut
    mask = np.zeros(np.shape(array),dtype=int)
    mask[np.where(array>=maxCut)]=1

    if verbose:
        try:
            plotArray(mask, title='cps>%i == 1'%maxCut)
        except:
            plt.matshow(mask)
            plt.show()

    array[np.where(mask==1)]=np.nan
    array[np.where(array==0)]=np.nan
    if verbose:
        try:
            plotArray(array, title='Image with mask 1')
        except:
            plt.matshow(array)
            plt.show()

    #second round of masking, where cps > mean+sigma*std
    mask2 = np.zeros(np.shape(array),dtype=int)
    with warnings.catch_warnings():
        # nan values will give an unnecessary RuntimeWarning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mask2[np.where(array>=np.nanmean(array)+sigma*np.nanstd(array))]=1

    #if coldCut is true, also mask cps < mean-sigma*std
    if coldCut==True:
        with warnings.catch_warnings():
            # nan values will give an unnecessary RuntimeWarning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mask2[np.where(array<=np.nanmean(array)-sigma*np.nanstd(array))]=1

    if verbose:
        try:
            plotArray(mask2, title='cps>mean+%i-sigma == 1'%sigma)
        except:
            plt.matshow(mask2)
            plt.show()

    array[np.where(mask2==1)]=np.nan
    if verbose:
        try:
            plotArray(array, title='Image with mask 2')
        except:
            plt.matshow(array)
            plt.show()
    saveMaskBasename = os.path.splitext(imageBasename)[0]+'Mask'
    saveImageBasename=os.path.splitext(imageBasename)[0]+'HPMasked'
    outfileMask = os.path.join(imageDirPath,saveMaskBasename)
    outfileImage = os.path.join(imageDirPath,saveImageBasename)
    finalMask = mask+mask2
    finalImage=array
    np.save(outfileMask, finalMask)
    np.save(outfileImage, finalImage)

def makeMultipleMaskedImages(filePath=None, verbose=False, sigma=None, maxCut=2450, coldCut=False):
    '''Grab all .npy files in a directory (assuming they are all from the same DitherStack sequence) and masks them
       Instead of specifying the path to the image, specify the path to the directory where the .npy files will be
       Verbose is False.
    '''
    ImgList = glob.glob(filePath+'*.npy')  
    print(ImgList)
    for i in np.arange(len(ImgList)):
        imagePath=ImgList[i]
        makeMaskedImage(imagePath=imagePath, verbose=False, sigma=sigma, maxCut=maxCut, coldCut=coldCut)

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("No arguments provided, running on test files")


