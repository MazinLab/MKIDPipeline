import glob
import inspect
import math
import os
import sys

import astropy.stats
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy
import scipy
import scipy.ndimage
import scipy.stats
import tables
from astropy import wcs
# import ds9
from numpy import linalg
from scipy.interpolate import griddata
from scipy.optimize.minpack import curve_fit

"""
Modules:

aperture(startpx=None, startpy=None, cenRA=None, cenDec=None, nPixRA=None, nPixDec=None, limits = None, degrees = False, radius=3)
bin12_9ToRad(binOffset12_9)
confirm(prompt,defaultResponse=True)
convertDegToHex(ra, dec)
convertHexToDeg(ra, dec)
gaussian_psf(fwhm, boxsize, oversample=50)
intervalSize(inter)
linearFit(x, y, err=None)
makeMovie( listOfFrameObj, frameTitles, outName, delay, listOfPixelsToMark,
              pixelMarkColor,**plotArrayKeys)              
mean_filterNaN(inputarray, size=3, *nkwarg, **kwarg)
median_filterNaN(inputarray, size=5, *nkwarg, **kwarg)
plotArray( 2darray, colormap=mpl.cm.gnuplot2, normMin=None, normMax=None, showMe=True,
              cbar=False, cbarticks=None, cbarlabels=None, plotFileName='arrayPlot.png',
              plotTitle='', sigma=None, pixelsToMark=[], pixelMarkColor='red')
printCalFileDescriptions( dir_path )
printObsFileDescriptions( dir_path )
rebin2D(a, ysize, xsize)
replaceNaN(inputarray, mode='mean', boxsize=3, iterate=True)
stdDev_filterNaN(inputarray, size=5, *nkwarg, **kwarg)
getGitStatus()
findNearestFinite(im,i,j,n=10)
nearestNstdDevFilter(inputArray,n=24)
nearestNmedFilter(inputArray,n=24)
interpolateImage(inputArray,method='linear')
showzcoord()
fitBlackbody(wvls,flux,fraction=1.0,newWvls=None,tempGuess=6000)
rebin(x,y,binedges)
gaussianConvolution(x,y,xEnMin=0.005,xEnMax=6.0,xdE=0.001,fluxUnits = "lambda",r=8, plots=False)
countsToApparentMag(cps, filterName = 'V', telescope = None)
"""

def aperture(startpx=None, startpy=None, cenRA=None, cenDec=None, nPixRA=None, nPixDec=None, limits = None, degrees = False, radius=3):
    """
    Creates a mask with specified radius and centered at specified pixel
    position.  Output mask is a 46x44 array with 0 represented pixels within
    the aperture and 1 representing pixels outside the aperture.

    Added by Neil (7/31/14):
    startpx/startpy - If working with an image in the detector fram specify these values as your starting x and y pixels
    cenRA/cenDec - This is synonemous to startpx/startpy except use this for virtual images (can be given in degrees or radians)
    nPixRA/nPixDec - These are the number of pixels in the virtual image. Need this to scale degrees/pixel in virtual image
    limits - If working with virtual image, this will be the window of RA and Dec values. (as of right now the input requires these to be in radians)
    degrees - set True if cenRA/cenDec are in degrees, otherwise False
    radius - this is the radius for the aperture. Set to 3 by default for detector images. If using virtual image this must be given in degrees (at least for right now).
    
    """
    r = radius         #sets the radius and calculates hight and length. Note that the radius should be given in whatever units the image will be plotted in
                       #that means for the detector frame it will be given in pixels and for the virtual image probably degrees... could add option to give in radians but
                       #julians RADecImage.display code as of now plots the image in degrees
    length = 2*r 
    height = length
    
    if startpx is not None and startpy is not None:  #If the x and y postions have been defined in the detector frame.
        print('----finding aperture mask in detector frame----')  #gathers all x and y positions within detector frame.
        allx = range(startpx-int(numpy.ceil(length/2.0)),startpx+int(numpy.floor(length/2.0))+1)
        ally = range(startpy-int(numpy.ceil(height/2.0)),startpy+int(numpy.floor(height/2.0))+1)
        pixx = []
        pixy = []
        mask=numpy.ones((46,44))  #sets all pixels in the detector frame equal to 1.
        for x in allx:
            for y in ally:
                if (numpy.abs(x-startpx))**2+(numpy.abs(y-startpy))**2 <= (r)**2 and 0 <= y and y < 46 and 0 <= x and x < 44:
                    mask[y,x]=0.  #If the allx and ally pixel positions are within the aperture radius, sets those pixels to 0
        return mask   #mask is a 46,44 array of all ones except where the aperture has been defined. The aperture will have 0's.
    
    
    
    elif cenRA is not None and cenDec is not None: #If the user wants to define an initial position in RA/Dec space.
                                                   #RADecImage.display shows the objects RA ad Dec in degrees but these can be given in radians as well
        print('----finding aperture mask in RA/Dec frame----')
        
        if nPixRA is not None and nPixDec is not None and limits is not None:  #really if choosing to make an aperture mask for the virtual image nPix should never be none,
            
            limits = numpy.array(limits,dtype = float)  #array of the limits in radians.
            limits = limits*(180./numpy.pi)             #converts limits into degree values
         
            stepRA = (limits[1]-limits[0])/nPixRA     #finds the degree/virtualpixel spacing for RA
            stepDec = (limits[3]-limits[2])/nPixDec   #finds the degree/virtualpixel spacing for Dec
            #print limits
            
            if degrees == True:   #If cenRA/cenDec are given in degrees.
                
                cenRA = ((limits[1]-cenRA)/stepRA)

                #cenRA = ((cenRA-limits[0])/stepRA)   #sets the center of the aperature by converting its location to pixel location on virtual grid.
                cenDec = ((cenDec-limits[2])/stepDec) #both RA and Dec ^^^^
                #print cenRA
                #print cenDec
            else:                   #if cenRA/cenDec are given in radians.

                cenRA = ((limits[1] - (cenRA*(180./numpy.pi)))/stepRA)  #converts the RA/Dec to degrees and converts to pixel location on virtual grid.
                #cenDec = ((limits[3] - (cenDec*(180./numpy.pi)))/stepDec)  
                cenDec = (((cenDec*(180./numpy.pi))-limits[2])/stepDec)            

            if numpy.allclose(stepRA, stepDec) == True:  #this ensures that the RA and Declination are being incremented by the same step values on the virtual grid.
                r = r/stepRA   #converts radius to a radius in the virtual image.
                length = length/stepRA
                height = height/stepRA
                #print r
                #print length
                #print height
            else:
                raise ValueError('cant calculate the radius')   #If stepRA!=stepDec then return erro message.
            
            allRA = range(int(cenRA)-int(numpy.ceil(length/2.0)),int(cenRA)+int(numpy.floor(length/2.0))+1)    #gathers all pixel positions in the virtual image.
            allDec = range(int(cenDec)-int(numpy.ceil(height/2.0)),int(cenDec)+int(numpy.floor(height/2.0))+1) 
            #allRA = numpy.arange((cenRA - length/2.0),(cenRA + length/2.0 + stepRA), step=stepRA)
            #allDec = numpy.arange((cenDec - height/2.0),(cenDec + height/2.0 + stepDec), step=stepDec)
            #print allRA
            #print allDec
        
            print('----creating aperture mask for virtual Image----')
        
            mask = numpy.ones((nPixDec, nPixRA)) #just as before, sets all pixels in the virtual frame equal to 1.
            for RA in allRA:
                for Dec in allDec:
                    if (numpy.abs(RA - cenRA))**2+(numpy.abs(Dec - cenDec))**2 <= (r)**2 and 0 <= Dec and Dec < nPixDec and 0 <= RA and RA < nPixRA:
                        mask[Dec, RA]=0  #If the pixels of allRA/allDec are located within the aperture radius, they are set equal to 0.
            return mask  #returns an array of size nPixRA by nPixDec with all 1's except where the aperture has been defined. these are 0's.
          
        else:
            raise ValueError('oops somethings not right') 






"""
        #conv = (vPlateScale*2*numpy.pi/1296000)  #no. of radians on sky per virtual pixel
        #conv = conv/.0174532925 #no of degrees per virtual pixel
        
            #temp = nPixRA/conv #number of radians in image
            
        
        print 'conv', conv
        print '----converting RA/Dec to virtual pixel locations----'
        r = int(r/conv)
        print 'r', r
        length = int(length/conv)
        print 'length', length
        height = int(height/conv)
        print 'height', height
        cenRA = int(cenRA/conv)
        print 'cenRA', cenRA
        cenDec = int(cenDec/conv) #converts the RA and Dec to location on virtual grid 
        print 'cenDec', cenDec       
        step = int(step/conv)        
        print 'step', step

        allRA = numpy.arange((cenRA - int(numpy.ceil(length/2.0))),(cenRA + int(numpy.floor(length/2.0))+step), step=step)
        allDec = numpy.arange((cenDec - int(numpy.ceil(height/2.0))),(cenDec + int(numpy.floor(height/2.0))+step), step=step)
        viRA = []
        viDec = []
        print allRA
        print allDec
        
"""
        
        
        
        
        
        

def bin12_9ToRad(binOffset12_9):
   """
   To convert one of the raw 12-bit unsigned values from the photon packet
   into a signed float in radians
   """
   x = binOffset12_9/2.**9. - 4.
   return x


def confirm(prompt,defaultResponse = True):
    """
    Displays a prompt, accepts a yes or no answer, and returns a boolean
    defaultResponse is the response returned if no response is given
    if an ill-formed response is given, the prompt is given again
    """
    if defaultResponse == True:
        optionsString = '[y]|n'
    else:
        optionsString = 'y|[n]'
    goodResponse = False
    while goodResponse == False:
        try:
            responseString = input('%s %s: '%(prompt,optionsString))
            if responseString in ['y','Y','yes','Yes','YES']:
                response = True
                goodResponse = True
            elif responseString in ['n','N','no','No','NO']:
                response = False
                goodResponse = True
            elif responseString == '':
                response = defaultResponse
                goodResponse = True
            else:
                goodResponse = False
        except:
            goodResponse = False
        if goodResponse == False:
            print('Unrecognized response. Try again.')
    return response
 
def convertDegToHex(ra, dec):
   """
   Convert RA, Dec in decimal degrees to (hh:mm:ss, dd:mm:ss)
   """

   if(ra<0):
      sign = -1
      ra   = -ra
   else:
      sign = 1
      ra   = ra

   h = int( ra/15. )
   ra -= h*15.
   m = int( ra*4.)
   ra -= m/4.
   s = ra*240.

   if(sign == -1):
      outra = '-%02d:%02d:%06.3f'%(h,m,s)
   else: outra = '+%02d:%02d:%06.3f'%(h,m,s)

   if(dec<0):
      sign = -1
      dec  = -dec
   else:
      sign = 1
      dec  = dec

   d = int( dec )
   dec -= d
   dec *= 100.
   m = int( dec*3./5. )
   dec -= m*5./3.
   s = dec*180./5.

   if(sign == -1):
      outdec = '-%02d:%02d:%06.3f'%(d,m,s)
   else: outdec = '+%02d:%02d:%06.3f'%(d,m,s)

   return outra, outdec


def convertHexToDeg(ra, dec):
   """
   Convert RA, Dec in ('hh:mm:ss', 'dd:mm:ss') into floating point degrees.
   """

   try :
      pieces = ra.split(':')
      hh=int(pieces[0])
      mm=int(pieces[1])
      ss=float(pieces[2])
   except:
      raise
   else:
      pass
   
   Csign=dec[0]
   if Csign=='-':
      sign=-1.
      off = 1
   elif Csign=='+':
      sign= 1.
      off = 1
   else:
      sign= 1.
      off = 0

   try :
      parts = dec.split(':')
      deg=int(parts[0][off:len(parts[0])])
      arcmin=int(parts[1])
      arcsec=float(parts[2])
   except:
      raise
   else:
      pass

   return(hh*15.+mm/4.+ss/240., sign*(deg+(arcmin*5./3.+arcsec*5./180.)/100.) )


def ds9Array(xyarray, colormap='B', normMin=None, normMax=None,
             sigma=None, scale=None,
             #pixelsToMark=[], pixelMarkColor='red',
             frame=None):
    """
    
    Display a 2D array as an image in DS9 if available. Similar to 'plotArray()'
    
    xyarray is the array to plot

    colormap - string, takes any value in the DS9 'color' menu.

    normMin minimum used for normalizing color values

    normMax maximum used for normalizing color values

    sigma calculate normMin and normMax as this number of sigmas away
    from the mean of positive values

    scale - string, can take any value allowed by ds9 xpa interface.
    Allowed values include:
        linear|log|pow|sqrt|squared|asinh|sinh|histequ
        mode minmax|<value>|zscale|zmax
        limits <minvalue> <maxvalue>
    e.g.: scale linear
        scale log 100
        scale datasec yes
        scale histequ
        scale limits 1 100
        scale mode zscale
        scale mode 99.5 
        ...etc.
    For more info see:
        http://hea-www.harvard.edu/saord/ds9/ref/xpa.html#scale

    ## Not yet implemented: pixelsToMark a list of pixels to mark in this image

    ## Not yet implemented: pixelMarkColor is the color to fill in marked pixels
    
    frame - to specify which DS9 frame number the array should be displayed in.
             Default is None. 
    
    """
    if sigma != None:
       # Chris S. does not know what accumulatePositive is supposed to do
       # so he changed the next two lines.
       #meanVal = numpy.mean(accumulatePositive(xyarray))
       #stdVal = numpy.std(accumulatePositive(xyarray))
       meanVal = numpy.mean(xyarray)
       stdVal = numpy.std(xyarray)
       normMin = meanVal - sigma*stdVal
       normMax = meanVal + sigma*stdVal


    d = ds9.ds9()   #Open a ds9 instance
    if type(frame) is int:
        d.set('frame '+str(frame))
        
    d.set_np2arr(xyarray)
    #d.view(xyarray, frame=frame)
    d.set('zoom to fit')
    d.set('cmap '+colormap)
    if normMin is not None and normMax is not None:
        d.set('scale '+str(normMin)+' '+str(normMax))
    if scale is not None:
        d.set('scale '+scale)

    
    #plt.matshow(xyarray, cmap=colormap, origin='lower',norm=norm, fignum=False)

    #for ptm in pixelsToMark:
    #    box = mpl.patches.Rectangle((ptm[0]-0.5,ptm[1]-0.5),\
    #                                    1,1,color=pixelMarkColor)
    #    #box = mpl.patches.Rectangle((1.5,2.5),1,1,color=pixelMarkColor)
    #    fig.axes[0].add_patch(box)

 

def gaussian_psf(fwhm, boxsize, oversample=50):
    
    """
    Returns a simulated Gaussian PSF: an array containing a 2D Gaussian function
    of width fwhm (in pixels), binned down to the requested box size. 
    JvE 12/28/12
    
    INPUTS:
        fwhm - full-width half-max of the Gaussian in pixels
        boxsize - size of (square) output array
        oversample (optional) - factor by which the raw (unbinned) model Gaussian
                                oversamples the final requested boxsize.
    
    OUTPUTS:
        2D boxsize x boxsize array containing the binned Gaussian PSF
    
    (Verified against IDL astro library daoerf routine)
        
    """
    fineboxsize = boxsize * oversample
    
    xcoord = ycoord = numpy.arange(-(fineboxsize - 1.) / 2., (fineboxsize - 1.) / 2. + 1.)
    xx, yy = numpy.meshgrid(xcoord, ycoord)
    xsigma = ysigma = fwhm / (2.*math.sqrt(2.*math.log(2.))) * oversample
    zx = (xx ** 2 / (2 * xsigma ** 2))
    zy = (yy ** 2 / (2 * ysigma ** 2))
    fineSampledGaussian = numpy.exp(-(zx + zy))

    #Bin down to the required output boxsize:
    binnedGaussian = rebin2D(fineSampledGaussian, boxsize, boxsize)

    return binnedGaussian


def intervalSize(inter):
    """
    INPUTS:
        inter - a pyinterval 'interval' instance.
    OUTPUTS:
        Returns the total size of an pyinterval '(multi)interval' instance. 
        So if an interval instance represents a set of time ranges, this returns
        the total amount of time covered by the ranges.
        E.g.:
            >>> from interval import interval
            >>> from util import utils
            >>> 
            >>> x=interval([10,15],[9],[2,3],[2.5,3.5]) #Sub-intervals are automatically unioned
            >>> x
            interval([2.0, 3.5], [9.0], [10.0, 15.0])
            >>> utils.intervalSize(x)
            6.5
    """
    size=0.0
    for eachComponent in inter.components:
        size+=(eachComponent[0][-1]-eachComponent[0][0])
    return size

   
def linearFit( x, y, err=None ):
    """
    Fit a linear function y as a function of x.  Optional parameter err is the
    vector of standard errors (in the y direction).

    Returns:  solution - where y = solution[0] + solution[1]*x
    """
    x = numpy.copy(x)
    y = numpy.copy(y)
    N = len(x)
    A = numpy.ones((2, N), x.dtype)
    A[1] = x
    if err!=None: A /= err
    A = numpy.transpose(A)
    if err!=None: y /= err

    solution, residuals, rank, s = scipy.linalg.lstsq(A, y)
    return solution

def fitRigidRotation(x,y,ra,dec,x0=0,y0=0,chatter=False):
    """
    calculate the rigid rotation from row,col positions to ra,dec positions

    return dictionary of theta,tx,ty, such that

    ra  = c*dx - s*dx + dra
    dec = s*dy + c*dy + ddec
    
    with c = scale*cos(theta) and s = scale*sin(theta)
         dx = x-x0 and dy = y-y0

    ra,dec are input in decimal degrees

    if chatter is True print some things to stdout

    The scale and rotation of the transform are recovered from the cd matrix;
      rm = w.wcs.cd
      wScale = math.sqrt(rm[0,0]**2+rm[0,1]**2) # degrees per pixel
      wTheta = math.atan2(rm[1,0],rm[0,0])      # radians


    """
    assert(len(x)==len(y)==len(ra)==len(dec)), "all inputs must be same length"
    assert(len(x) > 1), "need at least two points"

    dx = x-x0
    dy = y-y0
    a = numpy.zeros((2*len(x),4))
    b = numpy.zeros(2*len(x))
    for i in range(len(x)):
        a[2*i,0] = -dy[i]
        a[2*i,1] = dx[i]
        a[2*i,2] = 1
        b[2*i]   = ra[i]

        a[2*i+1,0] = dx[i]
        a[2*i+1,1] = dy[i]
        a[2*i+1,3] = 1
        b[2*i+1] = dec[i]
    answer,residuals,rank,s = linalg.lstsq(a,b)
    
    # put the fit parameters into the WCS structure
    sst = answer[0] # scaled sin theta
    sct = answer[1] # scaled cos theta
    dra = answer[2]
    ddec = answer[3]
    scale = math.sqrt(sst**2+sct**2)
    theta = math.degrees(math.atan2(sst,sct))
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [x0,y0]     # reference pixel position
    w.wcs.crval = [dra,ddec]  # reference sky position
    w.wcs.cd = [[sct,-sst],[sst,sct]] # scaled rotation matrix
    w.wcs.ctype = ["RA---TAN","DEC--TAN"]
    return w

def makeMovie( listOfFrameObj, frameTitles=None, outName='Test_movie',
              delay=0.1, listOfPixelsToMark=None, pixelMarkColor='red',
               **plotArrayKeys):
    """
    Makes a movie out of a list of frame objects (2-D arrays). If you
    specify other list inputs, these all need to be the same length as
    the list of frame objects.

    listOfFrameObj is a list of 2d arrays of numbers

    frameTitles is a list of titles to put on the frames

    outName is the file name to write, .gif will be appended

    delay in seconds between frames

    listOfPixelsToMark is a list.  Each entry is itself a list of
    pixels to mark pixelMarkColor is the color to fill in the marked
    pixels
    
    """
    # Looks like theres some sort of bug when normMax != None.
    # Causes frame to pop up in a window as gif is made.
    if len(listOfFrameObj) == 1:
        raise ValueError("I cannot make movie out of a list of one object!")

    if frameTitles != None:
        assert len(frameTitles) == len(listOfFrameObj), "Number of Frame titles\
        must equal number of frames"

    if os.path.exists("./.tmp_movie"):
        os.system("rm -rf .tmp_movie")

    os.mkdir(".tmp_movie")
    iFrame = 0
    print('Making individual frames ...')
    
    for frame in listOfFrameObj:

       if frameTitles!= None:
          plotTitle = frameTitles[iFrame]
       else:
          plotTitle=''

       if listOfPixelsToMark!= None:
           pixelsToMark = listOfPixelsToMark[iFrame]
       else:
           pixelsToMark = []
       pfn = '.tmp_movie/mov_'+repr(iFrame+10000)+'.png'
       fp = plotArray(frame, showMe=False, plotFileName=pfn,
                      plotTitle=plotTitle, pixelsToMark=pixelsToMark,
                      pixelMarkColor=pixelMarkColor, **plotArrayKeys)
       iFrame += 1
       del fp

    os.chdir('.tmp_movie')

    if outName[-4:-1]+outName[-1] != '.gif':
        outName += '.gif'

    delay *= 100
    delay = int(delay)
    print('Making Movie ...')

    if '/' in outName:
        os.system('convert -delay %s -loop 0 mov_* %s'%(repr(delay),outName))
    else:
        os.system('convert -delay %s -loop 0 mov_* ../%s'%(repr(delay),outName))
    os.chdir("../")
    os.system("rm -rf .tmp_movie")
    print('done.')
    
    
def mean_filterNaN(inputarray, size=3, *nkwarg, **kwarg):
    """
    Basically a box-car smoothing filter. Same as median_filterNaN, but calculates a mean instead. 
    Any NaN values in the input array are ignored in calculating means.
    See median_filterNaN for details.
    JvE 1/4/13
    """
    return scipy.ndimage.filters.generic_filter(inputarray, lambda x:numpy.mean(x[~numpy.isnan(x)]), size,
                                                 *nkwarg, **kwarg)

def median_filterNaN(inputarray, size=5, *nkwarg, **kwarg):
    """
    NaN-handling version of the scipy median filter function
    (scipy.ndimage.filters.median_filter). Any NaN values in the input array are
    simply ignored in calculating medians. Useful e.g. for filtering 'salt and pepper
    noise' (e.g. hot/dead pixels) from an image to make things clearer visually.
    (but note that quantitative applications are probably limited.)
    
    Works as a simple wrapper for scipy.ndimage.filters.generic-filter, to which
    calling arguments are passed.
    
    Note: mode='reflect' looks like it would repeat the edge row/column in the
    'reflection'; 'mirror' does not, and may make more sense for our applications.
    
    Arguments/return values are same as for scipy median_filter.
    INPUTS:
        inputarray : array-like, input array to filter (can be n-dimensional)
        size : scalar or tuple, optional, size of edge(s) of n-dimensional moving box. If 
                scalar, then same value is used for all dimensions.
    OUTPUTS:
        NaN-resistant median filtered version of inputarray.
    
    For other parameters see documentation for scipy.ndimage.filters.median_filter.

    e.g.:
        
        filteredImage = median_filterNaN(imageArray,size=3)
    
    -- returns median boxcar filtered image with a moving box size 3x3 pixels.
    
    JvE 12/28/12
    """
    return scipy.ndimage.filters.generic_filter(inputarray, lambda x:numpy.median(x[~numpy.isnan(x)]), size,
                                                 *nkwarg, **kwarg)

def nanStdDev(x):
    """
    NaN resistant standard deviation - basically scipy.stats.tstd, but
    with NaN rejection, and returning NaN if there aren't enough non-NaN
    input values, instead of just crashing. Used by stdDev_filterNaN.
    INPUTS:
        x - array of input values
    OUTPUTS:
        The standard deviation....
    """
    xClean = x[~numpy.isnan(x)]
    if numpy.size(xClean) > 1 and xClean.min() != xClean.max():
        return scipy.stats.tstd(xClean)
    #Otherwise...
    return numpy.nan
    
    
def plotArray( xyarray, colormap=mpl.cm.gnuplot2, 
               normMin=None, normMax=None, showMe=True,
               cbar=False, cbarticks=None, cbarlabels=None, 
               plotFileName='arrayPlot.png',
               plotTitle='', sigma=None, 
               pixelsToMark=[], pixelMarkColor='red',
               fignum=1, pclip=None):
    """
    Plots the 2D array to screen or if showMe is set to False, to
    file.  If normMin and normMax are None, the norm is just set to
    the full range of the array.

    xyarray is the array to plot

    colormap translates from a number in the range [0,1] to an rgb color,
    an existing matplotlib.cm value, or create your own

    normMin minimum used for normalizing color values

    normMax maximum used for normalizing color values

    showMe=True to show interactively; false makes a plot

    cbar to specify whether to add a colorbar
    
    cbarticks to specify whether to add ticks to the colorbar

    cbarlabels lables to put on the colorbar

    plotFileName where to write the file

    plotTitle put on the top of the plot

    sigma calculate normMin and normMax as this number of sigmas away
    from the mean of positive values

    pixelsToMark a list of pixels to mark in this image

    pixelMarkColor is the color to fill in marked pixels
    
    fignum - to specify which window the figure should be plotted in.
             Default is 1. If None, automatically selects a new figure number.
            Added 2013/7/19 2013, JvE
    
    pclip - set to percentile level (in percent) for setting the upper and
            lower colour stretch limits (overrides sigma).
    
    """
    if sigma != None:
       # Chris S. does not know what accumulatePositive is supposed to do
       # so he changed the next two lines.
       #meanVal = numpy.mean(accumulatePositive(xyarray))
       #stdVal = numpy.std(accumulatePositive(xyarray))
       meanVal = numpy.nanmean(xyarray)
       stdVal = numpy.nanstd(xyarray)
       normMin = meanVal - sigma*stdVal
       normMax = meanVal + sigma*stdVal
    if pclip != None:
        normMin = numpy.percentile(xyarray[numpy.isfinite(xyarray)], pclip)
        normMax = numpy.percentile(xyarray[numpy.isfinite(xyarray)], 100.-pclip)
    if normMin == None:
       normMin = xyarray.min()
    if normMax == None:
       normMax = xyarray.max()
    norm = mpl.colors.Normalize(vmin=normMin,vmax=normMax)

    figWidthPt = 550.0
    inchesPerPt = 1.0/72.27                 # Convert pt to inch
    figWidth = figWidthPt*inchesPerPt       # width in inches
    figHeight = figWidth*1.0                # height in inches
    figSize =  [figWidth,figHeight]
    params = {'backend': 'ps',
              'axes.labelsize': 10,
              'axes.titlesize': 12,
              'text.fontsize': 10,
              'legend.fontsize': 10,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'figure.figsize': figSize}


    fig = plt.figure(fignum) ##JvE - Changed fignum=1 to allow caller parameter
    plt.clf()
    plt.rcParams.update(params)
    plt.matshow(xyarray, cmap=colormap, origin='lower',norm=norm, fignum=False)

    for ptm in pixelsToMark:
        box = mpl.patches.Rectangle((ptm[0]-0.5,ptm[1]-0.5),\
                                        1,1,color=pixelMarkColor)
        #box = mpl.patches.Rectangle((1.5,2.5),1,1,color=pixelMarkColor)
        fig.axes[0].add_patch(box)

    if cbar:
        if cbarticks == None:
           cbar = plt.colorbar(shrink=0.8)
        else:
           cbar = plt.colorbar(ticks=cbarticks, shrink=0.8)
        if cbarlabels != None:
           cbar.ax.set_yticklabels(cbarlabels)
    
    plt.ylabel('Row Number')
    plt.xlabel('Column Number')
    plt.title(plotTitle)

    if showMe == False:
        plt.savefig(plotFileName)
    else:    
        plt.show()
 

def printCalFileDescriptions( dir_path ):
    """
    Prints the 'description' and 'target' header values for all calibration
    files in the specified directory
    """
    for obs in glob.glob(os.path.join(dir_path,'cal*.h5')):
       f=tables.openFile(obs,'r')
       hdr=f.root.header.header.read()
       print(obs,hdr['description'][0])
       target = f.root.header.header.col('target')[0]
       print(target)
       f.close()
    

def printObsFileDescriptions( dir_path ):
    """
    Prints the 'description' and 'target' header values for all observation
    files in the specified directory
    Added sorting to returned list - JvE Nov 7 2014
    """
    for obs in sorted(glob.glob(os.path.join(dir_path,'obs*.h5'))):
        f=tables.openFile(obs,'r')
    try:
            hdr=f.root.header.header.read()
            print(obs,hdr['description'][0])
    except:
        pass
        try:
            target = f.root.header.header.col('target')[0]
            print(target)
        except:
            pass
        f.close()
  

def rebin2D(a, ysize, xsize):
    """
    Rebin an array to a SMALLER array. Rescales the values such that each element
    in the output array is the mean of the elememts which it encloses in the input
    array (i.e., not the total). Similar to the IDL rebin function.
    Dimensions of binned array must be an integer factor of the input array.
    Adapted from SciPy cookbook - see http://www.scipy.org/Cookbook/Rebinning
    JvE 12/28/12

    INPUTS:
        a - array to be rebinned
        ysize - new ysize (must be integer factor of y-size of input array)
        xsize - new xsize (ditto for x-size of input array)

    OUTPUTS:
        Returns the original array rebinned to the new dimensions requested.        
    """
    
    yfactor, xfactor = numpy.asarray(a.shape) / numpy.array([ysize, xsize])
    return a.reshape(ysize, yfactor, xsize, xfactor,).mean(1).mean(2)


def replaceNaN(inputarray, mode='mean', boxsize=3, iterate=True):
    """
    Replace all NaN values in an array with the mean (or median)
    of the surrounding pixels. Should work for any number of dimensions, 
    but only fully tested for 2D arrays at the moment.
    
    INPUTS:
        inputarray - input array
        mode - 'mean', 'median', or 'nearestNmedian', to replace with the mean or median of
                the neighbouring pixels. In the first two cases, calculates on the basis of
                a surrounding box of side 'boxsize'. In the latter, calculates median on the 
                basis of the nearest N='boxsize' non-NaN pixels (this is probably a lot slower
                than the first two methods). 
        boxsize - scalar integer, length of edge of box surrounding bad pixels from which to
                  calculate the mean or median; or in the case that mode='nearestNmedian', the 
                  number of nearest non-NaN pixels from which to calculate the median.
        iterate - If iterate is set to True then iterate until there are no NaN values left.
                  (To deal with cases where there are many adjacent NaN's, where some NaN
                  elements may not have any valid neighbours to calculate a mean/median. 
                  Such elements will remain NaN if only a single pass is done.) In principle,
                  should be redundant if mode='nearestNmedian', as far as I can think right now
    
    OUTPUTS:
        Returns 'inputarray' with NaN values replaced.
        
    TO DO: currently spits out multiple 'invalid value encoutered' warnings if 
           NaNs are not all removed on the first pass. These can safely be ignored.
           Will implement some warning catching to suppress them.
    JvE 1/4/2013    
    """
    
    outputarray = numpy.copy(inputarray)
    while numpy.sum(numpy.isnan(outputarray)) > 0 and numpy.all(numpy.isnan(outputarray)) == False:
        
        #Calculate interpolates at *all* locations (because it's easier...)
        if mode=='mean':
            interpolates = mean_filterNaN(outputarray,size=boxsize,mode='mirror')
        elif mode=='median':
            interpolates = median_filterNaN(outputarray,size=boxsize,mode='mirror')
        elif mode=='nearestNmedian':
            interpolates = nearestNmedFilter(outputarray,n=boxsize)
        else:
            raise ValueError('Invalid mode selection - should be one of "mean", "median", or "nearestNmedian"')
        
        #Then substitute those values in wherever there are NaN values.
        outputarray[numpy.isnan(outputarray)] = interpolates[numpy.isnan(outputarray)]
        if not iterate: break 

    return outputarray


def stdDev_filterNaN(inputarray, size=5, *nkwarg, **kwarg):
    """
    Calculated a moving standard deviation across a 2D (image) array. The standard
    deviation is calculated for a box of side 'size', centered at each pixel (element)
    in the array. NaN values are ignored, and the center pixel at which the box is located
    is masked out, so that only the surrounding pixels are included in calculating the
    std. dev. Thus each element in the array can later be compared against
    this std. dev. map in order to effectively find outliers.
    
    Works as a simple wrapper for scipy.ndimage.filters.generic-filter, to which
    calling arguments are passed.
    
    Arguments/return values are same as for scipy median_filter.
    INPUTS:
        inputarray : array-like, input array to filter (can be n-dimensional)
        size : scalar or tuple, optional, size of edge(s) of n-dimensional moving box. If 
                scalar, then same value is used for all dimensions.
    OUTPUTS:
        NaN-resistant std. dev. filtered version of inputarray.
    
    """
    
    #Can set 'footprint' as follows to remove the central array element:
    #footprint = numpy.ones((size,size))
    #footprint[size/2,size/2] = 0
            
    return scipy.ndimage.filters.generic_filter(inputarray, nanStdDev, 
                                                size=size, *nkwarg, **kwarg)
    
    
    
def getGit():
    """
    return a Gittle, which controls the state of the git repository
    """
    utilInitFile = inspect.getsourcefile(sys.modules['Utils'])
    if (not os.path.isabs(utilInitFile)):
        utilInitFile = os.path.join(os.getcwd(),utilInitFile)
    darkRoot = os.path.split(os.path.split(utilInitFile)[0])[0]
    print("darkRoot=",darkRoot)
    git = Gittle(darkRoot)
    return git
    
def getGitStatus():
    """
    returns a dictionary with the following keys;

    repo : the repository (local directory)
    log0 : the most recent log entry
    remotes: the remote repository/repositories
    remote_branches:  name and tag for the remote branches
    head:  the current check out commit
    has_commits:  boolean -- True if there is code here that has not been committed
    modified_files:  a list of files that have been modified (including ones that are not tracked by git
    modified_unstaged_files:  a list of "important" files that are modified but not committed

    """
    git = getGit()
    return {"repo":git.repo,
            "log0":git.log()[0],
            "last_commit":git.last_commit,
            "remotes":git.remotes,
            "remote_branches":git.remote_branches,
            "head":git.head,
            "has_commits":git.has_commits,
            "modified_files":git.modified_files,
            "modified_unstaged_files":git.modified_unstaged_files
            }

    
    
def findNearestFinite(im,i,j,n=10):
    """
    JvE 2/25/2014
    Find the indices of the nearest n finite-valued (i.e. non-nan, non-infinity)
    elements to a given location within a 2D array. Pretty easily extendable
    to n dimensions in theory, but would probably mean slowing it down somewhat.
    
    The element i,j itself is *not* returned.
    
    If n is greater than the number of finite valued elements available,
    it will return the indices of only the valued elements that exist.
    If there are no finite valued elements, it will return a tuple of
    empty arrays. 
    
    No guarantees about the order
    in which elements are returned that are equidistant from i,j.
    
    
    
    INPUTS:
        im - a 2D numerical array
        i,j - position to search around (i=row, j=column)
        n - find the nearest n finite-valued elements

    OUTPUTS:
        (#Returns an boolean array matching the shape of im, with 'True' where the nearest
        #n finite values are, and False everywhere else. Seems to be outdated - see below. 06/11/2014,
        JvE. Should probably check to be sure there wasn't some mistake. ).
        
        Returns a tuple of index arrays (row_array, col_array), similar to results
        returned by the numpy 'where' function.
    """
    
    imShape = numpy.shape(im)
    assert len(imShape) == 2
    nRows,nCols = imShape
    ii2,jj2 = numpy.atleast_2d(numpy.arange(-i,nRows-i,dtype=float), numpy.arange(-j,nCols-j,dtype=float))
    distsq = ii2.T**2 + jj2**2
    good = numpy.isfinite(im)
    good[i,j] = False #Get rid of element i,j itself.
    ngood = numpy.sum(good)
    distsq[~good] = numpy.nan   #Get rid of non-finite valued elements
    #Find indices of the nearest finite values, and unravel the flattened results back into 2D arrays
    nearest = (numpy.unravel_index(
                                   (numpy.argsort(distsq,axis=None))[0:min(n,ngood)],imShape
                                   )) #Should ignore NaN values automatically
    
    #Below version is maybe slightly quicker, but at this stage doesn't give quite the same results -- not worth the trouble
    #to figure out right now.
    #nearest = (numpy.unravel_index(
    #                               (numpy.argpartition(distsq,min(n,ngood)-1,axis=None))[0:min(n,ngood)],
    #                               imShape
    #                              )) #Should ignore NaN values automatically
    
    return nearest



def nearestNstdDevFilter(inputArray,n=24):
    """
    JvE 2/25/2014
    Return an array of the same shape as the (2D) inputArray, with output values at each element
    corresponding to the standard deviation of the nearest n finite values in inputArray.
    
    INPUTS:
        inputArray - 2D input array of values.
        n - number of nearest finite neighbours to sample for calculating std. dev. around each pixel.
        
    OUTPUTS:
        A 2D array of standard deviations with the same shape as inputArray
    """
    
    outputArray = numpy.zeros_like(inputArray)
    outputArray.fill(numpy.nan)
    nRow,nCol = numpy.shape(inputArray)
    for iRow in numpy.arange(nRow):
        for iCol in numpy.arange(nCol):
            outputArray[iRow,iCol] = numpy.std(inputArray[findNearestFinite(inputArray,iRow,iCol,n=n)])
    return outputArray


def nearestNrobustSigmaFilter(inputArray,n=24):
    """
    JvE 4/8/2014
    Similar to nearestNstdDevFilter, but estimate the standard deviation using the 
    median absolute deviation instead, scaled to match 1-sigma (for a normal
    distribution - see http://en.wikipedia.org/wiki/Robust_measures_of_scale).
    Should be more robust to outliers than regular standard deviation.
    
    INPUTS:
        inputArray - 2D input array of values.
        n - number of nearest finite neighbours to sample for calculating std. dev. around each pixel.
        
    OUTPUTS:
        A 2D array of standard deviations with the same shape as inputArray
    """
    
    outputArray = numpy.zeros_like(inputArray)
    outputArray.fill(numpy.nan)
    nRow,nCol = numpy.shape(inputArray)
    for iRow in numpy.arange(nRow):
        for iCol in numpy.arange(nCol):
            vals = inputArray[findNearestFinite(inputArray,iRow,iCol,n=n)]
            #MAD seems to give best compromise between speed and reasonable results.
            #Biweight midvariance is good, somewhat slower.            
            #outputArray[iRow,iCol] = numpy.diff(numpy.percentile(vals,[15.87,84.13]))/2.
            outputArray[iRow,iCol] = astropy.stats.median_absolute_deviation(vals)*1.4826
            #outputArray[iRow,iCol] = astropy.stats.biweight_midvariance(vals)
    return outputArray



def nearestNmedFilter(inputArray,n=24):
    """
    JvE 2/25/2014
    Same idea as nearestNstdDevFilter, but returns medians instead of std. deviations.
    
    INPUTS:
        inputArray - 2D input array of values.
        n - number of nearest finite neighbours to sample for calculating median around each pixel.
        
    OUTPUTS:
        A 2D array of medians with the same shape as inputArray
    """
    
    outputArray = numpy.zeros_like(inputArray)
    outputArray.fill(numpy.nan)
    nRow,nCol = numpy.shape(inputArray)
    for iRow in numpy.arange(nRow):
        for iCol in numpy.arange(nCol):
            outputArray[iRow,iCol] = numpy.median(inputArray[findNearestFinite(inputArray,iRow,iCol,n=n)])
    return outputArray


def nearestNRobustMeanFilter(inputArray,n=24,nSigmaClip=3.,iters=None):
    """
    Matt 7/18/2014
    Same idea as nearestNstdDevFilter, but returns sigma clipped mean instead of std. deviations.
    
    INPUTS:
        inputArray - 2D input array of values.
        n - number of nearest finite neighbours to sample for calculating median around each pixel.
        
    OUTPUTS:
        A 2D array of medians with the same shape as inputArray
    """
    
    outputArray = numpy.zeros_like(inputArray)
    outputArray.fill(numpy.nan)
    nRow,nCol = numpy.shape(inputArray)
    for iRow in numpy.arange(nRow):
        for iCol in numpy.arange(nCol):
            outputArray[iRow,iCol] = numpy.ma.mean(astropy.stats.sigma_clip(inputArray[findNearestFinite(inputArray,iRow,iCol,n=n)],sig=nSigmaClip,iters=None))
    return outputArray


def interpolateImage(inputArray, method='linear'):
    """
    Seth 11/13/14
    2D interpolation to smooth over missing pixels using built-in scipy methods

    INPUTS:
        inputArray - 2D input array of values
        method - method of interpolation. Options are scipy.interpolate.griddata methods:
                 'linear' (default), 'cubic', or 'nearest'

    OUTPUTS:
        the interpolated image with same shape as input array
    """

    finalshape = numpy.shape(inputArray)

    dataPoints = numpy.where(numpy.logical_or(numpy.isnan(inputArray),inputArray==0)==False) #data points for interp are only pixels with counts
    data = inputArray[dataPoints]
    dataPoints = numpy.array((dataPoints[0],dataPoints[1]),dtype=numpy.int).transpose() #griddata expects them in this order
    
    interpPoints = numpy.where(inputArray!=numpy.nan) #should include all points as interpolation points
    interpPoints = numpy.array((interpPoints[0],interpPoints[1]),dtype=numpy.int).transpose()

    interpolatedFrame = griddata(dataPoints, data, interpPoints, method)
    interpolatedFrame = numpy.reshape(interpolatedFrame, finalshape) #reshape interpolated frame into original shape
    
    return interpolatedFrame


def showzcoord():
    """
    For arrays displayed using 'matshow', hack to set the cursor location
    display to include the value under the cursor, for the currently
    selected axes.
    
    NB - watch out if you have several windows open - sometimes it
    will show values in one window from another window. Avoid this by
    making sure you've clicked *within* the plot itself to make it
    the current active axis set (or hold down mouse button while scanning
    values). That should reset the values to the current window.
    
    JvE 5/28/2014
    
    """
    
    def format_coord(x,y):
        try:
            im = plt.gca().get_images()[0].get_array().data
            nrow,ncol = numpy.shape(im)
            row,col = int(y+0.5),int(x+0.5)
            z = im[row,col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
        except:
            return 'x=%1.4f, y=%1.4f, --'%(x, y)
    
    ax = plt.gca()
    ax.format_coord = format_coord


def fitBlackbody(wvls,flux,fraction=1.0,newWvls=None,tempGuess=6000):
    """
    Seth 11/13/14
    Simple blackbody fitting function that returns BB temperature and fluxes if requested.

    INPUTS:
        wvls - wavelengths of data points (in Angstroms!)
        flux - fluxes of data points (in ergs/s/cm^2/Angstrom!)
        fraction - what fraction of spectrum's red end to use for fit. Default is 1.0 (use whole spectrum).
                   for example, fraction=1.0/5.0 only fits BB to last 20% of spectrum.
        newWvls - 1D array of wavelengths in angstroms. If given, function returns blackbody fit at requested points
        tempGuess - manually adjust guess of BB temperature (in Kelvin) that fit starts with

    OUTPUTS:
        T - temperature in Kelvin of blackbody fit
        newFlux - fluxes calculated at newWvls using the BB equation generated by the fit
        
    """
    c=3.00E10 #cm/s
    h=6.626E-27 #erg*s
    k=1.3806488E-16 #erg/K
    
    x=wvls
    norm = flux.max()
    y= flux/norm

    print("BBfit using last ", fraction*100, "% of spectrum only")
    fitx = x[(1.0-fraction)*len(x)::]
    fity = y[(1.0-fraction)*len(x)::]

    guess_a, guess_b = 1/(2*h*c**2/1e-9), tempGuess #Constant, Temp
    guess = [guess_a, guess_b]

    blackbody = lambda fx, N, T: N * 2*h*c**2 / (fx)**5 * (numpy.exp(h*c/(k*T*(fx))) - 1)**-1 # Planck Law
    #blackbody = lambda fx, N, T: N*2*c*k*T/(fx)**4 #Rayleigh Jeans tail
    #blackbody = lambda fx, N, T: N*2*h*c**2/(fx**5) * exp(-h*c/(k*T*fx)) #Wein Approx

    params, cov = curve_fit(blackbody, fitx*1.0e-8, fity, p0=guess, maxfev=2000)
    N, T= params
    print("BBFit:\nN = %s\nT = %s\n"%(N, T))

    if newWvls !=None:
        best_fit = lambda fx: N * 2*h*c**2 / (fx)**5 * (numpy.exp(h*c/(k*T*(fx))) - 1)**-1 #Planck Law
        #best_fit = lambda fx: N*2*c*k*T/(fx)**4 # Rayleigh Jeans Tail
        #best_fit = lambda fx: N*2*h*c**2/(fx**5) * exp(-h*c/(k*T*fx)) #Wein Approx

        calcx = numpy.array(newWvls,dtype=float)
        newFlux = best_fit(calcx*1.0E-8)
        newFlux*=norm
        return T, newFlux
    else:
        return T


def rebin(x,y,binedges):
    """
    Seth Meeker 1-29-2013
    Given arrays of wavelengths and fluxes (x and y) rebins to specified bin size by taking average value of input data within each bin
    use: rebinnedData = rebin(x,y,binedges)
    binedges typically can be imported from a FlatCal after being applied in an ObsFile
    returns rebinned data as 2D array:
        rebinned[:,0] = centers of wvl bins
        rebinned[:,1] = average of y-values per new bins
    """
    #must be passed binedges array since spectra will not be binned with evenly sized bins
    start = binedges[0]
    stop = x[-1]
    #calculate how many new bins we will have
    nbins = len(binedges)-1
    #create output arrays
    rebinned = numpy.zeros((nbins,2),dtype = float)
    for i in range(nbins):
        rebinned[i,0] = binedges[i]+(binedges[i+1]-binedges[i])/2.0
    n=0
    binsize=binedges[n+1]-binedges[n]
    while start+(binsize/2.0)<stop:
        #print start
        #print binsize
        #print stop
        #print n
        rebinned[n,0] = (start+(binsize/2.0))
        ind = numpy.where((x>start) & (x<start+binsize))
        rebinned[n,1] = numpy.mean(y[ind])
        start += binsize
        n+=1
        try:
            binsize=binedges[n+1]-binedges[n]
        except IndexError:
            break
    return rebinned

def gaussianConvolution(x,y,xEnMin=0.005,xEnMax=6.0,xdE=0.001,fluxUnits = "lambda",r=8, plots=False):
    """
    Seth 2-16-2015
    Given arrays of wavelengths and fluxes (x and y) convolves with gaussian of a given energy resolution (r)
    Input spectrum is converted into F_nu units, where a Gaussian of a given R has a constant width unlike in
    wavelength space, and regridded to an even energy gridding defined by xEnMin, xEnMax, and xdE
    INPUTS:
        x - wavelengths of data points in Angstroms or Hz
        y - fluxes of data points in F_lambda (ergs/s/cm^2/Angs) or F_nu (ergs/s/cm^2/Hz)
        xEnMin - minimum value of evenly spaced energy grid that spectrum will be interpolated on to
        xEnMax - maximum value of evenly spaced energy grid that spectrum will be interpolated on to
        xdE - energy spacing of evenly spaced energy grid that spectrum will be interpolated on to
        fluxUnits - "lambda" for default F_lambda, x must be in Angstroms. "nu" for F_nu, x must be in Hz.
        r - energy resolution of gaussian to be convolved with. R=8 at 405nm by default.
    OUTPUTS:
        xOut - new x-values that convolution is calculated at (defined by xEnMin, xEnMax, xdE), returned in same units as original x
        yOut - fluxes calculated at new x-values are returned in same units as original y provided
    """
    ##=======================  Define some Constants     ============================
    c=3.00E10 #cm/s
    h=6.626E-27 #erg*s
    k=1.3806488E-16 #erg/K
    heV = 4.13566751E-15
    ##================  Convert to F_nu and put x-axis in frequency  ===================
    if fluxUnits == 'lambda':
        xEn = heV*(c*1.0E8)/x
        xNu = xEn/heV
        yNu = y * x**2 * 3.34E4 #convert Flambda to Fnu(Jy)
    elif fluxUnits =='nu':
        xNu = x
        xEn = xNu*heV
        yNu = y
    else:
        raise ValueError("fluxUnits must be either 'nu' or 'lambda'")
    ##============  regrid to a constant energy spacing for convolution  ===============
    xNuGrid = numpy.arange(xEnMin,xEnMax,xdE)/heV #make new x-axis gridding in constant freq bins
    yNuGrid = griddata(xNu,yNu,xNuGrid, 'linear', fill_value=0)
    if plots==True:
        plt.plot(xNuGrid,yNuGrid,label="Spectrum in energy space")
    ##====== Integrate curve to get total flux, required to ensure flux conservation later =======
    originalTotalFlux = scipy.integrate.simps(yNuGrid,x=xNuGrid)
    ##======  define gaussian for convolution, on same gridding as spectral data  ======
    #WARNING: right now flux is NOT conserved
    amp = 1.0
    offset = 0
    E0=heV*c/(450*1E-7) # 450rnm light is ~3eV
    dE = E0/r
    sig = dE/heV/2.355 #define sigma as FWHM converted to frequency
    gaussX = numpy.arange(-2,2,0.001)/heV
    gaussY = amp * numpy.exp(-1.0*(gaussX-offset)**2/(2.0*(sig**2)))
    if plots==True:
        plt.plot(gaussX, gaussY*yNuGrid.max(),label="Gaussian to be convolved")
        plt.legend()
        plt.show()
    ##================================    convolve    ==================================
    convY = numpy.convolve(yNuGrid,gaussY,'same')
    if plots==True:
        plt.plot(xNuGrid,convY,label="Convolved spectrum")
        plt.legend()
        plt.show()
    ##============ Conserve Flux ==============
    newTotalFlux = scipy.integrate.simps(convY,x=xNuGrid)
    convY*=(originalTotalFlux/newTotalFlux)
    ##==================   Convert back to wavelength space   ==========================
    if fluxUnits=='lambda':
        xOut = c/xNuGrid*1E8
        yOut = convY/(xOut**2)*3E-5 #convert Fnu(Jy) to Flambda
    else:
        xOut = xNuGrid
        yOut = convY
    if plots==True:
        plt.plot(xOut[xOut<25000], yOut[xOut<25000],label="Convolved Spectrum")
        plt.plot(x,y,label="Original spectrum")
        plt.legend()
        plt.ylabel('F_%s'%fluxUnits)
        plt.show()

    return [xOut,yOut]


def countsToApparentMag(cps, filterName = 'V', telescope = None):
    """
    routine to convert counts measured in a given filter to an apparent magnitude
    input: cps = counts/s to be converted to magnitude. Can accept np array of numbers.
           filterName = filter to have magnitude calculated for
           telescope = name of telescope used. This is necessary to determine the collecting
                       area since the counts need to be in units of counts/s/m^2 for conversion to mags
                       If telescope = None: assumes data was already given in correct units
    output: apparent magnitude. Returns same format as input (either single value or np array)
    """
    Jansky2Counts = 1.51E7
    dLambdaOverLambda = {'U':0.15,'B':0.22,'V':0.16,'R':0.23,'I':0.19,'g':0.14,'r':0.14,'i':0.16,'z':0.13}
    f0 = {'U':1810.,'B':4260.,'V':3640.,'R':3080.,'I':2550.,'g':3730.,'r':4490.,'i':4760.,'z':4810.}

    if filterName not in list(f0.keys()):
        raise ValueError("Not a valid filter. Please select from 'U','B','V','R','I','g','r','i','z'")    

    if telescope in ['Palomar','PAL','palomar','pal','Hale','hale']:
        telArea = 17.8421 #m^2 for Hale 200" primary
    elif telescope in ['Lick','LICK','Shane']:
        raise ValueError("LICK NOT IMPLEMENTED")
    elif telescope == None:
        print("WARNING: no telescope provided for conversion to apparent mag. Assuming data is in units of counts/s/m^2")
        telArea=1.0
    else:
        raise ValueError("No suitable argument provided for telescope name. Use None if data in counts/s/m^2 already.")

    cpsPerArea=cps/telArea
    mag = -2.5*numpy.log10(cpsPerArea/(f0[filterName]*Jansky2Counts*dLambdaOverLambda[filterName]))
    return mag
    
def medianStack(stack):
    return numpy.nanmedian(stack, axis=0)

