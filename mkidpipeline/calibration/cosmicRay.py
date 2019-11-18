import numpy as np
import time
import matplotlib.pyplot as plt
from mkidpipeline.hdf import photontable as pt
from scipy.stats import poisson
import math
import scipy
from scipy import signal

def cosmicCorrection(file, binsize = 10, removalRange = 10):
    '''
     function finds and deletes cosmic ray suspects. Assigns each arrival time to a time bin, 
     then returns number of counts for each time bin. Calculates average and threshold 
     for cosmic ray suspects. Assigns time bins with high counts to a dict (weirdones) with time as key and count 
     as value. Deletes time bins before and time bins after the suspected high count.

       Parameters
        ----------
        file: string or int
            H5 File name
        binsize: int
            size of time bins. Default is 10 microseconds.
        removalRange: int
            time bins to be deleted around cosmic ray occurrence. Default is 10 time bins.

       Returns
        -------
        Dictionary with keys:
            'returnDict': dictionary with keys as counts and values as # of bins with that count. Used in histogram.
            'peaks': list of times where peak of cosmic ray occurred
            'cutOut': list of all deleted times, including peaks and surrounding areas
            'goodones': list of times with no cosmic ray suspects      

    '''
    file = pt.ObsFile(str(file))  #grab data

    start_time = time.time()  # def bin size in microseconds
    photons = file.photonTable.read()  # grabs list for all photon arrivals (read)


    hist, bins = np.histogram(photons['Time'], bins=int((photons['Time'].max()+1)/binsize))  #returns tuple. hist -> counts/bin and bins -> bin
    unique, counts = np.unique(hist, return_counts=True)    
    returnDict = dict(zip(unique, counts))

    avgcounts = np.average(list(unique), weights=list(counts))
  
    threshold = np.ceil(6*poisson.std(avgcounts, loc=0) + avgcounts)

    localmax = scipy.signal.find_peaks(hist, height=threshold, threshold=10, distance=30)
    
    cutOut=[]
    for i in localmax[0]:
        for k in range(i-removalRange, i+removalRange +1):
            cutOut.append(k)

    hist = np.delete(hist, cutOut)
    bins = np.delete(bins, cutOut)

    print("--- %s seconds ---" % (time.time() - start_time))
    unique, counts = np.unique(hist, return_counts=True) 
    goodOnes = bins
    returnDict = dict(zip(unique, counts))
    peaks = localmax[0]
    return {'returnDict': returnDict, 'peaks': peaks,'cutOut': cutOut, 'goodOnes': goodOnes}
    

def cosmicFlag(file, binsize, startTime, endTime = -1):

    '''
     function returns Boolean for time or time range in file (time is given in microseconds). 
     This function assumes you have not yet run cosmicCorrection on file, so it will run it for you.

       Parameters
        ----------
        file: string or int
            H5 File name
        binsize: int
            size of time bins
        startTime: int
            start time to be analyzed.
        endTime: int
            end of time range. If not specified, only startTime will be analyzed 
            (will look at single time, not range).

       Returns
        -------
       True if there is cosmic ray suspect in specified time/range.
       False if there is no cosmic ray suspect in specified time/range.
    '''

    flagged = cosmicCorrection(file, binsize)

    if endTime == -1:
        return (startTime in flagged['cutOut'])
    else:
        return np.any(np.in1d(np.arange(startTime, endTime+1), flagged['cutOut']))



def fastCosmicFlag(cutOut, startTime, endTime = -1):

    '''
     function returns Boolean for time or time range in file (time is given in microseconds). 
     This function assumes you have already run cosmicCorrection on file, so you just have to provide it with
     cutOut times given by cosmicCorrection - must save cosmicCorrection's result to a variable name then use
     it to get cutOut (i.e. saved_name['cutOut'])

       Parameters
        ----------
        cutOut: array returned in cosmicCorrection's dictionary (input saved_name['cutOut'])
            array of cutOut times given by cosmicCorrection function.
        startTime: int
            start time to be analyzed.
        endTime: int
            end of time range. If not specified, only startTime will be analyzed 
            (will look at single time, not range).

       Returns
        -------
       True if there is cosmic ray suspect in specified time/range.
       False if there is no cosmic ray suspect in specified time/range.
    '''
    if endTime == -1:
        return (startTime in cutOut)
    else:
        return np.any(np.in1d(np.arange(startTime, endTime+1), cutOut))




def plotCosmicHist(dictUW):

    '''
     function returns plotted histogram of counts fitted into a Poisson distribution. 

       Parameters
        ----------
        dictUW: dictionary returned in cosmicCorrection's dictionary (input saved_name['returnDict'])
            dictionary with counts and number of time bins with such count
    '''
    start_time = time.time()

    unique = dictUW.keys()
    counts = dictUW.values()

    plt.bar(unique, np.divide(list(counts), sum(counts)), label="Real distribution")   #plot hist in prob distributions

    avgcounts = np.average(list(unique), weights = list(counts))    #calculate avg
   
    print("--- %s seconds ---" % (time.time() - start_time))
    X = np.arange(0, max(dictUW)-1) #fit curve to range of graph
    plt.plot( X, poisson.pmf(X,avgcounts), 'r-', label='Poisson Fit')        #fit Poisson w lambda automatically calculated
    plt.plot()    #plots histogram!
    print()
    plt.xlabel('Counts per 10 microseconds')
    plt.ylabel('Probability of Counts')
    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))



