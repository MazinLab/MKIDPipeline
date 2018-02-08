'''
SRM - Migrated to DARKNESS-pipeline unchanged 2017-04-09

Author: Julian van Eyken    Date: Apr 30 2013
Definitions of all data flags used by the pipeline.
Currently dictionaries to map flag descriptions to integer values.
May update to use pytables Enums at some point down the road....
'''

#Flat cal. flags:
flatCal = {
           'good':0,                #No flagging.
           'infWeight':1,           #Spurious infinite weight was calculated - weight set to 1.0
           'zeroWeight':2,          #Spurious zero weight was calculated - weight set to 1.0
           'belowWaveCalRange':10,  #Derived wavelength is below formal validity range of calibration
           'aboveWaveCalRange':11,  #Derived wavelength is above formal validity range of calibration
           'undefined':20,          #Flagged, but reason is undefined.
           'undetermined':99,       #Flag status is undetermined.
           }

#Flux cal. flags
fluxCal = {
           'good':0,                #No flagging.
           'infWeight':1,           #Spurious infinite weight was calculated - weight set to 1.0
           'LEzeroWeight':2,        #Spurious less-than-or-equal-to-zero weight was calculated - weight set to 1.0
           'nanWeight':3,           #NaN weight was calculated.
           'belowWaveCalRange':10,  #Derived wavelength is below formal validity range of calibration
           'aboveWaveCalRange':11,  #Derived wavelength is above formal validity range of calibration
           'undefined':20,          #Flagged, but reason is undefined.
           'undetermined':99        #Flag status is undetermined.
           }

#Wavelength calibration flags
waveCal = {0: "histogram fit - converged and validated",
           1: "histogram not fit - did not converge",
           2: "histogram not fit - converged but failed validation",
           3: "histogram not fit - not enough data",
           4: "energy fit - quadratic function",
           5: "energy fit - linear function",
           6: "energy not fit - not enough data points",
           7: "energy not fit - data not monotonic enough",
           8: "energy not fit - linear and quadratic fits failed",
           9: "energy fit - linear function through zero"}

#Bad pixel calibration flags (including hot pixels, cold pixels, etc.)
badPixCal = {
             'good':0,              #No flagging.
             'hot':1,               #Hot pixel
             'cold':2,              #Cold pixel
             'dead':3,              #Dead pixel
             'undefined':20,        #Flagged, but reason is undefined.
             'undetermined':99      #Flag status is undetermined.
             }

#Beammap flags (stored in beammap file)
beamMapFlags = {
                'good':0,           #No flagging
                'failed':1,         #Beammap failed to place pixel
                'yFailed':2,        #Beammap succeeded in x, failed in y
                'xFailed':3,        #Beammap succeeded in y, failed in x
                'wrongFeedline':4   #Beammap placed pixel in wrong feedline
                }

#Flags stored in HDF5 file. Works as a bitmask to allow for multiple flags
h5FileFlags = {
               'good':0b00000000,               #No flagging
               'beamMapFailed':0b00000001,      #Bad beammap
               'waveCalFailed':0b00000010       #No wavecal solution
               }
