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
waveCal = {
           'good':0,                #No flagging.
           'belowWaveCalRange':10,  #Derived wavelength is below formal validity range of calibration
           'aboveWaveCalRange':11,  #Derived wavelength is above formal validity range of calibration
           'noWaveCalSolution':12,  #Due to a bad pixel... or whatever....
           'undefined':20,          #Flagged, but reason is undefined.
           'undetermined':99        #Flag status is undetermined.
           }

#Bad pixel calibration flags (including hot pixels, cold pixels, etc.)
badPixCal = {
             'good':0,              #No flagging.
             'hot':1,               #Hot pixel
             'cold':2,              #Cold pixel
             'dead':3,              #Dead pixel
             'undefined':20,        #Flagged, but reason is undefined.
             'undetermined':99      #Flag status is undetermined.
             }
