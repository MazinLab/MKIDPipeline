import sys,os
import ast
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
import matplotlib
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
from configparser import ConfigParser
import tables
from DarknessPipeline.P3Utils.arrayPopup import PopUp,plotArray,pop
from DarknessPipeline.RawDataProcessing.darkObsFile import ObsFile
from DarknessPipeline.P3Utils.readDict import readDict
from DarknessPipeline.P3Utils.FileName import FileName
import DarknessPipeline.Cleaning.HotPix.darkHotPixMask as hp
from DarknessPipeline.Headers.CalHeaders import FlatCalSoln_Description
from DarknessPipeline.Headers import pipelineFlags
from DarknessPipeline.Calibration.WavelengthCal import plotWaveCal as p


def plotSinglePixelSolution(calsolnName, file_nameWvlCal, res_id=None, pixel=[], axis=None):  

	assert os.path.exists(calsolnName), "{0} does not exist".format(calsolnName)  
	flat_cal = tables.open_file(calsolnName, mode='r')
	calsoln = flat_cal.root.flatcal.calsoln.read()
	beamImage = flat_cal.root.header.beamMap.read()
	wavelengths= flat_cal.root.flatcal.wavelengthBins.read()

	if len(pixel) != 2 and res_id is None:
		flat_cal.close()
		raise ValueError('please supply resonator location or res_id')
	if len(pixel) == 2 and res_id is None:
                  row = pixel[0]
                  column = pixel[1]
                  res_id = beamImage[row][column]
                  index = np.where(res_id == np.array(calsoln['resid']))
	elif res_id is not None:
		index = np.where(res_id == np.array(calsoln['resid']))
		if len(index[0]) != 1:
			flat_cal.close()
			raise ValueError("res_id must exist and be unique")
		row = calsoln['pixel_row'][index][0]
		column = calsoln['pixel_col'][index][0]

	weights = calsoln['weights'][index]
	weightFlags=calsoln['weightFlags'][index]
	weightUncertainties=calsoln['weightUncertainties'][index]

	weights=np.array(weights)
	weights=weights.flatten()

	weightUncertainties=np.array(weightUncertainties)
	weightUncertainties=weightUncertainties.flatten()

	fig = plt.figure(figsize=(10,15),dpi=100)
	ax = fig.add_subplot(3,1,1)
	ax.set_ylim(.5,2.)
	ax.plot(wavelengths[0:len(wavelengths)-1],weights,label='weights %d'%index,alpha=.7,color=matplotlib.cm.Paired((1+1.)/1))
	ax.errorbar(wavelengths[0:len(wavelengths)-1],weights,yerr=weightUncertainties,label='weights',color='k')
                
	ax.set_title('p %d,%d'%(row,column))
	ax.set_ylabel('weight')
	ax.set_xlabel(r'$\lambda$ ($\AA$)')

	ax = fig.add_subplot(3,1,2)
	ax.set_ylim(.5,2.)
	my_pixel = [row, column]
	ax=p.plotEnergySolution(file_nameWvlCal, pixel=my_pixel,axis=ax)
	plt.show()

if __name__ == '__main__':
	calsolnName='flatcalsoln1.h5'
	file_nameWvlCal = '/mnt/data0/isabel/FlatConfiguration/wavecal/calsol_1528870743.h5' 
	plotSinglePixelSolution(calsolnName, file_nameWvlCal, res_id=None, pixel=[55,55], axis=None)
