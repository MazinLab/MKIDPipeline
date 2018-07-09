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
from DarknessPipeline.Utils.arrayPopup import PopUp,plotArray,pop
from DarknessPipeline.RawDataProcessing.darkObsFile import ObsFile
from DarknessPipeline.Utils.readDict import readDict
from DarknessPipeline.Utils.FileName import FileName
import DarknessPipeline.Cleaning.HotPix.darkHotPixMask as hp
from DarknessPipeline.Headers.CalHeaders import FlatCalSoln_Description
from DarknessPipeline.Headers import pipelineFlags
from DarknessPipeline.Calibration.WavelengthCal import plotWaveCal as p
from matplotlib.backends.backend_pdf import PdfPages


def plotSinglePixelSolution(calsolnName, file_nameWvlCal, res_id=None, pixel=[], save_plot=False):  

	'''
	Plots the weights and twilight spectrum of a single pixel (can be specified through the RES ID or pixel coordinates)
	Plots may be saved to a pdf if save_plot=True.  
	Also plots the energy solution for the pixel from Wavecal

	calsolnName= File path and name of wavecal solution
	res_id= RES ID of pixel (if known)
	pixel= Coordinates of pixel (if known)
	Note:  Either RES ID or pixel coordinates must be specified
	save_plot:  Should a plot be saved?  If FALSE, the plot will be displayed.  If TRUE, the plot will be saved to a pdf in the current working directory
	'''
	

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
	spectrum=calsoln['spectrum'][index]

	weights=np.array(weights)
	weights=weights.flatten()

	spectrum=np.array(spectrum)
	spectrum=spectrum.flatten()

	weightUncertainties=np.array(weightUncertainties)
	weightUncertainties=weightUncertainties.flatten()

	fig = plt.figure(figsize=(10,15),dpi=100)
	ax = fig.add_subplot(3,1,1)
	ax.set_ylim(.5,max(weights))
	ax.plot(wavelengths[0:len(wavelengths)-1],weights,label='weights %d'%index,alpha=.7,color=matplotlib.cm.Paired((1+1.)/1))
	ax.errorbar(wavelengths[0:len(wavelengths)-1],weights,yerr=weightUncertainties,label='weights',color='k')
                
	ax.set_title('Pixel %d,%d'%(row,column))
	ax.set_ylabel('Weight')
	ax.set_xlabel(r'$\lambda$ ($\AA$)')

	ax = fig.add_subplot(3,1,2)
	ax.set_ylim(.5,max(spectrum))
	ax.plot(wavelengths[0:len(wavelengths)-1],spectrum,label='Twilight Spectrum %d'%index,alpha=.7,color=matplotlib.cm.Paired((1+1.)/1))

	ax.set_ylabel('Twilight Spectrum')
	ax.set_xlabel(r'$\lambda$ ($\AA$)')

	ax = fig.add_subplot(3,1,3)
	ax.set_ylim(.5,2.)
	my_pixel = [row, column]
	ax=p.plotEnergySolution(file_nameWvlCal, pixel=my_pixel,axis=ax)

	if not save_plot:
		plt.show()
	else:
		pdf = PdfPages(os.path.join(os.getcwd(), str(res_id)+'.pdf'))
		pdf.savefig(fig)
		pdf.close()

