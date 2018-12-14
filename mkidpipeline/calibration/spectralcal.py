"""
Author: Isabel Lipartito        Date:Dec 11, 2018

Opens MKID observation of a spectrophotometric standard star and associated wavelength cal file,
reads in all photons and converts to energies.
Bins photons to generate a spectrum, then divides this into the known spectrum (published)
of the object to create a sensitivity curve.
This curve is then written out to the obs file as spectral weights

Flags are associated with each pixel - see mkidcore/pixelFlags.py for descriptions.
"""

import sys,os
import tables
import numpy as np
from mkidcore import pixelflags
from configparser import ConfigParser

from mkidpipeline.utils.utils import rebin, gaussianConvolution, fitBlackbody
from mkidpipeline.utils import MKIDStd
from mkidcore.corelog import getLogger
import mkidcore.corelog

class SpectralCal(object):
    """
    Opens flux file, prepares standard spectrum, and calculates flux factors for the file.
    Method is provided in param file. If 'relative' is selected, an obs file with standard star defocused over
    the entire array is expected, with accompanying sky file to do sky subtraction.
    If any other method is provided, 'absolute' will be done by default, wherein a point source is assumed
    to be present. The obs file is then broken into spectral frames with photometry (psf or aper) performed
    on each frame to generate the observed spectrum.
    """
    def __init__(self, config_file='default.cfg', cal_file_name = 'calsol_default.h5'):
        """
        Reads in the param file.
        """
        self.config_file = DEFAULT_CONFIG_FILE if config_file == 'default.cfg' else config_file
        assert os.path.isfile(self.config_file), \
            self.config_file + " is not a valid configuration file"
        # check the configuration file path and read it in
        self.config = ConfigParser()
        self.config.read(self.config_file)
        # check the configuration file format and load the parameters
        self.checksections()
        self.wvlcal_file = ast.literal_eval(self.config['Data']['wvlcal_file'])
        self.flatcal_file = ast.literal_eval(self.config['Data']['flatcal_file'])
        self.file_name = ast.literal_eval(self.config['Data']['file_name'])
        self.intTime = ast.literal_eval(self.config['Data']['intTime'])
        self.expTime = ast.literal_eval(self.config['Data']['expTime'])
        self.dataDir = ast.literal_eval(self.config['Data']['dataDir'])
        self.beamDir = ast.literal_eval(self.config['Data']['beamDir'])
        self.startTime = ast.literal_eval(self.config['Data']['startTime'])
        self.xpix = ast.literal_eval(self.config['Data']['xpix'])
        self.ypix = ast.literal_eval(self.config['Data']['ypix'])
        self.deadtime = ast.literal_eval(self.config['Instrument']['deadtime'])
        self.energyBinWidth = ast.literal_eval(self.config['Instrument']['energyBinWidth'])
        self.wvlStart = ast.literal_eval(self.config['Instrument']['wvlStart'])
        self.wvlStop = ast.literal_eval(self.config['Instrument']['wvlStop'])
        self.countRateCutoff = ast.literal_eval(self.config['Calibration']['countRateCutoff'])
        self.fractionOfChunksToTrim = ast.literal_eval(self.config['Calibration']['fractionOfChunksToTrim'])
        self.logging = ast.literal_eval(self.config['Output']['logging'])
        self.out_directory = ast.literal_eval(self.config['Output']['out_directory'])
        self.save_plots = ast.literal_eval(self.config['Output']['save_plots'])
        self.summary_plot = ast.literal_eval(self.config['Output']['summary_plot'])
        self.cal_file_name=cal_file_name
        self.timeSpacingCut = None


        run = self.params['run']
        sunsetDate = self.params['fluxSunsetLocalDate']
        self.fluxTstamp = self.params['fluxTimestamp']
        skyTstamp = self.params['skyTimestamp']
        wvlSunsetDate = self.params['wvlCalSunsetLocalDate']
        wvlTimestamp = self.params['wvlCalTimestamp']
        flatCalFileName = self.params['flatCalFileName']
        needTimeAdjust = self.params['needTimeAdjust']
        self.deadtime = float(self.params['deadtime']) #from firmware pulse detection
        self.timeSpacingCut = self.params['timeSpacingCut']
        bLoadBeammap = self.params.get('bLoadBeammap',False)
        self.method = self.params['method']
        self.objectName = self.params['object']
        self.r = float(self.params['energyResolution'])
        self.photometry = self.params['photometry']
        self.centroidRow = self.params['centroidRow']
        self.centroidCol = self.params['centroidCol']
        self.aperture = self.params['apertureRad']
        self.annulusInner = self.params['annulusInner']
        self.annulusOuter = self.params['annulusOuter']
        self.collectingArea = self.params['collectingArea']
        self.startTime = self.params['startTime']
        self.intTime = self.params['integrationTime']









        self.checktypes()
        self.checksections()
        self.checktypes()
        getLogger(__name__).info("Computing Factors for FlatCal")

    def load_absolutespectrum(self):
        '''
         Extract the ARCONS measured spectrum of the spectrophotometric standard by breaking data into spectral cube
         and performing photometry (aperture or psf) on each spectral frame
         '''
        cube_dict = self.fluxFile.getSpectralCube(firstSec=self.startTime, integrationTime=self.intTime, weighted=True,
                                                 fluxWeighted=False)
        cube = np.array(cube_dict['cube'], dtype=np.double)
        effIntTime = cube_dict['effIntTime']

        # add third dimension to effIntTime for broadcasting
        effIntTime = np.reshape(effIntTime, np.shape(effIntTime) + (1,))
        # put cube into counts/s in each pixel
        cube /= effIntTime
        # get dead time correction factors
        deadtime_corr = self.get_deadtimecorrection(self.flux_file)
        cube *= deadtime_corr  # cube now in units of counts/s and corrected for dead time

        light_curve = lightcurve.lightcurve()

        self.flux_spectrum = np.empty((self.nWvlBins), dtype=float)
        self.sky_spectrum = np.zeros((self.nWvlBins), dtype=float)

        for i in np.arange(self.nWvlBins):
            frame = cube[:, :, i]
            if self.photometry == 'aperture':
                flux_dict = light_curve.perform_photometry(self.photometry, frame, [[self.centroid_col, self.centroid_row]],
                                             expTime=None, aper_radius=self.aperture, annulus_inner=self.annulus_inner,
                                             annulus_outer=self.annulus_outer, interpolation="linear")
                self.flux_spectrum[i] = flux_dict['flux']
                self.sky_spectrum[i] = flux_dict['sky_flux']
            else:
                flux_dict = light_curve.perform_photometry(self.photometry, frame, [[self.centroid_col, self.centroid_row]],
                                             expTime=None, aper_radius=self.aperture)
                self.flux_spectrum[i] = flux_dict['flux']

        self.flux_spectrum = self.flux_spectrum / self.bin_widths / self.collecting_area  # spectrum now in counts/s/Angs/cm^2
        self.sky_spectrum = self.sky_spectrum / self.bin_widths / self.collecting_area

        return self.flux_spectrum, self.sky_spectrum

    def load_relativespectrum():
        self.flux_spectra = [[[] for i in np.arange(self.nCol)] for j in np.arange(self.nRow)]
        self.flux_effTime = [[[] for i in np.arange(self.nCol)] for j in np.arange(self.nRow)]
        for iRow in np.arange(self.nRow):
            for iCol in np.arange(self.nCol):
                count = self.flux_file.getPixelCount(iRow, iCol)
                flux_dict = self.flux_file.getPixelSpectrum(iRow, iCol, weighted=True, firstSec=0, integrationTime=-1)
                self.flux_spectra[iRow][iCol], self.flux_effTime[iRow][iCol] = flux_dict['spectrum'], flux_dict['effIntTime']
        self.flux_spectra = np.array(self.flux_spectra)
        self.flux_effTime = np.array(self.flux_effTime)
        deadtime_corr = self.get_deadtimecorrection(self.flux_file)
        self.flux_spectra = self.flux_spectra / self.bin_widths / self.flux_effTime * deadtime_corr
        self.flux_spectrum = self.calculate_median(self.flux_spectra)  # find median of subtracted spectra across whole array
        return self.flux_spectrum

    def load_skyspectrum():
        self.sky_spectra = [[[] for i in np.arange(self.nCol)] for j in np.arange(self.nRow)]
        self.sky_effTime = [[[] for i in np.arange(self.nCol)] for j in np.arange(self.nRow)]
        for iRow in np.arange(self.nRow):
            for iCol in np.arange(self.nCol):
                count = self.sky_file.getPixelCount(iRow, iCol)
                sky_dict = self.sky_file.getPixelSpectrum(iRow, iCol, weighted=True, firstSec=0, integrationTime=-1)
                self.sky_spectra[iRow][iCol], self.sky_effTime[iRow][iCol] = sky_dict['spectrum'], sky_dict['effIntTime']
        self.sky_spectra = np.array(self.sky_spectra)
        self.sky_effTime = np.array(self.sky_effTime)
        deadtime_corr = self.get_deadtimecorrection(self.sky_file)
        self.sky_spectra = self.sky_spectra / self.binWidths / self.sky_effTime * deadtime_corr
        self.sky_spectrum = self.calculate_median(self.sky_spectra)  # find median of subtracted spectra across whole array
        return self.sky_spectrum

    def load_stdspectrum(self, object_name="G158-100"):
        '''
        :param self:
        :param object_name:
        :return:

        import the known spectrum of the calibrator and rebin to the histogram parameters given
        '''
        standard = MKIDStd.MKIDStd()
        star_data = standard.load(object_name)
        star_data = standard.countsToErgs(star_data)  # convert standard star spectrum to ergs/s/Angs/cm^2 for BB fitting and cleaning
        self.std_wvls = np.array(star_data[:, 0])
        self.std_flux = np.array(star_data[:, 1])  # standard star object spectrum in ergs/s/Angs/cm^2

        convX_rev, convY_rev = self.clean_spectrum(self.std_wvls, self.std_flux)
        convX = convX_rev[::-1]  # convolved spectrum comes back sorted backwards, from long wvls to low which screws up rebinning
        convY = convY_rev[::-1]
        # rebin cleaned spectrum to flat cal's wvlBinEdges
        rebin_star_data = rebin(convX, convY, self.wvlBinEdges)
        rebin_std_wvls = np.array(rebin_star_data[:, 0])
        rebin_std_flux = np.array(rebin_star_data[:, 1])

        # convert standard spectrum back into counts/s/angstrom/cm^2
        rebin_star_data = standard.ergsToCounts(rebin_star_data)
        self.binned_spectrum = np.array(rebin_star_data[:, 1])


    def clean_spectrum(self, x, y):
        '''
        :param self:
        :param x:
        :param y:
        :return:

        BB Fit to extend spectrum beyond 11000 Angstroms
        '''

        fraction = 1.0 / 3.0
        nirX = np.arange(int(x[(1.0 - fraction) * len(x)]), 20000)
        T, nirY = fitBlackbody(x, y, fraction=fraction, newWvls=nirX, tempGuess=5600)

        extended_wvl = np.concatenate((x, nirX[nirX > max(x)]))
        extended_flux = np.concatenate((y, nirY[nirX > max(x)]))

        # Gaussian convolution to smooth std spectrum to MKIDs median resolution
        newX, newY = gaussianConvolution(extended_wvl, extended_flux, xEnMin=0.005, xEnMax=6.0, xdE=0.001,
                                             fluxUnits="lambda", r=self.r, plots=False)
        return newX, newY


    def calculate_specweights():
        """
        Calculate the sensitivity spectrum: the weighting factors that correct the flat calibrated spectra to the real spectra

        For relative calibration:
        First subtract sky spectrum from MKID observed spectrum. Then take median of this spectrum as it should be identical
        across the array, assuming the flat cal has done its job. Then divide this into the known spectrum of the object.

        For absolute calibration:
        self.flux_spectra already has sky subtraction included. Simply divide this spectrum into the known standard spectrum.
        """
        self.subtracted_spectrum = self.flux_spectrum - self.sky_spectrum
        self.subtracted_spectrum = np.array(self.subtracted_spectrum,
                                           dtype=float)  # cast as floats so division does not fail later

        if self.method == 'relative':
            norm_wvl = 5500  # Angstroms. Choose an arbitrary wvl to normalize the relative correction at
            ind = np.where(self.wvlBinEdges >= norm_wvl)[0][0] - 1
            self.subtracted_spectrum = self.subtracted_spectrum / (self.subtracted_spectrum[ind])  # normalize
            # normalize treated Std spectrum while we are at it
            self.binned_spectrum = self.binned_spectrum / (self.binned_spectrum[ind])

        # Calculate FluxCal factors
        self.flux_factors = self.binned_spectrum / self.subtracted_spectrum

        self.flux_flags = np.empty(np.shape(self.flux_factors), dtype='int')
        self.flux_flags.fill(pixelflags.speccal['good'])  # Initialise flag array filled with 'good' flags

        self.flux_flags[self.flux_factors == np.inf] = pixelflags.speccal['infWeight']
        self.flux_factors[self.flux_factors == np.inf] = 1.0
        self.flux_flags[np.isnan(self.flux_factors)] = pixelflags.speccal['nanWeight']
        self.flux_factors[np.isnan(self.flux_factors)] = 1.0
        self.flux_flags[self.flux_factors <= 0] = pixelflags.speccal['LEzeroWeight']
        self.flux_factors[self.flux_factors <= 0] = 1.0

    def write_speccal(self, speccal_filename):
        """
        Write flux cal weights to h5 file
        """

        speccal_file= tables.openFile(speccal_filename, mode='w')

        calgroup = fluxCalFile.createGroup(speccal_file.root, 'fluxcal', 'Table of flux calibration weights by wavelength')
        caltable = tables.Array(calgroup, 'weights', object=self.flux_factors,
                                title='Flux calibration Weights indexed by wavelength bin')
        flagtable = tables.Array(calgroup, 'flags', object=self.flux_flags,
                                 title='Flux cal flags indexed by wavelength bin. 0 is Good')
        bintable = tables.Array(calgroup, 'wavelengthBins', object=self.wvlBinEdges,
                                title='Wavelength bin edges corresponding to third dimension of weights array')
        speccal_file.flush()
        speccal_file.close()

    def make_specplots():
        pass

    def get_deadtimecorrection(self.flux_file):
        deadtime_corr=1.0
        return(deadtime_corr)

    def calculate_median(self, spectra):
        spectra2d = np.reshape(spectra,[self.nRow*self.nCol,self.nWvlBins])
        wvl_median = np.empty(self.nWvlBins,dtype=float)
        for iWvl in np.arange(self.nWvlBins):
            spectrum = spectra2d[:,iWvl]
            good_spectrum = spectrum[spectrum != 0]#dead pixels need to be taken out before calculating medians
            wvl_median[iWvl] = np.median(good_spectrum)
        return wvl_median



