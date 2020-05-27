"""
Author: Noah Swimmer 21 November 2019
TODO: Comment everything stop being lazy.
TODO: Convert print statements to logging
TODO: Make WaveCalComparer part of main and runnable from command line
"""
from mkidpipeline.calibration.wavecal import Solution
from mkidpipeline.hdf.photontable import Photontable
import numpy as np
import glob
import time
from datetime import datetime
import argparse
import scipy
from scipy import signal
from scipy.stats import poisson
from multiprocessing import Pool
import matplotlib.pyplot as plt

BF_LASER_WAVELENGTHS = [808, 920, 980, 1120, 1310]
PLANCK_CONST = 4.136e-15  # eV s
SPEED_OF_LIGHT = 2.998e17  # nm/s

DATA_COLUMN_NAMES = ['ResID', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin1cosmics', 'bin2cosmics', 'bin3cosmics', 'bin4cosmics', 'bin5cosmics', 'duration']


class WaveCalComparer(object):
    def __init__(self, waveCalPath1, waveCalPath2, nLasers=5):
        if waveCalPath1 is not None and waveCalPath2 is not None:
            self.solution1 = Solution(waveCalPath1)
            self.solution2 = Solution(waveCalPath2)

        self.nLasers = nLasers
        self._rvals1 = None
        self._rvals2 = None
        self.resIDs = None
        self.full_resIDs = None
        self._full_rvals1 = None
        self._full_rvals2 = None
        self._cleaningHasBeenRun = False

        if self.solution1 is not None and self.solution2 is not None:
            self.get_stable_res_ids()

    def get_stable_res_ids(self):
        '''
        Returns a list of resIDs which have wavecal solutions for all lasers used in the calibration. If there is not
        two wavecal solutions loaded in, function will fail, since there is nothing to compare. Also creates two arrays
        with the resolving powers of each resonator from each wavecal
        '''
        try:
            if self.resIDs is None:
                rvals1, resids1 = self.solution1.find_resolving_powers()
                rvals2, resids2 = self.solution2.find_resolving_powers()

                goodResids1 = [resids1[i] for i, j in enumerate(rvals1) if np.isfinite(j).all()]
                goodResids2 = [resids2[i] for i, j in enumerate(rvals2) if np.isfinite(j).all()]

                self.resIDs = np.array(list(set(goodResids1).intersection(goodResids2)))
                self._rvals1 = np.empty((len(self.resIDs), self.nLasers))
                self._rvals2 = np.empty((len(self.resIDs), self.nLasers))
                for i, j in enumerate(self.resIDs):
                    self._rvals1[i] = self.solution1.resolving_powers(res_id=j)
                    self._rvals2[i] = self.solution2.resolving_powers(res_id=j)
                return self.resIDs
            else:
                return self.resIDs
        except:
            ValueError("Either one or both of the wavecal solutions was not loaded in!")

    def clean_resonators(self, nsigma=2, r_cut=4):
        if self._cleaningHasBeenRun is True:
            pass
        else:
            self.full_resIDs = np.copy(self.resIDs)
            self._full_rvals1 = np.copy(self._rvals1)
            self._full_rvals2 = np.copy(self._rvals2)

        if len(self.resIDs) != len(self.full_resIDs):
            self.resIDs = np.copy(self.full_resIDs)
            self._rvals1 = np.copy(self._full_rvals1)
            self._rvals2 = np.copy(self._full_rvals2)

        diffs = [self._rvals2[i] - self._rvals1[i] for i in range(len(self.resIDs))]

        std_devs = np.std(diffs, axis=0)
        means = np.mean(diffs, axis=0)

        res_ids_to_cut = []
        for i, j in enumerate(self.resIDs):
            changes = diffs[i]-means
            diff_temp = changes < (nsigma * std_devs)
            cut_temp1 = self._rvals1[i] > r_cut
            cut_temp2 = self._rvals2[i] > r_cut
            if not diff_temp.all() or not cut_temp1.all() or not cut_temp2.all():
                res_ids_to_cut.append(j)

        mask = np.in1d(self.resIDs, np.array(res_ids_to_cut))
        self.resIDs = self.resIDs[~mask]
        self._rvals2 = self._rvals2[~mask]
        self._rvals1 = self._rvals1[~mask]
        self.avg_rvals = (self._rvals1 + self._rvals2) / 2

        self._cleaningHasBeenRun = True

    def load(self, waveCalPath1, waveCalPath2):
        self.solution1 = Solution(waveCalPath1)
        self.solution2 = Solution(waveCalPath2)

    def save(self):
        temp = np.concatenate((self.resIDs[:, None], self.avg_rvals), axis=1)
        data = np.core.records.fromarrays(temp.transpose(), names='ResID, R808, R920, R980, R1120, R1310',
                                          formats='i4, f8, f8, f8, f8, f8')
        np.save('stableResIDs', data)


class DataHandler(object):
    def __init__(self, residInfoPath, dataDirPath=None):
        print("Generating filenames")
        if dataDirPath is None:
            self.dataFileNames = sorted(glob.glob('*.h5'))
        else:
            self.dataFileNames = sorted(glob.glob(str(dataDirPath)+'/*.h5'))
        print("Loading in resonator info")
        self.residInfo = np.load(residInfoPath)
        self.resIDs = self.residInfo['ResID']
        self.wlLimits = None
        self.generateWavelengthLimits(mode='Constant Energy')

    def generateWavelengthLimits(self, mode=None):
        if mode.lower() == 'resolution':
            print("Generating wavelength bin limits")
            _bf_wl_dict = {808: 'R808', 920: 'R920', 980: 'R980', 1120: 'R1120', 1310: 'R1310'}

            self.medianRvals = np.array([np.median(self.residInfo[_bf_wl_dict[i]]) for i in BF_LASER_WAVELENGTHS])
            self.meanRvals = np.array([np.mean(self.residInfo[_bf_wl_dict[i]]) for i in BF_LASER_WAVELENGTHS])
            self.wl_std_devs = np.array([(j / (2.355 * self.medianRvals[i])) for i, j in enumerate(BF_LASER_WAVELENGTHS)])

            wlBinLowerLimits = BF_LASER_WAVELENGTHS - self.wl_std_devs
            wlBinUpperLimits = BF_LASER_WAVELENGTHS + self.wl_std_devs
            self.wlLimits = np.vstack((wlBinLowerLimits, wlBinUpperLimits))
        else:
            print("Generating wavelength bin limits")
            minE = PLANCK_CONST * SPEED_OF_LIGHT / BF_LASER_WAVELENGTHS[-1]
            maxE = PLANCK_CONST * SPEED_OF_LIGHT / BF_LASER_WAVELENGTHS[0]
            binEdges = np.linspace(minE, maxE, 6)
            binEdgesWVL = np.sort(PLANCK_CONST * SPEED_OF_LIGHT / binEdges)
            wlBinLowerLimits = binEdgesWVL[:5]
            wlBinUpperLimits = binEdgesWVL[1:]
            self.wlLimits = np.vstack((wlBinLowerLimits, wlBinUpperLimits))

        binInfo = [self.wlLimits[:, i] for i in range(len(self.wlLimits[0]))]
        np.save('WavelengthBins', binInfo)

    def getTotalCountsAndDuration(self, obsFilePath):
        counts = np.zeros((len(self.resIDs), len(BF_LASER_WAVELENGTHS)))
        cosmicCounts = np.zeros((len(self.resIDs), len(BF_LASER_WAVELENGTHS)))
        durations = np.zeros((len(self.resIDs), len(BF_LASER_WAVELENGTHS)))

        thresh = 4

        if thresh is not None:
            print(f"Using a threshold of {thresh} counts per bin to identify cosmic ray events in {obsFilePath}")

        obs = Photontable(obsFilePath)
        cosmicPeaks, cosmicCutout = self.findCosmicRayTimeStamps(obs.photonTable.read(), threshold=thresh)
        s1 = time.time()
        for count1, j in enumerate(self.resIDs):
            if (count1 % 40) == 0:
                print(f"{count1 / len(self.resIDs) * 100:.2f}% done with obsFile {obsFilePath}")

            if obs.duration >= 60:
                duration = 60
            else:
                duration = obs.duration

            timeRemoved = len(cosmicCutout) / 1e6

            for count2, _ in enumerate(BF_LASER_WAVELENGTHS):
                photonList = obs.getPixelPhotonList(resid=j, wvlStart=self.wlLimits[0][count2],
                                                    wvlStop=self.wlLimits[1][count2], integrationTime=60)

                cosmicCount = [np.any(np.in1d(cosmicCutout, k)) for k in photonList['Time']]
                photonsToRemove = np.sum(cosmicCount)

                counts[count1][count2] = len(photonList)
                durations[count1][count2] = (duration - timeRemoved)
                cosmicCounts[count1][count2] = photonsToRemove

        temp = np.concatenate((self.resIDs[:, None], counts, cosmicCounts, durations[:, 0][:, None]), axis=1)
        data = np.core.records.fromarrays(temp.transpose(), names='ResID, bin1, bin2, bin3, bin4, bin5, bin1cosmics, bin2cosmics, bin3cosmics, bin4cosmics, bin5cosmics, duration',
                                           formats='i4, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8')

        e1 = time.time()
        print(f"--- It took {int(e1 - s1)} seconds to generate counts from Photontable {obsFilePath} ---")
        obs.file.close()
        return data

    def findCosmicRayTimeStamps(self, pList, binSize=10, threshold=None):
        s = time.time()
        photons = pList

        # hist, bins = np.histogram(photons['Time'], bins=int((photons['Time'].max()+1)/binSize))
        hist = np.bincount((photons['Time']/10).astype(int))

        if threshold is None:
            unique, counts = np.unique(hist, return_counts=True)
            avgCounts = np.average(list(unique), weights=list(counts))
            threshold = np.ceil(6 * poisson.std(avgCounts, loc=0)+avgCounts)

        peaks = scipy.signal.find_peaks(hist, height=threshold, threshold=0, distance=20)[0] * 10
        cutOut = np.unique(np.array([np.arange(peak-50, peak+101, 1) for peak in peaks]))
        e = time.time()
        print(f"---- It took {int(e-s)} seconds to find the comic ray peaks ----")
        return peaks, cutOut

    def generateFinalData(self, dataIdentifier, save=True):
        dataFiles = glob.glob(dataIdentifier)

        datainit = np.zeros((len(self.resIDs), 12))
        self.data = np.core.records.fromarrays(datainit.transpose(), names='ResID, bin1, bin2, bin3, bin4, bin5, bin1cosmics, bin2cosmics, bin3cosmics, bin4cosmics, bin5cosmics, duration',
                                          formats='i4, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8')

        for filename in dataFiles:
            temp = np.load(filename)
            for col in DATA_COLUMN_NAMES:
                self.data[col] += temp[col]

        self.data['ResID'] = self.data['ResID']/len(dataFiles)
        if save:
            np.save('full_darkcount_data', data)

        print(f"--- Total stats ---")
        print(f"{self.wlLimits[0][0]:.2f} - {self.wlLimits[1][0]:.2f} nm bin : {((np.sum(self.data['bin1']-self.data['bin1cosmics']))/len(self.data['ResID']))/self.data['duration'][0]:.4e} cts/s/pixel")
        print(f"{self.wlLimits[0][1]:.2f} - {self.wlLimits[1][1]:.2f} nm bin : {((np.sum(self.data['bin2']-self.data['bin2cosmics']))/len(self.data['ResID']))/self.data['duration'][0]:.4e} cts/s/pixel")
        print(f"{self.wlLimits[0][2]:.2f} - {self.wlLimits[1][2]:.2f} nm bin : {((np.sum(self.data['bin3']-self.data['bin3cosmics']))/len(self.data['ResID']))/self.data['duration'][0]:.4e} cts/s/pixel")
        print(f"{self.wlLimits[0][3]:.2f} - {self.wlLimits[1][3]:.2f} nm bin : {((np.sum(self.data['bin4']-self.data['bin4cosmics']))/len(self.data['ResID']))/self.data['duration'][0]:.4e} cts/s/pixel")
        print(f"{self.wlLimits[0][4]:.2f} - {self.wlLimits[1][4]:.2f} nm bin : {((np.sum(self.data['bin5']-self.data['bin5cosmics']))/len(self.data['ResID']))/self.data['duration'][0]:.4e} cts/s/pixel")

    def plot_counts_on_array(self, obsFile, timeStep=10):
        obs = Photontable(obsFile)
        photons = obs.photonTable.read()

        hist = np.bincount((photons['Time']/timeStep).astype(int))
        timestamps = np.linspace(0, photons['Time'].max(), len(hist))

        plt.plot(timestamps, hist, linewidth=0.3)
        plt.xlabel('Time (microseconds)')
        plt.ylabel('Counts on Array')
        plt.title(f"Array Count Time Stream ({obsFile})")

    def summary_data(self, full_data_file=None, data_path=None):
        if full_data_file is not None:
            self.data = np.load(full_data_file)
        else:
            files = glob.glob(data_path)
            datainit = np.zeros((len(self.resIDs), 12))
            self.data = np.core.records.fromarrays(datainit.transpose(),
                                                   names='ResID, bin1, bin2, bin3, bin4, bin5, bin1cosmics, bin2cosmics, bin3cosmics, bin4cosmics, bin5cosmics, duration',
                                                   formats='i4, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8')
            for filename in files:
                temp = np.load(filename)
                for col in DATA_COLUMN_NAMES:
                    self.data[col] += temp[col]

            self.data['ResID'] = self.data['ResID'] / len(files)

        """
        TODO: Do something with the data! how do we want to report this? Ask Ben, see what the most effective reporting method is
        """


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Dark Count Data Analysis Utility')
    parser.add_argument('rFile', type=str, help='ResID Information File')
    parser.add_argument('--path', type=str, dest='dPath', help='Path to Data')
    parser.add_argument('--thresh', type=int, dest='maxCountRate', help='Threshold of cosmic ray counts/time bin')

    args = parser.parse_args()
    if args.dPath is None:
        handler = DataHandler(args.rFile)
    else:
        handler = DataHandler(args.rFile, args.dPath)

    files = np.array_split(handler.dataFileNames, 50)
    results = []
    for count, i in enumerate(files):
        print(f"Working on batch {count+1} of {len(files)} of obsFiles")
        names = i
        pool = Pool(processes=8)
        result = pool.map(handler.getTotalCountsAndDuration, names)
        datainit = np.zeros((len(handler.resIDs), 12))
        data = np.core.records.fromarrays(datainit.transpose(), names='ResID, bin1, bin2, bin3, bin4, bin5, bin1cosmics, bin2cosmics, bin3cosmics, bin4cosmics, bin5cosmics, duration',
                                          formats='i4, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8')

        for arr in result:
            for k in DATA_COLUMN_NAMES:
                data[k] += arr[k]

        pool.close()
        data['ResID'] = data['ResID']/len(result)
        np.save(str(datetime.now().strftime('%d%m%y_%H%M%S'))+'_dark_count_data', data)
        results.append(result)

        print(f"--- Batch {count}/{len(files)} stats ---")
        print(f"{handler.wlLimits[0][0]:.2f} - {handler.wlLimits[1][0]:.2f} nm bin : {((np.sum(data['bin1']-data['bin1cosmics']))/len(data['ResID']))/data['duration'][0]:.4e} cts/s/pixel")
        print(f"{handler.wlLimits[0][1]:.2f} - {handler.wlLimits[1][1]:.2f} nm bin : {((np.sum(data['bin2']-data['bin2cosmics']))/len(data['ResID']))/data['duration'][0]:.4e} cts/s/pixel")
        print(f"{handler.wlLimits[0][2]:.2f} - {handler.wlLimits[1][2]:.2f} nm bin : {((np.sum(data['bin3']-data['bin3cosmics']))/len(data['ResID']))/data['duration'][0]:.4e} cts/s/pixel")
        print(f"{handler.wlLimits[0][3]:.2f} - {handler.wlLimits[1][3]:.2f} nm bin : {((np.sum(data['bin4']-data['bin4cosmics']))/len(data['ResID']))/data['duration'][0]:.4e} cts/s/pixel")
        print(f"{handler.wlLimits[0][4]:.2f} - {handler.wlLimits[1][4]:.2f} nm bin : {((np.sum(data['bin5']-data['bin5cosmics']))/len(data['ResID']))/data['duration'][0]:.4e} cts/s/pixel")

        handler.generateFinalData('*_dark_count_data.npy')
