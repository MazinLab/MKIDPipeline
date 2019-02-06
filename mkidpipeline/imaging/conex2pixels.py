import photutils as pho
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from mkidpipeline.hdf.photontable import ObsFile


class Con2Pix(object):
    _template = ('{datadir}\n'
                 '{obs_files}\n'
                 '{x_initialguess}\n'
                 '{y_initialguess}\n'
                 '{x_conex}\n'
                 '{y_conex}\n'
                 'box_size')

    def __init__(self, datadir='./', wvl_start=None, wvl_stop=None, x_initialguess=None, y_initialguess=None, x_conex=None,
                 y_conex=None, obs_files=None, box_size=30, verbose=False):

        self.datadir = datadir
        self.wvl_start = wvl_start
        self.wvl_stop = wvl_stop
        self.obs_files=list(obs_files)
        self.x_initialguess = list(x_initialguess)
        self.y_initialguess = list(y_initialguess)
        self.x_conex = list(x_conex)
        self.y_conex = list(y_conex)
        self.box_size = box_size
        self.verbose = verbose

    def fit_centroids(self):
        self.xpos_new = []
        self.ypos_new = []
        for index, file in enumerate(self.obs_files):
            obsfile = os.path.join(self.datadir, file)
            obs = ObsFile(obsfile)
            data = obs.getPixelCountImage(applyWeight=False,flagToUse = 0,wvlStart=self.wvl_start,wvlStop=self.wvl_stop)['image']
            data = np.transpose(data)
            data[data == 0] = ['nan']
            x_initial = self.x_initialguess[index]
            y_initial = self.y_initialguess[index]
            positions_new = pho.centroids.centroid_sources(data, x_initial, y_initial, box_size=self.box_size)
            if self.verbose:
                fig, ax = plt.subplots(1, 1)
                ax.imshow(data, origin='lower', interpolation='nearest', cmap='viridis')
                ax.add_patch(Rectangle((x_initial - (self.box_size/2), y_initial - (self.box_size/2)), self.box_size, self.box_size, linewidth=1, edgecolor='r', fill=None))
                marker = '+'
                ms, mew = 30, 2.
                plt.plot(x_initial, y_initial, color='red', marker=marker, ms=ms, mew=mew)
                plt.plot(positions_new[0], positions_new[1], color='blue', marker=marker, ms=ms, mew=mew)
                plt.show()
            self.xpos_new.append(positions_new[0][0])
            self.ypos_new.append(positions_new[1][0])
        self.xpos_new=np.array(self.xpos_new)
        self.ypos_new=np.array(self.ypos_new)

    def calc_fit_params(self):
        self.con_xpos=np.unique(self.x_conex)
        self.con_ypos=np.unique(self.y_conex)

        self.xfit_array=[]
        self.yfit_array=[]

        self.xerr_array=[]
        self.yerr_array=[]

        for index in range(len(self.con_xpos)):
            sub_xarray = self.xpos_new[np.where(self.x_conex == self.con_xpos[index])]
            self.xfit_array.append(np.median(sub_xarray))
            if len(sub_xarray) > 1:
                self.xerr_array.append(np.std(sub_xarray))
            else:
                self.xerr_array.append(np.sqrt(sub_xarray[0]))

        for index in range(len(self.con_ypos)):
            sub_yarray = self.ypos_new[np.where(self.y_conex == self.con_ypos[index])]
            self.yfit_array.append(np.median(sub_yarray))
            if len(sub_yarray) > 1:
                self.yerr_array.append(np.std(sub_yarray))
            else:
                self.yerr_array.append(np.sqrt(sub_yarray[0]))

        np.array(self.xfit_array)
        np.array(self.yfit_array)
        np.array(self.xerr_array)
        np.array(self.yerr_array)

    def fit_two_linear_functions(self):
        self.x_linear, self.x_linearcov = curve_fit(linear_func, self.con_xpos, self.xfit_array, sigma=self.xerr_array)
        self.y_linear, self.y_linearcov = curve_fit(linear_func, self.con_ypos, self.yfit_array, sigma=self.yerr_array)

    def fit_geometric_transform(self):

        conex_fit = []
        for i in range(len(self.con_xpos)):
            conex_fit.append([self.con_xpos[i], self.con_ypos[i]])
        conex_fit = np.array(conex_fit)

        positions_fit = []
        for i in range(len(self.xfit_array)):
            positions_fit.append([self.xfit_array[i], self.yfit_array[i]])
        positions_fit = np.array(positions_fit)

        self.geo_transform = tf.estimate_transform('euclidean', conex_fit, positions_fit)

def linear_func(x, slope, intercept):
        return x * slope + intercept

if __name__ == '__main__':

    con=Con2Pix(datadir='/mnt/data0/isabel/highcontrastimaging/Jan2019Run/20190112/51Eri/51EriDither1/51Eri_wavecalib/', wvl_start=900, wvl_stop=1140,
                x_initialguess=[125.46, 124.79, 107.98, 106.54, 106.04, 93.81, 93.58, 91.51, 89.87],
                y_initialguess=[61.30, 88.13, 61.53, 90.77, 114.66, 36.54, 61.22, 90.85, 115.43], x_conex=[-0.035, -0.035, 0.23, 0.23, 0.23, 0.495, 0.495, 0.495, 0.495],
                 y_conex=[-0.38, 0.0, -0.38, 0.0, 0.38,-0.76, -0.38, 0.0, 0.38],
                obs_files=['1547356758.h5', '1547356819.h5', '1547357065.h5', '1547357126.h5', '1547357187.h5', '1547357310.h5', '1547357371.h5', '1547357432.h5', '1547357493.h5'],
                box_size=30, verbose=False)

    con.fit_centroids()
    con.calc_fit_params()
    con.fit_two_linear_functions()