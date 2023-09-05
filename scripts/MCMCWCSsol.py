import mkidcore
import mkidcore.config
import mkidpipeline.config as config

from mkidcore.instruments import CONEX2PIXEL
from mkidpipeline.photontable import Photontable

import matplotlib,emcee, ruamel.yaml, sys, operator, os,  math,warnings,scipy, corner, argparse, concurrent.futures
import numpy as np
import matplotlib.pylab as plt
import numpy.ma as ma
from tqdm import tqdm
from glob import glob
from astropy.modeling import models
from astropy.stats import gaussian_fwhm_to_sigma
from photutils.centroids import (centroid_sources, centroid_2dg)
from scipy import spatial
from astropy.visualization.mpl_normalize import simple_norm
from multiprocessing import Pool
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import display, Math
from IPython.utils import io
from itertools import repeat
from skimage.morphology.footprints import disk
from skimage.morphology import dilation
import warnings
warnings.filterwarnings("ignore")
class ClickCoords:
    """
    Class for choosing approximate location in the image for each point source to use for the wcscal. Associates the
    (RA/DEC) coordinates given using the source_loc keyword in the data.yaml with an approximate (x, y) pixel value.
    """

    def __init__(self, image, n_satspots, fig=None):
        self.coords = []
        self.image = image
        self.n_satspots=n_satspots
        self.n_sources = n_satspots+1
        self.counter = 0
        self.fig = plt.figure() if fig is None else fig
        self.cid1 = None
        self.cid2 = None
        self.redo = False

    def push_to_start(self):
        plt.imshow(self.image)
        plt.title('Click Anywhere to Start')
        if plt.waitforbuttonpress() is True:
            return True

    def get_coords(self):
        self.push_to_start()
        if self.n_satspots != 0: plt.title('Select the 4 satellite spots')
        else: plt.title('Select the the center of the star/coronograph')
        plt.draw()
        self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        return self.coords

    def __onclick__(self, click):
        try:
            point = (click.xdata, click.ydata)
        except TypeError:
            point = (None, None)
        self.coords.append(point)
        self.counter += 1
        if self.counter == self.n_sources:
            plt.scatter(*zip(*self.coords), c='red', marker='o', s=10)
            plt.draw()
            plt.title('Press "c" to Continue or "r" to Re-do')
            self.cid2 = self.fig.canvas.mpl_connect('key_press_event', self.__onkey__)
        else:
            plt.imshow(self.image)
            plt.scatter(*zip(*self.coords), c='red', marker='o', s=10)
            if self.counter < 4 and self.n_satspots != 0:
                plt.title('Select the 4 satellite spots')
            else:
                plt.title('Select the the center of the star/coronograph')
            plt.draw()
        return self.coords

    def __onkey__(self, event):
        cont_key = 'c'
        redo_key = 'r'
        cont = False
        while not cont:
            if cont_key in event.key:
                cont = True
            if redo_key in event.key:
                self.fig.canvas.mpl_disconnect(self.cid2)
                self.fig.canvas.mpl_disconnect(self.cid1)
                plt.close(self.fig)
                self.redo = True
                break
        self.fig.canvas.mpl_disconnect(self.cid2)
        self.fig.canvas.mpl_disconnect(self.cid1)
        plt.close(self.fig)
        return self.coords


def select_sources(image, n_satspots=4):
    """
    runs interactive plotting function for selecting which points in an image correspond to which sources that have
    well constrained on sky coordinates
    :param image: Image in which to select points
    :param source_locs: (RA, DEC) coordinates of the sources in the image to select
    :return: list of (x, y) pixel coordinates for each source_loc
    """
    fig = plt.figure()
    cc = ClickCoords(image, n_satspots=n_satspots, fig=fig)
    coords = cc.get_coords()
    if cc.redo:
        coords = select_sources(image, n_satspots)
    plt.close("all")
    return coords

class SATSPOT2PIXELREF:
    '''

    '''
    def __init__(self,data_list,header_list=None,var_list=None,guess_list=[[77, 62], [134, 62], [77, 7], [134, 7]],conex_guess=[-0.1,-0.6],letter_list=['A', 'B', 'C', 'D'], slopes = [-63.09, 67.61], xyCons = [0, 0], dmax=3):
        self.header_list=header_list
        self.data_list=data_list
        self.var_list=var_list
        self.letter_list=letter_list
        self.slopes=np.array(slopes)
        self.xCon, self.yCon =xyCons
        self.dmax=dmax
        self.guess_list=np.array(guess_list)
        self.conex_guess=np.array(conex_guess)

    def proj(self,args):
        x1, y1, ex1, ey1, x2, y2, ex2, ey2, cx, cy, ecx, ecy = args

        dx = x2 - x1
        dy = y2 - y1

        cdx = cx - x1
        cdy = cy - y1

        edx = np.sqrt(ex1 ** 2 + ex2 ** 2)
        edy = np.sqrt(ey1 ** 2 + ey2 ** 2)

        ecdx = np.sqrt(ex1 ** 2 + ecx ** 2)
        ecdy = np.sqrt(ey1 ** 2 + ecy ** 2)

        A = cdx * dx
        B = cdy * dy
        C = dx * dx
        D = dy * dy

        eA = np.abs(A) * np.sqrt((ecdx / cdx) ** 2 + (edx / dx) ** 2)
        eB = np.abs(B) * np.sqrt((ecdy / cdy) ** 2 + (edy / dy) ** 2)
        eC = np.abs(C) * np.sqrt(2 * (edx / dx) ** 2)
        eD = np.abs(D) * np.sqrt(2 * (edy / dy) ** 2)

        Alpha = (A + B)
        Beta = (C + D)
        t = Alpha / Beta

        eAlpha = np.sqrt(eA ** 2 + eB ** 2)
        eBeta = np.sqrt(eC ** 2 + eD ** 2)
        et = np.sqrt((eAlpha / Alpha) ** 2 + (eBeta / Beta) ** 2)

        if t < 0 or t > 1:
            print("Projection lies outside segment")

        A1 = t * dx
        B1 = t * dy

        eA1 = np.sqrt((et / t) ** 2 + (edx / dx) ** 2)
        eB1 = np.sqrt((et / t) ** 2 + (edy / dy) ** 2)

        px = x1 + A1
        py = y1 + B1

        epx = np.sqrt(ex1 ** 2 + eA1 ** 2)
        epy = np.sqrt(ey1 ** 2 + eB1 ** 2)

        return (px, py, epx, epy)


    def run(self, el, verbose=True,sat_spots=True):
        if self.header_list is not None:
            pixels_at_Con = np.array([CONEX2PIXEL(float(self.header_list[el]['E_CONEXX']),
                                                  float(self.header_list[el]['E_CONEXY']),  self.slopes, ref_pix, ref_con) for
                                      ref_pix, ref_con in zip(self.guess_list, [self.conex_guess] * 4)])  # Im assuming the CONEX2PIXEL has no errors
            x_init = pixels_at_Con[::, 0]
            y_init = pixels_at_Con[::, 1]
        else:
            x_init = self.guess_list[::, 0]
            y_init = self.guess_list[::, 1]

        data = ma.masked_values(self.data_list[el], 0)
        mask = data.mask
        if self.var_list is not None:
            err = np.sqrt(ma.masked_values(self.var_list[el], 0)) / 20
            kwargs = {'error': err, 'mask': mask}
        else:
            kwargs = {'mask': mask}

        x = []
        y = []
        xe = []
        ye = []
        for elno in range(len(x_init)):
            out = centroid_sources(data, x_init[elno], y_init[elno], box_size=15, **kwargs,
                                   centroid_func=centroid_2dg)
            x.extend(out[0])
            y.extend(out[1])
            xe.extend(out[2])
            ye.extend(out[3])

        x = np.array(x)
        y = np.array(y)
        xe = np.array(xe)
        ye = np.array(ye)

        if any((abs(x - x_init) > self.dmax) | (abs(y - y_init) > self.dmax)):
            print('> Skipping ',el)
        else:
            if sat_spots:
                reference_points = [[float(round(x[elno], 2)), float(round(y[elno], 2)), float(round(xe[elno], 2)),
                                     float(round(ye[elno], 2))] for elno in range(len(x))]
                reference_points_dict = {letter: points for letter, points in zip( self.letter_list, reference_points)}
                if np.any(np.array(list(reference_points_dict.values()))[::, 2:] < 1):
                    index = np.where(np.array(list(reference_points_dict.values()))[::, 2:] < 1)[0]
                    if len(index) > 1: print('> Skipping ',el)
                    selected_letter_list = [ self.letter_list[elno] for elno in range(len( self.letter_list)) if elno != index[0]]
                else:
                    print('V')
                    my_list = np.array(list(reference_points_dict.values()))[::, 2:].mean(axis=1)
                    index, value = max(enumerate(my_list), key=operator.itemgetter(1))
                    selected_letter_list = [ self.letter_list[elno] for elno in range(len( self.letter_list)) if elno != index]

                values = np.array([reference_points_dict[letter][:2] for letter in selected_letter_list])
                dist_mat = spatial.distance_matrix(values, values)
                i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
                input_letters = [selected_letter_list[i], selected_letter_list[j],
                                 list(set(selected_letter_list) - set([selected_letter_list[i], selected_letter_list[j]]))[
                                     0]]
                vars = np.array([reference_points_dict[letter] for letter in input_letters]).ravel()

                p_x, p_y, ep_x, ep_y = self.proj(vars)  # I'm assuming there is no correlation between variables
            else:
                input_letters=self.letter_list
                reference_points=[[x[0],y[0],xe[0],ye[0]]]
                p_x, p_y, ep_x, ep_y=np.array([x[0],y[0],xe[0],ye[0]])
                print(p_x,p_y)
                reference_points_dict = {letter: points for letter, points in zip(self.letter_list, reference_points)}
            if verbose:
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.title.set_text(el)
                cmap = matplotlib.cm.get_cmap("gray").copy()
                cmap.set_bad('black', 1.)
                im = ax.imshow(data, origin='lower', cmap=cmap, norm=simple_norm(data, stretch='sqrt'))
                for letter in  self.letter_list:
                    if letter in input_letters:
                        c = 'g'
                    else:
                        c = 'r'

                    xp, yp = reference_points_dict[letter][:2]
                    exp, eyp = reference_points_dict[letter][2:]
                    ax.errorbar(xp, yp, xerr=exp, yerr=eyp, marker='.', ecolor=c, ms=2, capsize=2, capthick=3,
                                elinewidth=3)

                    ax.plot(xp, yp, 'o%s' % c, ms=3)
                    ax.text(xp + 2, yp + 2, letter, color=c, fontsize=20)

                ax.errorbar(p_x, p_y, xerr=ep_x, yerr=ep_y, marker='.', ecolor='y', ms=2, capsize=2, capthick=3,
                            elinewidth=3)
                ax.plot(p_x, p_y, 'oy', ms=3)
                ax.text(p_x + 1, p_y + 1, 'P', color='y', fontsize=20)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                plt.show()

            if self.header_list is not None:
                conex_xy_ref= [float(self.header_list[el]['E_CONEXX']),
                                       float(self.header_list[el]['E_CONEXY'])]
            else: conex_xy_ref = [0,0]
            if sat_spots:
                data_dict = {'el': el ,'SSPOT_XY': reference_points_dict,
                             'conex_xy_ref': conex_xy_ref,
                             'P': [float(round(p_x, 2)), float(round(p_y, 2)), float(round(ep_x, 2)),
                                   float(round(ep_y, 2))]}
            else:
                data_dict = {'el': el,
                             'conex_xy_ref': conex_xy_ref,
                             'P': [float(round(p_x, 2)), float(round(p_y, 2)), float(round(ep_x, 2)),
                                   float(round(ep_y, 2))]}

            ref_pix = data_dict['P'][:2]
            eref_pix = data_dict['P'][2:]
            ref_con = data_dict['conex_xy_ref']


            pix_at_con_0= np.round(CONEX2PIXEL(self.xCon, self.yCon, self.slopes, ref_pix, ref_con),2)
            data_dict['pix_at_con_0'] = [pix_at_con_0[0],pix_at_con_0[1],eref_pix[0],eref_pix[1]]

            if verbose:
                print(data_dict)

            self.out=data_dict

class SATSPOT_MODEL:
    '''

    '''
    def __init__(self,shape_xy,angle=0,radial=False,factor=1,order=0):
        self.shape_xy=shape_xy
        self.angle=angle
        self.factor=factor
        self.order=order
        self.radial=radial

    def grid(self, start=[0,0], stop=[1,1], num=[10,10]):
        xy = [np.linspace(start[0],stop[0],num[0]+1),np.linspace(start[1],stop[1],num[1]+1)]
        return(np.meshgrid(xy[0], xy[1]))

    def psf_spot_loc(self,cen_xy,length,dangle):
        out=[]
        half_length = length/2
        radian = (self.angle+dangle)/180*math.pi
        dx = half_length*math.cos(radian)
        dy = half_length*math.sin(radian)
        return([[cen_xy[0]-dx, cen_xy[1]-dy], [cen_xy[0]+dx, cen_xy[1]+dy]],self.angle + dangle)

    def gaussian2d_model(self, init_amplitude, xy, fwhm_xy, theta, sat_xy, gaussian_fwhm_to_sigma):
        gauss = models.Gaussian2D(amplitude=init_amplitude, theta=theta,
                                  x_mean=sat_xy[0], y_mean=sat_xy[1],
                                  x_stddev=fwhm_xy[0] * gaussian_fwhm_to_sigma,
                                  y_stddev=fwhm_xy[1] * gaussian_fwhm_to_sigma)

        return gauss.evaluate(xy[0],xy[1],** {i:j for i,j in zip(gauss.param_names, gauss.parameters)})

    def create_psf_image(self, init_amplitude, cen_xy, fwhm_xy, lengths_list=None, abcd=None, d=0, pad=10, mask=[],flux=1,angles_list=None,sat_spots=True):
        if len(fwhm_xy)==1: fwhm_xy=fwhm_xy*2
        self.abcd=abcd
        self.d=d
        self.satspots_xy=[]
        self.rot_angle= []
        xy_grid = self.grid(stop=self.shape_xy, num=self.shape_xy)

        if sat_spots:
            sys_angle=angles_list[0]
            satspots_xy, rot_angles=self.psf_spot_loc(cen_xy, lengths_list[0], sys_angle)
            self.satspots_xy.extend(satspots_xy)
            self.rot_angle.append(rot_angles)


            z1 = self.gaussian2d_model(init_amplitude, xy_grid, fwhm_xy[0],np.deg2rad(sys_angle),satspots_xy[0],gaussian_fwhm_to_sigma)
            z2 = self.gaussian2d_model(init_amplitude, xy_grid, fwhm_xy[0],np.deg2rad(sys_angle),satspots_xy[1],gaussian_fwhm_to_sigma)

            rot_angle=angles_list[0]+angles_list[1]
            satspots_xy, rot_angles=self.psf_spot_loc(cen_xy, lengths_list[1], rot_angle)
            self.satspots_xy.extend(satspots_xy)
            self.rot_angle.append(rot_angles)

            z3 = self.gaussian2d_model(init_amplitude, xy_grid, fwhm_xy[1],np.deg2rad(rot_angle),satspots_xy[0],gaussian_fwhm_to_sigma)
            z4 = self.gaussian2d_model(init_amplitude, xy_grid, fwhm_xy[1],np.deg2rad(rot_angle),satspots_xy[1],gaussian_fwhm_to_sigma)


            self.model = z1 + z2 + z3 + z4 + self.d
        else:
            z1 = self.gaussian2d_model(init_amplitude, xy_grid, fwhm_xy[0],np.deg2rad(0),cen_xy,gaussian_fwhm_to_sigma)
            self.model = z1 + self.d

        self.psfs_img=self.model.copy()
        if self.abcd != None: self.apply_gradient(abcd=self.abcd)
        if self.factor != 1: self.resampling_img(factor=self.factor,order=self.order)
        if mask is not None: self.psfs_img[mask==0]=0
        self.psfs_img /= np.sum(self.psfs_img)
        self.psfs_img*=flux

    def sig2d(self,xy_grid,abcd_sig=[100,100,50,50]):
        self.abcd_sig=abcd_sig
        a, b, c, d = self.abcd_sig
        return (1/(1 + np.exp(a*xy_grid[0]+c)))*(1/(1 + np.exp(b*xy_grid[1]+d)))

    def radial_gradient(self,xy_grid):
        return(np.sqrt(xy_grid[0]**2 + xy_grid[1]**2))

    def linear_gradient(self,xy_grid):
        a,b,c,d = self.abcd
        return((d - a*xy_grid[0] - b*xy_grid[1]) / c)

    def apply_gradient(self,xy_start=[0,0],xy_stop=[1,1],abcd=[1,2,3,4]):
        self.abcd=abcd
        xy_grid=self.grid(start=xy_start,stop=xy_stop,num=self.shape_xy)
        if self.radial: gradientArray=self.radial_gradient(xy_grid)
        else: gradientArray=self.linear_gradient(xy_grid)
        self.gradientArray=gradientArray
        self.psfs_img+=gradientArray

    def resampling_img(self,factor=1,order=0):
        self.psfs_img=scipy.ndimage.zoom(self.psfs_img, self.factor, order=order)

    def plot_image(self,data,title=None,cen_xy=None, satspot_xy=None, norm=None, fcr=[7,7],rows=1, v_lim=None,save_output=False, path2savedir='./',filename='test.jpg'):
        if len(data.shape)==2:
            data=np.array([data])

        cols=data.shape[0]//rows
        if title is not None and len(title)==1: title = title * data.shape[0]
        if norm is not None and len(norm)==1:  norm = norm * data.shape[0]

        fig,ax = plt.subplots(rows,cols,figsize=(fcr[0]*cols,fcr[1]*rows),squeeze=False)
        elno=0
        for elno_r in range(rows):
            for elno_c in range(cols):
                if norm is not None: im = ax[elno_r][elno_c].imshow(data[elno],origin='lower',cmap='gray',norm=norm[elno])#,vmin=v_lim[elno][0],vmax=v_lim[elno][1])
                else: im = ax[elno_r][elno].imshow(data[elno_c],origin='lower',cmap='gray')#,vmin=v_lim[elno][0],vmax=v_lim[elno][1])
                ax[elno_r][elno_c].set_title(title[elno])
                if cen_xy is not None: ax[elno_r][elno_c].plot(cen_xy[0]*self.factor,cen_xy[1]*self.factor,'oy',ms=5)
                if satspot_xy is not None:
                    for x,y in satspot_xy: ax[elno_r][elno_c].plot(x*self.factor,y*self.factor,'og',ms=5)

                divider = make_axes_locatable(ax[elno_r][elno_c])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
                elno+=1
        plt.tight_layout()

        if save_output:
            fig.savefig(path2savedir+filename)
            plt.close('all')
        else: plt.show()

def create_mask(data,xyCons,slopes,positions0,E_CONEXXY0,factor,normalize=False):
    satspot_positions = np.round(np.array([CONEX2PIXEL(xyCons[0],
                                              xyCons[1],
                                              slopes,
                                              ref_pix,
                                              ref_con) for ref_pix, ref_con in zip(positions0,
                                                                                    [E_CONEXXY0] * len(positions0))]),2)

    data2mask=data.copy()
    data2mask=scipy.ndimage.zoom(data2mask, factor , order=0)
    data_norm = data2mask.copy()
    if normalize: data_norm/=np.sum(data)
    bright_pixel = np.zeros(data_norm.shape)
    for x,y in satspot_positions:
        bright_pixel[int(y)][int(x)]=1

    mask=dilation(bright_pixel, disk(14))

    masked_img=data_norm.copy()
    masked_img[mask==0]=0
    return(masked_img)

class MCMC_FIT:
    '''

    '''
    def __init__(self,  path='./', nwalkers = 10,  ncpu=10, steps = 1000, ndim=None, check_acor=25, Fconv=1, conv_thr=0.02, ndesired=100, multiplier=1e-2, ini_pos=True , progress=False, verbose=False, fwhm_xy=None,lengths=None, angles=None, abcd=None, fit_d=False, const=0, moves=None, labels=None,sat_spots=True, kwargs={}):
        self.labels=labels
        self.ndim=ndim
        self.ncpu=ncpu
        self.nwalkers=nwalkers
        self.steps=steps
        self.check_acor = check_acor
        self.Fconv = Fconv
        self.conv_thr = conv_thr
        self.ndesired = ndesired
        self.progress=progress
        self.kwargs = kwargs
        if self.kwargs: self.sp = SATSPOT_MODEL(**self.kwargs)
        self.path=path
        self.verbose=verbose
        self.const=const
        self.fwhm_xy=fwhm_xy
        self.fit_d=fit_d
        self.abcd=abcd
        self.moves=moves
        self.lengths=lengths
        self.angles=angles
        self.multiplier=multiplier
        self.ini_pos = ini_pos
        self.sat_spots=sat_spots
        if self.verbose: self.ncpu=1

    def get_id_from_key(self,key):
        if key in self.labels:
            return(np.where(np.array(self.labels) == key)[0][0])
        else:
            return False

    def log_likelihood(self, pos):  # ,a,b,c,d):
        masked_img=data.copy()
        amplitude=pos[self.get_id_from_key('amplitude')]
        cen_xy=[pos[self.get_id_from_key('cen_x')], pos[self.get_id_from_key('cen_y')]]
        if self.lengths is None: lengths = [pos[self.get_id_from_key('length1')],pos[self.get_id_from_key('length2')]]
        else: lengths = self.lengths
        if self.angles is None: angles = [pos[self.get_id_from_key('angle1')], pos[self.get_id_from_key('angle2')]]
        else: angles = self.angles
        if self.fwhm_xy is None:
            if self.sat_spots: fwhm_xy = [[pos[self.get_id_from_key('fwhm_x1')], pos[self.get_id_from_key('fwhm_y1')]],[pos[self.get_id_from_key('fwhm_x2')], pos[self.get_id_from_key('fwhm_y2')]]]
            else: fwhm_xy = [[pos[self.get_id_from_key('fwhm_x1')], pos[self.get_id_from_key('fwhm_y1')]]]
        else: fwhm_xy = self.fwhm_xy
        if self.const is None: const = pos[self.get_id_from_key('const')]
        else: const = self.const
        self.sp.create_psf_image(amplitude,
                                cen_xy,
                                fwhm_xy,
                                lengths_list=lengths,
                                angles_list=angles,
                                d=const,
                                mask=masked_img,
                                flux=np.nansum(masked_img),
                                sat_spots=self.sat_spots)

        chi2_map = (masked_img - self.sp.psfs_img) ** 2 / (self.sp.psfs_img)
        chi2_map[~np.isfinite(chi2_map)] = 0


        if self.verbose:
            datas = np.array([masked_img, self.sp.model, chi2_map / np.nanmax(data)])
            norm = [simple_norm(masked_img, stretch='sqrt'), simple_norm(self.sp.model, stretch='sqrt'),
                    simple_norm(chi2_map / np.nanmax(data), stretch='sqrt')]

            print('#############################################################')
            print('> pos: ',pos)
            self.sp.plot_image(datas,
                          title=['MaskedData', 'Model', 'Chi2 %.3f' % np.nansum(chi2_map)], cen_xy=[pos[self.get_id_from_key('cen_x')], pos[self.get_id_from_key('cen_y')]],
                          norm=norm, rows=1)

        return -np.nansum(chi2_map[np.isfinite(chi2_map)])/2

    def log_prior(self,pos):
        if np.all([self.pos_dict[list(self.labels)[elno]][1] <= pos[elno] <= self.pos_dict[list(self.labels)[elno]][2] for elno in range(len(self.labels))]):
            return 0.0
        else:
            return -np.inf

    def log_prob(self,pos):
        ll = self.log_prior(pos)
        lp = self.log_likelihood(pos)
        if np.isfinite(ll + lp):
            return ll + lp
        else:
            return (-np.inf)

    def sampler_convergence(self, sampler, pos):
        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(self.steps * 3)

        # This will be useful to testing convergence
        old_tau = np.inf
        converged = False
        for sample in sampler.sample(pos, iterations=self.steps, progress=self.progress, store=True):
            # Only check convergence every check_acor steps
            if converged == False:
                if sampler.iteration % self.check_acor:
                    continue

                tau = sampler.get_autocorr_time(tol=0)
                if np.any(np.isnan(tau)):
                    tau[:] = sampler.iteration / 100

                autocorr[index] = np.mean(tau)
                thin = int(0.5 * np.min(tau)) #np.int_(2 * np.max(tau))
                burnin = int(2 * np.max(tau)) #np.int_(sampler.iteration / 2)
                index += 1

                # Check convergence
                converged = np.all(tau * self.Fconv < sampler.iteration)
                converged &= np.all((np.abs(old_tau - tau) / tau) < self.conv_thr)
                if converged:
                    # Once converged, set the number of desired runs for further running the sampler
                    # until we have the desired number of post-convergence, iid samples
                    burnin = sampler.iteration
                    n_post_convergence_runs = int(self.ndesired *  thin)
                    n_to_go = 0
                    if self.progress:
                        print('Converged at iteration {}'.format(burnin))
                        print('Autocorrelation times equal to: {}'.format(tau))
                        print('Thinning equal to: {}'.format(thin))
                        print('Running {} iterations post-convergence'.format(n_post_convergence_runs))
                    sys.stdout.flush()

                old_tau = tau

            else:
                # Post-convergence samples
                n_to_go += 1
                if n_to_go > n_post_convergence_runs:
                    break
        return (sampler, autocorr, converged, burnin, thin, tau)

    def run(self, filename, pos_dict, masked_img):

        global data
        self.pos_dict=pos_dict
        self.labels=list(pos_dict.keys())
        self.ndim = len(self.labels)
        if self.ini_pos: pos = np.array([pos_dict[keys][0] for keys in pos_dict.keys()] + self.multiplier * np.random.randn(self.nwalkers, self.ndim))
        else: pos = np.array([np.array([np.random.uniform(pos_dict[keys][1],pos_dict[keys][2]) for keys in pos_dict.keys()])
                              for i in range(self.nwalkers)])

        backend = emcee.backends.HDFBackend(self.path+filename)
        backend.reset(self.nwalkers, self.ndim)

        data=masked_img
        if self.moves is None: self.moves = [(emcee.moves.DEMove(), 0.7), (emcee.moves.DESnookerMove(), 0.3), ]
        if self.ncpu==1:
            sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob, moves=self.moves, backend=backend)
            self.sampler, self.autocorr, self.converged, self.burnin, self.thin, self.tau = self.sampler_convergence(sampler, pos)
        else:
            with Pool(self.ncpu) as pool:
                sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_prob, moves=self.moves, backend=backend, pool=pool)
                self.sampler, self.autocorr, self.converged, self.burnin, self.thin, self.tau = self.sampler_convergence(
                    sampler, pos)

    def sample_posteriors(self, filename, slopes, full_posterior=True, verbose=False, save_output=False):
        if self.ndim is None: self.ndim = len(self.labels)

        reader = emcee.backends.HDFBackend(self.path+filename)
        tau = reader.get_autocorr_time(tol=0)
        if verbose: print('> tau: ',tau)
        thin = int(0.5 * np.nanmin(tau))  # np.int_(2 * np.max(tau))
        burnin = int(2 * np.nanmax(tau))  # np.int_(sampler.iteration / 2)

        if full_posterior: samples = reader.get_chain()
        else: samples = reader.get_chain(discard=burnin, thin=thin)
        flat_samples = reader.get_chain(flat=True, discard=burnin, thin=thin)

        sol = {}
        for i in range(self.ndim):
            mcmc_p = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc_p)
            if verbose:
                txt = "\mathrm{{{3}}} = {0:.5f}_{{-{1:.5f}}}^{{{2:.5f}}}"
                txt = txt.format(mcmc_p[1], q[0], q[1], self.labels[i])
                display(Math(txt))
            sol[self.labels[i]]=[mcmc_p[1], q[0], q[1]]

        if verbose or save_output:
            fig, axes = plt.subplots(self.ndim, figsize=(10, 10), sharex=True)

            for i in range(self.ndim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(self.labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)

            axes[-1].set_xlabel("step number");
            if save_output:
                fig.savefig(self.path + '/plots/posterior/%s' % filename.split('.h5')[0] + '.jpg')

            fig = corner.corner(flat_samples,
                                labels= self.labels,
                                quantiles=[0.16, 0.5, 0.84],
                                show_titles=True,
                                title_kwargs={"fontsize": 12})

            if not save_output: plt.show()
            else:
                fig.savefig(self.path + '/plots/corners/%s'%filename.split('.h5')[0]+'.jpg')
                plt.close('all')
        else: plt.close('all')

        return(sol)

class MCMCWCS:
    def __init__(self, pipe_cfg, data_cfg,verbose):
        self.data = mkidcore.config.load(data_cfg)
        pipe = mkidcore.config.load(pipe_cfg)

        self.paths={label: pipe['paths'][label] for label in ['data','out']}
        self.paths['data_cfg']=data_cfg
        self.paths['pipe_cfg']=pipe_cfg
        self.paths['MCMC_fit']=self.paths['out'] + 'MCMC_fit/'
        self.mcmc_config={label: pipe['mcmcwcssol'][label] for label in pipe['mcmcwcssol'].keys()}
        self.mcmc_config['ncpu']=pipe['ncpu']

        if not verbose:
            self.mcmc_config['verbose'] = pipe['mcmcwcssol']['verbose']
        else:
            self.mcmc_config['verbose'] = verbose

        mcmcmwcs_pos = [index for (index, d) in enumerate(self.data) if '!MKIDMCMCWCSol' in d.tag.value][0]
        data_names = self.data[mcmcmwcs_pos]['data'].split('+')
        data_pos = [index for (index, d) in enumerate(self.data) for out_data in data_names if
                          out_data in d['name']]
        data_list = [self.data[mcmcmwcs_pos]['data'] for mcmcmwcs_pos in data_pos]
        wcscal = self.data[mcmcmwcs_pos]['wcscal']
        start_offset = self.data[mcmcmwcs_pos]['start_offset']
        self.mcmc_config['mcmcmwcs_pos'] = mcmcmwcs_pos
        self.mcmc_config['start_offset'] = start_offset

        self.mcmc_setup={label:var for var,label in zip([data_names,data_list,wcscal],['data_names','data_list','wcscal'])}

    def make_dir(self):
        if not os.path.exists(self.paths['MCMC_fit']):
            print('> Making %s' % self.paths['MCMC_fit'])
            os.makedirs(self.paths['MCMC_fit'])
            if not os.path.exists(self.paths['MCMC_fit']+'/plots/'):
                print('> Making %s' % self.paths['MCMC_fit']+'plots/')
                os.makedirs(self.paths['MCMC_fit']+'plots/')
                if not os.path.exists(self.paths['MCMC_fit'] + '/plots/corners/'):
                    print('> Making %s' % self.paths['MCMC_fit']+'plots/corners/')
                os.makedirs(self.paths['MCMC_fit']+'plots/corners/')
                if not os.path.exists(self.paths['MCMC_fit'] + '/plots/debug/'):
                    print('> Making %s' % self.paths['MCMC_fit']+'plots/debug/')
                os.makedirs(self.paths['MCMC_fit']+'plots/debug/')
                if not os.path.exists(self.paths['MCMC_fit'] + '/plots/posterior/'):
                    print('> Making %s' % self.paths['MCMC_fit']+'plots/posterior/')
                os.makedirs(self.paths['MCMC_fit']+'plots/posterior/')

    def fetching_h5_names(self):
        start_times = []
        print('> Building list of start times from: %s' % self.mcmc_setup['data_names'])
        for name, data in zip(self.mcmc_setup['data_names'], self.mcmc_setup['data_list']):
            print('>> Looking for data associated to %s containing time %s,...' % (name, data))
            startt, _, _ = mkidcore.utils.get_ditherdata_for_time(self.paths['data'], data)
            start_times.extend([np.round(i) for i in startt])
            print('>> Found %i dithers.' % len(startt))

        h5_int_names = [int(i.split('/')[-1].split('.h5')[0]) for i in glob(self.paths['out'] + '*.h5')]

        self.mcmc_setup['start_times'] = start_times
        self.mcmc_setup['h5_names'] = [str(int(min(h5_int_names, key=lambda x: abs(x - i))))+'.h5' for i in self.mcmc_setup['start_times']]
        self.mcmc_setup['ntargets'] = len(start_times)

    def fetching_datas(self):
        datas = []
        headers = []
        dist_list = []
        elno_list = []
        data_names = []

        num_of_chunks = 3 * self.mcmc_config['ncpu']
        chunksize = self.mcmc_setup['ntargets'] // num_of_chunks
        if chunksize <= 0:
            chunksize = 1

        elno = 0
        if self.mcmc_config['ncpu'] == 1:
            workers_load = 10
        else:
            workers_load = self.mcmc_config['ncpu']
        print('> Using %i cpus to load a total of %i files ...' % (workers_load, self.mcmc_setup['ntargets']))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers_load) as executor:
            for filename, header, data in tqdm(executor.map(mcmcwcs.load_fits_task, self.mcmc_setup['h5_names'], chunksize=chunksize)):
                elno += 1
                data_names.append(filename)
                headers.append(header)
                datas.append(data)
                dist_list.append(np.sqrt((headers[0]['E_CONEXX'] - header['E_CONEXX']) ** 2 + (
                        headers[0]['E_CONEXY'] - header['E_CONEXY']) ** 2))
                elno_list.append(elno)

        sorted_data_pos = [x for _, x in sorted(zip(dist_list, elno_list))]
        
        self.mcmc_setup['data_names']=data_names
        self.mcmc_setup['sorted_data_pos']=sorted_data_pos
        return(datas,headers)


    def load_fits_task(self, filename):
        pt = Photontable(self.paths['out'] + filename)
        hdul = pt.get_fits(wave_start=950, wave_stop=1100, start=pt.start_time + self.mcmc_config['start_offset'])
        header = hdul[0].header
        data = hdul[1].data
        hdul.close()
        return (filename, header, data)

    def fetching_mcmc_parameters(self, datas, headers):
        if self.mcmc_config['redo'] or np.any([len(self.data[self.mcmc_config['mcmcmwcs_pos']][label]) == 0 for label in ['slopes']]):
            mcmcwcs.get_slope_and_conex(datas,headers)
        else:
            self.mcmc_setup['slopes'] = np.float64(self.data[self.mcmc_config['mcmcmwcs_pos']]['slopes'])
            self.mcmc_setup['conex_xy_ref'] = np.float64(self.data[self.mcmc_config['mcmcmwcs_pos']]['conex_ref'])

        if self.mcmc_config['sat_spots']:
            if self.mcmc_config['redo'] or np.any([len(self.data[self.mcmc_config['mcmcmwcs_pos']][label]) == 0 for label in
                               ['spot_ref1', 'spot_ref2', 'spot_ref3', 'spot_ref4', 'cor_spot_ref', 'conex_ref']]):
                mcmcwcs.get_satellite_spots_and_coronograph(datas[self.mcmc_config['ref_sat_spot_pos']],
                                                            headers[self.mcmc_config['ref_sat_spot_pos']])

            self.mcmc_setup['positions_ref'] = [np.float64(self.data[self.mcmc_config['mcmcmwcs_pos']]['spot_ref1']),
                             np.float64(self.data[self.mcmc_config['mcmcmwcs_pos']]['spot_ref2']),
                             np.float64(self.data[self.mcmc_config['mcmcmwcs_pos']]['spot_ref3']),
                             np.float64(self.data[self.mcmc_config['mcmcmwcs_pos']]['spot_ref4'])]
            self.mcmc_setup['coronograph_ref'] = np.float64(self.data[self.mcmc_config['mcmcmwcs_pos']]['cor_spot_ref'])
            self.mcmc_setup['conex_xy_ref'] = np.float64(self.data[self.mcmc_config['mcmcmwcs_pos']]['conex_ref'])
            self.mcmc_setup['slopes'] = np.float64(self.data[self.mcmc_config['mcmcmwcs_pos']]['slopes'])

            self.mcmc_setup['pos'] = {'amplitude': [self.mcmc_config['amplitude'][0], self.mcmc_config['amplitude'][1],
                                      self.mcmc_config['amplitude'][2]],
                        'length1': [self.mcmc_config['length'][0], self.mcmc_config['length'][1],
                                    self.mcmc_config['length'][2]],
                        'length2': [self.mcmc_config['length'][0], self.mcmc_config['length'][1],
                                    self.mcmc_config['length'][2]],
                        'angle1': [self.mcmc_config['angle1'][0], self.mcmc_config['angle1'][1],
                                   self.mcmc_config['angle1'][2]],
                        'angle2': [self.mcmc_config['angle2'][0], self.mcmc_config['angle2'][1],
                                   self.mcmc_config['angle2'][2]],
                        'fwhm_x1': [self.mcmc_config['fwhm_x'][0], self.mcmc_config['fwhm_x'][1],
                                    self.mcmc_config['fwhm_x'][2]],
                        'fwhm_y1': [self.mcmc_config['fwhm_y'][0], self.mcmc_config['fwhm_y'][1],
                                    self.mcmc_config['fwhm_y'][2]],
                        'fwhm_x2': [self.mcmc_config['fwhm_x'][0], self.mcmc_config['fwhm_x'][1],
                                    self.mcmc_config['fwhm_x'][2]],
                        'fwhm_y2': [self.mcmc_config['fwhm_y'][0], self.mcmc_config['fwhm_y'][1],
                                    self.mcmc_config['fwhm_y'][2]]}
        else:
            if self.mcmc_config['redo'] or np.any([len(self.data[self.mcmc_config['mcmcmwcs_pos']][label]) == 0 for label in
                               ['cor_spot_ref', 'conex_ref']]):
                mcmcwcs.get_satellite_spots_and_coronograph(datas[self.mcmc_config['ref_sat_spot_pos']],
                                                            headers[self.mcmc_config['ref_sat_spot_pos']])

            self.mcmc_setup['positions_ref'] = [np.float64(self.data[self.mcmc_config['mcmcmwcs_pos']]['cor_spot_ref'])]
            self.mcmc_setup['coronograph_ref'] = np.float64(self.data[self.mcmc_config['mcmcmwcs_pos']]['cor_spot_ref'])
            self.mcmc_setup['conex_xy_ref'] = np.float64(self.data[self.mcmc_config['mcmcmwcs_pos']]['conex_ref'])
            self.mcmc_setup['slopes'] = np.float64(self.data[self.mcmc_config['mcmcmwcs_pos']]['slopes'])

            self.mcmc_setup['pos']  = {'amplitude': [pipe_dict['mcmcwcssol']['amplitude'][0], pipe_dict['mcmcwcssol']['amplitude'][1],
                                      pipe_dict['mcmcwcssol']['amplitude'][2]],
                        'fwhm_x1': [pipe_dict['mcmcwcssol']['fwhm_x'][0], pipe_dict['mcmcwcssol']['fwhm_x'][1],
                                    pipe_dict['mcmcwcssol']['fwhm_x'][2]],
                        'fwhm_y1': [pipe_dict['mcmcwcssol']['fwhm_y'][0], pipe_dict['mcmcwcssol']['fwhm_y'][1],
                                    pipe_dict['mcmcwcssol']['fwhm_y'][2]]}

        config.dump_dataconfig(self.data, self.paths['data_cfg'])



    def fitting_paprameters(self,datas, headers):
            print('> Fitting parameters')

            num_of_chunks = 3 * self.mcmc_config['ncpu']
            chunksize = self.mcmc_setup['ntargets'] // num_of_chunks
            if chunksize <= 0:
                chunksize = 1

            if self.mcmc_config['parallel_runs']:
                print('> parallel runs %s, number of parallel runs %i, ncpu per target %i, ntargets %i, chunksize %i' % (self.mcmc_config['parallel_runs'], self.mcmc_config['ncpu'],self.mcmc_config['mcmc_ncpu_multiplier'], self.mcmc_setup['ntargets'], chunksize))
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.mcmc_config['ncpu']) as executor:
                    for _ in tqdm(executor.map(mcmcwcs.mcmc_task, self.mcmc_setup['data_names'], datas, headers, repeat(self.mcmc_config['mcmc_ncpu_multiplier']),
                                               chunksize=chunksize)):
                        pass
            else:
                print('> parallel_runs %s, ncpu per target %i ,ntargets %i' % (self.mcmc_config['parallel_runs'], self.mcmc_config['mcmc_ncpu_multiplier']*self.mcmc_config['ncpu'], self.mcmc_setup['ntargets']))
                for elno in tqdm(range(self.mcmc_setup['ntargets'])):
                    mcmcwcs.mcmc_task(self.mcmc_setup['data_names'][elno], datas[elno], headers[elno],self.mcmc_config['mcmc_ncpu_multiplier']*self.mcmc_config['ncpu'])


    def lsq_fit_dpdc(self,d,labels,path2savedir=None,filename='test.jpg',showplot=True,verbose=True,ext='_'):
        x = d[labels[0]]
        y= d[labels[1]]
        yerr = np.array([d[labels[2]]]*len(y))
        x0 = np.linspace(x.min(), x.max(), 500)
        A = np.vander(x, 2)
        C = np.diag(yerr * yerr)
        ATA = np.dot(A.T, A / (yerr**2)[:, None])
        cov = np.linalg.inv(ATA)
        w = np.linalg.solve(ATA, np.dot(A.T, y / yerr**2))
        if verbose:
            print("Least-squares estimates:")
            print("dp_dc = {0:.3f} ± {1:.3f}".format(w[0], np.sqrt(cov[0, 0])))
            print("pixel_at_conex_0 = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))
    
    
        fig=plt.figure()
        plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
        plt.plot(x0, np.dot(np.vander(x0, 2), [w[0],w[1]]), "-k", label="LSQ:\n"+"dpdc = {0:.3f} ± {1:.3f} \n".format(w[0], np.sqrt(cov[0, 0]))+"pc0 = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))
        plt.legend(fontsize=12)
        plt.xlabel("conex")
        plt.ylabel("pixel_at_conex");
        if path2savedir is not None: fig.savefig(path2savedir+'fit%s%s'%(ext,filename))
    
        if showplot: plt.show()
        else: plt.close()
    
        return([[np.round(w[0],3), np.round(np.sqrt(cov[0, 0]),3)],[np.round(w[1],3), np.round(np.sqrt(cov[1, 1]),3)]])
    
    def get_equidistant_points(self,p1, p2, parts):
        return zip(np.linspace(p1[0], p2[0], parts+1),
                   np.linspace(p1[1], p2[1], parts+1))

    def mcmc_task(self,h5_name,data,header,ncpu):
        filename = h5_name
        xyCons = [header['E_CONEXX'],header['E_CONEXY']]
        cen_xy=np.round(CONEX2PIXEL(xyCons[0],
                                    xyCons[1],
                                    self.mcmc_setup['slopes'],
                                    self.mcmc_setup['coronograph_ref'],
                                    self.mcmc_setup['conex_xy_ref']),2)

        self.mcmc_setup['pos']['cen_x'] = [cen_xy[0], cen_xy[0] - 5, cen_xy[0] + 5]
        self.mcmc_setup['pos']['cen_y'] = [cen_xy[1], cen_xy[1] - 5, cen_xy[1] + 5]

        masked_img=create_mask(data,xyCons,self.mcmc_setup['slopes'],self.mcmc_setup['positions_ref'],
                               self.mcmc_setup['conex_xy_ref'],self.mcmc_config['factor'])
        d=np.nanmedian(data[data>0])

        mcmc=MCMC_FIT(path=self.paths['MCMC_fit'], nwalkers=self.mcmc_config['nwalkers'], steps=self.mcmc_config['steps'],
                      ndesired=100, ncpu=ncpu,
                      progress=self.mcmc_config['progress'], verbose=self.mcmc_config['verbose'],
                      const=d, sat_spots=self.mcmc_config['sat_spots'],
                      kwargs={'shape_xy': np.array(data.shape[::-1])-1, 'factor': self.mcmc_config['factor']})
        mcmc.run(filename, self.mcmc_setup['pos'], masked_img)

    def sample_posteriors_task(self,data,header,h5_name,mcmc_config):
        with io.capture_output() as captured:

            d=np.nanmedian(data[data>0])

            xyCons=[float(header['E_CONEXX']), float(header['E_CONEXY'])]
            mcmc=MCMC_FIT(path=self.paths['MCMC_fit'], ncpu=self.mcmc_config['ncpu']*self.mcmc_config['mcmc_ncpu_multiplier'], progress=False, verbose=False,labels=self.mcmc_setup['mcmc_labels'])
            s=mcmc.sample_posteriors(h5_name, self.mcmc_setup['slopes'], full_posterior=True, verbose=False,save_output=True)
            pixel_cen=[s['cen_x'][0],s['cen_y'][0]]
            epixel_cen=[np.mean(s['cen_x'][1:3]),np.mean(s['cen_y'][1:3])]

            sp=SATSPOT_MODEL(np.array(data.shape[::-1])-1,factor=mcmc_config['factor'])
            masked_img=create_mask(data,xyCons,self.mcmc_setup['slopes'],self.mcmc_setup['positions_ref'],self.mcmc_setup['conex_xy_ref'],self.mcmc_config['factor'])
            if mcmc_config['sat_spots']: sp.create_psf_image(s['amplitude'][0],[s['cen_x'][0],s['cen_y'][0]], [[s['fwhm_x1'][0], s['fwhm_y1'][0]]],lengths_list=[s['length1'][0],s['length2'][0]],angles_list=[s['angle1'][0],s['angle2'][0]],d=d,mask=masked_img,flux=np.nansum(masked_img),sat_spots=mcmc_config['sat_spots'])
            else: sp.create_psf_image(s['amplitude'][0],[s['cen_x'][0],s['cen_y'][0]], [[s['fwhm_x1'][0], s['fwhm_y1'][0]]],d=d,mask=masked_img,flux=np.nansum(masked_img),sat_spots=mcmc_config['sat_spots'])
            chi2_map = (masked_img - sp.psfs_img) ** 2 / (sp.psfs_img+0.00000001)
            chi2_map[~np.isfinite(chi2_map)] = 0
            data_list=np.array([masked_img,sp.psfs_img,chi2_map/np.nanmax(data)])
            norm = [simple_norm(masked_img, stretch='sqrt', min_cut=mcmc_config['v_lim'][0][0], max_cut=mcmc_config['v_lim'][0][1]),simple_norm(sp.psfs_img, stretch='sqrt', min_cut=mcmc_config['v_lim'][1][0], max_cut=mcmc_config['v_lim'][1][1]),simple_norm(chi2_map/np.nanmax(data), stretch='sqrt', min_cut=mcmc_config['v_lim'][2][0], max_cut=mcmc_config['v_lim'][2][1])]

            sp.plot_image(np.array(data_list),
                      title=['MaskedData', 'Model', 'Chi2'], cen_xy=[s['cen_x'][0],s['cen_y'][0]], satspot_xy=sp.satspots_xy, norm=norm, rows=1, path2savedir=self.paths['MCMC_fit'] + 'plots/'+'debug/',filename="%s_MCMC_fit.jpg"%h5_name.split('.')[0],save_output=True)

        return(pixel_cen,epixel_cen,xyCons)

    def get_slope_and_conex(self,datas,headers):
        print('> Getting slope and conex postition from images.')
        N=int(self.mcmc_config['ref_el'])
        if N != 2: selected_pos=self.mcmc_setup['sorted_data_pos'][::int(np.ceil( len(self.mcmc_setup['sorted_data_pos']) / N ))]
        else: selected_pos=[self.mcmc_setup['sorted_data_pos'][0],self.mcmc_setup['sorted_data_pos'][-1]]
        print('> Selected N reference = %i, closest number of equidistant element = %i' % (
            N, len(selected_pos)))

        conex_ref_list=[]
        coronograph_ref_list=[]

        print(selected_pos)
        for elno in selected_pos:
            data=datas[elno]
            header=headers[elno]
            coords = select_sources(data, n_satspots=0)

            positions, coronograph = [coords[0], coords[0]]
            conex_xy = [float(header['E_CONEXX']), float(header['E_CONEXY'])]
            conex_ref_list.append(conex_xy)
            coronograph_ref_list.append(coronograph)

        coronograph_ref_list=np.array(coronograph_ref_list)
        conex_ref_list=np.array(conex_ref_list)

        self.mcmc_setup['guesses'] = {'conexx': conex_ref_list[:, 0],
             'conexy': conex_ref_list[:, 1],
             'pixel_at_conex_x': coronograph_ref_list[:, 0],
             'pixel_at_conex_y': coronograph_ref_list[:, 1],
             'std_pixel_at_conex_x':1,
             'std_pixel_at_conex_y':1}

        sol_x = mcmcwcs.lsq_fit_dpdc(self.mcmc_setup['guesses'], ['conexx', 'pixel_at_conex_x', 'std_pixel_at_conex_x'],
                                     showplot=mcmcwcs.mcmc_config['verbose'],
                                     verbose=mcmcwcs.mcmc_config['verbose'],
                                     path2savedir=self.path['MCMC_fit'] + 'plots/', ext='_x_test_')
        sol_y = mcmcwcs.lsq_fit_dpdc(self.mcmc_setup['guesses'], ['conexy', 'pixel_at_conex_y', 'std_pixel_at_conex_y'],
                                     showplot=mcmcwcs.mcmc_config['verbose'],
                                     verbose=mcmcwcs.mcmc_config['verbose'],
                                     path2savedir=self.path['MCMC_fit'] + 'plots/', ext='_y_test_')

        slopes = [float(np.round(sol_x[0][0],2)),float(np.round(sol_y[0][0],2))]
        self.data[self.mcmc_config['mcmcmwcs_pos']]['slopes'] =[float(np.round(x,2)) for x in slopes]

    def get_satellite_spots_and_coronograph(self,data,header):
        if self.pipe['mcmcwcssol']['sat_spots']:
            print('> Getting satellite spots and coronograph postions from images.')
            n_satspots=4
        else:
            print('> Getting coronograph postions from images.')
            n_satspots=0

        coords = select_sources(data, n_satspots=n_satspots)
        positions_ref, coronograph_ref = [coords[:-1], coords[-1]]
        conex_xy_ref = [float(header['E_CONEXX']), float(header['E_CONEXY'])]

        if self.pipe['mcmcwcssol']['sat_spots']:
            self.data[self.mcmc_config['mcmcmwcs_pos']]['spot_ref1'] = [float(np.round(x, 2)) for x in positions_ref[0]]
            self.data[self.mcmc_config['mcmcmwcs_pos']]['spot_ref2'] = [float(np.round(x, 2)) for x in positions_ref[1]]
            self.data[self.mcmc_config['mcmcmwcs_pos']]['spot_ref3'] = [float(np.round(x, 2)) for x in positions_ref[2]]
            self.data[self.mcmc_config['mcmcmwcs_pos']]['spot_ref4'] = [float(np.round(x, 2)) for x in positions_ref[3]]

        self.data[self.mcmc_config['mcmcmwcs_pos']]['cor_spot_ref'] = [float(np.round(x, 2)) for x in coronograph_ref]
        self.data[self.mcmc_config['mcmcmwcs_pos']]['conex_ref'] = [float(np.round(x, 2)) for x in conex_xy_ref]

    def make_outputs(self,datas,heders):
        print('> Working on outputs')
        print('> Looking for data in %s' % self.paths['MCMC_fit'])

        pixel_cen_list = []
        epixel_cen_list = []
        conexx_list = []
        if self.mcmc_config['sat_spots']:
            self.mcmc_setup['mcmc_labels'] = ['amplitude', 'length1', 'length2', 'angle1', 'angle2', 'fwhm_x1', 'fwhm_y1', 'fwhm_x2',
                           'fwhm_y2', 'cen_x',
                           'cen_y']
        else:
            self.mcmc_setup['mcmc_labels'] = ['amplitude', 'fwhm_x1', 'fwhm_y1', 'cen_x', 'cen_y']

        print('> Working on data:')
        for data,header,h5_name in tqdm(zip(datas,heders,self.mcmc_setup['data_names'])):
            pixel_cen, epixel_cen, xyCons = mcmcwcs.sample_posteriors_task(data,header,h5_name,self.mcmc_config)
            # out_filename_list.append(filename)
            pixel_cen_list.append(pixel_cen)
            epixel_cen_list.append(epixel_cen)
            conexx_list.append(xyCons)

        pixel_cen_list = np.array(pixel_cen_list)
        epixel_cen_list = np.array(epixel_cen_list)
        conexx_list = np.array(conexx_list)

        d = {'conexx': conexx_list[:, 0],
             'conexy': conexx_list[:, 1],
             'pixel_at_conex_x': pixel_cen_list[:, 0],
             'pixel_at_conex_y': pixel_cen_list[:, 1],
             'epixel_at_conex_x': epixel_cen_list[:, 0],
             'epixel_at_conex_y': epixel_cen_list[:, 1]}

        d['std_pixel_at_conex_x'] = np.nanmean(
            [np.std(d['pixel_at_conex_x'][np.where(d['conexx'] == conexx)[0]]) for conexx in set(d['conexx'])])

        d['std_pixel_at_conex_y'] = np.nanmean(
            [np.std(d['pixel_at_conex_y'][np.where(d['conexy'] == conexy)[0]]) for conexy in set(d['conexy'])])

        sol_x = mcmcwcs.lsq_fit_dpdc(d, ['conexx', 'pixel_at_conex_x', 'std_pixel_at_conex_x'], showplot=self.mcmc_config['verbose'],
                             verbose=self.mcmc_config['verbose'],
                             path2savedir=self.paths['MCMC_fit'] + 'plots/', ext='_x_')
        sol_y = mcmcwcs.lsq_fit_dpdc(d, ['conexy', 'pixel_at_conex_y', 'std_pixel_at_conex_y'], showplot=self.mcmc_config['verbose'],
                             verbose=self.mcmc_config['verbose'],
                             path2savedir=self.paths['MCMC_fit'] + 'plots/', ext='_y_')

        dout = {'x': {'dpdc': [float(np.round(i, 2)) for i in sol_x[0]],
                      'pc0': [float(np.round(i, 2)) for i in sol_x[1]],
                      'conex': 0},
                'y': {'dpdc': [float(np.round(i, 2)) for i in sol_y[0]],
                      'pc0': [float(np.round(i, 2)) for i in sol_y[1]],
                      'conex': 0}}

        self.data[self.mcmc_config['mcmcmwcs_pos']]['sol'] = dout

        self.data[self.mcmc_config['mcmcmwcs_pos']]['pixel_ref'] = [float(dout['x']['pc0'][0]), float(dout['y']['pc0'][0])]
        self.data[self.mcmc_config['mcmcmwcs_pos']]['conex_ref'] = [float(dout['x']['conex']), float(dout['y']['conex'])]
        self.data[self.mcmc_config['mcmcmwcs_pos']]['dp_dcx'] = float(dout['x']['dpdc'][0])
        self.data[self.mcmc_config['mcmcmwcs_pos']]['dp_dcy'] = float(dout['y']['dpdc'][0])

if __name__ == '__main__':
    def parse():
        # read in command line arguments
        parser = argparse.ArgumentParser(description='MKID Pipeline CLI')
        parser.add_argument('-p', type=str, help='A pipeline config file', default='./pipe.yaml', dest='pipe_cfg')
        parser.add_argument('-d', type=str, help='A input config file', default='./data.yaml', dest='data_cfg')
        parser.add_argument('--make-dir', dest='make_paths', help='Create all needed directories', action='store_true')
        parser.add_argument('--verbose', action='store_true', help='Verbose', dest='verbose')
        return parser.parse_args()

    ############################# VARIABLES DEFINITION ########################################

    args = parse()
    config.configure_pipeline(args.pipe_cfg)
    YAML = ruamel.yaml.YAML()
    mcmcwcs=MCMCWCS(args.pipe_cfg, args.data_cfg, args.verbose)

    if args.make_paths:
        mcmcwcs.make_dir()

    #################################### LOADING DATA #################################################
    mcmcwcs.fetching_h5_names()
    datas, headers = mcmcwcs.fetching_datas()

    ############################################################## PARAMETERS FIT #################################################################
    mcmcwcs.fetching_mcmc_parameters(datas,headers)
    w=np.where([mcmcwcs.paths['MCMC_fit']+i not in glob(mcmcwcs.paths['MCMC_fit']+'*.h5') for i in mcmcwcs.mcmc_setup['data_names']])[0]
    if mcmcwcs.mcmc_config['redo'] or len(w) > 0:
        mcmcwcs.fitting_paprameters([datas[i] for i in w.tolist()],[headers[i] for i in w.tolist()])

    #################################################### SAMPLIG POSTERIOR/MAKING OUTPUT ###########################################################
    mcmcwcs.make_outputs(datas,headers)

    #################################################### SAVE FINAL DATA YAML ###########################################################
    config.dump_dataconfig(mcmcwcs.data, mcmcwcs.paths['data_cfg'])

