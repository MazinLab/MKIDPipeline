# import mkidpipeline.definitions as definitions
import mkidcore
import mkidcore.config
# from mkidcore.corelog import getLogger
# import mkidpipeline.pipeline as pipe
import mkidpipeline.config as config

# import mkidcore.config
import mkidpipeline.steps as steps
from mkidcore.instruments import CONEX2PIXEL
from mkidpipeline.photontable import Photontable
# import mkidpipeline.config as config

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
        # self.source_locs = source_locs
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
        # except IndexError:
        #     pass
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

        # for elno in range(len(angles_list)):
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
        # if v_lim is None:
        #     v_lim = np.array([[None, None]] * data.shape[0])
        # elif len(v_lim) ==2:
        #     v_lim = np.array([[v_lim[0], v_lim[1]]] * data.shape[0])

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
            # if not os.path.exists(path2savedir):
            #     os.makedirs(path2savedir)
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

        # if not os.path.exists(path):
        #     os.makedirs(path)

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
                # print('VVVVVVVVVVVVVVVVVVVVVV')
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
                # print(index,tau * self.Fconv , sampler.iteration,tau * self.Fconv < sampler.iteration, (np.abs(old_tau - tau) / tau) , self.conv_thr, (np.abs(old_tau - tau) / tau) < self.conv_thr,converged)
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

                # elif index >= int(self.steps / self.check_acor):
                #     break
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

        # pixel_at_Con = CONEX2PIXEL(0, 0, slopes, [sol['cen_x'][0], sol['cen_y'][0]], [conex_x, conex_y])
        # if verbose:
        #     txt = "\mathrm{{{3}}} = {0:.5f}_{{-{1:.5f}}}^{{{2:.5f}}}"
        #     txt = txt.format(pixel_at_Con[0], sol['cen_x'][1], sol['cen_x'][2], 'x_{conex}')
        #     display(Math(txt))
        #     txt = "\mathrm{{{3}}} = {0:.5f}_{{-{1:.5f}}}^{{{2:.5f}}}"
        #     txt = txt.format(pixel_at_Con[1], sol['cen_y'][1], sol['cen_y'][2], 'y_{conex}')
        #     display(Math(txt))
        #
        # sol['pixel_x_at_Con_0']=[pixel_at_Con[0], sol['cen_x'][1], sol['cen_x'][2]]
        # sol['pixel_y_at_Con_0']=[pixel_at_Con[1], sol['cen_y'][1], sol['cen_y'][2]]

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
                # if not os.path.exists(self.path + '/plots/posterior/'):
                #     # Create a new directory because it does not exist
                #     os.makedirs(self.path + '/plots/posterior/')
                fig.savefig(self.path + '/plots/posterior/%s' % filename.split('.h5')[0] + '.jpg')

            fig = corner.corner(flat_samples,
                                labels= self.labels,
                                quantiles=[0.16, 0.5, 0.84],
                                show_titles=True,
                                title_kwargs={"fontsize": 12})

            if not save_output: plt.show()
            else:
                # if not os.path.exists(self.path + '/plots/corners/'):
                #     # Create a new directory because it does not exist
                #     os.makedirs(self.path + '/plots/corners/')
                fig.savefig(self.path + '/plots/corners/%s'%filename.split('.h5')[0]+'.jpg')
                plt.close('all')
        else: plt.close('all')

        return(sol)

class MCMCWCS:
    def __init__(self, pipe_cfg, data_cfg,verbose):

        self.data_dict = mkidcore.config.load(data_cfg)
        self.pipe_dict = mkidcore.config.load(pipe_cfg)

        self.path2out = self.pipe_dict['paths']['out']
        self.dither_path = self.pipe_dict['paths']['data']
        self.nwalkers = self.pipe_dict['mcmcwcssol']['nwalkers']
        self.steps = self.pipe_dict['mcmcwcssol']['steps']
        self.progress = self.pipe_dict['mcmcwcssol']['progress']
        self.workers = self.pipe_dict['mcmcwcssol']['workers']
        self.sat_spots = self.pipe_dict['mcmcwcssol']['sat_spots']
        self.ref_el = self.pipe_dict['mcmcwcssol']['ref_el']
        self.v_lim = self.pipe_dict['mcmcwcssol']['v_lim']
        self.factor = self.pipe_dict['mcmcwcssol']['factor']
        self.redo = self.pipe_dict['mcmcwcssol']['redo']
        if not verbose:
            self.verbose = self.pipe_dict['mcmcwcssol']['verbose']
        else:
            self.verbose = verbose

        self.data_elno = [index for (index, d) in enumerate(self.data_dict) if '!MKIDMCMCWCSol' in d.tag.value][0]

        self.out_data_list = self.data_dict[self.data_elno]['data'].split('+')

        self.data_elno_list = [index for (index, d) in enumerate(self.data_dict) for out_data in self.out_data_list if
                          out_data in d['name']]
        self.data_list = [self.data_dict[data_elno]['data'] for data_elno in self.data_elno_list[:5]]

        self.path2MCMC_fit = self.path2out + 'MCMC_fit/'
        # self.path2PLOTdir = self.path2MCMC_fit + 'plots/'
        # self.path2CORNERdir = self.path2MCMC_fit + 'plots/corners/'
        # self.path2DEBUGdir = self.path2MCMC_fit + 'plots/debug/'
        # self.path2POSTERIORdir = self.path2MCMC_fit + 'plots/posterior/'

        # return (data_dict, dither_path, nwalkers, steps, progress, workers, sat_spots, ref_el, v_lim, factor, redo,
        #         verbose, data_elno, data_list, path2out)

    def make_dir(self):
        if not os.path.exists(self.path2out + 'MCMC_fit/plots/posterior/'):
            print('> Making %s' % self.path2out + 'MCMC_fit/')
            os.makedirs(self.path2out + 'MCMC_fit/')
        if not os.path.exists(self.path2out + 'MCMC_fit/plots/'):
            print('> Making %s' % self.path2out + 'MCMC_fit/plots/')
            os.makedirs(self.path2out + 'MCMC_fit/plots/')
            print('> Making %s' % self.path2out + 'MCMC_fit/plots/corners/')
            os.makedirs(self.path2out + 'MCMC_fit/plots/corners/')
            print('> Making %s' % self.path2out + 'MCMC_fit/plots/debug/')
            os.makedirs(self.path2out + 'MCMC_fit/plots/debug/')
            print('> Making %s' % self.path2out + 'MCMC_fit/plots/posterior/')
            os.makedirs(self.path2out + 'MCMC_fit/plots/posterior/')
    #
    # def fetching_data(self, path2out, out_data_list, data_list, dither_path, workers):
    #     start_time_list = []
    #     print('> Building list of start times from: %s' % out_data_list)
    #     for name, data in zip(out_data_list, data_list):
    #         print('>> Looking for data associated to %s containing time %s,...' % (name, data))
    #         startt, _, _ = mkidcore.utils.get_ditherdata_for_time(dither_path, data)
    #         start_time_list.extend([np.round(i) for i in startt])
    #         print('>> Found %i dithers.' % len(startt))
    #
    #     # start_time_list=start_time_list[::10]
    #     image_list = []
    #     header_list = []
    #     dist_list = []
    #     elno_list = []
    #     h5_name_list = [int(i.split('/')[-1].split('.h5')[0]) for i in glob(path2out + '*.h5')]
    #     filename_list = []
    #
    #     ntargets = len(start_time_list)
    #     num_of_chunks = 3 * workers
    #     chunksize = ntargets // num_of_chunks
    #     if chunksize <= 0:
    #         chunksize = 1
    #
    #     elno = 0
    #     if workers == 1:
    #         workers_load = 10
    #     else:
    #         workers_load = workers
    #     print('> Using %i workers to load a total of %i files ...' % (workers_load, ntargets))
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=workers_load) as executor:
    #         for filename, header, data in tqdm(executor.map(load_fits_task, start_time_list, chunksize=chunksize)):
    #             elno += 1
    #             filename_list.append(filename)
    #             header_list.append(header)
    #             image_list.append(data)
    #             dist_list.append(np.sqrt((header_list[0]['E_CONEXX'] - header['E_CONEXX']) ** 2 + (
    #                     header_list[0]['E_CONEXY'] - header['E_CONEXY']) ** 2))
    #             elno_list.append(elno)
    #
    #     sorted_elno_list = [x for _, x in sorted(zip(dist_list, elno_list))]
    #     return (ntargets, filename_list, header_list, image_list, dist_list, elno_list, sorted_elno_list)
    #
    # def fetching_mcmc_parameters(self):
    #     print('> Fitting parameters')
    #     ntargets = len(filename_list)
    #
    #     num_of_chunks = 3 * workers
    #     chunksize = ntargets // num_of_chunks
    #     if chunksize <= 0:
    #         chunksize = 1
    #
    #     if redo or np.any([len(getattr(data_dict[data_elno], label)) == 0 for label in ['slopes']]):
    #         data_dict = get_slope_and_conex(data_elno, data_dict, sorted_elno_list, filename_list, image_list,
    #                                         header_list, ref_el)
    #
    #     if sat_spots:
    #         if redo or np.any([len(getattr(data_dict[data_elno], label)) == 0 for label in
    #                            ['spot_ref1', 'spot_ref2', 'spot_ref3', 'spot_ref4', 'cor_spot_ref', 'conex_ref']]):
    #             data_dict = get_satellite_spots_and_coronograph(data_elno, data_dict, image_list, header_list,
    #                                                             sat_spots)
    #
    #         positions_ref = [np.float64(data_dict[data_elno].spot_ref1),
    #                          np.float64(data_dict[data_elno].spot_ref2),
    #                          np.float64(data_dict[data_elno].spot_ref3),
    #                          np.float64(data_dict[data_elno].spot_ref4)]
    #         coronograph_ref = np.float64(data_dict[data_elno].cor_spot_ref)
    #         conex_xy_ref = np.float64(data_dict[data_elno].conex_ref)
    #         slopes = np.float64(data_dict[data_elno].slopes)
    #
    #         pos_dict = {'amplitude': [pipe_dict['mcmcwcssol']['amplitude'][0], pipe_dict['mcmcwcssol']['amplitude'][1],
    #                                   pipe_dict['mcmcwcssol']['amplitude'][2]],
    #                     'length1': [pipe_dict['mcmcwcssol']['length'][0], pipe_dict['mcmcwcssol']['length'][1],
    #                                 pipe_dict['mcmcwcssol']['length'][2]],
    #                     'length2': [pipe_dict['mcmcwcssol']['length'][0], pipe_dict['mcmcwcssol']['length'][1],
    #                                 pipe_dict['mcmcwcssol']['length'][2]],
    #                     'angle1': [pipe_dict['mcmcwcssol']['angle1'][0], pipe_dict['mcmcwcssol']['angle1'][1],
    #                                pipe_dict['mcmcwcssol']['angle1'][2]],
    #                     'angle2': [pipe_dict['mcmcwcssol']['angle2'][0], pipe_dict['mcmcwcssol']['angle2'][1],
    #                                pipe_dict['mcmcwcssol']['angle2'][2]],
    #                     'fwhm_x1': [pipe_dict['mcmcwcssol']['fwhm_x'][0], pipe_dict['mcmcwcssol']['fwhm_x'][1],
    #                                 pipe_dict['mcmcwcssol']['fwhm_x'][2]],
    #                     'fwhm_y1': [pipe_dict['mcmcwcssol']['fwhm_y'][0], pipe_dict['mcmcwcssol']['fwhm_y'][1],
    #                                 pipe_dict['mcmcwcssol']['fwhm_y'][2]],
    #                     'fwhm_x2': [pipe_dict['mcmcwcssol']['fwhm_x'][0], pipe_dict['mcmcwcssol']['fwhm_x'][1],
    #                                 pipe_dict['mcmcwcssol']['fwhm_x'][2]],
    #                     'fwhm_y2': [pipe_dict['mcmcwcssol']['fwhm_y'][0], pipe_dict['mcmcwcssol']['fwhm_y'][1],
    #                                 pipe_dict['mcmcwcssol']['fwhm_y'][2]]}
    #     else:
    #         if redo or np.any([len(getattr(data_dict[data_elno], label)) == 0 for label in
    #                            ['cor_spot_ref', 'conex_ref']]):
    #             data_dict = get_satellite_spots_and_coronograph(data_elno, data_dict, image_list, header_list,
    #                                                             sat_spots)
    #
    #         positions_ref = [np.float64(data_dict[data_elno].cor_spot_ref)]
    #         coronograph_ref = np.float64(data_dict[data_elno].cor_spot_ref)
    #         conex_xy_ref = np.float64(data_dict[data_elno].conex_ref)
    #         slopes = np.float64(data_dict[data_elno].slopes)
    #
    #         pos_dict = {'amplitude': [pipe_dict['mcmcwcssol']['amplitude'][0], pipe_dict['mcmcwcssol']['amplitude'][1],
    #                                   pipe_dict['mcmcwcssol']['amplitude'][2]],
    #                     'fwhm_x1': [pipe_dict['mcmcwcssol']['fwhm_x'][0], pipe_dict['mcmcwcssol']['fwhm_x'][1],
    #                                 pipe_dict['mcmcwcssol']['fwhm_x'][2]],
    #                     'fwhm_y1': [pipe_dict['mcmcwcssol']['fwhm_y'][0], pipe_dict['mcmcwcssol']['fwhm_y'][1],
    #                                 pipe_dict['mcmcwcssol']['fwhm_y'][2]]}
    #
    #     config.dump_dataconfig(data_dict, args.data_cfg)
    #
    #     if workers > 1:
    #         print('> workers %i,chunksize %i,ntargets %i' % (workers, chunksize, ntargets))
    #         with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
    #             for _ in tqdm(executor.map(mcmc_task, filename_list, image_list, header_list, repeat(pos_dict),
    #                                        chunksize=chunksize)): pass
    #     else:
    #         for elno in range(len(filename_list)): mcmc_task(filename_list[elno], image_list[elno], header_list[elno],
    #                                                          pos_dict)

    def lsq_fit_dpdc(self,d,labels,path2savedir=None,filename='test.jpg',showplot=True,verbose=True,ext='_'):
        # np.random.seed(42)
        x = d[labels[0]]
        y= d[labels[1]]
        print(y)
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
    
    def getEquidistantPoints(self,p1, p2, parts):
        return zip(np.linspace(p1[0], p2[0], parts+1),
                   np.linspace(p1[1], p2[1], parts+1))

    def load_fits_task(self,start_time):
        filename = min(h5_name_list, key=lambda x: abs(x - start_time))
        pt = Photontable(self.path2out + '%i.h5' % (filename))
        hdul = pt.get_fits(wave_start=950, wave_stop=1100, start=pt.start_time + start_offset)
        header = hdul[0].header
        data = hdul[1].data
        hdul.close()
        return (filename, header, data)

    def mcmc_task(self,name,data,header,pos_dict):
        filename = "%i_MCMC_fit.h5"%name
        pt = Photontable(self.path2out + '%i.h5' % (name))
        xyCons = [header['E_CONEXX'],header['E_CONEXY']]
        cen_xy=np.round(CONEX2PIXEL(xyCons[0],
                                    xyCons[1],
                                    slopes,
                                    coronograph_ref,
                                    conex_xy_ref),2)

        pos_dict['cen_x'] = [cen_xy[0], cen_xy[0] - 5, cen_xy[0] + 5]
        pos_dict['cen_y'] = [cen_xy[1], cen_xy[1] - 5, cen_xy[1] + 5]
        masked_img=create_mask(data,xyCons,slopes,positions_ref,conex_xy_ref,factor)
        d=np.nanmedian(data[data>0])

        mcmc=MCMC_FIT(path=self.path2out + 'MCMC_fit/', nwalkers=nwalkers, steps=steps, ndesired=100, ncpu=20, progress=progress, verbose=verbose,
                      const=d, sat_spots=sat_spots , kwargs={'shape_xy': np.array(data.shape[::-1])-1, 'factor' : factor})
        mcmc.run(filename, pos_dict, masked_img)

    def sample_posteriors_task(self,filename):
        with io.capture_output() as captured:
            MCMC_filename = "%i_MCMC_fit.h5"%filename
            pt = Photontable(self.path2out+'%i.h5'%filename)
            hdul = pt.get_fits(wave_start=950,wave_stop=1100,start=pt.start_time+start_offset)
            header = hdul[0].header
            data = hdul[1].data
            d=np.nanmedian(data[data>0])

            xyCons=[float(header['E_CONEXX']), float(header['E_CONEXY'])]
            mcmc=MCMC_FIT(path=self.path2out + 'MCMC_fit/', ncpu=1, progress=False, verbose=False,labels=MCMC_labels)
            s=mcmc.sample_posteriors(MCMC_filename, slopes, full_posterior=True, verbose=False,save_output=True)
            pixel_cen=[s['cen_x'][0],s['cen_y'][0]]
            epixel_cen=[np.mean(s['cen_x'][1:3]),np.mean(s['cen_y'][1:3])]

            sp=SATSPOT_MODEL(np.array(data.shape[::-1])-1,factor=factor)
            masked_img=create_mask(data,xyCons,slopes,positions_ref,conex_xy_ref,factor)
            if sat_spots: sp.create_psf_image(s['amplitude'][0],[s['cen_x'][0],s['cen_y'][0]], [[s['fwhm_x1'][0], s['fwhm_y1'][0]]],lengths_list=[s['length1'][0],s['length2'][0]],angles_list=[s['angle1'][0],s['angle2'][0]],d=d,mask=masked_img,flux=np.nansum(masked_img),sat_spots=sat_spots)
            else: sp.create_psf_image(s['amplitude'][0],[s['cen_x'][0],s['cen_y'][0]], [[s['fwhm_x1'][0], s['fwhm_y1'][0]]],d=d,mask=masked_img,flux=np.nansum(masked_img),sat_spots=sat_spots)
            chi2_map = (masked_img - sp.psfs_img) ** 2 / (sp.psfs_img+0.00000001)
            chi2_map[~np.isfinite(chi2_map)] = 0
            data_list=np.array([masked_img,sp.psfs_img,chi2_map/np.nanmax(data)])
            norm = [simple_norm(masked_img, stretch='sqrt', min_cut=v_lim[0][0], max_cut=v_lim[0][1]),simple_norm(sp.psfs_img, stretch='sqrt', min_cut=v_lim[1][0], max_cut=v_lim[1][1]),simple_norm(chi2_map/np.nanmax(data), stretch='sqrt', min_cut=v_lim[2][0], max_cut=v_lim[2][1])]

            sp.plot_image(np.array(data_list),
                      title=['MaskedData', 'Model', 'Chi2'], cen_xy=[s['cen_x'][0],s['cen_y'][0]], satspot_xy=sp.satspots_xy, norm=norm, rows=1, path2savedir=self.path2out + 'MCMC_fit/' + 'plots/'+'debug/',filename="%i_MCMC_fit.jpg"%filename,save_output=True)
            return(filename,pixel_cen,epixel_cen,xyCons)

    def get_slope_and_conex(self,data_elno,data_dict,sorted_elno_list,filename_list,image_list,header_list,ref_el):
        print('> Getting slope and conex postition from images.')
        N=int(ref_el)
        # if len(data_dict[data_elno].slopes) == 0:
        if N != 2: selected_elno_list=sorted_elno_list[::int(np.ceil( len(sorted_elno_list) / N ))]
        else: selected_elno_list=[sorted_elno_list[0],sorted_elno_list[-1]]
        print('> Selected N reference = %i, closest number of equidistant element = %i' % (
            N, len(selected_elno_list)))
        # else:
        #     selected_elno_list=[sorted_elno_list[0]]
        #     print('> Selected N reference = %i, closest number of equidistant element = %i' % (
        #         N, len(selected_elno_list)))
        conex_ref_list=[]
        coronograph_ref_list=[]

        store=True
        for elno in selected_elno_list:
            data=image_list[elno]
            header=header_list[elno]
            coords = select_sources(data, n_satspots=0)

            positions, coronograph = [coords[0], coords[0]]
            conex_xy = [float(header['E_CONEXX']), float(header['E_CONEXY'])]
            conex_ref_list.append(conex_xy)
            coronograph_ref_list.append(coronograph)
            # if store:
            #     data_dict[data_elno].cor_spot_ref = [float(np.round(x,2)) for x in coronograph]
            #     data_dict[data_elno].conex_ref = [ float(np.round(x,2)) for x in conex_xy]
            #     positions_ref, coronograph_ref = [[np.array(positions)], list(coronograph)]
            #     conex_xy_ref = np.array(conex_xy)
            #     store = False

        # if redo: #len(data_dict[data_elno].slopes) == 0:
        coronograph_ref_list=np.array(coronograph_ref_list)
        conex_ref_list=np.array(conex_ref_list)

        d = {'conexx': conex_ref_list[:, 0],
             'conexy': conex_ref_list[:, 1],
             'pixel_at_conex_x': coronograph_ref_list[:, 0],
             'pixel_at_conex_y': coronograph_ref_list[:, 1]}

        d['std_pixel_at_conex_x'] = 1
        d['std_pixel_at_conex_y'] = 1

        sol_x = lsq_fit_dpdc(d, ['conexx', 'pixel_at_conex_x', 'std_pixel_at_conex_x'], showplot=verbose,
                             verbose=verbose,
                             path2savedir=self.path2out + 'MCMC_fit/' + 'plots/', ext='_x_test_')
        sol_y = lsq_fit_dpdc(d, ['conexy', 'pixel_at_conex_y', 'std_pixel_at_conex_y'], showplot=verbose,
                             verbose=verbose,
                             path2savedir=self.path2out + 'MCMC_fit/' + 'plots/', ext='_y_test_')

        slopes = [float(np.round(sol_x[0][0],2)),float(np.round(sol_y[0][0],2))]
        data_dict[data_elno].slopes =[float(np.round(x,2)) for x in slopes]
        return(data_dict)

    def get_satellite_spots_and_coronograph(self,data_elno,data_dict,image_list,header_list,sat_spots):
        if sat_spots:
            print('> Getting satellite spots and coronograph postions from images.')
            n_satspots=4
        else:
            print('> Getting coronograph postions from images.')
            n_satspots=0
        header = header_list[0] # hdul[0].header
        data = image_list[0] #hdul[1].data
        coords = select_sources(data, n_satspots=n_satspots)
        positions_ref, coronograph_ref = [coords[:-1], coords[-1]]
        conex_xy_ref = [float(header['E_CONEXX']), float(header['E_CONEXY'])]
        if sat_spots:
            data_dict[data_elno].spot_ref1 = [float(np.round(x, 2)) for x in positions_ref[0]]
            data_dict[data_elno].spot_ref2 = [float(np.round(x, 2)) for x in positions_ref[1]]
            data_dict[data_elno].spot_ref3 = [float(np.round(x, 2)) for x in positions_ref[2]]
            data_dict[data_elno].spot_ref4 = [float(np.round(x, 2)) for x in positions_ref[3]]

        data_dict[data_elno].cor_spot_ref = [float(np.round(x, 2)) for x in coronograph_ref]
        data_dict[data_elno].conex_ref = [float(np.round(x, 2)) for x in conex_xy_ref]

        return(data_dict)



if __name__ == '__main__':
    def parse():
        # read in command line arguments
        parser = argparse.ArgumentParser(description='MKID Pipeline CLI')
        # parser.add_argument('-o', type=str, help='An output specification file', default='./out.yaml', dest='out_cfg')
        parser.add_argument('-p', type=str, help='A pipeline config file', default='./pipe.yaml', dest='pipe_cfg')
        parser.add_argument('-d', type=str, help='A input config file', default='./data.yaml', dest='data_cfg')
        # parser.add_argument('-s', type=str, help='A config file for the solution', default='./dpdc_pc0.yaml', dest='dpdc_pc0')
        parser.add_argument('--make-dir', dest='make_paths', help='Create all needed directories', action='store_true')
        parser.add_argument('--verbose', action='store_true', help='Verbose', dest='verbose')
        parser.add_argument('--make-outputs-only', dest='makeout', help='Run the pipeline on the outputs only', action='store_true')
        return parser.parse_args()

    ############################# VARIABLES DEFINITION ########################################
    # global MCMC_labels, v_lim, coronograph_ref,  slopes, positions_ref, conex_xy_ref, factor,  path2MCMC_fit, path2out, sat_spots, start_offset, nwalkers, steps, progress, verbose, h5_name_list

    args = parse()
    config.configure_pipeline(args.pipe_cfg)
    YAML = ruamel.yaml.YAML()
    mcmcwcs=MCMCWCS(args.pipe_cfg, args.data_cfg, args.verbose)
    # config.configure_pipeline(args.pipe_cfg)
    #
    # data_dict = mkidcore.config.load(args.data_cfg)
    # pipe_dict = mkidcore.config.load(args.pipe_cfg)
    #
    # path2out = pipe_dict['paths']['out']
    # dither_path= pipe_dict['paths']['data']
    # nwalkers = pipe_dict['mcmcwcssol']['nwalkers']
    # steps = pipe_dict['mcmcwcssol']['steps']
    # progress = pipe_dict['mcmcwcssol']['progress']
    # workers = pipe_dict['mcmcwcssol']['workers']
    # sat_spots=pipe_dict['mcmcwcssol']['sat_spots']
    # ref_el=pipe_dict['mcmcwcssol']['ref_el']
    # v_lim = pipe_dict['mcmcwcssol']['v_lim']
    # factor = pipe_dict['mcmcwcssol']['factor']
    # redo = pipe_dict['mcmcwcssol']['redo']

    # if not args.verbose:
    #     verbose=pipe_dict['mcmcwcssol']['verbose']
    # else: verbose = args.verbose

    # data_elno= [index for (index, d) in enumerate(data_dict) if '!MKIDmcmcwcssol' in d.yaml_tag][0]
    #
    # attribute_list = ['spot_ref1', 'spot_ref2', 'spot_ref3', 'spot_ref4']
    #
    # wcscal = data_dict[data_elno].wcscal
    # start_offset = data_dict[data_elno].start_offset
    # out_data_list = data_dict[data_elno].data.split('+')
    #
    # data_elno_list = [index for (index, d) in enumerate(data_dict) for out_data in out_data_list if out_data in d.name]
    # data_list = [data_dict[data_elno].data for data_elno in data_elno_list[:5]]

    #################################### Lodading paths #################################################

    # warnings.filterwarnings("ignore")

    # path2MCMC_fit = path2out + 'MCMC_fit/'
    # path2PLOTdir = path2MCMC_fit + 'plots/'
    # path2CORNERdir = path2MCMC_fit + 'plots/corners/'
    # path2DEBUGdir = path2MCMC_fit + 'plots/debug/'
    # path2POSTERIORdir = path2MCMC_fit + 'plots/posterior/'

    if args.make_paths:
        mcmcwcs.make_dir()
        # if not os.path.exists(path2MCMC_fit):
        #     print('> Making %s'%path2MCMC_fit)
        #     os.makedirs(path2MCMC_fit)
        # if not os.path.exists(path2MCMC_fit + 'plots/'):
        #     print('> Making %s'%path2MCMC_fit + 'plots/')
        #     os.makedirs(path2MCMC_fit + 'plots/')
        #     print('> Making %s'%path2MCMC_fit + 'plots/corners/')
        #     os.makedirs(path2MCMC_fit + 'plots/corners/')
        #     print('> Making %s'%path2MCMC_fit + 'plots/debug/')
        #     os.makedirs(path2MCMC_fit + 'plots/debug/')
        #     print('> Making %s'%path2MCMC_fit + 'plots/posterior/')
        #     os.makedirs(path2MCMC_fit + 'plots/posterior/')

    #################################### Lodading data #################################################
    if not args.makeout:

        print('> Looking for data in %s' % self.path2out)

        start_time_list=[]
        print('> Building list of start times from: %s'%out_data_list)
        for name,data in zip(out_data_list,data_list):
            print('>> Looking for data associated to %s containing time %s,...' % (name,data))
            startt, _, _ = mkidcore.utils.get_ditherdata_for_time(dither_path, data)
            start_time_list.extend([np.round(i) for i in startt])
            print('>> Found %i dithers.'%len(startt))

        # start_time_list=start_time_list[::10]
        image_list = []
        header_list = []
        dist_list = []
        elno_list = []
        h5_name_list=[int(i.split('/')[-1].split('.h5')[0]) for i in glob(self.path2out + '*.h5')]
        filename_list=[]

        ntargets = len(start_time_list)
        num_of_chunks = 3 * workers
        chunksize = ntargets // num_of_chunks
        if chunksize <= 0:
            chunksize = 1

        elno=0
        if workers == 1: workers_load=10
        else: workers_load=workers
        print('> Using %i workers to load a total of %i files ...'%(workers_load,ntargets))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers_load) as executor:
            for filename,header,data in tqdm(executor.map(load_fits_task, start_time_list,chunksize=chunksize)):
                elno+=1
                filename_list.append(filename)
                header_list.append(header)
                image_list.append(data)
                dist_list.append(np.sqrt((header_list[0]['E_CONEXX']-header['E_CONEXX'])**2+(header_list[0]['E_CONEXY']-header['E_CONEXY'])**2))
                elno_list.append(elno)

        sorted_elno_list = [x for _, x in sorted(zip(dist_list, elno_list))]

    ############################################################## PARAMETERS FIT #################################################################

        print('> Fitting parameters')
        ntargets = len(filename_list)

        num_of_chunks = 3 * workers
        chunksize = ntargets // num_of_chunks
        if chunksize <= 0:
            chunksize = 1

        if redo or np.any([len(getattr(data_dict[data_elno], label)) == 0 for label in [ 'slopes']]):
            out_dictc = get_slope_and_conex(data_elno,data_dict,sorted_elno_list,filename_list,image_list,header_list,ref_el)

        if sat_spots:
            if redo or np.any([len(getattr(data_dict[data_elno],label)) == 0 for label in
                        ['spot_ref1', 'spot_ref2', 'spot_ref3', 'spot_ref4', 'cor_spot_ref','conex_ref']]):
                data_dict =get_satellite_spots_and_coronograph(data_elno, data_dict,image_list,header_list,sat_spots)

            positions_ref = [np.float64(data_dict[data_elno].spot_ref1),
                             np.float64(data_dict[data_elno].spot_ref2),
                             np.float64(data_dict[data_elno].spot_ref3),
                             np.float64(data_dict[data_elno].spot_ref4)]
            coronograph_ref = np.float64(data_dict[data_elno].cor_spot_ref)
            conex_xy_ref = np.float64(data_dict[data_elno].conex_ref)
            slopes = np.float64(data_dict[data_elno].slopes)

            pos_dict = {'amplitude': [pipe_dict['mcmcwcssol']['amplitude'][0], pipe_dict['mcmcwcssol']['amplitude'][1], pipe_dict['mcmcwcssol']['amplitude'][2]],
                        'length1': [pipe_dict['mcmcwcssol']['length'][0], pipe_dict['mcmcwcssol']['length'][1], pipe_dict['mcmcwcssol']['length'][2]],
                        'length2': [pipe_dict['mcmcwcssol']['length'][0], pipe_dict['mcmcwcssol']['length'][1], pipe_dict['mcmcwcssol']['length'][2]],
                        'angle1': [pipe_dict['mcmcwcssol']['angle1'][0], pipe_dict['mcmcwcssol']['angle1'][1], pipe_dict['mcmcwcssol']['angle1'][2]],
                        'angle2': [pipe_dict['mcmcwcssol']['angle2'][0], pipe_dict['mcmcwcssol']['angle2'][1], pipe_dict['mcmcwcssol']['angle2'][2]],
                        'fwhm_x1': [pipe_dict['mcmcwcssol']['fwhm_x'][0], pipe_dict['mcmcwcssol']['fwhm_x'][1], pipe_dict['mcmcwcssol']['fwhm_x'][2]],
                        'fwhm_y1': [pipe_dict['mcmcwcssol']['fwhm_y'][0], pipe_dict['mcmcwcssol']['fwhm_y'][1], pipe_dict['mcmcwcssol']['fwhm_y'][2]],
                        'fwhm_x2': [pipe_dict['mcmcwcssol']['fwhm_x'][0], pipe_dict['mcmcwcssol']['fwhm_x'][1], pipe_dict['mcmcwcssol']['fwhm_x'][2]],
                        'fwhm_y2': [pipe_dict['mcmcwcssol']['fwhm_y'][0], pipe_dict['mcmcwcssol']['fwhm_y'][1], pipe_dict['mcmcwcssol']['fwhm_y'][2]]}
        else:
            if redo or np.any([len(getattr(data_dict[data_elno],label)) == 0 for label in
                        ['cor_spot_ref','conex_ref']]):
                data_dict =get_satellite_spots_and_coronograph(data_elno, data_dict,image_list,header_list,sat_spots)

            positions_ref = [np.float64(data_dict[data_elno].cor_spot_ref)]
            coronograph_ref = np.float64(data_dict[data_elno].cor_spot_ref)
            conex_xy_ref = np.float64(data_dict[data_elno].conex_ref)
            slopes = np.float64(data_dict[data_elno].slopes)

            pos_dict = {'amplitude': [pipe_dict['mcmcwcssol']['amplitude'][0], pipe_dict['mcmcwcssol']['amplitude'][1], pipe_dict['mcmcwcssol']['amplitude'][2]],
                        'fwhm_x1': [pipe_dict['mcmcwcssol']['fwhm_x'][0], pipe_dict['mcmcwcssol']['fwhm_x'][1], pipe_dict['mcmcwcssol']['fwhm_x'][2]],
                        'fwhm_y1': [pipe_dict['mcmcwcssol']['fwhm_y'][0], pipe_dict['mcmcwcssol']['fwhm_y'][1], pipe_dict['mcmcwcssol']['fwhm_y'][2]]}

        config.dump_dataconfig(data_dict, args.data_cfg)

        if workers >1:
            print('> workers %i,chunksize %i,ntargets %i'%(workers,chunksize,ntargets))
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                for _ in tqdm(executor.map(mcmc_task,filename_list,image_list,header_list,repeat(pos_dict),chunksize=chunksize)): pass

        else:
            for elno in range(len(filename_list)): mcmc_task(filename_list[elno],image_list[elno],header_list[elno],pos_dict)

    else:
        if sat_spots:
            positions_ref = [np.float64(data_dict[data_elno].spot_ref1),
                             np.float64(data_dict[data_elno].spot_ref2),
                             np.float64(data_dict[data_elno].spot_ref3),
                             np.float64(data_dict[data_elno].spot_ref4)]
            coronograph_ref = np.float64(data_dict[data_elno].cor_spot_ref)
            conex_xy_ref = np.float64(data_dict[data_elno].conex_ref)
            slopes = np.float64(data_dict[data_elno].slopes)

        else:
            positions_ref = [np.float64(data_dict[data_elno].cor_spot_ref)]
            coronograph_ref = np.float64(data_dict[data_elno].cor_spot_ref)
            conex_xy_ref = np.float64(data_dict[data_elno].conex_ref)
            slopes = np.float64(data_dict[data_elno].slopes)

    #################################################### OUTPUTS ###########################################################
    print('> Working on outputs')
    print('> Looking for data in %s'%self.path2out + 'MCMC_fit/')

    filename_list = [int(filename.split('/')[-1].split('_')[0]) for filename in glob(self.path2out + 'MCMC_fit/' + '*.h5')]

    ntargets=len(filename_list)
    num_of_chunks = 3 * workers
    chunksize = ntargets // num_of_chunks
    if chunksize <= 0:
        chunksize = 1

    out_filename_list = []
    pixel_cen_list = []
    epixel_cen_list = []
    conexx_list = []
    if sat_spots: MCMC_labels = ['amplitude', 'length1', 'length2', 'angle1', 'angle2', 'fwhm_x1', 'fwhm_y1', 'fwhm_x2', 'fwhm_y2', 'cen_x',
             'cen_y']
    else: MCMC_labels = ['amplitude', 'fwhm_x1', 'fwhm_y1', 'cen_x', 'cen_y']

    print('> Working on data:')
    for filename in tqdm(filename_list):
        try:
            filename, pixel_cen, epixel_cen, xyCons  = sample_posteriors_task(filename)
            out_filename_list.append(filename)
            pixel_cen_list.append(pixel_cen)
            epixel_cen_list.append(epixel_cen)
            conexx_list.append(xyCons)
        except:
            print('> Skipping %s'%filename)
    out_filename_list = np.array(out_filename_list)
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

    d['std_pixel_at_conex_y']= np.nanmean(
        [np.std(d['pixel_at_conex_y'][np.where(d['conexy'] == conexy)[0]]) for conexy in set(d['conexy'])])

    sol_x = lsq_fit_dpdc(d, ['conexx', 'pixel_at_conex_x', 'std_pixel_at_conex_x'], showplot=verbose, verbose=verbose,
                         path2savedir=self.path2out + 'MCMC_fit/' + 'plots/', ext='_x_')
    sol_y = lsq_fit_dpdc(d, ['conexy', 'pixel_at_conex_y', 'std_pixel_at_conex_y'], showplot=verbose, verbose=verbose,
                         path2savedir=self.path2out + 'MCMC_fit/' + 'plots/', ext='_y_')

    dout = {'x': {'dpdc': [float(np.round(i,2)) for i in sol_x[0]],
                  'pc0': [float(np.round(i,2)) for i in sol_x[1]],
                  'conex': 0},
            'y': {'dpdc': [float(np.round(i,2)) for i in sol_y[0]],
                  'pc0': [float(np.round(i,2)) for i in sol_y[1]],
                  'conex': 0}}

    data_dict[data_elno].sol = dout

    data_elno = next((index for (index, d) in enumerate(data_dict) if wcscal in d.name),None)
    data_dict[data_elno].pixel_ref = [float(dout['x']['pc0'][0]),float(dout['y']['pc0'][0])]
    data_dict[data_elno].conex_ref = [float(dout['x']['conex']),float(dout['y']['conex'])]
    data_dict[data_elno].dp_dcx = float(dout['x']['dpdc'][0])
    data_dict[data_elno].dp_dcy = float(dout['y']['dpdc'][0])

    # with open(args.data_cfg, "w") as file:
    #     YAML.dump(data_dict, file)
    config.dump_dataconfig(data_dict, args.data_cfg)

