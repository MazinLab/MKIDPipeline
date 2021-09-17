import mkidpipeline.config
import numpy as np
import matplotlib.pyplot as plt
from mkidpipeline.utils.smoothing import astropy_convolve
from mkidpipeline.utils.photometry import fit_sources

class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!wcscal_cfg'
    REQUIRED_KEYS = (('plot', 'none', 'none|all|summary'),
                     ('interpolate', 'True', 'whether to inerpolate the image before PSF fitting. Recommended if an '
                                             'MKIDObservation is used or data is noisy'),
                     ('sigma_psf', 2.0, 'standard deviation of the point spread functions to fit in the image '),
                     ('source_loc', [], 'locations of the sources in the image in (RA, DEC)'))


HEADER_KEYS = tuple()


PROBLEM_FLAGS = ('pixcal.hot', 'pixcal.cold', 'pixcal.dead', 'beammap.noDacTone', 'wavecal.bad',
                 'wavecal.failed_validation', 'wavecal.failed_convergence', 'wavecal.not_monotonic',
                 'wavecal.not_enough_histogram_fits', 'wavecal.no_histograms',
                 'wavecal.not_attempted')

class ClickCoords:
    def __init__(self, image, source_locs):
        self.coords = []
        self.image = image
        self.n_sources = len(source_locs)
        self.source_locs = source_locs
        self.counter = 0
        self.fig = plt.figure()
        self.cid = None

    def push_to_start(self):
        plt.imshow(self.image)
        plt.title('Click Anywhere to Start')
        if plt.waitforbuttonpress() is True:
            return True

    def get_coords(self):
        self.push_to_start()
        plt.clf()
        plt.imshow(self.image)
        plt.title(f'Select Locations of Source at {self.source_locs[self.counter]}')
        plt.draw()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        return self.coords

    def __onclick__(self, click):
        point = (click.xdata, click.ydata)
        self.coords.append(point)
        self.counter += 1
        if self.counter == self.n_sources:
            self.fig.canvas.mpl_disconnect(self.cid)
            plt.close()
        plt.clf()
        plt.imshow(self.image)
        plt.scatter(*zip(*self.coords), c='red', marker='o', s=10)
        plt.title(f'Select Locations of Source at {self.source_locs[self.counter]}')
        plt.draw()
        return self.coords

def crop_image(image): #TODO move to utils
    """
    crops out buffer of pixels with 0 counts from an image
    :param image: 2D image array to be cropped
    :return: cropped 2D image array.
    """
    live_idxs = np.argwhere(image != 0)
    center_x, center_y = int(image.shape[0]/2), int(image.shape[1]/2)
    min_x, min_y = min(live_idxs[:,0]), min(live_idxs[0,:])
    max_x, max_y = max(live_idxs[:,0]), max(live_idxs[0,:])
    use_x = int(max(center_x - min_x, max_x - center_x))
    use_y = int(max(center_y - min_y, max_y - center_y))
    return image[center_x - use_x:center_x + use_x, center_y - use_y:center_y + use_y]

def select_sources(image, source_locs):
    cc = ClickCoords(image, source_locs=source_locs)
    coords = cc.get_coords()
    return coords

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

def difference(a, b):
    diff = np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    return diff

def angle(a, b):
    theta = np.arctan((a[0] - b[0])/(a[1] - b[1]))
    return theta

def dict_dist_compare(coord_dict):
    result = []
    for key1, val1 in coord_dict.items():
        for key2, val2 in coord_dict.items():
            if key1==key2:
                pass
            else:
                result.append(difference(key1, key2)/difference(val1, val2))
        return result

def dict_theta_compare(coord_dict):
    result = []
    for key1, val1 in coord_dict.items():
        for key2, val2 in coord_dict.items():
            if key1==key2:
                pass
            else:
                true_theta = angle(key1, key2)
                measured_theta = angle(val1, val2)
                result.append(true_theta - measured_theta)
        return result

def get_platescale(coord_dict):
    pltscl = []
    dict_copy = coord_dict.copy()
    for key, value in coord_dict.items():
        pltscl += dict_dist_compare(dict_copy)
        dict_copy.pop(key)
    return np.mean(pltscl) # TODO errors

def get_rotation_angle(coord_dict):
    angles = []
    dict_copy = coord_dict.copy()
    for key, value in coord_dict.items():
        angles += dict_theta_compare(dict_copy)
        dict_copy.pop(key)
    return np.mean(angles) # TODO errors

def resolve_units(coords):
    # TODO write
    return coords

def calculate_wcs_solution(image, source_locs=None, sigma_psf=2.0, convolve=False):
    #TODO get source_locs in consistent units no matter what is given by user
    source_locs = resolve_units(source_locs)
    if convolve:
        im = astropy_convolve(image)
    else:
        im = image
    cropped = crop_image(im)
    #returns clicked coordinates corresponding in order to the given ra/dec coords
    coord_dict = {k: None for k in source_locs}
    selected_coords = select_sources(cropped, source_locs=source_locs)
    sources, residuals = fit_sources(cropped, sigma_psf=sigma_psf)#, guesses=selected_coords)
    fit_coords = [(sources['x_0'][i], sources['y_0'][i]) for i in range(len(sources))]
    use_idxs = [closest_node(selected_coords[i], fit_coords) for i in range(len(selected_coords))]
    use_coord = [fit_coords[idx] for idx in use_idxs]
    for i, key in enumerate(coord_dict.keys()):
        coord_dict[key] = use_coord[i]
    pltscl = get_platescale(coord_dict)
    rot_angle = get_rotation_angle(coord_dict)
    return pltscl, rot_angle


def fetch(data):
    pass