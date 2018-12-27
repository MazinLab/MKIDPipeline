#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photon list class for doing SSD analysis.
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import datetime
import time

import mkidpipeline.speckle.genphotonlist_IcIsIr as gpl
import mkidpipeline.speckle.binned_rician as binMR
import mkidpipeline.speckle.optimize_IcIsIr as binfree
import mkidpipeline.speckle.binFreeRicianEstimate as bf # alex's code
from scipy import optimize
import pickle


class photon_list(object):
    def __init__(self,Ic=-1,Is=-1,Ir=-1,Ttot=-1,tau=-1):
        self.Ic = Ic # nominal Ic
        self.Is = Is
        self.Ir = Ir
        self.tau = tau # decorrelation time, seconds
        self.Ttot = Ttot # total length of photon list in seconds
        self.ts = np.array([])  # units microseconds
        self.p0 = np.array([]) # seed for calling the optimize routine
        self.p1 = np.array([]) # result of the optimize routine
        self.cube = np.array([]) # this is a loglike cube
        self.Ic_list = np.array([])
        self.Is_list = np.array([])
        self.Ir_list = np.array([])
        self.deadtime = 0
        self.dt = np.array([])
        self.cube_max_params = np.array([]) # [Ic, Is, Ir] that give the maximum loglike in self.cube
        self.p0_list = np.array([])
        self.p1_list = np.array([])
        self.loglike_max_check = np.array([],dtype=bool)
        self.loglike_max_list = np.array([])
        self.loglike_true_params = -1
        self.statsModels_result = np.array([])
        self.logLike_statsModels = -1


        if Ic>0:
            self.gen_plist(Ic,Is,Ir,Ttot,tau)
            self.setup_cube_lists()
            self.get_cube()
            self.p0_get_simple()
            self.find_max_like()
            self.p0_cube_max()
            self.find_max_like()
            self.loglike_true_params = -binfree.loglike([Ic, Is, Ir], self.dt, self.deadtime)
            self.do_stats_models()


    def load(self,filename):
        self.filename = filename
        npzfile = np.load(self.filename)
        self.ts = npzfile['ts']
        self.p1 = npzfile['p1']

        self.dt = (self.ts[1:] - self.ts[:-1]) * 1e-6

    # def save(self): # this doesn't work- not sure why :(
    #     with open("/Users/clint/Dropbox/mazinlab/speckle/20181226/junk.p", "wb") as f:
    #         pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def gen_plist(self,Ic,Is,Ir,Ttot,tau):
        self.Ic = Ic
        self.Is = Is
        self.Ir = Ir
        self.Ttot = Ttot
        self.tau = tau
        self.ts = gpl.genphotonlist(Ic, Is, Ir, Ttot, tau)
        self.dt = (self.ts[1:] - self.ts[:-1]) * 1e-6

    def get_cube(self):
        if len(self.Ic_list) < 2 or len(self.Is_list) < 2 or len(self.Ir_list) < 2:
            print('one of the lists has a length < 2')
            return
        self.cube = binMR.logL_cube(self.ts, self.Ic_list, self.Is_list, self.Ir_list)


    def find_max_like(self):
        if self.p0.size == 0:
            print('you need to define a seed: self.p0')
            return
        self.p1 = optimize.minimize(binfree.loglike, self.p0, (self.dt, self.deadtime), method='Newton-CG', jac=binfree._jacobean,
                               hess=binfree._hessian).x
        self.p0_list = np.append(self.p0_list, self.p0)
        self.p1_list = np.append(self.p1_list, self.p1)
        self.loglike_max_check = np.append(self.loglike_max_check, binMR.check_binfree_loglike_max(self.ts, self.p1, deadtime=self.deadtime))
        self.loglike_max_list = np.append(self.loglike_max_list,-binfree.loglike(self.p1,self.dt,self.deadtime))


    def p0_get_simple(self):
        I = 1 / np.mean(self.dt)
        self.p0 = I*np.ones(3)/3.

    def p0_cube_max(self):
        if self.cube.size == 0:
            print('self.cube doesnt exist yet. Run self.get_cube first.')
            return
        index = np.unravel_index(self.cube.argmax(), np.shape(self.cube))
        temp = [self.Ic_list[index[2]], self.Is_list[index[1]], self.Ir_list[index[0]]]
        self.cube_max_params = np.copy(temp)
        self.p0 = np.copy(temp)


    def setup_cube_lists(self):
        # I = 1 / np.mean(self.dt)
        I = self.Ic + self.Is + self.Ir
        self.Ic_list = np.linspace(1, I, 100)
        self.Is_list = np.linspace(1, I, 100)
        self.Ir_list = np.linspace(0, 100, 10)


    def save_cubeslices(self,path,plist_number):
        try:
            os.makedirs(path)
        except:
            pass

        for ii in range(len(self.cube)):
            fig = plt.figure()
            plt.imshow(self.cube[ii] - np.amax(self.cube),extent=[np.amin(self.Ic_list), np.amax(self.Ic_list), np.amin(self.Is_list), np.amax(self.Is_list)], aspect = 'auto', origin = 'lower', cmap = 'hot_r', vmin = -8, vmax = 0, interpolation = 'spline16')
            plt.ylabel('Is [/s]')
            plt.xlabel('Ic [/s]')
            plt.title('Ic,Is,Ir, Ttot, tau = [{:g},{:g},{:g},{:g},{:g}]      Ir slice = {:.3g}'.format(self.Ic,self.Is,self.Ir,self.Ttot,self.tau,self.Ir_list[ii]))

            cb_ax = fig.add_axes([0.88, 0.1, 0.02, 0.8])  # [0.83, 0.1, 0.02, 0.8] [x0, y0, width, height]
            cbar = fig.colorbar(img, cax=cb_ax)
            cbar.set_label(r'ln$\mathcal{L}$ - ln$\mathcal{L}_{max}$')

            filename = os.path.join(path, 'ph_list_{:g}_Ir_slice_{:.3g}.png'.format(plist_number, self.Ir_list[ii]))
            plt.savefig(filename, dpi=500)
            # plt.show()
            plt.close(fig)


    def do_stats_models(self):
        # now use Alex's code to estimate the parameters
        m = bf.MR_SpeckleModel(self.ts, deadtime=self.deadtime)
        res = m.fit()
        self.logLike_statsModels = -binfree.loglike([res.params[0], res.params[1], res.params[2]], self.dt, self.deadtime)

        self.p0_list = np.append(self.p0_list, np.array([-1,-1,-1]))
        self.p1_list = np.append(self.p1_list, np.array([res.params[0], res.params[1], res.params[2]]))
