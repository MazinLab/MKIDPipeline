#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Alex Walter; Clint Bockstiegel
Date: March 15, 2019

mock Photon list class for doing SSD analysis.
The logL cubes that explore the Ic, Is, Ir parameter space take a while to calculate.
This class is used to easily save and load the data later

"""

import numpy as np
import matplotlib.pyplot as plt

from functools import partial

import os, glob
import datetime
import time
import multiprocessing

from mkidpipeline.speckle.genphotonlist_IcIsIr import genphotonlist

import mkidpipeline.speckle.genphotonlist_IcIsIr as gpl
import mkidpipeline.speckle.binned_rician as binMR
import mkidpipeline.speckle.binFreeRicianEstimate as binfree # alex's code
from scipy import optimize, integrate
import pickle


def probIr(logLcube, p_lists, cumulative = False):
    """
    This function returns the (cumulative) probability density of Ir. 
    Essentially, it integrates the likelihood over Ic, Is for values of Ir.
    It should be normalized

    INPUTS:
        logLcube - cube of log likelihood values
        p_lists - dictionary containg the values of Ic, Is, Ir for each axis
                    ie. p_lists[0] is the Ic_list
        cumulative - Calculate the cumulative probability density

    OUTPUTS:
        Ir_list - list of Ir values
        intL - corresponding probability density
    """
    
    likelihoodCube = np.exp(logLcube - np.amax(logLcube))   #partially normalize so we don't have a numerical overflow
    prob = integrate.trapz(likelihoodCube, p_lists[0],axis=0)   #integrate over Ic
    prob = integrate.trapz(prob, p_lists[1],axis=0)       #integrate over Is
    if cumulative:
        prob=integrate.cumtrapz(prob,p_lists[2],initial=0)               #cumulative integral over Ir
        norm=prob[-1]
    else:
        norm = integrate.trapz(prob, p_lists[2])         #integrate over Ir to get normalization
    prob/=norm
    return prob


def getLogLCube(logLfunc, p_lists={0:[], 1:[], 2:[]}, relmin=10.**-8., p_opt=None, rounddec=2):
    """
    This function maps out the log likelihood space for both the binned and binfree model.

    If p_lists is None it tries to use smart values for the sampling of Ic, Is, Ir:
        - The sample step size is np.sqrt(I)/20. (right now it's hardcoded to 1/20)
        - First it calculates where the max Log liklihood is
        - Fixing Ic=Ic_optimal, Is=Is_optimal, it increments Ir until the logL gets insignificant.
        - Then it does the same thing for Ic, and Is
        - Using the obtained Ic, Is, Ir valid ranges --> it populates the entire logL cube
        WARNING: This won't work well if p_opt isn't the max likelihood. It should be fine as long as it's close though.

    INPUTS:
        logLfunc - function to get logL with signature logL(p)
                   ie. logLfunc=lambda p: binfree.MRlogL(p, dt, deadtime)
        p_lists - If any of the keys are empty try to create a smart sampling along that axis
        relmin - starting at the p_opt, continue mapping out in Ic/Is/Ir until logL is less than maxLogL*relmin
                 Used to help auto-populate p_lists
                 relmin is ignored if p_lists is given already
        p_opt - The location [Ic, Is, Ir] with the maximum logL
                Used to help auto-populate p_lists
                p_opt is ignored if p_lists is given already
                if p_lists[2] is empty for example then still give the entire p_opt=[Ic_opt, Is_opt, Ir_opt] (don't skip out on Ir_opt)
                if any p_lists[i] is empty then give p_opt cannot be none
        rounddec - round the new values of the parameters (Ic,Is,Ip) to the nearst rounddec decimals
                   negative means no rounding

    OUTPUTS:
        logLmap - cube containing the logL. shape=( len(Ic_list), len(Is_list), len(Ir_list) )
        p_lists - dictionary continaing Ic_list, Is_list, Ir_list
                  p_lists[0] returns the Ic_list, etc...
    """
    
    populate_inds = []
    if len(p_lists[0])==0 or len(p_lists[1])==0 or len(p_lists[2])==0:
        sampling=-1./20.
        p_inc = np.sqrt(p_opt)*sampling
        p_inc[p_inc>sampling]=sampling   # minimum sampling of parameters
        if rounddec>=0:
            p_opt=p_opt.round(rounddec)
            p_inc=p_inc.round(rounddec)
            p_inc[p_inc==0]=-10.**-rounddec
        maxLogL=logLfunc(p_opt)

        tmp_logL={}
        for k in range(3):
            if len(p_lists[k])==0:
                p_lists[k]=[p_opt[k]]
                populate_inds.append(k)
                tmp_logL[k]=[maxLogL]
            elif p_opt[k] not in p_lists[k]:
                p_lists[k].append(p_opt[k])

        #populate p_lists
        for ind in populate_inds:
            p_tmp=np.copy(p_opt)
            logL=maxLogL
            loopSafety=1000 #just in case...
            while loopSafety >=0:   
                loopSafety-=1
                
                #if logL>maxLogL*relmin and p_tmp[ind]+p_inc[ind]>0.:    #continue sweeping
                if logL>(maxLogL + np.log(relmin)) and p_tmp[ind]+p_inc[ind]>0.:    #continue sweeping
                    p_tmp[ind]+=p_inc[ind]
                else:   # finished sweeping in one direction
                    if p_inc[ind]<0.: # It was sweeping down. Sweep It up now!
                        p_inc[ind]*=-1
                        p_tmp[ind]=p_opt[ind]+p_inc[ind]
                    else: break     # move on to next parameter
                        
                logL=logLfunc(p_tmp)
                p_lists[ind].append(p_tmp[ind])
                tmp_logL[ind].append(logL)
        
        

        #Force Ir=0 to be in logLmap
        if (0 not in p_lists[2]):
            p_lists[2].append(0)
            if 2 in populate_inds: tmp_logL[2].append(-np.inf)
        #sort p_lists lists
        for k in p_lists.keys():
            sort_ind = np.argsort(p_lists[k])
            p_lists[k]=np.asarray(p_lists[k])[sort_ind]
            if k in populate_inds: tmp_logL[k]=np.asarray(tmp_logL[k])[sort_ind]

            #plt.plot(p_lists[k], tmp_logL[k],'.-',label=k)
        #plt.legend()
        #print("\tshape: ({}, {}, {})".format(len(p_lists[0]),len(p_lists[1]),len(p_lists[2])))
        #keep only 150 points so that the map isn't too big
        maxListSize=150.
        for k in p_lists.keys():
            if len(p_lists[k])>maxListSize:
                l = len(p_lists[k])
                keep = np.floor(np.arange(l)%(1.0*l/maxListSize))==0
                p_lists[k] = p_lists[k][keep]
                tmp_logL[k] = tmp_logL[k][keep]
                

        # pre-populate log likelihood map
        logLmap = np.full([len(p_lists[i]) for i in range(len(p_opt))], -np.inf)
        if 0 in populate_inds: logLmap[:,p_lists[1]==p_opt[1],p_lists[2]==p_opt[2]] = tmp_logL[0][:,np.newaxis]
        if 1 in populate_inds: logLmap[p_lists[0]==p_opt[0],:,p_lists[2]==p_opt[2]] = tmp_logL[1][np.newaxis,:]
        if 2 in populate_inds: logLmap[p_lists[0]==p_opt[0],p_lists[1]==p_opt[1],:] = tmp_logL[2][np.newaxis,:]

    else:
        #p_opt=[-1,-1,-1]
        logLmap = np.full([len(p_lists[i]) for i in range(3)], -np.inf)
        

    #print(logLmap.shape)
    # now fully populate the logL cube
    #for i,c in enumerate(p_lists[0]):
    #    for j,s in enumerate(p_lists[1]):
    #        for k,r in enumerate(p_lists[2]):
    #            #if ((c==p_opt[0]) + (s==p_opt[1]) + (r==p_opt[2])) >=2: continue
    #            if np.isfinite(logLmap[i,j,k]): continue
    #            logLmap[i,j,k]=logLfunc([c,s,r])
    #            
    #return logLmap, p_lists
    
    #print("\tlogLCube shape: "+str(logLmap.shape)+" p_opt: "+str(p_opt))
    if p_opt is None:
        print("\tCube: {}. Ic={:.3f}..?..{:.3f}, Is={:.3f}..?..{:.3f}, Ip={:.3f}..?..{:.3f}".format(logLmap.shape,
            p_lists[0][0], p_lists[0][-1],
            p_lists[1][0], p_lists[1][-1],
            p_lists[2][0],  p_lists[2][-1]))
    else:
        print("\tCube: {}. Ic={:.3f}..{:.3f}..{:.3f}, Is={:.3f}..{:.3f}..{:.3f}, Ip={:.3f}..{:.3f}..{:.3f}".format(logLmap.shape,
            p_lists[0][0], p_opt[0], p_lists[0][-1],
            p_lists[1][0], p_opt[1], p_lists[1][-1],
            p_lists[2][0], p_opt[2], p_lists[2][-1]))

    plt.show()

    return _multiprocessGetLogLCube(logLmap, p_lists, logLfunc)

def _worker(arg_dict,logLfunc):
    """
    Used in _multiprocessGetLogLCube
    """
    logLmap_slice=arg_dict['map_slice']
    p_lists_slice=arg_dict['p_slice']
    #logLfunc = arg_dict['logLfunc']
    for i,c in enumerate(p_lists_slice[0]):
        for j,s in enumerate(p_lists_slice[1]):
            for k,r in enumerate(p_lists_slice[2]):
                #if ((c==p_opt[0]) + (s==p_opt[1]) + (r==p_opt[2])) >=2: continue
                if np.isfinite(logLmap_slice[i,j,k]): continue
                logLmap_slice[i,j,k]=logLfunc([c,s,r])
                #logLmap_slice[i,j,k]=np.sum([c,s,r])
    return logLmap_slice

def _multiprocessGetLogLCube(logLmap, p_lists, logLfunc):
    """
    This function fills in a logLCube with a pool of processes.
    nProc = nCPU - 1
    The logLCube is split on the Ir axis into nProc arrays and each one is sent to _worker() in a different process
    The results are combined and returned

    INPUTS:
        logLmap - 3d array of shape ( len(p_lists[0]),len(p_lists[1]),len(p_lists[2]) )
        p_lists - dictionary containing parameter lists
                  p_lists[0] is the Ic_list etc...
        logLfunc - function with signature logLfunc([Ic,Is,Ir]) that returns the log likelihood
    OUTPUTS:
        logLmap - same shape as input but now filled in
        p_lists - identical to input
    """

    nProc = multiprocessing.cpu_count()-1
    arg_dicts=[]
    nSlices = int(np.ceil(1.0*logLmap.shape[-1]/nProc))
    ind_sections = np.ceil(np.linspace(0,logLmap.shape[2],nSlices+1)[1:]).astype(np.int)
    map_slices = np.array_split(logLmap, ind_sections,axis=2)
    Ir_lists = np.array_split(p_lists[2], ind_sections)


    map_slices_n = [map_slices[i] for i in np.where(list(map(lambda x: map_slices[x].shape[2]>0, range(len(map_slices)))))[0]]
    Ir_lists_n=np.asarray(Ir_lists)[list(map(lambda x: len(Ir_lists[x])>0, range(len(Ir_lists))))]

    for i in range(len(map_slices_n)):
        p_slice=p_lists.copy()
        p_slice[2]=Ir_lists_n[i]
        arg_dicts.append({'map_slice':map_slices_n[i], 'p_slice':p_slice})#, 'logLfunc':logLfunc})
        #print(arg_dicts[-1]['map_slice'].shape)
        #print(arg_dicts[-1]['p_slice'][2].shape)
        #print(arg_dicts[-1]['logLfunc'])

    pool = multiprocessing.Pool(processes=nProc)
    map_slices_3=pool.map(partial(_worker,logLfunc=logLfunc), arg_dicts)
    pool.close()    
    pool.join()
    
    logLmap=np.concatenate(map_slices_3, axis=2, out=logLmap)
    return logLmap, p_lists




def _savePhotonlist(photonlist, fn):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, 'wb') as output:
        pickle.dump(photonlist, output, pickle.HIGHEST_PROTOCOL)
    print('saving: '+fn)

def _loadPhotonlist(fn):
    with open(fn, 'rb') as infile:
        photonlist = pickle.load(infile)
    return photonlist

def savePhotonlist(photonlist, Ic, Is, Ir, Ttot, loc='/Data/SSD/logLmaps/'):
    fn=loc+'data_{}_{}_{}_{}/{}.pickle'.format(Ic, Is, Ir, Ttot,int(time.time()))
    _savePhotonlist(photonlist, fn)

def loadPhotonlist(Ic, Is, Ir, Ttot, loc='/Data/SSD/logLmaps/'):
    """
    loads all photons list objects with the requested parameters
    """
    directory = loc+'data_{}_{}_{}_{}/'.format(Ic, Is, Ir, Ttot)
    fns = glob.glob(directory+'*.pickle')
    return [_loadPhotonlist(fn) for fn in fns]
    

class mock_photonlist():
    """
    mock photonlist class to hold logLCubes and other miscellanious data
    Is pickle-able for easy save / load

    This class replaces photon_list() but I haven't deleted photon_list() yet for 
    """
    def __init__(self,Ic,Is,Ir,Ttot=30,tau=0.1, deadtime=10.e-6,return_IDs=True):
        self.p_true = np.asarray([Ic, Is, Ir])
        self.Ttot=Ttot
        self.tau=tau
        self.deadtime=deadtime
        self.ts, self.ts_star, _ = genphotonlist(Ic, Is, Ir, Ttot, tau, deadtime*10.**6., return_IDs=return_IDs)
        self.dt = (self.ts[1:] - self.ts[:-1]) * 1e-6
        self.dt_star = (self.ts_star[1:] - self.ts_star[:-1]) * 1e-6

        #self.p_seed=np.asarray([len(self.ts)/self.Ttot/3.]*3)   # I/3
        self.p_seed=np.copy(self.p_true)

        self.logLCubes={}         # list of logLCubes. key:value --> binsize:logLCube.
        self.logLCubes_star={}
        self.logLCubes_lists={} # list of p_lists for corresponding logLCubes
        self.logLCubes_star_lists={}

    def save(self,loc='/Data/SSD/logLmaps/'):
        savePhotonlist(self, self.p_true[0], self.p_true[1], self.p_true[2], self.Ttot, loc=loc)
        #savePhotonlist(self, fn)

    def getLogLCubes(self,binSize_list, matchIr_lists=True, relmin=10.**-8.):
        """
        Get a bunch of logLcubes for different binSizes. 

        INPUTS:
            binSize_list - list of binSizes. binSize<=0 means binfree
            marchIr_ists - if True, make sure the Ir_list for star and star+planet have the same values
            relmin - passed to self.getLogLCube()
        """
        for b in binSize_list:
            print("Getting data for star+planet. BinSize: "+str(b))
            logLmap, p_lists, binSize = self._getLogLCube(b, star=False, p_lists={0:[], 1:[], 2:[]},relmin=relmin)
            self.logLCubes[binSize]=logLmap
            self.logLCubes_lists[binSize]=p_lists

            print("Getting data for star. BinSize: "+str(b))
            logLmap_star, p_lists_star, binSize = self._getLogLCube(b, star=True, p_lists={0:[], 1:[], 2:[]}, relmin=relmin)
            self.logLCubes_star[binSize]=logLmap_star
            self.logLCubes_star_lists[binSize]=p_lists_star

            print("Matching Ir_lists. BinSize: "+str(b))
            if matchIr_lists:
                _, p_list_new, __ = self.appendVals2Cube(p_lists_star[2], 2, binSize, star=False, relmin=relmin)
                _,__,___=self.appendVals2Cube(p_list_new[2], 2, binSize, star=True, relmin=relmin)
                

        return self.logLCubes, self.logLCubes_lists, self.logLCubes_star, self.logLCubes_star_lists

    def appendVals2Cube(self,I_list, ind, binSize, star=False, relmin=10.**-8.):
        """
        This function is for when we've populated a logLCube but we want to add data points.
        ie. It ranges over Ir_list = [10,11,12,13] and we want to add Ir_list=[0,1,2,3]

        INPUTS:
            I_list - The additional points we want to append to a list. 
                     If we've already calculated logL at some points then those points are ignored
            ind - 0 means interpret I_list as the additional Ic_list. Similiar for ind=1 or ind=2
            binSize - Used to grab the logLmap from self.logLCubes (_star)
            star - Used to grab the logLmap from self.logLCubes (_star)
            relmin - argument passed to getLogCube()

        OUTPUTS:
            logLmap - cube containing the new logLmap
            p_lists - dictionary continaing Ic_list, Is_list, Ir_list
                      p_lists[0] returns the Ic_list, etc...
            binSize - 

        It outputs the new cube but also updates it in the class attribute self.logLCubes (_star) so you don't need to
        """
        logLcube = self.logLCubes_star[binSize] if star else self.logLCubes[binSize]
        p_list = self.logLCubes_star_lists[binSize] if star else self.logLCubes_lists[binSize]
        I_old = p_list[ind]
        #I_diff = np.setdiff1d(I_list, I_old, assume_unique=True)
        I_diff = np.setdiff1d(I_list, I_old)
        if len(I_diff)==0: return logLcube, p_list, binSize
        p_list_diff = p_list.copy()
        p_list_diff[ind]=I_diff

        #logLcube_diff, p_list_diff, binSize = getLogLCube(binSize, star=star, p_lists=p_list_diff, relmin=relmin)
        logLcube_diff, _, __ = self._getLogLCube(binSize, star=star, p_lists=p_list_diff, relmin=relmin)
        logLcube_new = np.append(logLcube, logLcube_diff, axis=ind)
        I_new = np.append(I_old, I_diff)
        sort_inds = np.argsort(I_new)
        logLcube_new = np.take(logLcube_new, sort_inds, axis=ind)
        I_new = I_new[sort_inds]
        p_list_new = p_list_diff
        p_list_new[ind]=I_new

        if star:
            self.logLCubes_star[binSize] = logLcube_new
            self.logLCubes_star_lists[binSize] = p_list_new
        else:
            self.logLCubes[binSize] = logLcube_new
            self.logLCubes_lists[binSize] = p_list_new

        return logLcube_new, p_list_new, binSize



    def _getLogLCube(self,binSize, star=False, p_lists={0:[], 1:[], 2:[]}, relmin=10.**-8.):
        """
        This function maps out the log likelihood space for both the binned and binfree model.
        It calls the getLogLCube() that's defined outside of this class

        INPUTS:
            binSize - bin size of lightcurve. If <=0 then do binfree method
            star - use the list of photons with only stellar light. No planet light
            p_lists - If None, then auto-generate. Otherwise pass this to getLogLCube()
            relmin - passed to getLogLCube()

        OUTPUTS:
            logLmap - cube containing the logL. shape=( len(Ic_list), len(Is_list), len(Ir_list) )
            p_lists - dictionary continaing Ic_list, Is_list, Ir_list
                    p_lists[0] returns the Ic_list, etc...
            binSize - 
        """
        #print("Starting _getLogLCube. p_lists= "+str(p_lists))

        p_seed=np.copy(self.p_seed)
        if star: p_seed[2]=10.  # Ir is 0 in this case but guessing Ir=10. seems to work better...

        if np.isfinite(binSize) and binSize>0:  # binned case
            # n is the light curve
            ts = self.ts_star if star else self.ts
            n = binMR.getLightCurve(photonTimeStamps=ts*1e-6,startTime=ts[0]*1e-6,stopTime=ts[-1]*1e-6,effExpTime=binSize)[0] # get the light curve, units = [cts/bin]
            dist = np.bincount(n)
            #I = 1 / np.mean(dt)
            #p0 = I * np.ones(3) / 3.
            if len(p_lists[0])==0 or len(p_lists[1])==0 or len(p_lists[2])==0:
                p_opt = optimize.minimize(binMR._bin_logL, p_seed, dist, bounds=((0.001, np.inf), (0.001, np.inf), (.001, np.inf))).x / binSize  # units are [cts/sec]
            else: p_opt = None
            logLfunc = partial(binMR.bin_logL, dist=dist)

        else:  # binfree case
            binSize=-1
            dt=self.dt_star if star else self.dt
            if len(p_lists[0])==0 or len(p_lists[1])==0 or len(p_lists[2])==0:
                res = binfree.optimize_IcIsIr(dt, p_seed, self.deadtime)
                p_opt=res.params
                #print(res.summary())
            else: p_opt=None
            #logLfunc = lambda p: binfree.MRlogL(p, dt, self.deadtime)
            logLfunc = partial(binfree.MRlogL, dt=dt, deadtime=self.deadtime)


        logLmap, p_lists_new = getLogLCube(logLfunc, p_lists, relmin, p_opt)
        return logLmap, p_lists_new, binSize

class photon_list(object):
    def __init__(self,Ic=-1,Is=-1,Ir=-1,Ttot=-1,tau=-1, deadtime=0,return_IDs=False):
        # attributes to create original photonlist
        self.Ic = Ic # nominal Ic. phtons/second
        self.Is = Is
        self.Ir = Ir
        self.tau = tau # decorrelation time, seconds
        self.Ttot = Ttot # total length of photon list in seconds
        self.deadtime = deadtime    # seconds

        # photonlist data
        self.ts = np.array([])  # units microseconds. Includes star+planet photons. 
        self.ts_star = np.array([]) #ts except only includes star photons
        self.dt = np.array([]) # units in seconds
        self.dt_star = np.array([])
        self.cube = np.array([]) # this is a loglike cube
        self.Ic_list = np.array([])
        self.Is_list = np.array([])
        self.Ir_list = np.array([])

        # max likelihood data
        self.loglike_true_params = -1
        self.logLike_statsModels = -1
        self.p0 = np.array([]) # seed for calling the optimize routine
        self.p1 = np.array([]) # result of the optimize routine
        self.p0_list = np.array([])
        self.p1_list = np.array([])
        self.cube_max_params = np.array([]) # [Ic, Is, Ir] that give the maximum loglike in self.cube
        self.loglike_max_check = np.array([],dtype=bool)
        self.loglike_max_list = np.array([])
        self.statsModels_result = np.array([])
        self.optimize_res_list = []
        self.eval_time = []


        if Ic>0:
            self.gen_plist(Ic,Is,Ir,Ttot,tau,deadtime,return_IDs)
            self.setup_cube_lists()
            self.p0_get_simple()
            self.find_max_like()
            # self.get_cube()
            # self.p0_cube_max()
            # self.find_max_like()
            self.loglike_true_params = binfree.MRlogL([Ic, Is, Ir], self.dt, self.deadtime)
            # self.do_stats_models()


    def load(self,filename):
        self.filename = filename
        npzfile = np.load(self.filename)
        self.ts = npzfile['ts']
        self.p1 = npzfile['p1']

        self.dt = (self.ts[1:] - self.ts[:-1]) * 1e-6

    # def save(self): # this doesn't work- not sure why :(
    #     with open("/Users/clint/Dropbox/mazinlab/speckle/20181226/junk.p", "wb") as f:
    #         pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def gen_plist(self,Ic,Is,Ir,Ttot,tau,deadtime,return_IDs=False):
        self.Ic = Ic
        self.Is = Is
        self.Ir = Ir
        self.Ttot = Ttot
        self.tau = tau
        self.deadtime=deadtime  #should be units of seconds
        if return_IDs:
            self.ts, self.ts_star, _ = gpl.genphotonlist(Ic, Is, Ir, Ttot, tau, deadtime*10.**6., return_IDs)
            self.dt = (self.ts[1:] - self.ts[:-1]) * 1e-6
            self.dt_star = (self.ts_star[1:] - self.ts_star[:-1]) * 1e-6
        else:
            self.ts = gpl.genphotonlist(Ic, Is, Ir, Ttot, tau, deadtime*10.**6., return_IDs)
            self.dt = (self.ts[1:] - self.ts[:-1]) * 1e-6

    def get_cube(self, Ic_list=None, Is_list=None, Ir_list=None):
        if Ic_list is not None:
            self.Ic_list=Ic_list
        if Is_list is not None:
            self.Is_list=Is_list
        if Ir_list is not None:
            self.Ir_list=Ir_list
        if len(self.Ic_list) < 2 or len(self.Is_list) < 2 or len(self.Ir_list) < 2:
            print('one of the lists has a length < 2')
            return
        self.cube = binMR.logL_cube(self.ts, self.Ic_list, self.Is_list, self.Ir_list, partial_cube=True)


    def find_max_like(self, p0=None):
        if p0 is not None: self.p0=p0
        if self.p0.size == 0:
            print('you need to define a seed: self.p0')
            return
        t1 = time.time()
        res = optimize.minimize(binfree.MRlogL, self.p0, (self.dt, self.deadtime), method='Newton-CG', jac=binfree.MRlogL_Jacobian,
                               hess=binfree.MRlogL_Hessian)
        t2 = time.time()
        self.eval_time.append(t2-t1)
        self.p1 = res.x
        self.optimize_res_list.append(res)
        self.p0_list = np.append(self.p0_list, self.p0)
        self.p1_list = np.append(self.p1_list, self.p1)
        self.loglike_max_check = np.append(self.loglike_max_check, binMR.check_binfree_loglike_max(self.ts, self.p1, deadtime=self.deadtime))
        self.loglike_max_list = np.append(self.loglike_max_list,binfree.MRlogL(self.p1,self.dt,self.deadtime))


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
        self.Ir_list = np.linspace(0, 100, 21)


    def save_cubeslices(self,path,plist_number):
        try:
            os.makedirs(path)
        except:
            pass

        for ii in range(len(self.cube)):
            if np.all(self.cube[ii] < np.amax(self.cube)-8):
                continue
            fig, ax = plt.subplots()
            cax = ax.imshow(self.cube[ii] - np.amax(self.cube),extent=[np.amin(self.Ic_list), np.amax(self.Ic_list), np.amin(self.Is_list), np.amax(self.Is_list)], aspect = 'auto', origin = 'lower', cmap = 'hot_r', vmin = -8, vmax = 0, interpolation = 'spline16')
            plt.ylabel('Is [/s]')
            plt.xlabel('Ic [/s]')
            plt.title('Ic,Is,Ir, Ttot, tau = [{:g},{:g},{:g},{:g},{:g}]      Ir slice = {:.3g}'.format(self.Ic,self.Is,self.Ir,self.Ttot,self.tau,self.Ir_list[ii]))

            cb_ax = fig.add_axes([0.88, 0.1, 0.02, 0.8])  # [0.83, 0.1, 0.02, 0.8] [x0, y0, width, height]
            cbar = fig.colorbar(cax,cb_ax)
            fig.subplots_adjust(bottom=0.12, top=0.88, left=0.1, right=0.86, wspace=0.05, hspace=0.0)
            cbar.set_label(r'ln$\mathcal{L}$ - ln$\mathcal{L}_{max}$')

            filename = os.path.join(path, 'ph_list_{:g}_Ir_slice_{:.3g}.png'.format(plist_number, self.Ir_list[ii]))
            plt.savefig(filename, dpi=500)
            # plt.show()
            plt.close(fig)


    def do_stats_models(self):
        # now use Alex's code to estimate the parameters
        I = 1 / np.mean(self.dt)
        p0 = I*np.ones(3)/3.
        t1 = time.time()
        res = binfree.optimize_IcIsIr(self.dt, guessParams=p0, deadtime=self.deadtime, method='ncg')
        t2 = time.time()
        self.eval_time.append(t2 - t1)
        self.logLike_statsModels = binfree.MRlogL(res.params, self.dt, self.deadtime)

        #self.p0_list = np.append(self.p0_list, np.array([-1,-1,-1]))
        self.p0_list = np.append(self.p0_list, p0)
        self.p1_list = np.append(self.p1_list, np.asarray(res.params))



