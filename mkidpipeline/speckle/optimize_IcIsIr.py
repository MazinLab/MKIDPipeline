from __future__ import print_function
import numpy as np
from scipy import optimize, ndimage
import time
from mkidpipeline.speckle.genphotonlist_IcIsIr import genphotonlist
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
import matplotlib.ticker as ticker

def loglike(p, dt, deadtime=0, Ir_slice = 0):

    Ic = p[0]
    Is = p[1]
    if len(p) > 2:
        Ir = p[2]
    else:
        Ir = Ir_slice

    ###################################################################
    # Intensity should be strictly positive, and each Ic, Is, Ir
    # should be nonegative.
    ###################################################################

    if Ic < 0 or Is < 0 or Ir < 0 or Ic + Is + Ir == 0:
        return 1e100 
    
    ###################################################################
    # Pre-compute combinations of variables
    ###################################################################

    nslice = 20
    u = 1./(1 + dt*Is)
    u2 = u*u
    u3 = u2*u
    u4 = u2*u2
    u5 = u4*u
    N = len(u)
    
    umax = 1./(1 + deadtime*Is)
    arg_log = (Ic**2)*u5 + (4*Ic*Is)*u4 + (2*Is**2 + 2*Ir*Ic)*u3
    if Ir > 0:
        arg_log += (2*Ir*Is)*u2 + (Ir**2)*u
    
    ###################################################################
    # Reshape the array and multiply along the short axis to reduce
    # the number of calls to np.log.
    ###################################################################

    _n = (len(u)//nslice + 1)*nslice
    _arg_log = np.ones(_n)
    _arg_log[:len(u)] = arg_log
    _arg_log = np.prod(np.reshape(_arg_log, (nslice, -1)), axis=0)
    lnL = np.sum(np.log(_arg_log))

    lnL += -Ir*np.sum(dt) + Ic/Is*np.sum(u) - N*Ic/Is
    lnL -= N*(umax - 1)*(Ir + Ic*umax)/(Is*umax)
    lnL -= N*np.log(Ic*umax**3 + Is*umax**2 + Ir*umax)
    if np.isfinite(lnL):
        return -lnL
    else:
        return 1e100

def _jacobean(p, dt, deadtime=0, Ir_slice = 0):
    if len(p) > 2:
        return -1*jacobean(p[0], p[1], p[2], dt, deadtime)
    else:
        return -1*jacobean(p[0], p[1], Ir_slice, dt, deadtime)[:2]

def jacobean(Ic, Is, Ir, dt, deadtime=0):

    ###################################################################
    # Pre-compute combinations of variables
    ###################################################################

    u = 1./(1 + dt*Is)
    umax = 1./(1 + deadtime*Is)
    u2 = u*u
    u3 = u2*u
    u4 = u2*u2
    sum_u = np.sum(u)
    N = len(u)
    denom_inv = 1/(Ic**2*u4 + 4*Ic*Is*u3 + (2*Is**2 + 2*Ir*Ic)*u2 + 2*Ir*Is*u + Ir**2)

    d_Ic = 2*np.sum((Ic*u4 + 2*Is*u3 + Ir*u2)*denom_inv)
    d_Ic += (sum_u - N)/Is
    d_Ic -= N*umax**2/(Ic*umax**2 + Is*umax + Ir)
    d_Ic += N*(1 - umax)/Is

    d_Ir = 2*np.sum((Ic*u2 + Is*u + Ir)*denom_inv)
    d_Ir -= np.sum(dt)
    d_Ir += N*deadtime
    d_Ir -= N/(Ic*umax**2 + Is*umax + Ir)

    num = (4*Ic**2/Is)*u4*u + (12*Ic - 4*Ic**2/Is)*u4
    num += (4*Is - 8*Ic + 4*Ir*Ic/Is)*u3 + (2*Ir - 4*Ir*Ic/Is)*u2

    d_Is = np.sum(num*denom_inv)
    d_Is += Ic/Is**2*(np.sum(u2) - 2*sum_u + N)
    d_Is += (sum_u - N)/Is
    d_Is -= N*Ic*((umax - 1)/Is)**2
    d_Is -= N*(umax - 1)/Is
    d_Is -= N*umax**2*(2*Ic*(umax - 1)/Is + 1)/(Ic*umax**2 + Is*umax + Ir)

    return np.asarray([d_Ic, d_Is, d_Ir])

def _hessian(p, dt, deadtime=0, Ir_slice = 0):
    if len(p) > 2:
        return -1*hessian(p[0], p[1], p[2], dt, deadtime)
    else:
        return -1*hessian(p[0], p[1], Ir_slice, dt, deadtime)[:2, :2]

def hessian(Ic, Is, Ir, dt, deadtime=0):
    
    ###################################################################
    # Pre-compute combinations of variables
    ###################################################################

    u = 1./(1 + dt*Is)
    umax = 1./(1 + deadtime*Is)
    u2 = u*u
    u3 = u2*u
    u4 = u2*u2
    u5 = u4*u
    u_m_1 = u - 1.
    u_m_1_sq = u_m_1*u_m_1
    N = len(u)
    
    denom_inv = 1./((Ic**2)*u4 + (4*Ic*Is)*u3 + (2*Is**2 + 2*Ir*Ic)*u2 + (2*Ir*Is)*u + Ir**2)
    denom2_inv = denom_inv*denom_inv

    num_Ic = Ic*u4 + (2*Is)*u3 + Ir*u2
    d_IcIc = -4*np.sum(num_Ic*num_Ic*denom2_inv)
    d_IcIc += 2*np.sum(u4*denom_inv)
    d_IcIc += N*(umax**2/(Ic*umax**2 + Is*umax + Ir))**2

    num_Ir = Ic*u2 + Is*u + Ir
    d_IrIr = -4*np.sum(num_Ir*num_Ir*denom2_inv)
    d_IrIr += 2*np.sum(denom_inv)
    d_IrIr += N/(Ic*umax**2 + Is*umax + Ir)**2

    d_IcIr = -4*np.sum(num_Ir*num_Ic*denom2_inv)
    d_IcIr += 2*np.sum(u2*denom_inv)
    d_IcIr += N*(umax/(Ic*umax**2 + Is*umax + Ir))**2

    num_Is = (4*Ic**2/Is)*u5 + (12*Ic - 4*Ic**2/Is)*u4
    num_Is += (4*Is - 8*Ic + 4*Ir*Ic/Is)*u3 + (2*Ir - 4*Ir*Ic/Is)*u2

    argsum = (u_m_1*((2.*Ic/Is)*u2 + 3.*u + Ir/Is) + u)*denom_inv
    argsum -= ((Ic/2.)*u2 + Is*u + Ir/2.)*num_Is*denom2_inv
    d_IcIs = 4*np.sum(u2*argsum)
    d_IcIs  += np.sum(u_m_1_sq)/Is**2

    d_IcIs -= N*((umax - 1)/Is)**2
    d_IcIs += N*umax**4*(2*Ic*(umax - 1)/Is + 1)/(Ic*umax**2 + Is*umax + Ir)**2
    d_IcIs -= 2*N*umax**2*(umax - 1)/Is/(Ic*umax**2 + Is*umax + Ir)

    d_IrIs = np.sum(((4*Ic/Is)*u3 + (2 - 4*Ic/Is)*u2)*denom_inv)
    d_IrIs -= 2*np.sum((Ic*u2 + Is*u + Ir)*num_Is*denom2_inv)
    d_IrIs += N*umax**2*(2*Ic*(umax - 1)/Is + 1)/(Ic*umax**2 + Is*umax + Ir)**2

    argsum = u_m_1_sq*((5*Ic**2/Is**2)*u2 + (6*Ic/Is)*u + 3*Ic*Ir/Is**2)
    argsum += u_m_1*((6*Ic/Is)*u2 + 3*u + Ir/Is) + u
    d_IsIs = 4*np.sum(u2*argsum*denom_inv)

    d_IsIs -= np.sum(num_Is*num_Is*denom2_inv)
    d_IsIs += 2*Ic/Is**3*np.sum(u_m_1_sq*u_m_1)
    d_IsIs += np.sum(u_m_1_sq)/Is**2
    d_IsIs -= 2*N*Ic*((umax - 1)/Is)**3
    d_IsIs -= N*((umax - 1)/Is)**2
    d_IsIs += N*(umax**2*(2*Ic*(umax - 1)/Is + 1)/(Ic*umax**2 + Is*umax + Ir))**2
    d_IsIs -= N*(6*Ic*umax**2*(umax - 1)**2/Is**2 + 2*umax**2*(umax - 1)/Is)/(Ic*umax**2 + Is*umax + Ir)

    return np.asarray([[d_IcIc, d_IcIs, d_IcIr],
                       [d_IcIs, d_IsIs, d_IrIs],
                       [d_IcIr, d_IrIs, d_IrIr]])



def _slice(arr, k, maxval, njp, nkp, jmax, lnLmin, thresh):

    ####################################################################
    # Step along in j from j1 to j2.  Take advantage of the smoothness
    # of the distribution to avoid doing any calculations where the
    # log likelihood is below the maximum minus thresh.  Save the
    # best-fit j to start the march in the other direction.
    ####################################################################

    lnLbest = np.inf
    
    for j in range(jmax, njp):
        Ic = k*maxval/nkp
        Is = j*maxval/njp - Ic
            
        if Ic < 0 or Is < 0 or j > njp or k > nkp:
            continue
        lnL = loglike([Ic + 1e-5, Is + 1e-5], dt_us, deadtime_us)
        if lnL < lnLbest:
            jmax = j
            lnLbest = lnL
        arr[j, k] = lnL
        if lnL > lnLmin + thresh:
            break
        
    for j in range(jmax, 0, -1):
        Ic = k*maxval/nkp
        Is = j*maxval/njp - Ic
            
        if Ic < 0 or Is < 0 or j > njp or k > nkp:
            continue
        lnL = loglike([Ic + 1e-5, Is + 1e-5], dt_us, deadtime_us)
        if lnL < lnLbest:
            jmax = j
            lnLbest = lnL
        arr[j, k] = lnL
        if lnL > lnLmin + thresh:
            break

    return jmax

if __name__ == "__main__":
    
    Ic, Is, Ir, Ttot, tau, deadtime = [30, 300, 0, 30, 0.1, 10]
    t = genphotonlist(Ic, Is, Ir, Ttot, tau, deadtime=deadtime)    

    do_Ir = False
    doIcpIs = False #y-axis is Ic+Is?  If False, y-axis is Ic.
    
    #####################################################################
    # All units to microseconds.  Initial guess for parameters: equal
    # photons from each component of the distribution
    #####################################################################

    deadtime_us = deadtime*1e-6
    dt_us = (t[1:] - t[:-1])*1e-6  # units of dt_us are seconds, not microseconds. CB 20180926
    I = 1/np.mean(dt_us)

    if do_Ir:
        p0 = I*np.ones(3)/3.
    else:
        p0 = I*np.ones(2)/2.
        
    p1 = optimize.minimize(loglike, p0, (dt_us, deadtime_us),
                           method='Newton-CG', jac=_jacobean, hess=_hessian).x
    print(p1)

    # Diagonalize covariance matrix
    
    C = np.linalg.inv(_hessian(p1, dt_us, deadtime_us))
    u, s, v = np.linalg.svd(_hessian(p1, dt_us, deadtime_us))
    sig1, sig2 = np.sqrt(1./s)
    maxval = np.sum(p1)*1.5

    #####################################################################
    # Calculate likelihood on a sparse array.  We should be able to do
    # a better job of this by being fancier; right now the best
    # results come with njp and nkp (number of points in each
    # direction) manually set.  An explicit loop with an uneven tiling
    # using interpolate.griddata can probably beat the approach given
    # here; we should try to make this work and make it reasonably
    # efficient.
    #####################################################################
    
    njp = 100 #5*int(maxval/(sig2 + 0.1*(sig1**2 + sig2**2)**0.5)) + 5 #100
    nkp = 200 #5*int(maxval/(sig1 + 0.1*(sig1**2 + sig2**2)**0.5)) + 5 #200

    outarr = np.zeros((njp + 1, nkp + 1)) + np.inf
    
    #####################################################################
    # Only bother calculating for points with log likelihood no more
    # than thresh below the best-fit.  Then start with the best
    # log-likelihood and march in each direction.
    #####################################################################
    
    lnLmin = loglike(p1, dt_us, deadtime_us)
    thresh = 20
    jmax = int(np.sum(p1)*njp/maxval)
    kmax = int(p1[0]*nkp/maxval)
    for k in range(kmax, nkp + 1):
        jmax = _slice(outarr, k, maxval, njp, nkp, jmax, lnLmin, thresh)

    jmax = int(np.sum(p1)*njp/maxval)
    for k in range(kmax, -1, -1):
        jmax = _slice(outarr, k, maxval, njp, nkp, jmax, lnLmin, thresh)
    
    #####################################################################
    # Array should be zero below lnLmax - thresh.  Then interpolate
    # onto the desired array defined by x, y.
    #####################################################################
    
    outarr = lnLmin + thresh - outarr
    outarr[np.where(np.logical_not(np.isfinite(outarr)))] = 0
    outarr *= outarr > 0
    outarr -= thresh - 8
    x = np.arange(451)/450.*nkp
    y = np.arange(451)/450.*njp
    x, y = np.meshgrid(x, y)
    arr_interp = ndimage.map_coordinates(outarr, [y, x])

    if not doIcpIs:
        _x = np.arange(x.shape[1])
        _y = np.arange(y.shape[0])
        _x, _y = np.meshgrid(_x, _y)
        _y += _x
        arr_interp = ndimage.map_coordinates(arr_interp, [_y, _x])
            
    outarr = arr_interp*(arr_interp > 0)
    x *= maxval/nkp
    y *= maxval/njp

    f, ax = plt.subplots(1, figsize=(4, 4))
    ax.imshow(outarr[::-1], extent=(x.min(), x.max(), y.min(), y.max()), aspect=1, interpolation='bilinear', cmap=cm.hot_r)
    if doIcpIs:
        ax.set_ylabel('$I_c + I_s$')
    else:
        ax.set_ylabel('$I_s$')
    ax.set_xlabel('$I_c$')
    sm = plt.cm.ScalarMappable(cmap=cm.hot_r, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbaxes = f.add_axes([0.55, 0.75, 0.3, 0.03]) 
    cb = plt.colorbar(sm, cax=cbaxes, ticks=[0, 0.25, 0.5, 0.75, 1],
                      orientation='horizontal')
    cb.ax.set_xticklabels(['$-8$', '$-6$', '$-4$', '$-2$', '$0$'])
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(axis='both', which='major', labelsize=8, pad=-1)
    cbaxes.set_xlabel("$\\ln\\mathcal{L} - \\ln\\mathcal{L}_\mathrm{max}$",
                      fontsize='medium', labelpad=6)
    
    plt.savefig('sample.png', bbox_inches='tight', dpi=300)

    
