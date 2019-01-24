import numpy as np



def multiple_2d_circ_gauss_func(p_guess):

    def f(p, fjac=None, data=None, err=None,return_models=False):
        #p[0] = background
        #p[1] = amplitude
        #p[2] = x_offset
        #p[3] = y_offset
        #p[4] = sigma for both x and y
        #And so on for the 2nd and 3rd gaussians etc...
        #A+Be^-((x-xo)^2+(y-y0)^2)/2s^2 + Ce^-((x-x1)^2+(y-y1)^2)/2d^2 + ...
        #print p_guess
        #print p
        #print range(len(p[1:])%4)
        
        models=[p[0]*p_guess[0]*np.ones(np.asarray(data).shape)]
        x=np.tile(range(len(data[0])),(len(data),1))
        y=np.tile(range(len(data)),(len(data[0]),1)).transpose()
        for i in range(len(p[1:])/4):
            m = p[1+4*i]*p_guess[1+4*i] * np.exp( - (pow(x-p[2+i*4]*p_guess[2+i*4],2)+pow(y-p[3+i*4]*p_guess[3+i*4],2)) / (2 * pow(p[4+i*4]*p_guess[4+i*4],2)) )
            models.append(m)
        if return_models: return models
        model = models[0]
        for m in models[1:]: model+=m
        status = 0
        return([status,np.ravel((data-model)/err)])

    return f

def benitez2(p, fjac=None, x=None, y=None, err=None):
    model = pow(x,p[0]) * np.exp( -(pow(x/p[1]),p[2]))
    status = 0
    return([status, (y-model)/err])    

def parabola_old(p, fjac=None, x=None, y=None, err=None,return_models=False):
    #p[0] = x_offset
    #p[1] = y_offset
    #p[2] = amplitude
    model = p[2] * (pow( (x - p[0]), 2 )) + p[1]
    if return_models:
        return [model]
    status = 0
    return([status, (y-model)/err])

def parabola(p, fjac=None, x=None, y=None, err=None,return_models=False):
    #p[0] = constant term
    #p[1] = linear term
    #p[2] = quadratic term
    model = p[0]+p[1]*x+p[2]*x**2
    if return_models:
        return [model]
    status = 0
    return([status, (y-model)/err])

def gaussian(p, fjac=None, x=None, y=None, err=None,return_models=False):
    #p[0] = sigma
    #p[1] = x_offset
    #p[2] = amplitude
    #p[3] = y_offset
#    model = p[3] + p[2] * np.exp( - (pow(( x - p[1]),2) / ( 2. * pow(p[0],2))))
    model = p[2] * np.exp( - (pow(( x - p[1]),2) / ( 2. * pow(p[0],2))))
    if return_models:
        return [model]
    # Non-negative status value means MPFIT should continue, negative means
    # stop the calculation.
    status = 0
    return([status, (y-model)/err])

def twogaussian(p, fjac=None, x=None, y=None, err=None):
    #p[0] = sigma1
    #p[1] = x_offset1
    #p[2] = amplitude1
    #p[3] = sigma2
    #p[4] = x_offset2
    #p[5] = amplitude2
    gauss1 = p[2] * np.exp( - (pow(( x - p[1]),2) / ( 2. * pow(p[0],2))))
    gauss2 = p[5] * np.exp( - (pow(( x - p[4]),2) / ( 2. * pow(p[3],2))))
    model = gauss1 + gauss2 
    status = 0
    return([status, (y-model)/err])

def twogaussianexp(p, fjac=None, x=None, y=None, err=None):
    #p[0] = sigma1
    #p[1] = x_offset1
    #p[2] = amplitude1
    #p[3] = sigma2
    #p[4] = x_offset2
    #p[5] = amplitude2
    #p[6] = scalefactor
    #p[7] = x_offset3
    #p[8] = amplitude3
    gauss1 = p[2] * np.exp( - (pow(( x - p[1]),2) / ( 2. * pow(p[0],2))))
    gauss2 = p[5] * np.exp( - (pow(( x - p[4]),2) / ( 2. * pow(p[3],2))))
    expo = p[8] * np.exp(p[6] * (x - p[7]))
    model = gauss1 + gauss2 + expo 
    status = 0
    return([status, (y-model)/err])

def threegaussian(p, fjac=None, x=None, y=None, err=None):
    #p[0] = sigma1
    #p[1] = x_offset1
    #p[2] = amplitude1
    #p[3] = sigma2
    #p[4] = x_offset2
    #p[5] = amplitude2
    #p[6] = sigma3
    #p[7] = x_offset3
    #p[8] = amplitude3
    gauss1 = p[2] * np.exp( - (pow(( x - p[1]),2) / ( 2. * pow(p[0],2))))
    gauss2 = p[5] * np.exp( - (pow(( x - p[4]),2) / ( 2. * pow(p[3],2))))
    gauss3 = p[8] * np.exp( - (pow(( x - p[7]),2) / ( 2. * pow(p[6],2))))
    model = gauss1 + gauss2 + gauss3
    status = 0
    return([status, (y-model)/err])

def fourgaussian(p, fjac=None, x=None, y=None, err=None,return_models=False):
    #p[0] = sigma1
    #p[1] = x_offset1
    #p[2] = amplitude1
    #p[3] = sigma2
    #p[4] = x_offset2-x_offset1
    #p[5] = amplitude2
    #p[6] = sigma3
    #p[7] = x_offset3-xoffset2
    #p[8] = amplitude3
    #p[9] = sigma4
    #p[10] = x_offset4
    #p[11] = amplitude4
    gauss1 = p[2] * np.exp( - (pow(( x - p[1]),2) / ( 2. * pow(p[0],2))))
    gauss2 = p[5] * np.exp( - (pow(( x - p[4]-p[1]),2) / ( 2. * pow(p[3],2))))
    gauss3 = p[8] * np.exp( - (pow(( x - p[7]-p[4]-p[1]),2) / ( 2. * pow(p[6],2))))
    gauss4 = p[11] * np.exp( - (pow(( x - p[10]),2) / ( 2. * pow(p[9],2))))
    model = gauss1 + gauss2 + gauss3 + gauss4
    if return_models:
        return [gauss1, gauss2, gauss3, gauss4]
    status = 0
    return([status, (y-model)/err])
    
def fourgaussian_pow(p, fjac=None, x=None, y=None, err=None,return_models=False):
    #p[0] = sigma1
    #p[1] = x_offset1
    #p[2] = amplitude1
    #p[3] = sigma2
    #p[4] = x_offset2-x_offset1
    #p[5] = amplitude2
    #p[6] = sigma3
    #p[7] = x_offset3-xoffset2
    #p[8] = amplitude3
    #p[9] = sigma4
    #p[10] = x_offset4
    #p[11] = amplitude4
    gauss1 = p[2] * np.exp( - (pow(( x - p[1]),2) / ( 2. * pow(p[0],2))))
    gauss2 = p[5] * np.exp( - (pow(( x - p[4]-p[1]),2) / ( 2. * pow(p[3],2))))
    gauss3 = p[8] * np.exp( - (pow(( x - p[7]-p[4]-p[1]),2) / ( 2. * pow(p[6],2))))
    gauss4 = p[11] * np.exp( - (pow(( x - p[10]),2) / ( 2. * pow(p[9],2))))
    model = gauss1 + gauss2 + gauss3 + gauss4
    if return_models:
        return [gauss1, gauss2, gauss3, gauss4]
    status = 0
    return([status, (y-model)/err])

def threegaussian_exp(p, fjac=None, x=None, y=None, err=None,return_models=False):
    #p[0] = sigma1
    #p[1] = x_offset1
    #p[2] = amplitude1
    #p[3] = sigma2
    #p[4] = x_offset2-x_offset1
    #p[5] = amplitude2
    #p[6] = sigma3
    #p[7] = x_offset3-xoffset2
    #p[8] = amplitude3
    #p[9] = scale_factor
    #p[10] = x_offset4
    #p[11] = amplitude4
    gauss1 = p[2] * np.exp( - (pow(( x - p[1]),2) / ( 2. * pow(p[0],2))))
    gauss2 = p[5] * np.exp( - (pow(( x - p[4]-p[1]),2) / ( 2. * pow(p[3],2))))
    gauss3 = p[8] * np.exp( - (pow(( x - p[7]-p[4]-p[1]),2) / ( 2. * pow(p[6],2))))
    expo = p[11] * np.exp(p[9] * (x - p[10]))
    model = gauss1 + gauss2 + gauss3 + expo
    if return_models:
        return [gauss1, gauss2, gauss3, expo]
    status = 0
    return([status, (y-model)/err])


def threegaussian_exppow(p, fjac=None, x=None, y=None, err=None,return_models=False):
    #p[0] = sigma1
    #p[1] = x_offset1
    #p[2] = amplitude1
    #p[3] = sigma2
    #p[4] = x_offset2
    #p[5] = amplitude2
    #p[6] = sigma3
    #p[7] = x_offset3
    #p[8] = amplitude3
    #p[9] = scale_factor
    #p[10] = x_offset4
    #p[11] = amplitude4
    #p[12] = power4
    gauss1 = p[2] * np.exp( - (pow(( x - p[1]),2) / ( 2. * pow(p[0],2))))
    gauss2 = p[5] * np.exp( - (pow(( x - p[4]),2) / ( 2. * pow(p[3],2))))
    gauss3 = p[8] * np.exp( - (pow(( x - p[7]),2) / ( 2. * pow(p[6],2))))
    expo = p[11] * np.exp(p[9] * (-(p[10] - x)**p[12]))
    model = gauss1 + gauss2 + gauss3 + expo
    if return_models:
        return [gauss1, gauss2, gauss3, expo]
    status = 0
    return([status, (y-model)/err])

def threegaussian_moyal(p, fjac=None, x=None, y=None, err=None,return_models=False):
    #p[0] = sigma1
    #p[1] = x_offset1
    #p[2] = amplitude1
    #p[3] = sigma2
    #p[4] = x_offset2
    #p[5] = amplitude2
    #p[6] = sigma3
    #p[7] = x_offset3
    #p[8] = amplitude3
    #p[9] = sigma4
    #p[10] = x_offset4
    #p[11] = amplitude4
    gauss1 = p[2] * np.exp( - (pow(( x - p[1]),2) / ( 2. * pow(p[0],2))))
    gauss2 = p[5] * np.exp( - (pow(( x - p[4]),2) / ( 2. * pow(p[3],2))))
    gauss3 = p[8] * np.exp( - (pow(( x - p[7]),2) / ( 2. * pow(p[6],2))))
    moyal = p[11] * np.exp( - 0.5 * (np.exp( - (-x - p[10])/p[9]) + (-x - p[10])/p[9] + 1))
    model = gauss1 + gauss2 + gauss3 + moyal
    if return_models:
        return [gauss1, gauss2, gauss3, moyal]
    status = 0
    return([status, (y-model)/err])

def threegaussian_power(p, fjac=None, x=None, y=None, err=None,return_models=False):
    #p[0] = sigma1
    #p[1] = x_offset1
    #p[2] = amplitude1
    #p[3] = sigma2
    #p[4] = x_offset2-x_offset1
    #p[5] = amplitude2
    #p[6] = sigma3
    #p[7] = x_offset3-xoffset2
    #p[8] = amplitude3
    #p[9] = scale_factor4
    #p[10] = x_offset4
    #p[11] = amplitude4
    gauss1 = p[2] * np.exp( - (pow(( x - p[1]),2) / ( 2. * pow(p[0],2))))
    gauss2 = p[5] * np.exp( - (pow(( x - p[4]-p[1]),2) / ( 2. * pow(p[3],2))))
    gauss3 = p[8] * np.exp( - (pow(( x - p[7]-p[4]-p[1]),2) / ( 2. * pow(p[6],2))))
    power4 = p[11] * np.maximum((x - p[10]),0)**p[9]
    model = gauss1 + gauss2 + gauss3 + power4
    if return_models:
        return [gauss1, gauss2, gauss3, power4]
    status = 0
    return([status, (y-model)/err])

def threegaussian_lorentzian(p, fjac=None, x=None, y=None, err=None,return_models=False):
    #p[0] = sigma1
    #p[1] = x_offset1
    #p[2] = amplitude1
    #p[3] = sigma2
    #p[4] = x_offset2
    #p[5] = amplitude2
    #p[6] = sigma3
    #p[7] = x_offset3
    #p[8] = amplitude3
    #p[9] = scale_factor4
    #p[10] = x_offset4
    #p[11] = amplitude4
    gauss1 = p[2] * np.exp( - (pow(( x - p[1]),2) / ( 2. * pow(p[0],2))))
    gauss2 = p[5] * np.exp( - (pow(( x - p[4]),2) / ( 2. * pow(p[3],2))))
    gauss3 = p[8] * np.exp( - (pow(( x - p[7]),2) / ( 2. * pow(p[6],2))))
    lorentz4 = p[11] / (1 + ((x - p[10])/p[9])**2)
    model = gauss1 + gauss2 + gauss3 + lorentz
    if return_models:
        return [gauss1, gauss2, gauss3, lorentz]
    status = 0
    return([status, (y-model)/err])
    
model_list = {
    'parabola': parabola,
    'gaussian': gaussian,
    'fourgaussian': fourgaussian,
    'threegaussian_exp': threegaussian_exp,
    'threegaussian_exppow': threegaussian_exppow,
    'threegaussian_moyal': threegaussian_moyal,
    'threegaussian_power': threegaussian_power,
    'threegaussian_lorentzian': threegaussian_lorentzian,
    'multiple_2d_circ_gauss_func': multiple_2d_circ_gauss_func}

