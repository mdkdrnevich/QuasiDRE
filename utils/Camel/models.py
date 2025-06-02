import numpy as np
import scipy as sp

def radial_density(r, scale=1, norm=1, logprob=False):
    scaled_r = r/scale
    if logprob is True:
        return -scaled_r**2 / 2 + np.log(norm/scale**2)
    else:
        p = r*np.exp(-scaled_r**2 / 2)
        p *= norm/scale**2
        return p


def radial_cdf(r, scale=1, norm=1):
    scaled_r = r/scale
    return norm*(1 - np.exp(-scaled_r**2 / 2))


def radial_quantile(z, scale=1):
    return np.sqrt(-2*scale**2 * np.log(1-z))
    

def angular_density(t, scale=1, norm=1, logprob=False):
    p = norm/(2*np.pi)
    if logprob is True:
        return np.log(p)
    else:
        return p
        

def bivariate_density(x, y, scale=1, norm=1, basis="cartesian", logprob=False):
    """
    (x,y): if cartesian, as is. If polar, then (r, \phi)
    basis: "cartesian" or "polar"

    p(x,y) = (np.e-1)/(np.pi*(x**2+y**2+1)*(x**2+y**2+np.e))
    p(r,\phi) = r*(np.e-1)/(np.pi*(r**2+1)*(r**2+np.e))
    """
    if basis=="cartesian":
        scaled_x = x/scale
        scaled_y = y/scale
        if logprob is True:
            p = -(scaled_x**2 + scaled_y**2)/2 + np.log(norm/(2*np.pi*scale**2))
        else:
            p = np.exp(-(scaled_x**2 + scaled_y**2)/2)
            p *= norm/(2*np.pi*scale**2)
            
    elif basis=="polar":
        if logprob is True:
            p = np.log(norm) + radial_density(x, scale=scale, logprob=True) + angular_density(y, scale=scale, logprob=True)
        else:
            p = norm*radial_density(x, scale=scale, logprob=False)*angular_density(y, scale=scale, logprob=False)
    return p
    


def camel_density(x, y, mixture_coef, scales=(1,1), basis="cartesian", logprob=False):
    """
    composed as: p_camel(x,y) = (coef1*p(x/scale1, y/scale1) + coef2*p(x/scale2, y/scale2))/(coef1 + coef2)
    """
    mixture_coef = np.array(mixture_coef)
    norm = np.sum(mixture_coef)

    p = mixture_coef[0]*bivariate_density(x, y, scale=scales[0], basis=basis, logprob=False)
    p += mixture_coef[1]*bivariate_density(x, y, scale=scales[1], basis=basis, logprob=False)
    p /= norm
    if logprob is True:
        return np.log(p)
    else:
        return p


def camel_sampling_density(x, y, mixture_coef, scales=(1,1), basis="cartesian", logprob=False):
    """
    composed as: p_camel(x,y) = (coef1*p(x/scale1, y/scale1) + coef2*p(x/scale2, y/scale2))/(coef1 + coef2)
    """
    mixture_coef = np.array(mixture_coef)
    norm = np.sum(mixture_coef)
    c = mixture_coef[0] / norm
    new_norm = 2*c-1

    p = c*bivariate_density(x, y, scale=scales[0], basis=basis, logprob=False)
    p += (c-1)*bivariate_density(x, y, scale=scales[1], basis=basis, logprob=False)
    p /= new_norm
    if logprob is True:
        return np.log(p)
    else:
        return p


def camel_radial_density(r, mixture_coef, scales=(1,1), logprob=False):
    mixture_coef = np.array(mixture_coef)
    norm = np.sum(mixture_coef)

    p = mixture_coef[0] * radial_density(r, scale=scales[0], logprob=False)
    p += mixture_coef[1] * radial_density(r, scale=scales[1], logprob=False)
    p /= norm
    if logprob is True:
        return np.log(p)
    else:
        return p


def camel_radial_sampling_density(r, mixture_coef, scales=(1,1), logprob=False):
    mixture_coef = np.array(mixture_coef)
    norm = np.sum(mixture_coef)
    c = mixture_coef[0] / norm
    new_norm = 2*c-1

    p = c * radial_density(r, scale=scales[0], logprob=False)
    p += (c-1) * radial_density(r, scale=scales[1], logprob=False)
    p /= new_norm
    if logprob is True:
        return np.log(p)
    else:
        return p


def camel_radial_cdf(r, mixture_coef, scales=(1,1)):
    norms = (mixture_coef[0]/(mixture_coef[0]+mixture_coef[1]), mixture_coef[1]/(mixture_coef[0]+mixture_coef[1]))
    z = radial_cdf(r, scale=scales[0], norm=norms[0]) + radial_cdf(r, scale=scales[1], norm=norms[1])
    return z


def camel_radial_quantile(z, mixture_coef, scales=(1,1)):
    if np.equal(*scales) is True:
        return radial_quantile(z, scale=scales[0])
    else:
        return sp.optimize.fsolve(lambda r: camel_radial_cdf(r, mixture_coef, scales=scales) - z, 1)