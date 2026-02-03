import numpy as np
from .models import camel_density, camel_sampling_density

def entropy(x, p, weights=None):
    """
    x: data in the correct format to be plugged into p
    p: probability density function over x
    weights: per event weights if applicable
    """
    H = p(x) * np.log(p(x))
    if weights is not None:
        H *= weights
    H = -np.sum(H)
    return H


def cross_entropy(x, p, q, weights=None):
    """
    x: data in the correct format to be plugged into p
    p: reference probability density function over x
    q: probability density function over x (usually an estimate of p)
    weights: per event weights if applicable
    """
    H = p(x) * np.log(q(x))
    if weights is not None:
        H *= weights
    H = -np.sum(H)
    return H


def optimal_binary_classifier(source_mixture_coef, source_scales, target_mixture_coef, target_scales):
    """
    Returns a function of x,y under the assumptions of our toy problem
    """
    def f(x): # x can be multidimensional
        num = camel_density(x[:,0], x[:,1], target_mixture_coef, scales=target_scales)
        denom = num + camel_density(x[:,0], x[:,1], source_mixture_coef, scales=source_scales)
        #denom = num + models.bivariate_density(x[:,0], x[:,1], scale=source_scale)
        return num/denom
    return f


def optimal_sampling_binary_classifier(source_mixture_coef, source_scales, target_mixture_coef, target_scales):
    """
    Returns a function of x,y under the assumptions of our toy problem
    """
    def f(x): # x can be multidimensional
        num = camel_sampling_density(x[:,0], x[:,1], target_mixture_coef, scales=target_scales)
        denom = num + camel_sampling_density(x[:,0], x[:,1], source_mixture_coef, scales=source_scales)
        #denom = num + models.bivariate_density(x[:,0], x[:,1], scale=source_scale)
        return num/denom
    return f


def optimal_likelihood_ratio(source_mixture_coef, source_scales, target_mixture_coef, target_scales):
    """
    Returns a function of x,y under the assumptions of our toy problem
    """
    def f(x): # x can be multidimensional
        num = camel_density(x[:,0], x[:,1], target_mixture_coef, scales=target_scales)
        denom = camel_density(x[:,0], x[:,1], source_mixture_coef, scales=source_scales)
        #denom = models.bivariate_density(x[:,0], x[:,1], scale=source_scale)
        return num/denom
    return f


def optimal_sampling_likelihood_ratio(source_mixture_coef, source_scales, target_mixture_coef, target_scales):
    """
    Returns a function of x,y under the assumptions of our toy problem
    """
    def f(x): # x can be multidimensional
        num = camel_sampling_density(x[:,0], x[:,1], target_mixture_coef, scales=target_scales)
        denom = camel_sampling_density(x[:,0], x[:,1], source_mixture_coef, scales=source_scales)
        #denom = models.bivariate_density(x[:,0], x[:,1], scale=source_scale)
        return num/denom
    return f


def weight_function(mixture_coef, scales):
    """
    Returns a function of x,y under the assumptions of our toy problem
    """
    def f(x): # x can be multidimensional
        num = camel_density(x[:,0], x[:,1], mixture_coef, scales=scales)
        denom = camel_sampling_density(x[:,0], x[:,1], mixture_coef, scales=scales)
        #denom = models.bivariate_density(x[:,0], x[:,1], scale=source_scale)
        return num/denom
    return f