import numpy as np
from .models import radial_quantile, camel_radial_quantile


def sample_radial(n_samples, scale=1):
    u = np.random.sample(n_samples)
    return radial_quantile(u, scale=scale)


def sample_camel(n_samples, mixture_coef, scales=(1,1), basis="cartesian"):
    u = np.random.sample(n_samples)
    if np.equal(*scales) is True: # analytical
        r = camel_radial_quantile(u, mixture_coef, scales=scales)
    else: # numerical
        r = np.concatenate([camel_radial_quantile(u_i, mixture_coef, scales=scales) for u_i in u])
    phi = 2*np.pi*np.random.sample(n_samples)
    if basis == "polar":
        return np.hstack([r.reshape(-1,1), phi.reshape(-1,1)])
    elif basis == "cartesian":
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return np.hstack([x.reshape(-1,1), y.reshape(-1,1)])


def sample_camel_mc(n_samples, mixture_coef, scales=(1,1), basis="cartesian", weight_scales=(1,1), weight_spreads=(0,0), weight_dist="gaussian"):
    min_arg, max_arg = np.argmin(mixture_coef), np.argmax(mixture_coef)
    min_coef, max_coef = mixture_coef[min_arg]/sum(mixture_coef), mixture_coef[max_arg]/sum(mixture_coef)

    phi = 2*np.pi*np.random.sample(n_samples)
    r_arr = np.empty(n_samples)
    w_arr = np.empty(n_samples)
    i=0
    while i < n_samples:
        # If we have a negative component
        if min_coef < 0:
            flip = (1-2*min_coef)*np.random.sample() + min_coef  # sample from (2*|b|+1)U - |b| ~ U(b, 1-b)
            if flip < 0: # sample from negative only
                if weight_scales[min_arg] > 1:
                    cutoff = 1/weight_scales[min_arg]
                    if np.random.sample() < cutoff:
                        r_arr[i] = sample_radial(1, scale=scales[min_arg]).item()  # get samples
                        if weight_dist == "gaussian":
                            w_arr[i] = np.random.normal(loc=-weight_scales[min_arg], scale=weight_spreads[min_arg])
                        elif weight_dist == "lognormal":
                            mu = np.log(weight_scales[min_arg]) - weight_spreads[min_arg]**2 / 2
                            w_arr[i] = -np.random.lognormal(mean=mu, sigma=weight_spreads[min_arg])
                        else:
                            raise Exception("NOT IMPLEMENTED YET. PLEASE CHOOSE EITHER GAUSSIAN OR LOGNORMAL DISTRIBUTIONS")
                        i += 1
                    else:
                        continue
                elif weight_scales[min_arg] == 1:
                    r_arr[i] = sample_radial(1, scale=scales[min_arg]).item()  # get samples
                    if weight_dist == "gaussian":
                        w_arr[i] = np.random.normal(loc=-1, scale=weight_spreads[min_arg])
                    elif weight_dist == "lognormal":
                        mu = -weight_spreads[min_arg]**2 / 2
                        w_arr[i] = -np.random.lognormal(mean=mu, sigma=weight_spreads[min_arg])
                    else:
                        raise Exception("NOT IMPLEMENTED YET. PLEASE CHOOSE EITHER GAUSSIAN OR LOGNORMAL DISTRIBUTIONS")
                    i += 1
                else:
                    raise Exception("NOT IMPLEMENTED YET. PLEASE CHOOSE A WEIGHT SCALE >= 1")
            else: # sample from positive only
                if weight_scales[max_arg] > 1:
                    cutoff = 1/weight_scales[max_arg]
                    if np.random.sample() < cutoff:
                        r_arr[i] = sample_radial(1, scale=scales[max_arg]).item()  # get samples
                        if weight_dist == "gaussian":
                            w_arr[i] = np.random.normal(loc=weight_scales[max_arg], scale=weight_spreads[max_arg])
                        elif weight_dist == "lognormal":
                            mu = np.log(weight_scales[max_arg]) - weight_spreads[max_arg]**2 / 2
                            w_arr[i] = np.random.lognormal(mean=mu, sigma=weight_spreads[max_arg])
                        else:
                            raise Exception("NOT IMPLEMENTED YET. PLEASE CHOOSE EITHER GAUSSIAN OR LOGNORMAL DISTRIBUTIONS")
                        i += 1
                    else:
                        continue
                elif weight_scales[max_arg] == 1:
                    r_arr[i] = sample_radial(1, scale=scales[max_arg]).item()  # get samples
                    if weight_dist == "gaussian":
                        w_arr[i] = np.random.normal(loc=1, scale=weight_spreads[max_arg])
                    elif weight_dist == "lognormal":
                        mu = -weight_spreads[max_arg]**2 / 2
                        w_arr[i] = np.random.lognormal(mean=mu, sigma=weight_spreads[max_arg])
                    else:
                        raise Exception("NOT IMPLEMENTED YET. PLEASE CHOOSE EITHER GAUSSIAN OR LOGNORMAL DISTRIBUTIONS")
                    i += 1
                else:
                    raise Exception("NOT IMPLEMENTED YET. PLEASE CHOOSE A WEIGHT SCALE >= 1")
        else: # Normal positive distributions
            flip = np.random.sample()  # sample from U(0,1)
            if flip < min_coef:
                r_arr[i] = sample_radial(1, scale=scales[min_arg]).item()  # get samples
                if weight_dist == "gaussian":
                    w_arr[i] = np.random.normal(loc=weight_scales[min_arg], scale=weight_spreads[min_arg])
                elif weight_dist == "lognormal":
                    mu = np.log(weight_scales[min_arg]) - weight_spreads[min_arg]**2 / 2
                    w_arr[i] = np.random.lognormal(mean=mu, sigma=weight_spreads[min_arg])
                else:
                    raise Exception("NOT IMPLEMENTED YET. PLEASE CHOOSE EITHER GAUSSIAN OR LOGNORMAL DISTRIBUTIONS")
                i += 1
            else:
                r_arr[i] = sample_radial(1, scale=scales[max_arg]).item()  # get samples
                if weight_dist == "gaussian":
                    w_arr[i] = np.random.normal(loc=weight_scales[max_arg], scale=weight_spreads[max_arg])
                elif weight_dist == "lognormal":
                    mu = np.log(weight_scales[max_arg]) - weight_spreads[max_arg]**2 / 2
                    w_arr[i] = np.random.lognormal(mean=mu, sigma=weight_spreads[max_arg])
                else:
                    raise Exception("NOT IMPLEMENTED YET. PLEASE CHOOSE EITHER GAUSSIAN OR LOGNORMAL DISTRIBUTIONS")
                i += 1
                
    if basis == "polar":
        return np.hstack([r_arr.reshape(-1,1), phi.reshape(-1,1), w_arr.reshape(-1,1)])
    elif basis == "cartesian":
        x = r_arr * np.cos(phi)
        y = r_arr * np.sin(phi)
        return np.hstack([x.reshape(-1,1), y.reshape(-1,1), w_arr.reshape(-1,1)])


def __sample_camel_mc(n_samples, mixture_coef, scales=(1,1), basis="cartesian", weight_scale=1, weight_spread=0):
    min_arg, max_arg = np.argmin(mixture_coef), np.argmax(mixture_coef)
    min_coef, max_coef = mixture_coef[min_arg]/sum(mixture_coef), mixture_coef[max_arg]/sum(mixture_coef)

    phi = 2*np.pi*np.random.sample(n_samples)
    r_arr = np.empty(n_samples)
    w_arr = np.empty(n_samples)
    for i in  range(n_samples):
        if min_coef < 0:
            flip = (1-2*min_coef)*np.random.sample() + min_coef  # sample from (2*|b|+1)U - |b| ~ U(b, 1-b)
            if flip < 0: # sample from negative only
                r_arr[i] = sample_radial(1, scale=scales[min_arg]).item()  # get samples
                w_arr[i] = np.random.normal(loc=-1, scale=weight_spread)
            else: # sample from positive only
                r_arr[i] = sample_radial(1, scale=scales[max_arg]).item()  # get samples
                w_arr[i] = np.random.normal(loc=1, scale=weight_spread)
        else: # Normal positive distributions
            flip = np.random.sample()  # sample from U(0,1)
            if flip < min_coef:
                r_arr[i] = sample_radial(1, scale=scales[min_arg]).item()  # get samples
                w_arr[i] = np.random.normal(loc=1, scale=weight_spread)
            else:
                r_arr[i] = sample_radial(1, scale=scales[max_arg]).item()  # get samples
                w_arr[i] = np.random.normal(loc=1, scale=weight_spread)
                
    if basis == "polar":
        return np.hstack([r_arr.reshape(-1,1), phi.reshape(-1,1), w_arr.reshape(-1,1)])
    elif basis == "cartesian":
        x = r_arr * np.cos(phi)
        y = r_arr * np.sin(phi)
        return np.hstack([x.reshape(-1,1), y.reshape(-1,1), w_arr.reshape(-1,1)])