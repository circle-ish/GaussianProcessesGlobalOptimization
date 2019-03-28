import numpy as np
import scipy.interpolate
import datasets


# comment on how ranges looks like
def create_input_space(dimensions, ranges):
    if dimensions != len(ranges):
        raise AssertionError("'ranges' need to be of size 'dimensions', i.e. it has one range for each dimension")
    spaces = []
    for i in range(dimensions):
        spaces.append(np.arange(*ranges[i]))

    mesh = np.meshgrid(*spaces)
    mesh_values = np.concatenate(mesh).flatten()
    return mesh_values.reshape((dimensions,-1)).T

def grid_with_interpolation(xy, z, resX=100, resY=100, method='linear', xmin=None, xmax=None, ymin=None, ymax=None, smooth=0.001):
    x = xy[:,0]
    y = xy[:,1]
    if xmin is not None and xmax is not None:
        xi = np.linspace(xmin, xmax, resX)
    else:
        xi = np.linspace(np.min(x), np.max(x), resX)
    if ymin is not None and ymax is not None:
        yi = np.linspace(ymin, ymax, resY)
    else:
        yi = np.linspace(np.min(y), np.max(y), resY)
    X, Y = np.meshgrid(xi, yi)
    #Z = scipy.interpolate.griddata(xy, z, (X, Y), method=method)
    rbf_func = scipy.interpolate.Rbf(x, y, z, method=method, smooth=smooth)
    Z = rbf_func(X, Y)
    return X, Y, Z

def get_index(X_range, x):
    for i in range(X_range.shape[0]):
        if np.allclose(X_range[i,:], x):
            arg_x = i
    return arg_x

def usa_goal_func(x, data, X_range, noise=True, sign=1):
    y = data[get_index(X_range, x)][np.newaxis, :]
    if noise:
        y += np.random.normal(0,1)
    return sign * y


def d1_goal_func(x, noise=True, sign=1):
    y = 5 + 3 * datasets.toy_func(x)
    if noise:
        y += np.random.normal(0, 0.2, size=x.shape)
    return sign * y

def gap_measure(goal_func, x_first, x_best, y_opt):
    return ((goal_func(x_first) - goal_func(x_best)) / (goal_func(x_first) - y_opt)).flatten()[0]

def closeness_measure(x_best, x_opt):
    return np.linalg.norm(x_best - x_opt)

def euc_distance(u, v):
    return np.linalg.norm(u - v, axis=1)

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def check_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
