import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.stats import multivariate_normal
import csv
import utils
from functools import partial


# dataset environment measurements
# http://java.epa.gov/castnet/datatypepage.do?reportTypeLabel=Measurement%20(Raw%20Data)&reportTypeId=REP_001

def toy():
    """
    Simple One-Dimensional Dataset
    """

    # number of examples
    n = 40

    # function parameters
    a = 1
    b = 2.65
    offset = 1
    noise = 0.2

    rstate = np.random.mtrand.RandomState(12345)

    X = rstate.uniform(-3, 3, [n, 1])

    Y1 = np.sin(a * X) + np.cos(b * X + offset)
    Y2 = 0.5 * np.cos(2 * b * X - offset) + 0.3 * np.sin(0.1 * a * X - 2)
    Y3 = rstate.normal(0, noise, X.shape)

    Y = (Y1 + Y2 + Y3).flatten()

    return X, Y


def toy_func(X):
    # function parameters
    a = 1
    b = 2.65
    offset = 1
    Y1 = np.sin(a * X) + np.cos(b * X + offset)
    Y2 = 0.5 * np.cos(2 * b * X - offset) + 0.3 * np.sin(0.1 * a * X - 2)
    return Y1 + Y2

def toy_goal_func(x):
    return 5 + 3*toy_func(x) + np.random.normal(0,0.2, size=x.shape)

def yacht():
    """
    Yacht Hydrodynamics Data Set
    """

    D = np.loadtxt(open("yacht_hydrodynamics.csv", "rb"), delimiter=",", skiprows=1)

    X, Y = D[:, :-1], D[:, -1]

    return X, Y


def reading(filename):
    with open(filename, 'rb') as st:
        source = pickle.loads(st.read())
    return source


def load_data(name, preprocessed=False, country_id=111, noise=True): #111=Germany
    if name == "1D":
        X_range = np.arange(-3*np.pi, 3*np.pi, 0.01)[:,np.newaxis]
        data = utils.d1_goal_func(X_range, noise=False)
        return data, X_range, None, None, None, None, partial(utils.d1_goal_func, noise=noise)
    if name == "population":
        # ----------------------------------------
        # Create a matrix of population density 
        # ----------------------------------------
        population = reading('data/population.pkl')
        population = np.array([population[i::5, j::5] for i in range(5) for j in range(5)]).sum(axis=0)

        # ----------------------------------------
        # Create a matrix of country indicators 
        # ----------------------------------------
        countries = reading('data/countries.pkl')
        countries = countries[2::5, 2::5]

        # ----------------------------------------
        # Size of the map
        # ----------------------------------------
        assert (population.shape[0] == countries.shape[0])
        assert (population.shape[1] == countries.shape[1])
        nx = population.shape[0]
        ny = population.shape[1]

        if not preprocessed:
            return population
        else:
            
            """with open("europe_X_20", 'rb') as f:
                X = pickle.load(f)
            with open("europe_Y_20", 'rb') as f:
                Y = pickle.load(f)
            with open("europe_Z_20", 'rb') as f:
                Z = pickle.load(f)
            
            population[50:180, 70:200]
                """
            
            #find and cut out a country (Germany =111)
            mask = (countries == country_id)
            tmp = np.arange(0, len(mask))
            x = np.asarray(np.sum(mask, axis=1), dtype=bool)
            y = np.asarray(np.sum(mask, axis=0), dtype=bool)

            x_window = np.arange(np.nanmin(tmp[x]), np.nanmax(tmp[x]))
            y_window = np.arange(np.nanmin(tmp[y]), np.nanmax(tmp[y]))
            x_window, y_window = np.meshgrid(x_window, y_window)

            mask = mask[x_window, y_window]
            Z = np.log(population[x_window, y_window] + 1)
            Z[~mask] = np.nan
            Z = Z[:,::-1]
            data = Z.reshape((-1, 1))
            y = np.linspace(0, Z.shape[0] - 1, Z.shape[0])
            x = np.linspace(0, Z.shape[1] - 1, Z.shape[1])
            Y, X = np.meshgrid(x,y)
            x = X.reshape((-1, 1))
            y = Y.reshape((-1, 1))
            X_range = np.concatenate((x, y), axis=1)
            
            goal_func = partial(utils.usa_goal_func, data=data, X_range=X_range, noise=noise)
            return data, X_range, X, Y, Z, mask, goal_func

    if name == "ozone":
        ozone = pd.read_csv('data/ozone_data.csv')
        ozone = ozone[["LONGITUDE", "LATITUDE", "OZONE"]].as_matrix()
        if preprocessed:
            is_us = []
            with open("data/meshgrid_USA_all") as f:
                for line in f:
                    state = line.split(",")[1]
                    is_us.append(state == " United States of America\n" or state == "United States of America\n")

            is_us = np.array(is_us).reshape((100, 100))
            X, Y, Z = utils.grid_with_interpolation(ozone[:, 0:2], ozone[:, 2], xmin=-127.0, xmax=-74.0, ymin=23.0, ymax=50.0)
            # Z = np.log(Z)
            Z[np.invert(is_us)] = np.nan
            x = X[is_us].reshape((-1, 1))
            y = Y[is_us].reshape((-1, 1))
            X_range = np.concatenate((x, y), axis=1)
            data = Z[is_us].reshape((-1, 1))
            goal_func = partial(utils.usa_goal_func, data=data, X_range=X_range, noise=noise)
            return data, X_range, X, Y, Z, is_us, goal_func
        else:
            return ozone

    if name.startswith("random"):
        ran = np.genfromtxt('data/' + name, delimiter=',')
        l = np.sqrt(len(ran))
        x = np.linspace(0, l - 1, l)
        X,Y = np.meshgrid(x,x)
        Z = np.reshape(ran, (l, l), order='C')
        
        return ranDataParams(X, Y, Z)

# ----------------------------------------
# Plots geographical locations on a map
#
# input:
# - an array of latitudes
# - an array of longitudes
#
# ----------------------------------------
def plot_pop(latitudes, longitudes):
    borders = countries * 0

    borders[1:-1, 1:-1] = ((countries[1:-1, 1:-1] != countries[:-2, 1:-1]) +
                           (countries[1:-1, 1:-1] != countries[2:, 1:-1]) +
                           (countries[1:-1, 1:-1] != countries[1:-1, :-2]) +
                           (countries[1:-1, 1:-1] != countries[1:-1, 2:]))

    plt.figure(figsize=(14, 10))
    plt.imshow(np.log(1 + population) * (1 - borders))
    plt.plot(longitudes, latitudes, 's', ms=5, markeredgewidth=1.5, mfc='white', mec='black')
    plt.axis([0, population.shape[1], population.shape[0], 0])


# ----------------------------------------
# Create random 2D data 
#
# input:
# -step
# -no
# -window=5
# -dim=100
# -low=0
# -factor=2
# -s=1
# -amplitude=1
# -save=0
#
# ----------------------------------------
def ranDataExample():
    dim = 100
    low = 0
    amplitude = 100

    step = dim * 10
    no = 40
    window = 5
    factor = 10 * 2
    s = 1
    
    #for above values:
    #two peaks: 123456; one peak: 1234567; brain: 1234; two peaks and ring: 123; small areas: 1
    seed = 123456 

    x, y, z = ranDataCreate(step, no, window, dim, low, factor, s, amplitude, seed, save=1)
    ranDataPlot(x, y, z)

def ranDataCreate(step, no, window=5, dim=100, low=0, factor=2, s=1, amplitude=1, seed=123456, save=0):
    # random generator
    ranGen = np.random.RandomState()
    ranGen.seed(seed)

    x, y, z = buildSpace(no, -1, 1, step, window, low, factor, ranGen)
    tck = interpolate.bisplrep(x, y, z, s=s)

    xynew = np.linspace(-1, 1, step)
    Z = interpolate.bisplev(xynew, xynew, tck)

    Z = setAmplitude(amplitude, Z)
    x = np.linspace(0, step - 1, step)
    X, Y = np.meshgrid(x, x)

    if (save == 0):
        np.savetxt('data/random_2d_data.csv', Z.ravel(order='C'), delimiter=',')

    return ranDataParams(X, Y, Z)

def buildSpace(itr, start, dim, step, window, mean, factor, ranGen):
    smallstep = step * factor / step
    x, y = np.mgrid[start:dim:1j * smallstep, start:dim:1j * smallstep]

    z = np.zeros((smallstep, smallstep))
    z.fill(mean)
    for i in range(itr):
        xwindow, ywindow = randomWindow(smallstep, window, ranGen)

        mean = [0, 0]
        cov = [0.8, 0.3]
        z[xwindow, ywindow] += normalDistr(mean, cov, start, dim, window, ranGen)
    return x, y, z

def normalDistr(mean, cov, start, dim, window, ranGen):
    # preparations
    step = ranGen.random_integers(low=window + 1, high=100)
    x, y = np.mgrid[start:dim:1j * step, start:dim:1j * step]
    xwindow, ywindow = randomWindow(step, window, ranGen)

    # getting values
    pos = np.dstack((x, y))
    fct = multivariate_normal(mean, cov)
    z = fct.pdf(pos)[xwindow, ywindow]

    return z

def randomWindow(step, window, ranGen):
    xloc = step
    yloc = step
    while (xloc + window >= step or yloc + window >= step):
        xloc = ranGen.random_integers(low=0, high=step)
        yloc = ranGen.random_integers(low=0, high=step)

    ywindow = np.arange(yloc, np.minimum(yloc + window, step - 1), dtype='uint')
    xwindow = np.arange(xloc, np.minimum(xloc + window, step - 1), dtype='uint')
    xwindow, ywindow = np.meshgrid(xwindow, ywindow)
    return xwindow.astype(int), ywindow.astype(int)

def ranDataParams(X, Y, Z):
    x = X.reshape((-1, 1))
    y = Y.reshape((-1, 1))
    X_range = np.concatenate((x, y), axis=1)
    data = Z.reshape((-1,1))
    is_true = np.ones(shape=X.shape, dtype=bool)
    #TODO: make this more elegant
    if noise:
        goal_func = (lambda xi: np.array([Z[xi[0, 1],xi[0, 0]]], ndmin=2) + np.random.normal(0,1,size=xi.shape))
    else:
        goal_func = (lambda xi: np.array([Z[xi[0, 1],xi[0, 0]]], ndmin=2))
    
    return data, X_range, X, Y, Z, is_true, goal_func

def ranDataPlot(x, y, z):
    plt.contourf(x, y, z)
    plt.colorbar()
    plt.show()

def setAmplitude(amp, z):
    tmp = z.ravel()
    mi = np.amin(tmp)
    tmp = (tmp + np.abs(mi))
    ma = np.amax(tmp)
    return ((tmp / ma) * amp).reshape(z.shape)
