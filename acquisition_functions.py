import sys
import numpy as np
from functools import partial
from scipy.stats import multivariate_normal
import utils

#function getter
def get_function(fct, kappa=None, gamma=None):
    if fct == "IG":
        return ig_acqui
    if fct == "UCB":
        return partial(ucb_acqui, kappa=kappa)
    if fct == "DUCB":
        return partial(ducb_acqui, kappa=kappa, gamma=gamma)
    if fct == "TS":
        return thompson_sampl_acqui

#information gain based on predicted variance
#based on IEM paper p.4
def ig_acqui(x, gp, prt=False):
    _, cov_predict = gp.predict(x)
    if prt:
        print "Var:", np.diag(cov_predict) ** 0.5
    return (np.diag(cov_predict) ** 0.5)[:,np.newaxis]
    
#upper confidence bound
#kappa default value from IEM paper
#28.6 1d, 
def ucb_acqui(x, gp, kappa = 4.5, prt=False):
    mean_predict, cov_predict = gp.predict(x)
    if prt:
        print "\t\t\tMean:", mean_predict, 
        print "Var:", np.diag(cov_predict) ** 0.5, 
        print "Final value:", mean_predict + kappa * np.diag(cov_predict) ** 0.5
    return mean_predict + kappa * (np.diag(cov_predict) ** 0.5)[:,np.newaxis]
    
#distance-based upper confidence bound
#kappa and gamma default value from IEM paper
#kappa = 28.6, gamma = 0.72
def ducb_acqui(x, gp, kappa = 4.5, gamma = 0.72, prt=False):
    last_x = gp.Xtrain[-1, :]
    mean_predict, cov_predict = gp.predict(x)
    if prt:
        print "Mean:", mean_predict
        print "Var:", np.diag(cov_predict) ** 0.5
        print "Dist:", utils.euc_distance(x, last_x)
        print "Final value:", mean_predict + kappa * (np.diag(cov_predict) ** 0.5) + gamma * utils.euc_distance(x, last_x)
    return mean_predict + kappa * (np.diag(cov_predict) ** 0.5)[:,np.newaxis] + gamma * utils.euc_distance(x, last_x)[:,np.newaxis]

#thompson sampling
def thompson_sampl_acqui(x, gp, prt=False):
    mean_predict, cov_predict = gp.predict(x)
    L = np.linalg.cholesky(cov_predict)
    rand_samples = np.matrix(np.random.normal(size=(x.shape[0],1)))
    return np.array(mean_predict + (L * rand_samples))[:,0][:,np.newaxis] 

"""
HOW TO USE
original call:
    acqui_func = acquisition_functions.get_function("UCB", 2)
with ACQUI object:
    acqui_func = acquisition_functions.get_function("UCB", 2)
    aa = ACQUI(X_range, acqui_func, mode='last', thres=1, goal_shape=X_range.shape)
    acqui_func = aa.get()
    
FUNCTIONS    
get_last(): blocks the last 'lastSize' and their surroundings till 'thres' distance from being chosen
    (by setting them to zero)
get_prob(): weights the chosen point and its surroundings till 'thres' distance negative so that it is less likely to be chosen again

PARAMS
thres: a (x-axis and y-axis) distance not a index distance; example values: USA data: 8
lastSize: number of last chosen points to block
goal_shape: shape of the real data (like it would be plotted)
space_mask: for the USA dataset: = is_us
"""
class ACQUI(object):
    def __init__(self, X_range, acqui_fun, mode, thres, lastSize=10, goal_shape=(0,0), space_mask=None):
        self.x_range = X_range
        self.fun = acqui_fun
        self.mode = mode
        self.i =0
        if mode == 'last':
            #get_last params
            self.thres = thres
            self.last = np.empty(shape=(0,self.x_range.shape[1]))
            self.lastSize = lastSize
        
        elif mode == 'prob':
            #get_prob params
            self.goal_shape = goal_shape
            self.space_mask = np.ones(shape=self.goal_shape, dtype='bool')
            if space_mask is not None:
                self.space_mask = space_mask
            self.space = np.ones(shape=self.goal_shape, order='C')
            self.space[~self.space_mask] = np.nan
            self.thres = np.ceil(thres / 2 * (np.abs(X_range[1,0] - X_range[0, 0])))

            #update_matrix
            x = np.arange(-self.thres, self.thres + 1)
            x, y = np.meshgrid(x,x)
            pos = np.dstack((x, y))
            fct = multivariate_normal([0,0], [[self.thres, 0], [0, self.thres]])
            self.update_matrix = fct.pdf(pos)
            self.update_matrix /= self.update_matrix[self.thres, self.thres]
        
    def get(self):        
        if self.mode == 'last': return self.get_last
        elif self.mode == 'prob': return self.get_prob
        else: return None
        
    def get_last(self, x, gp, **dict): 
        #call acqui function
        res = self.fun(x, gp, **dict)[:,0][:,np.newaxis]
        if dict and dict['prt']:
            return res
        
        #block indices and nearbyes from the last lastSize rounds
        for i in self.last:
            mask = np.linalg.norm(self.x_range - i, axis=1)
            res[mask < self.thres] = 0
       
        #find index of max value
        idx_2d = np.where(res == res[np.argmax(res)])
        idx_1d = np.random.choice(len(idx_2d[0]), 1, replace=False)
        idx_1d = idx_2d[0][idx_1d][0]

        #update params
        if len(self.last) == self.lastSize:
            self.last = np.delete(self.last, 0, axis=0)
        self.last = np.concatenate((self.last, self.x_range[idx_1d, :][np.newaxis,:]), axis=0)

        res = np.zeros(shape=res.shape)
        res[idx_1d, :] = 1
        return res
    
    def get_prob(self, x, gp, **dict):        
        #call acqui function
        res = self.fun(x, gp, **dict)[:,0][:,np.newaxis] 
        if dict and dict['prt']:
            return res
        
        #apply weights
        res = np.multiply(res.T, 1 - self.space[self.space_mask].flatten()).reshape(res.shape)
        np.nan_to_num(res) #once in a while there is an error with a nan value in random.choice
        
        """plot new distribution
        tmp = np.zeros(self.goal_shape)
        tmp[self.space_mask] = res.ravel()
        p.contourf(X, Y, tmp)
        p.colorbar()
        p.show()"""
        
        #find index of max value
        idx_2d = np.where(res == res[np.argmax(res)])
        idx_1d = np.random.choice(idx_2d[0], 1, replace=False)[0]
        
        res[~idx_1d, :] = 0        
        tmp = np.zeros(self.goal_shape)
        tmp[self.space_mask] = res.ravel()
        idx_2d = np.unravel_index(tmp.argmax(), tmp.shape)
        
        #find weights to update
        xwindow = np.arange(idx_2d[0] - self.thres, idx_2d[0] + self.thres + 1)
        ywindow = np.arange(idx_2d[1] - self.thres, idx_2d[1] + self.thres + 1)  
        ymask = (ywindow >= 0) & (ywindow < self.goal_shape[1])
        xmask = (xwindow >= 0) & (xwindow < self.goal_shape[0])
        
        xwindow, ywindow = np.meshgrid(xwindow[xmask], ywindow[ymask])
        xmask, ymask = np.meshgrid(xmask, ymask)
        mask = np.array((xmask & ymask), dtype=bool)
        
        #update params
        x_shape = np.sum(mask[:, np.amax(np.argmax(mask, axis=1))])
        y_shape = np.sum(mask[np.amax(np.argmax(mask, axis=0)), :])
        update_matrix_shape = (x_shape, y_shape)
        self.space[xwindow.astype(int), ywindow.astype(int)] += self.update_matrix[mask].reshape((update_matrix_shape))  
        self.space /= np.nanmax(self.space)
        
        return res 