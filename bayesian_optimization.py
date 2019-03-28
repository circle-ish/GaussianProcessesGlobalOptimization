import numpy as np
# import matplotlib.pyplot as p
import copy
import time
import stat_utils

def bayesian_optimization(gp, iterations, goal_func, acqui_func, X_range, bounds, save_every=0, start_opt=0):
    #X_new = X_range[np.random.randint(len(X_range)), :][np.newaxis, :]
    #print X_new.shape
    #Y_new = goal_func(X_new)
    # print Y_new.shape
    #gp.update(X_new, Y_new)
    # print save_every
    
    #initialize hp statistics
    hp_list = []
    for i in range(5): #number of hyper params
        hp_list.append([])

    gp_list = []
    sample_list = []
    for i in range(iterations):
        Y_range = np.zeros((X_range.shape[0], 1))
        st = time.time()
        Y_range = acqui_func(X_range, gp)
        et = time.time()        
        print '\n========================Iteration', i, '======================='
        print '\t time:', et - st
        idx = np.where(Y_range == Y_range[np.argmax(Y_range)])
        idx = np.random.choice(idx[0])

        """
        if X_range.shape == (6299, 2): #usa
            print "\tAcquisition Insides:"
            print "\t\t<Chosen>"        
            acqui_func(X_range[idx, :][np.newaxis, :], gp, prt=True)
            print "\t\t\tIndex", idx
            print "\t\t<Best>"
            acqui_func(X_range[1515, :][np.newaxis, :], gp, prt=True)
        """

        X_new = X_range[idx, :][np.newaxis, :]
        Y_new = goal_func(X_new)

        gp.update(X_new, Y_new)

        if i >= start_opt:
            res = gp.update_hyperparams_ml(bounds)
        
            #collect the results of the hyperparameter optimization
            stat_utils.statistic_collect_hyperparams(res, hp_list)
        
        if save_every and ((i + 1) % save_every == 0):
            print "Saving interation", i + 1
            gp_list.append(copy.copy(gp))
            sample_list.append((X_new, Y_new))
            
    #plot hyperparameter statistic
    #stat_utils.statistic_plot_hyperparams(hp_list)

    if save_every:
        return gp, gp_list, sample_list, hp_list
    else:
        return gp

"""
if __name__ == '__main__':
    # do we want to make it command line executable?
    if len(sys.argv) == np.inf:
        print "Missing arguments: bayesian_optimization <>"
    else:
        #parameters gaussian process
        dimensions = 1
        noise = 1
        cov_func = covariance_functions.gaussian_kernel
        cov_grad = covariance_functions.gaussian_kernel_gradient
        width = 1

        #initialize gaussian process
        gp = gaussian_process.GP_Regressor(dimensions, noise, cov_func, cov_grad, width)

        #parameters bayesian optimization
        iterations = 100
        goal_func = (lambda x: np.sin(x) + np.random.normal(0,0.5))
        acqui_func = acquisition_functions.get_function("IG")
        #maybe have something like this
        X_range = np.arange(-np.pi, np.pi, 0.01)[:,np.newaxis]

        #execute bayesian optimization
        bayesian_optimization(gp, iterations, goal_func, acqui_func, X_range)
"""
