import gaussian_process, covariance_functions, acquisition_functions, bayesian_optimization, datasets, utils, GP_lookahead
from functools import partial
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as p
import numpy as np
import os
import time
import sys
import json

current_milli_time = lambda: int(round(time.time() * 1000))

def run_experiment(params, output_path):
    folder_name = get_folder_name(params, output_path)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    print "Started EXPERIMENTS for", folder_name

    k = 0
    for p in params["first_points"]:
        curr_time = current_milli_time() % 100000000000

        out_file = open(folder_name + str(curr_time) + "_stdout", 'w')
        save_stdout = sys.stdout
        sys.stdout = out_file

        data, X_range, X, Y, Z, is_us, goal_func = datasets.load_data(params["data_name"], True)

        if params["acqui_name"] == "lookahead":
            w_s, w_e = params["widths_range"]
            a_s, a_e = params["amps_range"]
            n_s, n_e = params["noise_range"]
            widths = np.logspace(np.log(w_s), np.log(w_e), num=10, base=np.e)
            amps = np.logspace(np.log(a_s), np.log(a_e), num=5, base=np.e)
            noises = np.logspace(np.log(n_s), np.log(n_e), num=5, base=np.e)

        # initialize gaussian process
        gp = gaussian_process.GP_Regressor(params["dimensions"], params["noise"], params["cov_func"],
                                           params["cov_grad"], [params["width"], params["amp"]])

        # print 'Max location', np.unravel_index(np.argmax(data), data.shape)

        #add first point
        X_new = X_range[p, :][np.newaxis, :]
        Y_new = goal_func(X_new)

        gp.update(X_new, Y_new)

        if params["acqui_name"] == "lookahead":
            goal_func = partial(goal_func, sign=-1)
            gp, gp_list, s_list, aux_list = GP_lookahead.lookahead_optimization(
                gp,
                params["iterations"],
                goal_func,
                X_range,
                [widths, amps, noises],
                lookahead_steps=params["steps"],
                save_every=params["save_every"])
        else:
            acqui_func = acquisition_functions.get_function(params["acqui_name"], kappa=params["kappa"], gamma=params["gamma"])
            # execute bayesian optimization
            gp, gp_list, s_list, aux_list = bayesian_optimization.bayesian_optimization(gp, params["iterations"], goal_func,
                                                                              acqui_func, X_range, params["bounds"],
                                                                              save_every=params["save_every"])

        pickle_data(folder_name, data=[gp, gp_list, s_list, aux_list], curr_time=curr_time)
        sys.stdout = save_stdout
        out_file.close()

        k += 1
        print "Experiment", k, "of", len(params["first_points"]), "done"

def get_folder_name(params, output_path):
    folder_name = output_path + params["data_name"] + "_" + params["acqui_name"]
    if params["acqui_name"] == "UCB" or params["acqui_name"] == "DUCB":
        folder_name += "_kappa" + str(params["kappa"])
    if params["acqui_name"] == "DUCB":
        folder_name += "_gamma" + str(params["gamma"])
    folder_name += "/"
    return folder_name

# if data is None then this function reads data, if data is given then it writes the data
# data = [gp, gp_list, s_list] in this order for writing
def pickle_data(folder_name, data=None, curr_time=None, file_prefix=None):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if file_prefix is not None:
        """
        random_millisec = ""
        while not utils.check_int(random_millisec):
            random_millisec = np.random.choice(os.listdir(folder_name))[:11]
        print random_millisec
        """

        with open(folder_name + file_prefix + "_gp", 'rb') as f:
            gp = pickle.load(f)
        with open(folder_name + file_prefix + "_gplist", 'rb') as f:
            gp_list = pickle.load(f)
        with open(folder_name + file_prefix + "_slist", 'rb') as f:
            s_list = pickle.load(f)
    elif data is not None:
        with open(folder_name + str(curr_time) + "_gp", 'wb') as f:
            pickle.dump(data[0], f)
        with open(folder_name + str(curr_time) + "_gplist", 'wb') as f:
            pickle.dump(data[1], f)
        with open(folder_name + str(curr_time) + "_slist", 'wb') as f:
            pickle.dump(data[2], f)
        with open(folder_name + str(curr_time) + "_auxlist", 'wb') as f:
            pickle.dump(data[3], f)

    if file_prefix is not None:
        return (gp, gp_list, s_list)


def plot_experiment(params, output_path):
    folder_name = get_folder_name(params, output_path)
    all_experiments = set([f[:11] for f in os.listdir(folder_name) if utils.check_int(f[:11])])
    all_experiments = sorted(all_experiments)
    for experiment in all_experiments:
        gp, gp_list, s_list = pickle_data(folder_name, file_prefix=experiment)

        data, X_range, X, Y, Z, is_us, goal_func = datasets.load_data(params["data_name"], True)

        print X_range.shape
        h, w = X_range.shape
        for i in range(len(gp_list)):

            gpi = gp_list[i]
            X_new, Y_new = s_list[i]
            print X_new.shape
            mean = np.zeros((X_range.shape[0], 1))
            var = np.zeros((X_range.shape[0], 1))
            for j in range(X_range.shape[0]):
                mean[j], var[j] = gpi.predict(X_range[j, :][np.newaxis, :])
            print gpi.Xtrain.shape
            print gpi.Ytrain.shape
            print Z.shape, is_us.shape, mean.flatten().shape, var.flatten().shape
            Z[is_us] = mean.flatten()
            p.contourf(X, Y, Z)  # GP mean
            p.colorbar()
            p.scatter(gpi.Xtrain[:, 0], gpi.Xtrain[:, 1], color='green', marker='x', s=50)  # training data
            p.scatter(X_new[:, 0], X_new[:, 1], color='purple', marker='*', s=100)  # test data
            p.show()
            Z[is_us] = var.flatten() ** 0.5
            p.contourf(X, Y, Z)  # GP mean
            p.colorbar()
            p.scatter(gpi.Xtrain[:, 0], gpi.Xtrain[:, 1], color='green', marker='x', s=50)  # training data
            p.scatter(X_new[:, 0], X_new[:, 1], color='purple', marker='*', s=100)  # test data
            p.show()
            Z[is_us] = data.flatten()
            p.contourf(X, Y, Z)  # GP mean
            p.colorbar()
            p.scatter(gpi.Xtrain[:, 0], gpi.Xtrain[:, 1], color='green', marker='x', s=50)  # training data
            p.scatter(X_new[:, 0], X_new[:, 1], color='purple', marker='*', s=100)  # test data
            p.show()

            Z[is_us] = mean.flatten()
            fig = p.figure()
            ax = fig.gca(projection='3d')
            # surf = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.coolwarm,
            #                   linewidth=0, antialiased=False)
            ax.contourf(X, Y, Z)
            # fig.colorbar(surf, shrink=0.5, aspect=5)
            ax.set_zlim(-15, 90)
            # p.colorbar()
            ax.scatter(gpi.Xtrain[:, 0], gpi.Xtrain[:, 1], gpi.Ytrain_original, color='green', marker='x',
                       s=50)  # training data
            # p.scatter(X_new[:,0],X_new[:,1],color='purple',marker='*', s=100)   # test data
            p.show()
            print X_new, Y_new


def evaluate_experiment(params, output_path):
    folder_name = get_folder_name(params, output_path)
    all_experiments = set([f[:11] for f in os.listdir(folder_name) if utils.check_int(f[:11])])
    all_experiments = sorted(all_experiments)

    print "Starting EVALUATION for", folder_name

    data, X_range, X, Y, Z, is_us, goal_func = datasets.load_data(params["data_name"], True, noise=False)

    all_gap_m = []
    all_closeness_m = []
    all_dist_m = []

    k = 0

    for experiment in all_experiments:
        gp, gp_list, s_list = pickle_data(folder_name, file_prefix=experiment)

        gap_measure = []
        closeness_measure = []
        dist_m = []
        dist = 0

        y_opt = data[np.argmax(data)]
        x_opt = X_range[np.argmax(data), :]

        for i in range(len(gp_list)):
            gpi = gp_list[i]
            if i % 5 == 0 or i == (len(gp_list) - 1):
                mean = np.zeros((X_range.shape[0], 1))
                var = np.zeros((X_range.shape[0], 1))
                for j in range(X_range.shape[0]):
                    mean[j], var[j] = gpi.predict(X_range[j, :][np.newaxis, :])

                x_best = X_range[np.argmax(mean), :]
                x_first = gpi.Xtrain[0, :]

                gap_measure.append(utils.gap_measure(goal_func, x_first, x_best, y_opt))
                closeness_measure.append(utils.closeness_measure(x_best, x_opt))

                #print "Gap measure:", utils.gap_measure(goal_func, x_first, x_best, y_opt)
                #print "Closeness measure:", utils.closeness_measure(x_best, x_opt)
            if params["data_name"] == "ozone" or params["data_name"] == "population":
                dist += utils.haversine(gpi.Xtrain[-1,0], gpi.Xtrain[-1,1], gpi.Xtrain[-2,0], gpi.Xtrain[-2,1])
            else:
                dist += np.linalg.norm(gpi.Xtrain[-1,:] - gpi.Xtrain[-2,:])
            dist_m.append(dist)

        all_gap_m.append(gap_measure)
        all_closeness_m.append(closeness_measure)
        all_dist_m.append(dist_m)

        k += 1
        print "Evaluation", k, "of", len(all_experiments), "done"


    with open(folder_name + "evaluation_gap_mtx", 'wb') as f:
        pickle.dump(np.matrix(all_gap_m), f)
    with open(folder_name + "evaluation_closeness_mtx", 'wb') as f:
        pickle.dump(np.matrix(all_closeness_m), f)
    with open(folder_name + "evaluation_distance_mtx", 'wb') as f:
        pickle.dump(np.matrix(all_dist_m), f)


def create_plots(params, output_path):
    print "Creating plots ..."

    plots_folder = output_path + "plots/"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    folder_name = get_folder_name(params, output_path)

    with open(folder_name + "evaluation_gap_mtx", 'rb') as f:
        gap = pickle.load(f)
    with open(folder_name + "evaluation_closeness_mtx", 'rb') as f:
        closeness = pickle.load(f)
    with open(folder_name + "evaluation_distance_mtx", 'rb') as f:
        distance = pickle.load(f)

    exp_name = get_folder_name(params, "")

    p.clf()
    p.boxplot(gap)
    p.ylim(0, 1)
    p.title(exp_name[:-1] + " Gap Measure")
    p.xticks([i for i in range(1, 12)], [i for i in range(1,50,5)] + [50])
    p.savefig(plots_folder + exp_name[:-1] + "gap_measure.png")

    p.clf()
    p.boxplot(closeness)
    p.ylim(0, 50)
    p.title(exp_name[:-1] + " Closeness Measure")
    p.xticks([i for i in range(1, 12)], [i for i in range(1,50,5)] + [50])
    p.savefig(plots_folder + exp_name[:-1] + "closeness_measure.png")

    p.clf()
    p.boxplot(distance)
    p.ylim(0, 250000)
    p.title(exp_name[:-1] + " Distance Measure")
    p.xticks([i for i in range(5, 51, 5)], [i for i in range(5,51,5)])
    p.savefig(plots_folder + exp_name[:-1] + "distance_measure.png")


if __name__ == "__main__":

    params_file = sys.argv[1]

    with open(params_file) as par:
        params = json.load(par)

    params["cov_func"] = covariance_functions.gaussian_kernel
    params["cov_grad"] = covariance_functions.gaussian_kernel_gradient

    output_path = "experiments/"
    """
    #acquis = ["IG", "UCB", "DUCB"]
    acquis = ["TS"]
    kappas = np.arange(2.8, 5.2, 0.4)
    kappas = [4.4]
    gammas = np.arange(-0.01, -0.06, -0.02)
    gammas = [-0.07]

    params = {
        "data_name": "ozone",
        "acqui_name":"TS",
        "dimensions": 2,
        "noise": 1,
        "width": 5,
        "amp": 50,
        "cov_func": covariance_functions.gaussian_kernel,
        "cov_grad": covariance_functions.gaussian_kernel_gradient,
        "iterations": 50,
        "save_every": 1,
        "kappa": 4.4,
        "gamma": -0.05,
        "bounds": [[0.01, 100], [0.01, 1000], [0.01, 100]]
    }
    """


    """
    # in order worst to best starting point
    first_points = [1236, 570, 332, 2225, 475, 599, 245, 317, 2874, 3450, 763, 59, 6258, 2113, 5212, 1391, 1554, 1207, 3519,
              6185, 4574, 4001, 3615, 2586, 2292, 3418, 2938, 3901, 3514, 3439, 3223, 5551, 2201, 1398, 2196, 5597,
              5026, 3239, 4746, 5500, 3991, 4562, 3833, 3633, 5999, 2467, 5732, 4950, 1998, 2486, 3836, 5806, 4897,
              5610, 4771, 4087, 53, 4405, 5611, 2756, 3262, 781, 2465, 3116, 2352, 2259, 3397, 5404, 3060, 1725, 2925,
              3593, 5092, 5218, 3061, 4466, 3678, 392, 1810, 1719, 1457, 1282, 230, 793, 4881, 2614, 1351, 4453, 430,
              2530, 4270, 2057, 2977, 3172, 1870, 3077, 1692, 1081, 2050, 1428]
    first_points_ran = [7801, 8531, 5887, 2918, 2332, 7150, 1281, 3590, 6741, 148, 6742, 7760, 7846, 4037, 9080, 7627, 6566, 8165,
                        1825, 3119, 5184, 351, 3835, 7002, 4674, 5517, 1659, 1309, 6238, 8233, 5777, 7751, 3354, 6168, 6867, 1709,
                       1099, 8417, 9665, 4480, 8640, 299, 1876, 2642, 2689, 118, 7926, 4554, 6026, 570, 2419, 6746, 9228, 4009, 754,
                        1914, 5746, 6800, 8480, 9258, 955, 264, 3766, 1631, 1297, 646, 2844, 1474, 7748, 9089, 768, 8948, 1124, 9538,
                        4924, 3250, 3194, 398, 7853, 2801, 6386, 9791, 514, 2351, 9225, 1756, 2716, 4877, 6065, 4102, 7395, 1189,
                        6605, 8881, 3544, 1861, 55, 7749, 6994, 8866]
    #first points for 1D data set
    first_points = range(0, 1885, 1+(1885//10))
    """
    print "Doing", len(params["first_points"]), "experiments (each) for", params["acqui_name"], "with kappas", params["kappas"], "and gammas", params["gammas"], "\n"

    #plot_experiment(acqui_name, params, output_path)

    for kappa in params["kappas"]:
        params["kappa"] = kappa
        for gamma in params["gammas"]:
            params["gamma"] = gamma
            run_experiment(params, output_path)
            evaluate_experiment(params, output_path)
            create_plots(params, output_path)
            print "----- FINISHED", params["acqui_name"], "with kappa", kappa, "and gamma", gamma, "-----\n"

    #plot_experiment(params, output_path)

    """
    #preliminary code to plot 1D data set

    folder_name = get_folder_name(params, output_path)
    all_experiments = set([f[:11] for f in os.listdir(folder_name) if utils.check_int(f[:11])])
    all_experiments = sorted(all_experiments)

    data, X_range, X, Y, Z, is_us = datasets.load_data(params["data_name"], True)

    print folder_name
    print all_experiments

    for experiment in all_experiments:
        gp, gp_list, s_list = pickle_data(folder_name, file_prefix=experiment)
        for i in range(len(gp_list)):
            gpi = gp_list[i]
            X_new, Y_new = s_list[i]
            mean,cov = gpi.predict(X_range)
            var = cov.diagonal()[:,np.newaxis]
            p.title("Iteration: " + str(i*params["save_every"]) + " HP: " + str(gpi.cov_hyperparams) + " " + str(gpi.noise))
            p.xlabel('x')
            p.ylabel('y')
            p.scatter(gpi.Xtrain,gpi.Ytrain_original,color='green',marker='x', s=50) # training data
            p.plot(X_range,mean,color='blue')                  # GP mean
            p.plot(X_range,mean+var**.5,color='red')           # GP mean + std
            p.plot(X_range,mean-var**.5,color='red')           # GP mean - std
            p.plot(X_range, 5 + 3*datasets.toy_func(X_range),color='black')           # GP mean - std
            p.scatter(X_new,Y_new,color='purple',marker='*', s=100)   # test data
            p.xlim(-3*np.pi,3*np.pi)
            p.show()

    """
    """
    # code to get first sample points

    points = np.random.randint(0, X_range.shape[0], size=100)
    sorted = np.argsort(data[points].flatten())
    points = points[sorted]
    print ",".join(str(p) for p in points)
    print data[points]

    data, X_range, X, Y, Z, is_us = datasets.load_data(params["data_name"], True)

    p.contourf(X, Y, Z)  # GP mean
    p.colorbar()
    p.scatter(X_range[points, 0], X_range[points, 1], color='green', marker='x', s=50)  # training data
    p.show()
    """
