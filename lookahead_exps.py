import gaussian_process
from mpl_toolkits.mplot3d import Axes3D
from covariance_functions import gaussian_kernel, gaussian_kernel_gradient
import json
import GP_lookahead, datasets, utils
from functools import partial
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import sys

params_file = sys.argv[1]

current_milli_time = lambda: int(round(time.time() * 1000))


def run_experiments(params, first_points, output_path):
    for p in first_points:
        curr_time = current_milli_time() % 100000000000

        folder_name = get_folder_name(params,
                                      output_path)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        out_file = open(folder_name + str(curr_time) + "_stdout", 'w')
        save_stdout = sys.stdout
        sys.stdout = out_file

        if params["data_name"] == "1d":
            data, X_range = datasets.load_data(
                params["data_name"])
        else:
            data, X_range, X, Y, Z, is_us = datasets.load_data(
                params["data_name"], True)

        w_s, w_e = params["widths_range"]
        a_s, a_e = params["amps_range"]
        n_s, n_e = params["noise_range"]
        widths = np.logspace(np.log(w_s), np.log(w_e), num=10, base=np.e)
        amps = np.logspace(np.log(a_s), np.log(a_e), num=5, base=np.e)
        noises = np.logspace(np.log(n_s), np.log(n_e), num=5, base=np.e)

        # initialize gaussian process
        gp = gaussian_process.GP_Regressor(params["dimensions"],
                                           params["noise"],
                                           gaussian_kernel,
                                           gaussian_kernel_gradient,
                                           [params["width"], params["amp"]])

        # print 'Max location', np.unravel_index(np.argmax(data), data.shape)

        # parameters bayesian optimization
        if params["data_name"] == "ozone" or params["data_name"] == "population":
            goal_func = partial(utils.usa_goal_func, data=data,
                                X_range=X_range, sign=-1)
        elif params["data_name"] == "1d":
            goal_func = partial(utils.d1_goal_func, sign=-1)
        else:
            raise NameError("Data set name unknown.")

        # add first point
        X_new = X_range[p, :][np.newaxis, :]
        Y_new = goal_func(X_new)

        gp.update(X_new, Y_new)

        # execute lookahead optimization
        gp, gp_list, s_list, loss_list = GP_lookahead.lookahead_optimization(
            gp,
            params["iterations"],
            goal_func,
            X_range,
            [widths, amps, noises],
            lookahead_steps=params["steps"],
            save_every=params["save_every"])

        pickle_data(get_folder_name(params, output_path),
                    data=[gp, gp_list, s_list, loss_list], curr_time=curr_time)
        sys.stdout = save_stdout
        out_file.close()

    print "Experiment for " + params["data_name"] + " done!"


def get_folder_name(params, output_path):
    folder_name = output_path + params["data_name"] + '_' + \
        str(params["iterations"]) + '_' + str(params["steps"])
    folder_name += "/"
    return folder_name


# if data is None then this function reads data, if data is given then it
# writes the data as data = [gp, gp_list, s_list] in this order for writing
def pickle_data(folder_name, data=None, curr_time=None):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if data is None:
        random_millisec = ""
        while not utils.check_int(random_millisec):
            random_millisec = np.random.choice(os.listdir(folder_name))[:11]
        print random_millisec

        with open(folder_name + random_millisec + "_gp", 'rb') as f:
            gp = pickle.load(f)
        with open(folder_name + random_millisec + "_gplist", 'rb') as f:
            gp_list = pickle.load(f)
        with open(folder_name + random_millisec + "_slist", 'rb') as f:
            s_list = pickle.load(f)
        with open(folder_name + random_millisec + "_losslist", 'rb') as f:
            loss_list = pickle.load(f)
    else:
        with open(folder_name + str(curr_time) + "_gp", 'wb') as f:
            pickle.dump(data[0], f)
        with open(folder_name + str(curr_time) + "_gplist", 'wb') as f:
            pickle.dump(data[1], f)
        with open(folder_name + str(curr_time) + "_slist", 'wb') as f:
            pickle.dump(data[2], f)
        with open(folder_name + str(curr_time) + "_losslist", 'wb') as f:
            pickle.dump(data[3], f)

    if data is None:
        return (gp, gp_list, s_list, loss_list)


def plot_1d_experiment(params, output_path):
    gp, gp_list, s_list, loss_list = pickle_data(get_folder_name(params,
                                                                 output_path))

    data, X_range = datasets.load_data(params["data_name"])

    print X_range.shape
    h, w = X_range.shape
    for i in range(len(gp_list)):
        gpi = gp_list[i]
        gpi.Ytrain = -gpi.Ytrain
        gpi.Ytrain_original = -gpi.Ytrain_original
        gpi.mean = -gpi.mean
        X_new, Y_new = s_list[i]
        Y_new = -Y_new
        plt.title("Iteration: " + str((i + 1) * params["save_every"]))
        plt.xlabel('x')
        plt.ylabel('y')
        mean, cov = gpi.predict(X_range)
        var = cov.diagonal()[:, np.newaxis]
        plt.scatter(gpi.Xtrain, gpi.Ytrain_original, color='green',
                    marker='x', s=50)  # training data
        plt.plot(X_range, mean, color='blue')  # GP mean
        plt.plot(X_range, mean + var, color='red')  # GP mean
        plt.plot(X_range, mean - var, color='red')  # GP mean
        plt.plot(X_range, 5 + 3 * data, color='black')  # GP mean - std
        plt.scatter(X_new, Y_new, color='purple', marker='*', s=100)  # data
        plt.xlim(-3 * np.pi, 3 * np.pi)
        plt.show()

        print X_new, Y_new


def plot_2d_experiment(params, output_path):
    gp, gp_list, s_list, loss_list = pickle_data(get_folder_name(params,
                                                                 output_path))

    data, X_range, X, Y, Z, is_us = datasets.load_data(
        params["data_name"],
        True)

    print X_range.shape
    h, w = X_range.shape
    for i in range(len(gp_list)):

        gpi = gp_list[i]
        gpi.update_hyperparams_ml(params["bounds"])
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
        plt.contourf(X, Y, Z)  # GP mean
        plt.colorbar()
        plt.scatter(gpi.Xtrain[:, 0], gpi.Xtrain[:, 1],
                    color='green', marker='x', s=50)  # training data
        plt.scatter(X_new[:, 0], X_new[:, 1], color='purple',
                    marker='*', s=100)  # test data
        plt.show()
        Z[is_us] = var.flatten() ** 0.5
        plt.contourf(X, Y, Z)  # GP mean
        plt.colorbar()
        plt.scatter(gpi.Xtrain[:, 0], gpi.Xtrain[:, 1],
                    color='green', marker='x', s=50)  # training data
        plt.scatter(X_new[:, 0], X_new[:, 1], color='purple',
                    marker='*', s=100)  # test data
        plt.show()
        Z[is_us] = data.flatten()
        plt.contourf(X, Y, Z)  # GP mean
        plt.colorbar()
        plt.scatter(gpi.Xtrain[:, 0], gpi.Xtrain[:, 1],
                    color='green', marker='x', s=50)  # training data
        plt.scatter(X_new[:, 0], X_new[:, 1], color='purple',
                    marker='*', s=100)  # test data
        plt.show()

        Z[is_us] = mean.flatten()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # surf = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        #                   linewidth=0, antialiased=False)
        ax.contourf(X, Y, Z)
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_zlim(-15, 90)
        # plt.colorbar()
        ax.scatter(gpi.Xtrain[:, 0], gpi.Xtrain[:, 1],
                   gpi.Ytrain_original, color='green', marker='x',
                   s=50)  # training data
        # plt.scatter(X_new[:,0],X_new[:,1],color='purple',
        #             marker='*', s=100)   # test data
        plt.show()
        print X_new, Y_new


def evaluate_experiment(params, output_path):
    if params["data_name"] == "1d":
        data, X_range = datasets.load_data(
            params["data_name"])
    else:
        data, X_range, X, Y, Z, is_us = datasets.load_data(
            params["data_name"], True)

    gp, gp_list, s_list, loss_list = pickle_data(get_folder_name(params,
                                                                 output_path))

    if params["data_name"] == "ozone":
        goal_func = partial(utils.usa_goal_func, data=data,
                            X_range=X_range, sign=-1)
    elif params["data_name"] == "1d":
        goal_func = partial(utils.d1_goal_func, sign=-1)
    else:
        raise NameError("Data set name unknown.")

    y_opt = data[np.argmax(data)]
    x_opt = X_range[np.argmax(data), :]

    for i in range(len(gp_list)):
        gpi = gp_list[i]
        gpi.update_hyperparams_ml(params["bounds"])
        mean = np.zeros((X_range.shape[0], 1))
        var = np.zeros((X_range.shape[0], 1))
        for j in range(X_range.shape[0]):
            mean[j], var[j] = gpi.predict(X_range[j, :][np.newaxis, :])

        x_best = X_range[np.argmax(mean), :]
        x_first = gpi.Xtrain[0, :]

        print "Iteration:", (i + 1) * params["save_every"]
        print "Gap measure:", utils.gap_measure(goal_func, x_first,
                                                x_best, y_opt)
        print "Closeness measure:", utils.closeness_measure(x_best, x_opt)


if __name__ == "__main__":
    output_path = "lookahead_experiments/"

    # params = {
    #     "data_name": "ozone",
    #     "dimensions": 2,
    #     "noise": 1,
    #     "width": 5,
    #     "amp": 50,
    #     "cov_func": covariance_functions.gaussian_kernel,
    #     "cov_grad": covariance_functions.gaussian_kernel_gradient,
    #     "iterations": 50,
    #     "save_every": 1,
    #     "kappa": 4,
    #     "gamma": -0.05,
    #     "bounds": [(0.01, 100), (0.01, 1000), (0.01, 100)]
    # }

    # in order worst to best starting point
    # first_points = [1236, 570, 332, 2225, 475, 599, 245, 317, 2874, 3450,
    #                 763, 59, 6258, 2113, 5212, 1391, 1554, 1207, 3519,
    #                 6185, 4574, 4001, 3615, 2586, 2292, 3418, 2938,
    #                 3901, 3514, 3439, 3223, 5551, 2201, 1398, 2196, 5597,
    #                 5026, 3239, 4746, 5501, 3991, 4562, 3833, 3633, 5999, 2467,
    #                 5732, 4950, 1998, 2486, 3836, 5806, 4897,
    #                 5610, 4771, 4087, 53, 4405, 5611, 2756, 3262, 781, 2465,
    #                 3116, 2352, 2259, 3397, 5404, 3060, 1725, 2925,
    #                 3593, 5092, 5218, 3061, 4466, 3678, 392, 1810, 1719, 1457,
    #                 1282, 230, 793, 4881, 2614, 1351, 4453, 430, 2530, 4270,
    #                 2057, 2977, 3172, 1870, 3077, 1692, 1081, 2050, 1428]

    first_points = [1236, 570, 332, 599, 245]
    with open(params_file) as par:
        params = json.load(par)

    # run_experiments(params, first_points, output_path)

    plot_2d_experiment(params, output_path)

    # evaluate_experiment(params, output_path)

    """
    # code to get first sample points
    points = np.random.randint(0, X_range.shape[0], size=100)
    sorted = np.argsort(data[points].flatten())
    points = points[sorted]
    print ",".join(str(p) for p in points)
    print data[points]

    data, X_range, X, Y, Z, is_us = datasets.load_data(
        params["data_name"], True)

    plt.contourf(X, Y, Z)  # GP mean
    plt.colorbar()
    plt.scatter(X_range[points, 0], X_range[points, 1],
                color='green', marker='x', s=50)  # training data
    plt.show()
    """
