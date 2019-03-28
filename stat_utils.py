import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_current_state(gp, goal_func, X_range, X_new, Y_new):
    # Compute the predicted mean and variance for test data        
    mean,cov = gp.predict(X_range)
    var = np.sqrt(np.diag(cov))[:,np.newaxis]

    # Plot the data
    #plt.title("Iteration: " + str(i))
    plt.xlabel('x')
    plt.ylabel('y')    
    plt.xlim(np.min(X_range), np.max(X_range))
    #plt.ylim(-2,2)
    plt.scatter(gp.Xtrain, gp.Ytrain,     color='green',  marker='x')          # training data
    plt.plot(X_range, mean,               color='blue')                        # GP mean
    plt.plot(X_range, mean + var,         color='red')                         # GP mean + std
    plt.plot(X_range, mean - var,         color='red')                         # GP mean - std
    plt.plot(X_range, goal_func(X_range), color='black')                       # goal function
    plt.scatter(X_new, Y_new,             color='purple', marker='*', s=100)   # test data
    plt.show()
    
def statistic_plot_hyperparams(hp_list):
    iters = range(len(hp_list[0]))
    colors = ['blue', 'red', 'green', 'black', 'purple']
    
    plt.title("Hyperparams statistic")
    plt.xlabel('iteration')
    plt.ylabel('value')    
    plt.xlim(0, len(iters))
    #plt.ylim(-2,2)
    i = 0
    labels = ['width', 'amp', 'noise']
    for l in hp_list:
        if l:
            plt.plot(iters, l, color=colors[i], label=labels[i])   
            i += 1
    plt.legend(loc='upper right',fontsize=8)
    plt.show()
    
def statistic_collect_hyperparams(optimization_res, hp_list):
    for i in range(len(optimization_res.get('x'))):
        hp_list[i].append(optimization_res.get('x')[i])
    
def statistic_export(path_file, data):
    with open(path_file, 'w') as f:
        writer = csv.writer(f)

        #then the data
        for l in data:
            writer.writerow(dl)

        f.close()
    
def statistic_import(fil):
    hp_list = []
    with open(fil, 'rb') as f:
        data = csv.reader(f)
       
        for i in range(5):
            hp_list.append(data.next())

        f.close()
    
    return hp_list