import numpy as np
from scipy import optimize



class GP_Regressor(object):
    # currently no prior mean function possible
    def __init__(self, dimensions, noise, cov_func, cov_grad, cov_hyperparams):
        self.Sigma_inv = None
        self.Xtrain = np.empty([0, dimensions], dtype='float64')
        self.Ytrain = np.empty([0, 1], dtype='float64')
        self.Ytrain_original = np.empty([0, 1], dtype='float64')
        self.noise = noise
        #self.original_noise = noise
        self.cov_func = cov_func
        self.cov_grad = cov_grad
        self.cov_hyperparams = cov_hyperparams
        #self.original_cov_hyperparams = cov_hyperparams
        self.mean = 0
        #self.original_min = np.Inf
        #self.original_max = -np.Inf

    def normalize_data(self):
        """
        self.original_max = np.max(self.Ytrain_original)
        self.original_min = np.min(self.Ytrain_original)
        self.Ytrain = self.Ytrain_original - self.original_min
        if self.original_max != 0:
            self.Ytrain = 2 * self.Ytrain / np.max(self.Ytrain)
        self.Ytrain = self.Ytrain - np.mean(self.Ytrain)
        """
        self.Ytrain = self.Ytrain_original - np.mean(self.Ytrain_original)

    def rescale(self, mean, variance):
        """
        scaling_factor = 0.5 * (self.original_max - self.original_min)
        new_mean = (mean - np.min(self.Ytrain)) * scaling_factor + self.original_min
        new_variance = variance * (scaling_factor ** 2)
        return new_mean, new_variance
        """
        return mean + np.mean(self.Ytrain_original), variance


    def update(self, Xtrain_new, Ytrain_new):
        #Ytrain_new = Ytrain_new[:, np.newaxis]

        if Xtrain_new.size:
            self.Xtrain = np.concatenate((self.Xtrain, Xtrain_new), axis=0)
        if Ytrain_new.size:
            self.Ytrain_original = np.concatenate((self.Ytrain_original, Ytrain_new), axis=0)
            #self.Ytrain = np.concatenate((self.Ytrain, Ytrain_new), axis=0)
            #self.normalize_data()
        self.mean = np.mean(self.Ytrain_original)
        self.Ytrain = self.Ytrain_original - self.mean
        self.Sigma_inv = self.cov_func(self.cov_hyperparams, self.Xtrain, self.Xtrain)
        self.Sigma_inv += (np.power(self.noise, 2)) * np.identity(self.Xtrain.shape[0])
        self.Sigma_inv = np.linalg.pinv(self.Sigma_inv)
        # todo: hyperparameter adjustment

    def predict(self, Xtest):
        C_s = self.cov_func(self.cov_hyperparams, Xtest, Xtest)
        C_s += (np.power(self.noise, 2)) * np.identity(Xtest.shape[0])

        if self.Xtrain.shape[0] == 0:
            print "don't go here! often"
            mu_s = np.zeros(Xtest.shape)
        else:
            Sigma_s = self.cov_func(self.cov_hyperparams, self.Xtrain, Xtest)
            mu_s = np.dot(Sigma_s.T, np.dot(self.Sigma_inv, self.Ytrain))
            C_s -= np.dot(Sigma_s.T, np.dot(self.Sigma_inv, Sigma_s))


        return mu_s + self.mean, C_s

    def loglikelihood(self, Xtest, Ytest):
        #Ytest = Ytest[:, np.newaxis]
        mu_s, C_s = self.predict(Xtest)
        diff = Ytest - mu_s
        f1 = np.dot(diff.T, np.dot(np.linalg.pinv(C_s), diff))
        s, logdet = np.linalg.slogdet(C_s)
        f2 = logdet
        f3 = Xtest.shape[0] * np.log(2 * np.pi)
        return -0.5 * (f1 + f2 + f3)

    def loglikelihood_par(self, pars, sign=1):
        w = pars[:-2]
        a = pars[-2]
        n = pars[-1]
        Sigma_inv = self.cov_func([w,a], self.Xtrain, self.Xtrain)
        Sigma_inv += (np.power(n, 2)) * np.identity(self.Xtrain.shape[0])
        Sigma = Sigma_inv
        Sigma_inv = np.linalg.pinv(Sigma_inv)
        f1 = np.dot(self.Ytrain.T, np.dot(Sigma_inv, self.Ytrain))
        s, f2 = np.linalg.slogdet(Sigma)
        f3 = self.Xtrain.shape[0] * np.log(2 * np.pi)
        return sign * (-0.5 * (f1 + f2 + f3))

    def logtrain(self):
        f1 = np.dot(self.Ytrain.T, np.dot(self.Sigma_inv, self.Ytrain))
        s, f2 = np.linalg.slogdet(np.linalg.pinv(self.Sigma_inv))
        f3 = self.Xtrain.shape[0] * np.log(2 * np.pi)
        return -0.5 * (f1 + f2 + f3)

    def update_hyperparams(self):
        old_pars = self.cov_hyperparams
        converged = False
        alpha = np.dot(self.Sigma_inv, self.Ytrain)
        eta = 1 / (10 * np.trace(np.dot(alpha, alpha.T)))
        log1 = np.log(eta)
        log2 = np.log(eta / 10)
        etas = np.exp(np.linspace(log1, log2, 100000))
        iters = 1
        while not converged and iters < 100000:
            dSigmas = self.cov_grad(old_pars, self.Xtrain, self.Xtrain)
            grad_pars = np.zeros(old_pars.shape)
            for i in range(len(dSigmas)):
                # print(np.dot(alpha, alpha.T).shape)
                # pr)nt(dSigmas[i].shape)
                # print np.sum(dSigmas[i])
                grad_pars[i] = 0.5 * np.trace(np.dot(np.dot(alpha, alpha.T) - self.Sigma_inv, dSigmas[i]))
            # print "gradients:", grad_pars, 'sum ', np.sum(dSigmas[i])
            th = 0.7
            if abs(etas[i] * grad_pars) > th:
                new_pars = old_pars + np.sign(etas[i] * grad_pars) * th
            else:
                new_pars = old_pars + etas[i] * grad_pars

            converged = np.allclose(new_pars, old_pars)  # np.allclose(new_pars, old_pars)
            old_pars = new_pars
            # print old_pars
            iters += 1
        print "iterations: ", iters
        self.cov_hyperparams = old_pars
        self.update(np.array([]), np.array([]))

    def update_hyperparams_ml(self, bounds):
        pars = np.append(self.cov_hyperparams,self.noise)
        print '\tHP Optimization:\n\t\tbefore: width=', pars[0], '- amp=', pars[1], '- noise=', pars[2]
        res = optimize.minimize(fun=self.loglikelihood_par, x0=pars , args=(-1),
                                bounds=bounds)#, method="SLSQP") #jac=False)
        print '\t\tafter: width=', res.get('x')[0], '- amp=', res.get('x')[1], '- noise=', res.get('x')[2]
        self.cov_hyperparams = abs(res.x[:-1])
        self.noise = abs(res.x[-1])
        """
        if np.isclose(self.noise, np.array(bounds)[-1,0]) or np.any(np.isclose(self.cov_hyperparams, np.array(bounds)[:-1,0])):
            print "HPs at lower bound"
            self.cov_hyperparams = self.original_cov_hyperparams
            self.noise = self.original_noise
            print self.cov_hyperparams#
        """
        self.update(np.array([]), np.array([]))
        return res
