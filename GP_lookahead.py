import time
import numpy as np
import scipy.stats as sp
import scipy.special as ssp
import copy
from multiprocessing import Pool


def get_probabilities(gp, phi_space):
    # print hyperparams
    # print hyperparams[0]
    gp_hyp = copy.deepcopy(gp)
    phi_log = np.log(phi_space)
    num_space, num_params = phi_space.shape

    # print 'hyperparams shape', phi_space.shape
    nu = np.mean(phi_log, axis=0)
    lamb = np.cov(phi_log.T)
    # print 'lambda shape', lamb.shape
    Ns = np.zeros((num_space, num_space))
    K = np.zeros(Ns.shape)
    p_nu = np.concatenate((nu, nu), axis=0)
    # print 'p nu  shape', p_nu.shape

    # pred_mean = np.concatenate()
    # print 'Ns shape', Ns.shape
    W = np.identity(num_params)
    # print 'W shape', W.shape
    p_cov = np.bmat([[lamb + W, lamb], [lamb, lamb + W]])
    # print 'p covariance shape', p_cov.shape
    N_distr = sp.multivariate_normal(mean=p_nu, cov=p_cov)
    K_distr = sp.multivariate_normal(mean=np.zeros((num_params,)), cov=W)
    phi_x, phi_y = np.meshgrid(range(num_space),
                               range(num_space), indexing='ij')
    phi_mat = np.zeros((num_space, num_space, 2 * num_params))
    phi_mat[:, :, :num_params] = phi_log[phi_x, :]
    phi_mat[:, :, num_params:] = phi_log[phi_y, :]
    Ns = N_distr.pdf(phi_mat)
    diff_mat = np.zeros((num_space, num_space, num_params))
    diff_mat = phi_space[phi_x, :] - phi_space[phi_y, :]
    K = K_distr.pdf(diff_mat)
    K_inv = np.abs(np.linalg.pinv(K))
    rs = np.zeros((num_space, 1))
    for i in range(rs.shape[0]):
        gp_hyp.cov_hyperparams = phi_space[i, :-1]
        gp_hyp.noise = phi_space[i, -1]
        gp_hyp.update(np.array([]), np.array([]))
        mu, C = gp_hyp.predict(gp.Xtrain)
        prob = gp_hyp.loglikelihood_par(phi_space[i, :])
        rs[i] = np.abs(np.exp(prob))
    # print rs

    e = np.ones(rs.shape)
    # F1 = np.dot(K_inv, np.dot(Ns, np.dot(K_inv, rs)))
    ps = np.dot(K_inv, np.dot(Ns, np.dot(K_inv, rs)))
    ps /= np.dot(e.T, ps)
    # for p, h in zip(ps, phi_space):
    #     print p, h
    # print sum(ps)
    return ps


def get_phis(hyperparams):
    phi_space = [[x, y] for x in hyperparams[0] for y in hyperparams[1]]
    for i in range(2, len(hyperparams)):
        # print hyperparams[i]
        phi_space = [x + [y] for x in phi_space for y in hyperparams[i]]

    phi_space = np.array(phi_space)
    return phi_space


def gaussian_cdf(x, mu, sig):
    new_x = (x - mu) / (np.sqrt(2) * sig)
    return 0.5 * (1 + ssp.erf(new_x))


def gaussian_pdf(x, mu, sig):
    return 1 / (sig * np.sqrt(2 * np.pi)) *\
        np.exp(-(x - mu) ** 2 / (2 * sig ** 2))


def calc_loss(X_range, gp, phi_space, ps, ni, chosen=None):
    gp_hyp = copy.deepcopy(gp)
    if chosen:
        print chosen
    loss = np.zeros(X_range.shape[0])
    for i in range(phi_space.shape[0]):
        # print 'phi', i
        gp_hyp.cov_hyperparams = phi_space[i, :-1]
        gp_hyp.noise = phi_space[i, -1]
        gp_hyp.update(np.array([]), np.array([]))
        for j in range(X_range.shape[0]):
            x = X_range[j, :][np.newaxis, :]
            mu, C = gp_hyp.predict(x)
            sig = np.sqrt(C)
            # v = (mu - ni) * (1 - sp.norm.cdf(ni, loc=mu, scale=C)) +
            # C * sp.norm.pdf(ni, loc=mu, scale=C)
            pdf = gaussian_pdf(ni, mu, sig)
            # print np.isclose(pdf, sp.norm.pdf(ni, loc=mu, scale=C))
            cdf = gaussian_cdf(ni, mu, sig)
            if chosen and (j == 470 or j == chosen):
                print 'idx =', j, 'e1 =', (mu - ni) * (1 - cdf), ' e2 =', \
                    C * pdf, 'mi =', mu, 'C =', C, 'ni =', ni

            # v = ni + (mu - ni) * (1 - cdf) + C * pdf
            v = ni + (mu - ni) * cdf - C * pdf
            loss[j] += v * ps[i]
    return loss


def calc_loss_multi((X_range, gp_hyp, ni)):
    # if chosen:
    #     print chosen
    loss = np.zeros(X_range.shape[0])
    for j in range(X_range.shape[0]):
        x = X_range[j, :][np.newaxis, :]
        mu, C = gp_hyp.predict(x)
        sig = np.sqrt(C)
        # v = (mu - ni) * (1 - sp.norm.cdf(ni, loc=mu, scale=C)) +
        # C * sp.norm.pdf(ni, loc=mu, scale=C)
        pdf = gaussian_pdf(ni, mu, sig)
        # print np.isclose(pdf, sp.norm.pdf(ni, loc=mu, scale=C))
        cdf = gaussian_cdf(ni, mu, sig)
        # if chosen and (j == 470 or j == chosen):
        #     print 'idx =', j, 'e1 =', (mu - ni) * (1 - cdf), ' e2 =',
        # C * pdf, 'mi =', mu, 'C =', C, 'ni =', ni

        # v = ni + (mu - ni) * (1 - cdf) + C * pdf
        v = ni + (mu - ni) * cdf - C * pdf
        loss[j] += v
    return loss


def calc_lookahead_multi((X_range, gp, ni, steps, phi_space, X_orig)):
    loss = np.ones(X_range.shape[0])
    rands = np.random.randint(len(X_orig), size=100)
    X_samp = X_orig[rands, :]
    for i in range(X_range.shape[0]):
        idx = i
        gp_tmp = copy.deepcopy(gp)
        x = X_range[idx, :][np.newaxis, :]
        # print 'i:', i, 'x:', x
        for j in range(steps):
            mu, C = gp_tmp.predict(x)
            sig = np.sqrt(C)
            Y_new = sp.norm.rvs(loc=mu, scale=sig)
            # print 'idx:', idx, 'Y new:', Y_new
            # loss[i] *= gaussian_pdf(Y_new, mu, sig)
            gp_tmp.update(x, Y_new)
            ps = get_probabilities(gp_tmp, phi_space)
            Y_range = calc_loss(X_samp, gp_tmp, phi_space, ps, ni)
            Y_range = np.array(Y_range)
            idx = np.argmin(Y_range)
            # idx = np.argmax(Y_range)
            x = X_samp[idx, :][np.newaxis, :]
            ni = np.min(gp_tmp.Ytrain_original)
            # ni = np.max(gp_tmp.Ytrain_original)
        loss[i] *= calc_loss_multi((x, gp_tmp, ni))
    return loss


def lookahead_optimization(gp, iterations, goal_func, X_range, hyperparams,
                           lookahead_steps=0, save_every=0):
    # ni = np.max(gp.Ytrain_original)
    ni = np.min(gp.Ytrain_original)
    # print 'max coord', gp.Xtrain[np.argmax(gp.Ytrain), :]
    print 'min coord', gp.Xtrain[np.argmin(gp.Ytrain), :]
    gp_list = []
    sample_list = []
    loss_list = []
    idx = None
    phi_space = get_phis(hyperparams)
    p = Pool(8)
    for i in xrange(iterations):
        rands = np.random.randint(len(X_range), size=100)
        X_samp = X_range[rands, :]
        steps = min(lookahead_steps, iterations - i)
        print 'steps', steps
        print 'Iteration', i
        Y_range = np.zeros((X_samp.shape[0], 1))
        st = time.time()
        ps = get_probabilities(gp, phi_space)
        args = []
        for j in range(phi_space.shape[0]):
            gp_hyp = copy.deepcopy(gp)
            gp_hyp.cov_hyperparams = phi_space[j, :-1]
            gp_hyp.noise = phi_space[j, -1]
            gp_hyp.update(np.array([]), np.array([]))
            args.append((X_samp, gp_hyp, ni, steps, phi_space, X_range))
            # args.append((X_range, gp_hyp, ni))
        # Y_range = p.map(calc_lookahead_multi, args)
        Y_range = p.map_async(calc_lookahead_multi, args).get(9999999)
        Y_range = np.array(Y_range)
        print Y_range.shape
        Y_range = np.dot(Y_range.T, ps)
        loss_list.append(Y_range)
        # Y_range = calc_loss(X_range, gp, phi_space, ps, ni, idx)
        et = time.time()
        print et - st
        idx = np.argmin(Y_range)
        # idx = np.argmax(Y_range)
        print 'Acquisition Insides:'
        print '<Chosen>'
        print Y_range[idx]
        print 'Index', idx
        print '<Best>'
        print Y_range[24]
        X_new = X_samp[idx, :][np.newaxis, :]
        Y_new = goal_func(X_new)

        gp.update(X_new, Y_new)
        # ni = np.max(gp.Ytrain_original)
        # print 'max coord', gp.Xtrain[np.argmax(gp.Ytrain), :]
        ni = np.min(gp.Ytrain_original)
        print 'min coord', gp.Xtrain[np.argmin(gp.Ytrain), :]
        if save_every and ((i + 1) % save_every) == 0:
            print "Saving interation", i + 1
            gp_list.append(copy.deepcopy(gp))
            sample_list.append((X_new, Y_new))

    if save_every:
        return gp, gp_list, sample_list, loss_list
    else:
        return gp
