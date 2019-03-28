import numpy as np, scipy.spatial.distance

# kernel cookbook
# http://people.seas.harvard.edu/~dduvenaud/cookbook/


def gaussian_kernel(params, X1, X2):
        width = params[0]
        amplitude = params[1]
        """
        Generates the Gaussian kernel matrix K with K[i,j]=k(X1[i,:],X2[j,:])
        """
        # print X1.shape, X2.shape
        return (amplitude ** 2) * np.exp(-scipy.spatial.distance.cdist(X1, X2, metric='sqeuclidean') / (2 * np.power(width, 2)))


def gaussian_kernel_gradient(width, X1, X2):
        """
        Generates the Gaussian kernel matrix K with K[i,j]=k(X1[i,:],X2[j,:])
        """
        dist_mat = scipy.spatial.distance.cdist(X1, X2, metric='sqeuclidean')
        # print 'dist mat: ', dist_mat.shape
        # print 'exp: ', (np.exp(-dist_mat / (2 * width ** 2)) / width ** 3).shape
        return [dist_mat * np.exp(-dist_mat / (2 * np.power(width, 2))) / np.power(width, 3)]


def split(X, Y):
        """
        Partitions a dataset into a training and test set
        """
        n = len(X)

        rstate = np.random.mtrand.RandomState(2345)

        R = rstate.permutation(n)
        Rtrain = R[:n / 2]
        Rtest = R[n / 2:]

        Xtrain = X[Rtrain]  # Training data
        Ytrain = Y[Rtrain]  # Training targets

        Xtest = X[Rtest]  # Test data
        Ytest = Y[Rtest]  # Test targets

        return Xtrain, Ytrain, Xtest, Ytest
