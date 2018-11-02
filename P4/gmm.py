import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE
            k_means = KMeans(self.n_cluster, self.max_iter, self.e)
            self.means, membership, _ = k_means.fit(x)
            gamma = np.zeros((self.n_cluster,N))
            gamma[membership,np.arange(N)] = 1
            self.variances = np.zeros((self.n_cluster, D, D))
            Nk = np.sum(gamma,axis=1)
            for k in range(self.n_cluster):
                diff = x - self.means[k,:].reshape(1,-1)
                self.variances[k,:,:] = np.dot(diff.T,gamma[k,:].reshape(-1,1) * diff) / Nk[k]
            self.pi_k = Nk / N
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE
            self.means = np.random.rand(self.n_cluster, D)
            self.variances = np.tile(np.identity(D).reshape(1,D,D),(self.n_cluster,1,1))
            self.pi_k = np.ones(self.n_cluster) / self.n_cluster
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        # DONOT MODIFY CODE ABOVE THIS LINE
        log_likelihood = -1e10
        for iteration in range(self.max_iter):
            # E step
            gamma = np.zeros((self.n_cluster, N))
            for k in range(self.n_cluster):
                gaus = self.Gaussian_pdf(self.means[k,:].reshape(1,-1),self.variances[k,:,:])
                for i in range(N):
                    Normal = gaus.getLikelihood(x[i,:].reshape(1,-1))
                    gamma[k,i] = self.pi_k[k] * Normal
            gamma/=np.sum(gamma,axis=0,keepdims=True)

            # M step:
            Nk = np.sum(gamma,axis=1) # (K,)
            self.means = np.dot(gamma,x) / Nk.reshape(-1,1)
            self.variances = np.zeros((self.n_cluster, D, D))
            for k in range(self.n_cluster):
                diff = x - self.means[k,:].reshape(1,-1)
                self.variances[k,:,:] = np.dot(diff.T,gamma[k,:].reshape(-1,1) * diff) / Nk[k]
            self.pi_k = Nk / N
            new_log_likelihood = self.compute_log_likelihood(x)
            if np.absolute(log_likelihood - new_log_likelihood)<= self.e:
                break
            log_likelihood = new_log_likelihood
        return iteration + 1
        # DONOT MODIFY CODE BELOW THIS LINE

		
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        D = self.means.shape[1]
        samples = np.zeros((N,D))
        ks = np.random.choice(np.arange(self.n_cluster), N, p=self.pi_k) 
        for n in range(N):
            samples[n,:] = np.random.normal(self.means[ks[n],:],self.variances[ks[n],:,:].diagonal())
        # DONOT MODIFY CODE BELOW THIS LINE
        return samples        

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k    
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        N,D = x.shape
        ps = np.zeros((N,self.n_cluster))
        for k in range(self.n_cluster):
            gaus = self.Gaussian_pdf(self.means[k,:].reshape(1,-1),self.variances[k,:,:])
            for i in range(N):
                ps[i,k] = gaus.getLikelihood(x[i,:].reshape(1,-1))
        log_likelihood = np.sum(np.log(np.dot(self.pi_k, ps.T)))
        # DONOT MODIFY CODE BELOW THIS LINE
        return np.asscalar(log_likelihood)

    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            # Ensure full rank
            D = mean.size
            while True:
                try:
                    self.inv = np.linalg.inv(variance)
                    break
                except np.linalg.LinAlgError:
                    variance+=1e-3 * np.identity(D)
            self.c = np.power(2*np.pi,D) * np.linalg.det(variance)
            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            diff = x - self.mean
            expotent = -0.5*np.dot(np.dot(diff,self.inv),diff.T)
            p = np.exp(expotent) / np.sqrt(self.c)
            # DONOT MODIFY CODE BELOW THIS LINE
            return p
