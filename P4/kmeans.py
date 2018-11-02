import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        # Initialization
        J = 1e10
        random_ind = np.random.choice(N,self.n_cluster)
        means = x[random_ind,:]

        def compute_square_distances(Xtrain, X):
            """
            Compute the distance between each test point in X and each training point
            in Xtrain.
            Inputs:
            - Xtrain: A numpy array of shape (num_train, D) containing training data
            - X: A numpy array of shape (num_test, D) containing test data.
            Returns:
            - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
              is the Euclidean distance between the ith test point and the jth training
              point.
            """
            num_train = Xtrain.shape[0]
            num_test =  X.shape[0]
            # Fast matrix operation implementation: (x-y)^2 = x^2 - 2*x*y + y^2
            Xsq = np.sum(np.square(X),axis=1)
            Xsq = Xsq.reshape(num_test,1)
            Xtrsq = np.sum(np.square(Xtrain),axis=1)
            Xtrsq = Xtrsq.reshape(1,num_train)
            dot = np.dot(X,Xtrain.T)
            dists = Xsq - 2*dot + Xtrsq
            return dists

        for iteration in range(self.max_iter):
            square_dist = compute_square_distances(x, means)
            membership = np.argmin(square_dist,axis=0)
            r = np.zeros((self.n_cluster, N))
            r[membership,np.arange(N)] = 1
            J_new = np.mean(np.sum(r * square_dist, axis=0))
            if np.absolute(J - J_new) <= self.e:
                break
            J = J_new
            means = np.dot(r,x) / np.sum(r,axis=1,keepdims = True)

        return (means, membership, iteration+1)
        # DONOT CHANGE CODE BELOW THIS LINE

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float) 
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by 
                    majority voting ((N,) numpy array) 
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        kmeans = KMeans(self.n_cluster, self.max_iter, self.e)
        centroids, membership, _ = kmeans.fit(x)
        r = -np.ones((self.n_cluster, N))
        r[membership,np.arange(N)] = 1
        label_matrix = r * y.reshape(1,-1)
        centroid_labels = []
        for k in range(self.n_cluster):
            to_choose = label_matrix[k,:]
            mask = to_choose >=0
            cluster_labels = to_choose[mask]
            (_, idx, counts) = np.unique(cluster_labels, return_index=True, return_counts=True)
            index = idx[np.argmax(counts)]
            centroid_labels.append(cluster_labels[index])
        centroid_labels = np.array(centroid_labels)
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        def compute_square_distances(Xtrain, X):
            """
            Compute the distance between each test point in X and each training point
            in Xtrain.
            Inputs:
            - Xtrain: A numpy array of shape (num_train, D) containing training data
            - X: A numpy array of shape (num_test, D) containing test data.
            Returns:
            - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
              is the Euclidean distance between the ith test point and the jth training
              point.
            """
            num_train = Xtrain.shape[0]
            num_test =  X.shape[0]
            # Fast matrix operation implementation: (x-y)^2 = x^2 - 2*x*y + y^2
            Xsq = np.sum(np.square(X),axis=1)
            Xsq = Xsq.reshape(num_test,1)
            Xtrsq = np.sum(np.square(Xtrain),axis=1)
            Xtrsq = Xtrsq.reshape(1,num_train)
            dot = np.dot(X,Xtrain.T)
            dists = Xsq - 2*dot + Xtrsq
            return dists

        dists = compute_square_distances(x,self.centroids)
        membership = np.argmin(dists,axis=0)
        labels = self.centroid_labels[membership]
        # DONOT CHANGE CODE BELOW THIS LINE
        return labels

