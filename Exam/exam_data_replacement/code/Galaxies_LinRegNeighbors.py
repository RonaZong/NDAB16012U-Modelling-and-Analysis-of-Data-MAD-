import csv
import numpy
import matplotlib.pyplot as plt

train_data = numpy.loadtxt("../data/galaxies_train.csv", delimiter=",", skiprows=1)
test_data = numpy.loadtxt("../data/galaxies_test.csv", delimiter=",", skiprows=1)

X_train = train_data[:,1:]
t_train = train_data[:,0]
X_test = test_data[:,1:]
t_test = test_data[:,0]
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of attributes: %i" % X_train.shape[1])

# TODO: ADD YOUR CODE HERE
class NearestNeighborRegressor:

    def __init__(self, n_neighbors=1, dist_measure="euclidean", dist_matrix=None):
        """
        Initializes the model.

        Parameters
        ----------
        n_neighbors : The number of nearest neighbors (default 1)
        dist_measure : The distance measure used (default "euclidean")
        dist_matrix : The distance matrix if needed (default "None")
        """

        self.n_neighbors = n_neighbors
        self.dist_measure = dist_measure
        self.dist_matrix = dist_matrix

    def fit(self, X, t):
        """
        Fits the nearest neighbor regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of length n_samples
        """

        X = numpy.array(X).reshape((len(X), -1))
        t = numpy.array(t).reshape((len(t), 1))

        # prepend a column of ones
        ones = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((ones, X), axis=1)

        # compute weights  (matrix inverse)
        self.w = numpy.linalg.inv((numpy.dot(X.T, X)))
        self.w = numpy.dot(self.w, X.T)
        self.w = numpy.dot(self.w, t)

    def predict(self, X):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of length n_samples
        """

        predictions = []

        # TODO: ADD YOUR CODE HERE
        X = numpy.array(X).reshape((len(X), -1))

        # prepend a column of ones
        ones = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((ones, X), axis=1)
        X = numpy.dot(X, self.w)

        sum = 0
        for i in range(10):
            sum += numpy.square(self.w[i])

        predictions = numpy.argmin(self.w)*(1/self.n_neighbors)*numpy.square(X-t_train)+sum
        predictions = numpy.array(predictions)

        return predictions

# TODO: ADD YOUR CODE HERE
k = 13

for n in range(1,k+1):
    model = NearestNeighborRegressor()
    model.__init__(n_neighbors=n)
    model.fit(X_train, t_train)
    predictions = model.predict(X_train)
    RMSE = numpy.sqrt(numpy.mean((t_test-predictions)**2))
    print(RMSE)

xplot = range(X_train.shape[0])
y1 = t_test
y2 = numpy.full(shape=X_train.shape[0], fill_value=predictions)
plt.scatter(xplot, y1)
plt.scatter(xplot, y2)
plt.savefig("true redshift vs predicted redshift")
plt.show()