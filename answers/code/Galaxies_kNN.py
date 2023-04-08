import csv
import numpy
import matplotlib.pyplot as plt

train_data = numpy.loadtxt("../data/galaxies_train.csv", delimiter=",", skiprows=1)
test_data = numpy.loadtxt("../data/galaxies_test.csv", delimiter=",", skiprows=1)

X_train = train_data[:,1:] # 10 attributes
t_train = train_data[:,0] # target value
X_test = test_data[:,1:]
t_test = test_data[:,0]
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of attributes: %i" % X_train.shape[1])

# NOTE: You are supposed to use this structure, i.e.,
# the pre-defined functions and variables. If you
# have difficulties to keep this structure, you ARE
# ALLOWED to adapt/change the code structure slightly!
# You might also want to add additional functions or
# variables.

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

        self.X_train = X
        self.t_train = t

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
        self.X_train = X
        predictions = (1/self.n_neighbors)*self.t_train
        predictions = numpy.array(predictions)

        return predictions

# TODO: ADD YOUR CODE HERE
k = 3
d = numpy.square(X_train-X_test)
D = d.shape[1]
eulidean = []
for i in range(d.shape[0]):
    sum = 0
    for l in range(D):
        sum += d[i][l]
    sum = numpy.sqrt(sum)
    eulidean.append(sum)

for n in range(1,k+1):
    model = NearestNeighborRegressor()
    model.__init__(n_neighbors=n, dist_measure=eulidean)
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
