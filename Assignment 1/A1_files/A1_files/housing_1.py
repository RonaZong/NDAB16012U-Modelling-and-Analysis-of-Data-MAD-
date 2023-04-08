import numpy
import matplotlib.pyplot as plt

# load data
train_data = numpy.loadtxt("boston_train.csv", delimiter=",")
test_data = numpy.loadtxt("boston_test.csv", delimiter=",")
# X_train: first 13 values correspond to features
# t_train: target house price = the last value MEDV(median value of owner-occupied homes in $1000's)
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (a) compute mean of prices on training set
train_mean = numpy.mean(t_train)
print(train_mean)

# (b) RMSE function
def rmse(t, tp):
    # numpy.linalg.norm(t-tp)/numpy.sqrt(len(tp))
    return numpy.sqrt(numpy.mean((t-tp)**2))
RMSE = rmse(t_test, train_mean)
print(RMSE)

# (c) visualization of results
xplot = range(253)
y1 = t_test
y2 = numpy.full(shape=253, fill_value=train_mean)
plt.scatter(xplot, y1)
plt.scatter(xplot, y2)
plt.savefig("housing_1")
plt.show()