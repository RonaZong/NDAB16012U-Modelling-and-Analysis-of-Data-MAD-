import numpy
# import pandas
import linweighreg
import matplotlib.pyplot as plt

# load data
train_data = numpy.loadtxt("boston_train.csv", delimiter=",")
test_data = numpy.loadtxt("boston_test.csv", delimiter=",")
# X_train: first 13 values correspond to features
# t_train: talist(range(len(self.w)))rget house price = the last value MEDV(median value of owner-occupied homes in $1000's)
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])


# (b) fit linear regression model using all features
W = linweighreg.LinearRegression()
W.fit(X_train, t_train)
# (14,1)
print(W.w)
# compute predictions
predictions = W.predict(X_test)
print(predictions)

# scatter plot for the model that is based on all features
predictions = numpy.array(predictions).reshape(len(predictions))
xplot = range(253)
y1 = t_test
y2 = numpy.full(shape=253, fill_value=predictions)
plt.scatter(xplot, y1)
plt.scatter(xplot, y2)
plt.savefig("housing_2")
plt.show()