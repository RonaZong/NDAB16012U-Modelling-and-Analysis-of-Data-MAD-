import numpy
# import pandas
import linreg
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

# (b) fit linear regression using only the first feature
model_single = linreg.LinearRegression()
model_single.fit(X_train[:,0], t_train)
print(model_single.w)

# (c) fit linear regression model using all features
W = linreg.LinearRegression()
W.fit(X_train, t_train)
# (14,1)
print(W.w)

# (d) evaluation of results
# (i) the model that is based on the first feature
prediction = model_single.predict(X_train[:,0])
print(prediction)
RMSE = numpy.sqrt(numpy.mean((t_test-prediction)**2))
print(RMSE)
prediction = numpy.array(prediction).reshape(len(prediction))
xplot = range(253)
y1 = t_test
y2 = numpy.full(shape=253, fill_value=prediction)
plt.scatter(xplot, y1)
plt.scatter(xplot, y2)
plt.savefig("housing_2_predict")
plt.show()

# (ii) the model that is based on all features
predictions = W.predict(X_train)
print(predictions)
RMSEs = numpy.sqrt(numpy.mean((t_test-predictions)**2))
print(RMSEs)
predictions = numpy.array(predictions).reshape(len(predictions))
xplot = range(253)
y1 = t_test
y2 = numpy.full(shape=253, fill_value=predictions)
plt.scatter(xplot, y1)
plt.scatter(xplot, y2)
plt.savefig("housing_2_predicts")
plt.show()