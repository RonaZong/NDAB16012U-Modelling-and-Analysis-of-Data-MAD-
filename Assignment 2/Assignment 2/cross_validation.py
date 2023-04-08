import numpy as np
import matplotlib.pyplot as plt
import linreg

raw = np.genfromtxt("men-olympics-100.txt", delimiter=" ")
X, t = raw[:,0], raw[:,1]

# maximum degree
max_degree = 2

# reshape
X = X.reshape((len(X), 1))
t = t.reshape((len(t), 1))
print("Shape of our data matrix: %s" % str(X.shape))
print("Shape of our data vector: %s" % str(t.shape))

model = linreg.LinearRegression()
model.fit(X,t)
predictions = model.predict(X)

plt.plot(X, t, "o")
plt.plot(X, predictions, "x", color = "red")
plt.show()


def augment(X, max_order):
    """ Augments a given data
    matrix by adding additional
    columns.

    NOTE: In case max_order is very large,
    numerical inaccuracies might occur
    """

    X_augmented = X

    for i in range(2, max_order + 1):
        print("Augmented with order %i ..." % i)
        X_augmented = np.concatenate([X_augmented, X ** i], axis=1)

    return X_augmented

Xnew = augment(X, max_degree)
max_degree += 1
print("Shape of augmented data matrix: %s" % str(Xnew.shape))

# fit linear regression model using the augmented data matrix
model = linreg.LinearRegression()
model.fit(Xnew, t)
predictions = model.predict(Xnew)

# plot the results: only use the first column of the augmented
# data matrix, which corresponds to the original one-dimensional
# input variables
plt.plot(Xnew[:,0], t, 'o')
plt.plot(Xnew[:,0], predictions, 'x', color='red')
plt.show()

Xnew = augment(X, max_degree)
max_degree += 1
print("Shape of augmented data matrix: %s" % str(Xnew.shape))

# fit linear regression model using the augmented data matrix
model = linreg.LinearRegression()
model.fit(Xnew, t)
predictions = model.predict(Xnew)

# plot the results: only use the first column of the augmented
# data matrix, which corresponds to the original one-dimensional
# input variables
plt.plot(Xnew[:,0], t, 'o')
plt.plot(Xnew[:,0], predictions, 'x', color='red')
plt.show()

# same plot as before but with some more points
# for plotting the model ...
Xplot = np.arange(X.min(), X.max(), 0.01)
Xplot = Xplot.reshape((len(Xplot), 1))
Xplot = augment(Xplot, max_degree)
preds_plot = model.predict(Xplot)

plt.plot(Xnew[:,0], t, 'o')
plt.plot(Xnew[:,0], predictions, 'x', color='red')
plt.plot(Xplot[:,0], preds_plot, '-', color='red')
plt.show()