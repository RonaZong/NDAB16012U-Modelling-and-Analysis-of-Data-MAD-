import numpy as np
import matplotlib.pyplot as plt

# diatoms contains 780 diatoms dexcrived by 90 successive "landmark points" (x_i, y_i) along the outline, recorded as (x_0, y_0, x_1, y_1, ..., x_89, y_89)
diatoms = np.loadtxt("diatoms.txt", delimiter=",").T
# diatoms_classes contains one class assignment per diatom, into species classified by the integers 1-37
diatoms_classes = np.loadtxt("diatoms_classes.txt", delimiter=",")
print('Shape of diatoms:', diatoms.shape)
print('Shape of diatoms_classes:', diatoms_classes.shape)
# print('Classes:', diatoms_classes)

d, N = diatoms.shape
print('Dimension:', d)
print('Sample size:', N)

# Task 1
def pca(data):
    """ 1) Noramilize data subtracting the mean shape. No need to use Procrustes Analysis or other more complex types of normalization
    2) Compute covariance matrix (check np.cov)
    3) Compute eigenvectors and values (check np.linalg.eigh)"""

    # Normalize data and center data points
    data_cent = data - data.mean(axis=1).reshape((-1,1))

    # Get covariance matrix
    cov = np.cov(data_cent)
    # Eigendecomposition extracts eigenvalues and corresponding eigenvectors of a matrix
    # PCevals is a vector of eigenvalues in decreasing order.
    # PCevecs is a matrix whose columns are the eigenvectors listed in the order of decreasing eigenvectors
    PCevals, PCevecs = np.linalg.eigh(cov)
    # Sort descending and get sorted indices
    idx = PCevals.argsort()[::-1]
    # Use indices on eigenvalue vector
    PCevals = PCevals[idx]
    PCevecs = PCevecs[:, idx]

    return PCevals, PCevecs, data_cent

PCevals, PCevecs, data_cent = pca(diatoms)

# Projection matrix
# print(data_cent.T.dot(PCevecs[:, :data_cent.shape[0]]))
np.dot(data_cent.T, PCevecs)

variance_explained_per_component = PCevals / np.sum(PCevals)
cumulative_variance_explained = np.cumsum(variance_explained_per_component)

plt.plot(cumulative_variance_explained)
plt.xlabel("Number of principal components included")
plt.ylabel("Proportion of variance explained")
plt.title("Proportion of variance explained as a function of number of PCs included")
plt.savefig("Proportion of variance explained as a function of number of PCs included")
plt.show()

# Let's print out the proportion of variance explained by the first 10 PCs
for i in range(10):
    print("Proportion of variance explained by the first " + str(i+1) + " principal components:", cumulative_variance_explained[i])

# Task 2
def plot_diatom(diatom):
    xs = np.zeros(91)
    ys = np.zeros(91)
    for i in range(90):
        xs[i] = diatom[2 * i]
        ys[i] = diatom[2 * i + 1]

    # Loop around to first landmark point to get a connected shape
    xs[90] = xs[0]
    ys[90] = ys[0]

    plt.plot(xs, ys)
    plt.axis('equal')

plot_diatom(diatoms[:, 0])
plt.title("Diatom")
plt.savefig("Diatom")
plt.show()

mean_diatom = np.mean(diatoms, 1)
plot_diatom(mean_diatom)
plt.title("Mean_diatom")
plt.savefig("Mean_diatom")
plt.show()

# Gets the fourth eigenvector
e4 = PCevecs[:, 3]
# Gets the fourth eigenvalue
lambda4 = PCevals[3]
# In case the naming std is confusing -- the eigenvalues have a statistical interpretation
std4 = np.sqrt(lambda4)
std4 *= e4
standard_deviation = [-3,-2,-1,0,1,2,3]

for i in range(7):
    """Please fill the gaps in the code to plot mean diatom shape with added FOURTH eigenvector
    mulitplied by [-3,-2,-1,0,1,2,3] standard deviations corresponding to this eigenvector.
    Submit the resulting plot for grading."""
    mean_diatom += std4*standard_deviation[i]
    plot_diatom(mean_diatom)
    mean_diatom = np.mean(diatoms, 1)
plt.title("Diatom shape along fourth eigenvector")
plt.savefig("Diatom shape along fourth eigenvector")
plt.show()

# diatoms_along_pc = np.zeros((7, 180))
# for i in range(7):
#     plot_diatom(diatoms_along_pc[i])
# plt.title("Diatom shape along PC1")
