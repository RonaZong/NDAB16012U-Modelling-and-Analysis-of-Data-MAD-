import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# We will consider the sum of i.i.d. uniformly distributed random variables on the interval [0,1]

# True mean and standard deviation of the Uniform(0,1) distribution
# Needed for standardization purposes
mu = 0.5
sigma = np.sqrt(1.0/12.0)


# Lets consider increasing values of n
nvals=[1, 2, 10, 100]

# Define plotting interval
plot_limits = (-3.0,3.0)
plot_range = np.linspace(plot_limits[0], plot_limits[1], 100)


idx = 1
for n in nvals:
    xmeans = []
    for i in range(1000): # Generate many sets of random samples 
        x = np.random.uniform(0,1, n) # Generate n random samples from the uniform distribution on the interval [0,1]
        xmeans.append(np.sqrt(n)*(np.mean(x)-mu)/sigma) # Standardize sample mean estimator
    
    plt.subplot(2,2,idx)
    plt.hist(xmeans, 10, plot_limits, density=True) # Make histogram and normalize to represent a density
    plt.plot(plot_range, norm.pdf(plot_range), 'r') # Plot the standard normal distribution N(0,1) for comparison
    plt.title("n = " + str(n))
    
    idx+=1
    

plt.show()