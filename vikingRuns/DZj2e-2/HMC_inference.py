import numpy as np
import GPy as gpy


## OPTIONS ##
name = 'DZ'
thin = 1
num_samples = 1000

# Load data
if name == 'DZ':
    data = np.loadtxt('DZ_residuals.dat')
elif name == 'LD':
    data = np.loadtxt('LDM_residuals.dat')

X = data[::thin,1:3]
Y = np.atleast_2d(data[::thin,3]).T

# Create kernel
kernel = gpy.kern.RBF(input_dim=2, ARD=True)

# Create GPy model
    # Fix GP noise parameter to average scale of experimental uncertainty, 0.0235 MeV, as done in Neufcourt et al.
model = gpy.models.GPRegression(X, Y, kernel=kernel, noise_var=0.0235)
model.Gaussian_noise.variance.fix()
print('RBF parameters after model initialisation:')
print(model.rbf.param_array)

# Constrain RBF kernel hyperparameters to reasonable ranges
model.rbf.variance.constrain_bounded(0., 20.)
model.rbf.lengthscale.constrain_bounded(0.1, 10.)

# First optimise using ML to get good starting point for HMC
model.optimize()
print('RBF parameters after ML optimisation')
print(model.rbf.param_array)

# Change start position of parameters to avoid HMC getting stuck
model.rbf.lengthscale = model.rbf.lengthscale + 1e-4 * np.random.randn(2)
model.rbf.variance = model.rbf.variance + 1e-4 * np.random.randn(1)
print('RBF parameters after adding small normal random numbers')
print(model.rbf.param_array)

# HMC inference of kernel hyperparameters
hmc = gpy.inference.mcmc.HMC(model, stepsize=2e-2)
samples = hmc.sample(num_samples=num_samples)
print('RBF parameters as means of HMC samples')
print(samples[:,:].mean(axis=0))

# Save samples
np.savetxt(name+'_samples.dat',samples)
