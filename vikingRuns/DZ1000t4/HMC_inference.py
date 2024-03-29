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
kernel = gpy.kern.RBF(input_dim=2, variance=1., lengthscale=(1., 1.), ARD=True)

# Create GPy model
    # Fix GP noise parameter to average scale of experimental uncertainty, 0.0235 MeV, as done in Neufcourt et al.
model = gpy.models.GPRegression(X, Y, kernel=kernel, noise_var=0.0235)
model.Gaussian_noise.variance.fix()

# Constrain RBF kernel hyperparameters to reasonable ranges
model.rbf.variance.constrain_bounded(0., 20.)
model.rbf.lengthscale.constrain_bounded(0.1, 10.)

# First optimise using ML to get good starting point for HMC
model.optimize()

# HMC inference of kernel hyperparameters
hmc = gpy.inference.mcmc.HMC(model, stepsize=5e-2)
samples = hmc.sample(num_samples=num_samples)

# Save samples
np.savetxt(name+'_samples.dat',samples)
