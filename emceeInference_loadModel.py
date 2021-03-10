import pickle
from multiprocessing import Pool

import numpy as np

import GPy as gpy
import emcee


# Log-probability function
def log_prob(p, model):
    if any(p <= 0.):
        return -np.inf
        
    else:
        model[''] = list(p) + [model.param_array[3]]
        return model.log_likelihood()


# Load model
with open('inferenceModels/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Fix GP noise parameter to average scale of experimental uncertainty
model.Gaussian_noise.variance.fix()


## OPTIONS ##
# MCMC Options
nwalkers = 32
nchain = 2000

# RBF variance and lengthscales from ML optimisation
x0 = model.rbf.param_array

# Number of parameters
ndim = x0.shape[0]

print(f'Dimensions = {ndim}')
print(f'Walkers = {nwalkers}')
print(f'Chain length = {nchain}')
print(f'Function calls = {nwalkers*nchain}')

# Initialise walkers in 'small Gaussian ball' around minimum
p0 = x0 + 1e-5 * np.random.randn(nwalkers, ndim)


# Enable (shared-memory) parallelisation
with Pool() as pool:
    # Create sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(model,), pool=pool)
    
    # Run sampler with progress bar
    sampler.run_mcmc(p0, nchain, progress=True)

# Autocorrelation time
tau = sampler.get_autocorr_time()

# Number of samples to discard from beginning of chain
burnin = int(2 * np.max(tau))

# How much to thin chain
thin = int(0.5 * np.min(tau))

print(f'tau = {tau}')
print(f'Burn-in = {burnin}')
print(f'Thin = {thin}')

# 3D array of samples
samples = sampler.get_chain()

# Save all samples for reuse
np.save('samples', samples)

# Remove 'burn-in' samples, and thin samples so not too many correlated samples are plotted
flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

# Save modified samples for plotting again
np.savetxt('flat_samples.dat', flat_samples)

# Extract log probabilites of chain
log_prob_samples = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)

# Save log-probs for plotting
np.savetxt('log_probs.dat', log_prob_samples)