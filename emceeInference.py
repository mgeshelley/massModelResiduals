import numpy as np
import GPy as gpy
from multiprocessing import Pool
import emcee


# Log-probability function
def log_prob(p,model,noise):
    if any(p <= 0.):
        return -np.inf
        
    else:
        model[''] = list(p) + [noise]
        return model.log_likelihood()


## OPTIONS ##
# Seed
np.random.seed(100)

# Fraction of data to use for training
trainFrac = 0.5

# GP noise
noise = 0.0235

# More walkers than parameters
nwalkers = 40

# Chain length
nchain = 2000

# Show progress bar
progress_bar = False


## LOAD DATA ##
data = np.loadtxt('DZ_residuals.dat')


## CREATE TWO DATA SETS ##
# Shuffle
shuffled = np.copy(data)
rng = np.random.default_rng()
rng.shuffle(shuffled, axis=0)

# Select random indices of data for training, at random
trainIdx = np.random.choice(shuffled.shape[0], int(round(trainFrac*shuffled.shape[0])), replace=False)

# Create training and testing data sets
train = shuffled[trainIdx,:]
test = np.delete(shuffled, trainIdx, axis=0)

# Save to file the data sets
np.save('trainingData', train)
np.save('testingData', test)

# Input parameters and observations for training data
X = train[:,1:3]
Y = np.atleast_2d(train[:,3]).T


## MODEL SETUP ##
# Create 2D RBF kernel, with ARD
kernel = gpy.kern.RBF(input_dim=2, ARD=True)

# Create GPy model
    # Fix GP noise parameter to average scale of experimental uncertainty, 0.0235 MeV, as done in Neufcourt et al.
model = gpy.models.GPRegression(X, Y, kernel=kernel, noise_var=noise)
model.Gaussian_noise.variance.fix()
print('RBF parameters after model initialisation:')
print(model.rbf.param_array)

# Constrain RBF kernel hyperparameters to reasonable ranges
#model.rbf.variance.constrain_bounded(0., 20.)
#model.rbf.lengthscale.constrain_bounded(0.1, 10.)

# First optimise using ML to get good starting point for HMC
model.optimize()
print('RBF parameters after ML optimisation:')
print(model.rbf.param_array)

# Save ML parameters
np.savetxt('ML_values.dat', model.rbf.param_array)


## WALKER START POSITIONS ##
# RBF variance and lengthscales from ML optimisation
x0 = model.param_array[:3]

# Number of parameters
ndim = x0.shape[0]

print('Dimensions = {:d}'.format(ndim))
print('Walkers = {:d}'.format(nwalkers))
print('Chain length = {:d}'.format(nchain))
print('Function calls = {:d}'.format(nwalkers*nchain))

# Initialise walkers in 'small Gaussian ball' around minimum
p0 = x0 + 1e-5 * np.random.randn(nwalkers, ndim)


## SAMPLE ##
# Enable (shared-memory) parallelisation
with Pool() as pool:
    # Create sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(model, noise), pool=pool)
    
    # Run sampler with progress bar
    sampler.run_mcmc(p0, nchain, progress=progress_bar)

# Autocorrelation time
tau = sampler.get_autocorr_time()

# Number of samples to discard from beginning of chain
burnin = int(2 * np.max(tau))

# How much to thin chain
thin = int(0.5 * np.min(tau))

print('tau = ', tau)
print('Burn-in = {:d}'.format(burnin))
print('Thin = {:d}'.format(thin))

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