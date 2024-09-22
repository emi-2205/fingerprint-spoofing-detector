# Gaussian Model Fitting

Try fitting univariate Gaussian models to the different features of the different classes of the project dataset. 

- For each class, for each component of the feature vector of that class, compute the ML (Maximum Likelihood) estimate for the parameters of a 1D Gaussian distribution.
  
- Plot the distribution density on top of the normalized histogram:
  - Remember that you have to exponentiate the log-density.
  - Set `density=True` when creating the histogram (refer to Laboratory 2 for details).

### Observations
- What do you observe?
- Are there features for which the Gaussian densities provide a good fit?
- Are there features for which the Gaussian model seems significantly less accurate?