# Project: Analysis of MVG Classifier Performance

## Applications Overview

Analyze the performance of the MVG classifier and its variants for different applications defined by the triplet (π1, Cfn, Cfp):

1. **Uniform Prior and Costs**: 
   - (0.5, 1.0, 1.0)
   
2. **Higher Prior Probability of Genuine Samples**:
   - (0.9, 1.0, 1.0)
   - (Most users are legitimate)
   
3. **Higher Prior Probability of Fake Samples**:
   - (0.1, 1.0, 1.0)
   - (Most users are impostors)

4. **Increased Cost of Accepting a Fake Image**:
   - (0.5, 1.0, 9.0)
   - (Stronger security needed)

5. **Increased Cost of Rejecting a Legit Image**:
   - (0.5, 9.0, 1.0)
   - (Ease of use for legitimate users)

### Effective Priors Representation

Represent the applications in terms of effective prior. Analyze how the costs of misclassifications are reflected in the prior. Observations:
- Stronger security (higher false positive cost) corresponds to a lower effective prior probability of a legitimate user.

## Focus on Three Applications

Now focus on the applications represented by effective priors:
- π̃ = 0.1
- π̃ = 0.5
- π̃ = 0.9

For each application, compute the optimal Bayes decisions for the validation set using the MVG models and their variants, both with and without PCA (try different values of m).

### Metrics Calculation

- Compute **DCF (actual)** and **minimum DCF** for the different models.
- Compare the models in terms of minimum DCF.
  - Which models perform best?
  - Are the relative performance results consistent across the different applications?

### Calibration Analysis

Now consider actual DCFs. Analyze the models' calibration:
- Are the models well calibrated (i.e., calibration loss within a few percent of the minimum DCF value) for the applications?
- Are there models that are better calibrated than others for the considered applications?

## PCA Setup for Best Results

Consider the PCA setup that gave the best results for the π̃ = 0.1 configuration (main application):

### Bayes Error Plots

Compute the Bayes error plots for the MVG, Tied, and Naive Bayes Gaussian classifiers. 

- Compare the minimum DCF of the three models for different applications.
- For each model, plot both minimum and actual DCF.

### Prior Log Odds Analysis

Consider prior log odds in the range (-4, +4):

- What do you observe?
- Are model rankings consistent across applications (minimum DCF)?
- Are models well-calibrated over the considered range?