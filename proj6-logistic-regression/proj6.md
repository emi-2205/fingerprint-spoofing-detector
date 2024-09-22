# Project: Binary Logistic Regression Analysis

## Overview

We analyze the binary logistic regression model on the project data, starting with the standard, non-weighted version of the model without any pre-processing.

### Model Training

1. **Regularization Parameter (λ)**:
   - Train the model using different values for λ.
   - Use `numpy.logspace` to generate logarithmically spaced values for λ, e.g., `numpy.logspace(-4, 2, 13)` for good coverage.

2. **Validation**:
   - Score the validation samples and compute the corresponding **actual DCF** and **minimum DCF** for the primary application \( \pi_T = 0.1 \).
   - Remember to remove the log-odds of the training set empirical prior when computing actual DCF.

3. **Plotting Metrics**:
   - Plot actual DCF and minimum DCF as a function of λ using a logarithmic scale for the x-axis:
     ```python
     import matplotlib.pyplot as plt
     plt.xscale('log', base=10)
     ```

### Observations

- What do you observe in the metrics as λ varies?
- Are there significant differences for the different values of λ?
- How does the regularization coefficient affect the two metrics?

## Analysis with Fewer Training Samples

Given the large number of samples, regularization may seem ineffective and could degrade actual DCF as regularized models lose the probabilistic interpretation of scores.

1. **Sample Reduction**:
   - Repeat the previous analysis but keep only 1 out of 50 model training samples:
     ```python
     DTR_reduced = DTR[:, ::50]
     LTR_reduced = LTR[::50]
     ```

2. **Observations**:
   - What do you observe in this case?
   - Explain the results considering the effects of regularization: 
     - Lower values of the regularizer imply a larger risk of overfitting.
     - Higher values reduce overfitting but may lead to underfitting.

## Prior-Weighted Model Analysis

1. **Full Dataset**:
   - Repeat the analysis with the prior-weighted version of the model.
   - Remember to transform the scores to LLRs by removing the log-odds of the chosen prior during training.

2. **Comparison**:
   - Are there significant differences in performance?
   - Discuss the advantages of using the prior-weighted model for the application.

## Quadratic Logistic Regression Model

1. **Feature Expansion**:
   - Expand the features and train the quadratic logistic regression model (focus on the standard, non-prior-weighted model).
   - Consider different values for λ.

2. **Observations**:
   - What do you observe regarding regularization effectiveness?
   - How does it affect the two metrics?

## Effects of Data Centering

1. **Affine Transformations**:
   - Analyze the effects of centering on model results.
   - Optionally, try different strategies like Z-normalization, whitening, or PCA.

2. **Implementation**:
   - Center both datasets with respect to the model training dataset mean, avoiding using validation data for transformation estimation.

## Model Calibration and Comparison

1. **Calibration**:
   - Best models in terms of minimum DCF may not provide the best actual DCFs, indicating potential mis-calibration.

2. **Model Comparison**:
   - Compare all models trained (including Gaussian models) in terms of minimum DCF for the target application \( \pi_T = 0.1 \).
   - Which model(s) achieve(s) the best results?
   - Identify the separation rules or distribution assumptions characterizing these models.

3. **Feature Characteristics**:
   - Discuss how the results relate to the characteristics of the dataset features.