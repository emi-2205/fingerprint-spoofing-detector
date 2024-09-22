# Project: SVM Model Analysis

## Overview

We analyze the performance of SVM models on the project data. To speed up training times, start by setting up the experiments with only a fraction of the model training data. Once the code is tested, re-run all experiments with the full dataset.

## Linear SVM

### Model Training

1. **Linear Model Setup**:
   - We begin by using a linear model with K = 1.0 to reduce training time.
   - Train the model with different values of the regularization coefficient C, using logarithmically spaced values. Reasonable values for C are given by:
     ```python
     numpy.logspace(-5, 0, 11)
     ```

### Evaluation

1. **Validation**:
   - Score the validation samples and compute the **minimum DCF (minDCF)** and **actual DCF (actDCF)** for the primary application with target prior pi_T = 0.1.
   - Plot the two metrics as a function of C using a logarithmic scale for the x-axis:
     ```python
     import matplotlib.pyplot as plt
     plt.xscale('log', base=10)
     ```

### Observations

- How does the regularization coefficient C affect the two metrics? 
  - **Low values of C** imply strong regularization.
  - **Large values of C** imply weak regularization.
- Are the scores well calibrated for the target application?
- How does linear SVM perform compared to other linear models, such as logistic regression?
- Repeat the analysis with **centered data**:
  - Are the results significantly different with centered data?

## Polynomial Kernel SVM

### Model Setup

1. **Kernel Configuration**:
   - Consider a **polynomial kernel** with parameters:
     - d = 2
     - c = 1 (accounts for the bias term)
     - xi = 0 (bias term implicitly handled by the kernel)
   - Use the **original, non-centered features**.

2. **Training**:
   - Train the model with different values of C.

### Observations

- Compare **minDCF** and **actDCF** with the quadratic model.
- Are the results consistent with previous models (e.g., logistic regression and MVG) in terms of minDCF and actDCF?

## RBF Kernel SVM

### Model Setup

1. **Parameter Search**:
   - For the **RBF kernel**, we need to optimize both gamma (γ) and C.
   - Suggested values for gamma are {e^(-4), e^(-3), e^(-2), e^(-1)}.
   - For C, use log-spaced values:
     ```python
     numpy.logspace(-3, 2, 11)
     ```

### Grid Search

1. **Training**:
   - Train the models with all possible combinations of gamma and C.
   - Plot **minDCF** and **actDCF** as a function of C, with a different line for each value of gamma (i.e., four lines for minDCF and four lines for actDCF).

### Observations

- Analyze the results for each combination of gamma and C.
- Are there values of gamma and C that provide better results?
- Are the scores well calibrated?
- How do these results compare to previous models?
- Are there dataset characteristics that can be better captured by the RBF kernel?

## Optional: Higher-Degree Polynomial Kernel SVM

### Model Setup

1. **Polynomial Kernel**:
   - Consider a higher-degree **polynomial kernel** with parameters:
     - d = 4
     - c = 1
     - xi = 0
   - Train the model with different values of C using:
     ```python
     numpy.logspace(-5, 0, 11)
     ```

### Observations

- Compare **minDCF** and **actDCF** for this model.
- Are there better results compared to the quadratic model?

### Analysis:

1. **Feature Transformation**:
   - Focus on the last two features of each sample (i.e., `x_i` from indices 4 to 6).
   - Consider how these features would be transformed by a simple **degree 2 kernel** that maps each sample as follows:
     - Let `y_i` represent a feature vector, and the transformation is: `z_i = y_i[0:1] * y_i[1:2]`
   - Draw a few samples on paper and explore how **1-D linear rules** and **quadratic rules** would separate the transformed features.
   - **Linear rules** divide the space into two regions by a single threshold.
   - **Quadratic rules** involve inequalities that create intervals (e.g., features lie inside or outside a range).
   - Discuss how this transformation affects the classification rules for separating clusters.