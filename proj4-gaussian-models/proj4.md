# Project: MVG Model and Feature Analysis

## Applying the MVG Model

Apply the MVG model to the project data. 

- Split the dataset into model training and validation subsets (important: use the same splits for all models, including those from other laboratories).
- Train the model parameters on the training portion of the dataset.
- Compute the **Log-Likelihood Ratios (LLRs)**:

  s(xt) = llr(xt) = fX|C(xt|1) / fX|C(xt|0)

  Where class **True** (label 1) is on top of the ratio.

- For the validation subset:
  - Obtain predictions from LLRs, assuming uniform class priors P(C=1) = P(C=0) = 1/2.
  - Compute the corresponding error rate.

**Note**: In upcoming labs, we will modify how we compute predictions from LLRs, so it's recommended to separate the functions that compute LLRs, those that compute predictions from LLRs, and those that compute the error rate from predictions.

## Tied Gaussian Model and Comparisons

- Apply the **tied Gaussian model**.
- Compare the results with those from the **MVG** and **LDA** models.
  - Which model performs better?

## Naive Bayes Gaussian Model

- Test the **Naive Bayes Gaussian model**.
- Compare its performance with the MVG and tied Gaussian models.

## Feature Covariance and Correlation Analysis

- Print the covariance matrix of each class (extracted from the MVG model parameters).
  - The diagonal contains the variances for the features, and the off-diagonal elements contain the feature covariances.
  
- Compare the covariances of different feature pairs with their respective variances.
  - What do you observe? 
  - Are the covariance values large or small compared to the variances?

- To better visualize covariance strength relative to variances, compute the **Pearson correlation coefficient** for a pair of features i, j:

  Corr(i, j) = Cov(i, j) / (sqrt(Var(i)) * sqrt(Var(j)))

  In matrix form:

  Corr = C / (vcol(C.diagonal()**0.5) * vrow(C.diagonal()**0.5))

  Where C is the covariance matrix. Diagonal elements will be 1, while off-diagonal elements will be the correlation coefficients for feature pairs, with -1 ≤ Corr(i, j) ≤ 1.

- Compute the correlation matrices for the two classes.
  - What can you conclude about the features? 
  - Are the features strongly or weakly correlated? 
  - How does this relate to the Naive Bayes results?

## Evaluating the Gaussian Model Assumptions

The Gaussian model assumes that features can be jointly modeled by Gaussian distributions. The model's accuracy depends on this assumption.

- Although visualizing 6-dimensional distributions is unfeasible, analyze how well the assumption holds for single or pairs of features.
- In **Laboratory 4**, we fitted Gaussian densities over each feature for each class (corresponding to the Naive Bayes model).
  - What can you conclude about the goodness of the Gaussian assumption? 
  - Is it accurate for all 6 features? 
  - Are there features for which the assumption does not hold?

## Feature Selection

To assess whether the last set of features negatively affects the classifier due to poor modeling assumptions:
- Try repeating the classification using only **features 1 to 4** (i.e., discarding the last 2 features).
- Repeat the analysis for the three models.
  - What do you obtain?
  - What can you conclude from discarding the last two features?
  - Despite the poor assumptions for these features, do the Gaussian models still extract useful information to improve classification accuracy?

## Feature Distribution Characteristics

In **Laboratory 2** and **Laboratory 4**, we analyzed the distribution of **features 1-2** and **features 3-4**:

- For **features 1-2**: The means are similar, but variances are not.
- For **features 3-4**: The two classes differ mainly in mean, but show similar variances.
- Additionally, both sets of features exhibit limited correlation in both classes.

### Model Comparison by Feature Pairs

- Repeat the classification using **features 1-2** (jointly), then using **features 3-4** (jointly). Compare the results of the **MVG** and **tied MVG** models.
  - In the first case, which model is better?
  - And in the second case?
  - How is this related to the characteristics of the two classifiers?
  - Is the tied model effective for the first two features? Why? 
  - How does the MVG model perform in both cases?

## PCA as Pre-Processing

- Finally, analyze the effects of **PCA** as pre-processing.
  - Use PCA to reduce the dimensionality of the feature space, and apply the three classification approaches.
  - What do you observe? 
  - Is PCA effective for this dataset with the Gaussian models?
  
### Overall Conclusion

- What model provided the best accuracy on the validation set?