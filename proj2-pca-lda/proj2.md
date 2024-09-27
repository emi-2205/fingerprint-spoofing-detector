# Project

This project applies Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) to a dataset. The aim is to analyze the effects of PCA on feature distributions and to evaluate the performance of LDA as a classifier. Below is a detailed analysis of the findings.

## Analysis of the Effect of PCA on Class Distributions

In this analysis, we transformed our original six features into six principal components (or directions) using PCA. The focus is on the impact of PCA on the class distributions of `Counterfeit` and `Genuine`.

---

### First Principal Component (Largest Variance)

The distributions of the `Counterfeit` and `Genuine` classes were initially centered around zero, with the `Counterfeit` class showing a denser, more pronounced peak compared to the wider spread of the `Genuine` class. The significant overlap between these distributions posed challenges for effective class separation.

After applying PCA, the distributions became more similar in shape and are now separated by their means: the `Counterfeit` class has shifted to the left, while the `Genuine` class has moved to the right. This transformation effectively reduced overlap, resulting in improved class separability and enhanced classification capabilities.

---

### Second Principal Component

Originally, each class formed a cluster around zero, with the `Genuine` class being denser and exhibiting a more distinct peak compared to the greater spread of the `Counterfeit` class. The significant overlap complicated the task of distinguishing between the classes.

Following PCA, the situation reversed: the `Genuine` class now displays a broader distribution with a less distinct peak, while the `Counterfeit` class has become denser with a sharper peak. However, this transformation has increased overlap, making classification based on this projected feature more challenging.

---

### Third and Fourth Principal Components

Initially, these components showed distinct class distributions with means around -1 for `Counterfeit` and 0.5 for `Genuine`, allowing for reasonable separation due to slight overlap near zero.

Post-PCA, the third and fourth principal components no longer maintain this separation. The class distributions have become nearly indistinguishable, complicating differentiation between `Counterfeit` and `Genuine`.

---

### Fifth and Sixth Principal Components (Least Variance)

Both components displayed noticeable overlap between classes, with the `Genuine` class exhibiting a bimodal distribution and the `Counterfeit` class concentrating around zero.

After PCA, the class distributions appear more symmetric and Gaussian, centered around zero. This suggests that PCA has centered and scaled the data. However, the overlap has significantly increased, making it more difficult to distinguish between the two classes using these principal components.

---

### Conclusion

The application of PCA has varied effects on class separability depending on the principal component analyzed. While the first principal component demonstrates clear improvements in separating the classes, subsequent components often result in increased overlap and reduced distinction. This indicates that while PCA captures the largest variance, not all principal components are equally useful for classification.

## Analysis of the Effect of LDA

Upon applying LDA with \( m = 1 \), we observed that the distributions of the two classes are much more similar, both resembling Gaussian distributions but shifted: `Counterfeit` to the left and `Genuine` to the right. The reduced overlap around zero allows for better class separation and improved classification performance.

## LDA as a Classifier

We applied LDA as a classifier on the dataset, and the following results were obtained for different scenarios:

1. **Initial LDA Results**:
   - **Threshold**: -0.0185
   - **Error Rate**: 9.3%

2. **Best Threshold Found**:
   - **Best Threshold**: -0.1071
   - **Lowest Error Rate**: 9.1%

Adjusting the threshold showed that specific values could enhance classification accuracy. The best threshold improved the error rate from 9.3% to 9.1%, indicating a slight performance improvement.

## Combining PCA and LDA

We evaluated the impact of preprocessing features with PCA before applying LDA. The following table summarizes our findings for different values of `m` (number of PCA dimensions):

| **PCA Dimensions (m)** | **Threshold**         | **Error Rate (%)** | **Best Threshold**     | **Lowest Error Rate (%)** |
|-------------------------|-----------------------|---------------------|-------------------------|---------------------------|
| 6                       | -0.0185               | 9.3                 | -0.1071                 | 9.1                       |
| 5                       | -0.0185               | 9.3                 | -0.0949                 | 9.05                      |
| 4                       | -0.0183               | 9.25                | 0.0141                  | 9.15                      |
| 3                       | -0.0184               | 9.25                | -0.0343                 | 9.15                      |
| 2                       | -0.0183               | 9.25                | 0.0061                  | 8.95                      |
| 1                       | -0.0176               | 9.35                | 0.0141                  | 8.85                      |

### Observations:

- **Threshold Impact**: Adjusting the threshold revealed that certain values significantly influenced classification accuracy. For example, at `m=2`, the lowest error rate of 8.95% was achieved with a best threshold of 0.0061. This indicates that the threshold choice significantly affects performance, with small changes leading to substantial improvements.

- **Trends with Standard Threshold vs. Best Threshold**: Using the standard threshold resulted in a stable error rate, primarily around 9.3% for the first two PCA dimensions. While exploring lower dimensions, error rates showed slight fluctuations but remained within a narrow band. In contrast, utilizing the best thresholds allowed for better optimization. For instance, at \( m=1 \), the standard threshold resulted in an error of 9.35%, while the best threshold lowered it to 8.85%. This highlights the trade-off between dimensionality reduction and classification performance.

- **Effectiveness of PCA**: The results indicate that reducing the number of PCA dimensions (\( m \)) facilitated the identification of values that improved validation set accuracy. Particularly, with \( m=2 \), the best threshold yielded a significant error reduction to 8.95%.

- **PCA's Contribution**: Overall, the combination of PCA and LDA proved advantageous, enhancing class separation and leading to lower error rates. This suggests that PCA assists not only in dimensionality reduction but also in highlighting the most discriminative features, which enhances LDA classification.

In conclusion, the findings underscore the importance of carefully adjusting the classification threshold to maximize accuracy. The analysis suggests that PCA can significantly improve classification performance when integrated with the LDA classifier, especially when optimal thresholds are employed for various dimensionality configurations.