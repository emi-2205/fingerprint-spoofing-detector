# Fingerprint Spoofing Detection Project

## Project Overview

The project task consists of a binary classification problem aimed at performing fingerprint spoofing detection, specifically identifying genuine versus counterfeit fingerprint images. The dataset consists of labeled samples corresponding to the genuine (True, label 1) class and the fake (False, label 0) class. Samples are computed by a feature extractor summarizing high-level characteristics of fingerprint images. The data is 6-dimensional.

The training files for the project are stored in `Project/trainData.txt`. The format of the file is similar to that of the Iris dataset, being a CSV file where each row represents a sample. The first 6 values of each row are the features, while the last value represents the class (1 or 0). The samples are not ordered.

## Task

Load the dataset and plot the histogram and pair-wise scatter plots of the different features. Analyze the plots.

### Data Summary

- **Means**:
  - Feature 1: 
    - Class 0: `0.00287744`
    - Class 1: `0.00054455`
  - Feature 2:
    - Class 0: `0.01869316`
    - Class 1: `-0.00852437`
  - Feature 3:
    - Class 0: `-0.68094016`
    - Class 1: `0.66523785`
  - Feature 4:
    - Class 0: `0.6708362`
    - Class 1: `-0.66419535`
  - Feature 5:
    - Class 0: `0.02795697`
    - Class 1: `-0.04172519`
  - Feature 6:
    - Class 0: `-0.0058274`
    - Class 1: `0.02393849`

- **Variances**:
  - Class 0: `[0.56958105, 1.42086571, 0.54997702, 0.53604266, 0.6800736, 0.70503844]`
  - Class 1: `[1.43023345, 0.57827792, 0.5489026, 0.55334275, 1.31776792, 1.28702609]`

- **Standard Deviations**:
  - Class 0: `[0.75470594, 1.19200072, 0.74160436, 0.73214934, 0.82466575, 0.83966567]`
  - Class 1: `[1.19592368, 0.76044587, 0.74087962, 0.74387012, 1.14794073, 1.13447172]`

Overall, we have a good picture of our data.

## Analysis

### 1. First Two Features

#### i. What do you observe?

- **Feature 1**:
  - Class 0 and class 1 values are concentrated around their means.
  - Mean for class 0: `0.00287744`, Mean for class 1: `0.00054455`.
  - Close proximity of means leads to significant overlap in the range of `[-2, 2]`.
  - Class 0 has smaller variance, leading to a sharper peak compared to class 1.

- **Feature 2**:
  - Means are also close to zero: class 0: `0.01869316`, class 1: `-0.00852437`.
  - Class 1 has a smaller variance, resulting in a higher peak, while class 0 has a broader distribution.
  - Scatter plot shows greater variance along feature 2 for class 0 and along feature 1 for class 1.

#### ii. Do the classes overlap? If so, where?

Yes, significant overlap occurs primarily in the range of values between `[-2, 2]` for both features.

#### iii. Do the classes show similar means for the first two features?

Yes, the means for both classes are very similar, with minimal differences observed.

#### iv. Are the variances similar for the two classes?

No, the variances differ. Class 0 has a smaller variance in feature 1, while class 1 has a smaller variance in feature 2.

#### v. How many modes are evident from the histograms?

Each feature histogram shows one peak per class, leading to two peaks in total per feature.

---

### 2. Third and Fourth Features

#### i. What do you observe?

- **Feature 3**:
  - Means are more separated: class 0: `-0.68094016`, class 1: `0.66523785`.
  - Low variance for both classes leads to pronounced peaks in histograms.

- **Feature 4**:
  - Similar situation with reversed means: class 0: `0.6708362`, class 1: `-0.66419535`.
  - Low variances yield well-defined peaks in histograms.

- Scatter plot shows two distinct clusters corresponding to the two classes.

#### ii. Do the classes overlap? If so, where?

Some overlap exists primarily in the range of `[-2, 2]`, but the separation between means reduces the extent of the overlap.

#### iii. Do the classes show similar means for the third and fourth features?

No, the means for the two classes are distinct and separated for both features.

#### iv. Are the variances similar for the two classes?

Yes, the variances for both classes are similar in features 3 and 4.

#### v. How many modes are evident from the histograms?

Each histogram shows two peaks (one for each class).

---

### 3. Last Two Features

#### i. What do you observe?

- **Feature 5**:
  - Class 0 has lower variance with a sharp peak; class 1 shows higher variance and two distinct peaks.

- **Feature 6**:
  - Similar pattern to feature 5; class 0 has a tight cluster, while class 1 has a wider spread with multiple peaks.

- Scatter plot indicates multiple clusters, with four clusters for both classes.

#### ii. Do the classes overlap? If so, where?

Yes, there is overlap due to the close means of both classes, but distinct peaks in class 1 reduce the extent of overlap.

#### iii. How many modes are evident from the histograms?

Feature 5 has three modes (one for class 0, two for class 1). The same applies for feature 6.

#### iv. How many clusters can you notice from the scatter plots for each class?

Four distinct clusters are observed for each class in the scatter plot. Importantly, these clusters are not strictly aligned along the axes, as the values for feature 5 being zero do not necessarily correspond to feature 6 also being zero. Therefore, high concentrations around `(0, 0)` in each feature do not imply that the same samples have both features equal to zero.

---

## Conclusion

This analysis provides insights into the structure and distribution of the data in the fingerprint spoofing detection task. The differences in means, variances, and the presence of clusters across the features will be critical for developing an effective classification model.