# Project

Apply PCA and LDA to the project data. Start by analyzing the effects of PCA on the features. 

- Plot the histogram of the projected features for the 6 PCA directions, starting from the principal (largest variance). 
  - What do you observe? 
  - What are the effects on the class distributions? 
  - Can you spot the different clusters inside each class?

Apply LDA (1-dimensional, since we have just two classes), and compute the histogram of the projected LDA samples. 
- What do you observe? 
- Do the classes overlap? 
- Compared to the histogram of the 6 features you computed in Laboratory 2, is LDA finding a good direction with little class overlap?

### LDA as a Classifier

- Try applying LDA as a classifier. 
  - Divide the dataset into model training and validation sets (you can reuse the previous function to split the dataset).
  - Apply LDA, and select the threshold as in the previous sections.
  - Compute the predictions and the error rate.

- Now try changing the value of the threshold. 
  - What do you observe? 
  - Can you find values that improve the classification accuracy?

### Combining PCA and LDA

- Finally, try pre-processing the features with PCA. 
  - Apply PCA (estimated on the model training data only), and then classify the validation data with LDA.
  - Analyze the performance as a function of the number of PCA dimensions `m`. 
    - What do you observe? 
    - Can you find values of `m` that improve the accuracy on the validation set? 
    - Is PCA beneficial for the task when combined with the LDA classifier?