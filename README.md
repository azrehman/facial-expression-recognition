# Facial Expression Recognition

### CS 4664 - Capstone Project - Facial Expression Recognition

To accomplish our goal of facial expression recognition (FER), we apply ResNet-50, a pretrained deep-CNN, on two different datasets. In addition to classifying facial expressions, we explore many different aspects of computer vision by experimenting with our final trained classifier. We also intend to create different visualizations for our model to increase insight and understanding. This git repository has the code to the various investigative experiments that we conducted, as well as our final trained classifier.

## How to Run Our Code
(insert)

## Investigative Experiments
- Experimenting Different Model Sizes
- Experimenting Different Dataset Sizes
- Support Vector Machine Classifier vs Final Trained Classifier
- Exploring Bias in Our Model
- Adversarial Attack 
- Introducing Noise
- t-SNE Feature Visualizations
- Saliency Map using Guided Back Propagation
- Bonus: Image to Image Emotion Transfer

## Results

### Experimenting Different Model Sizes
### Experimenting Different Dataset Sizes
### Support Vector Machine Classifier vs Final Trained Classifier
**Motivation** Investigate how a simple SVM model (our non-deep-learning baseline) performs compared to our fine tuned CNN classifier.
We want to show that a simple ML model will not be as accurate for our task. An SVM model will not be as comparable for the task of accurately classifying emotions.  

**Steps of SVM Experiment**
1. Load and preprocess all dataset images.
2. Split the dataset into train and test sets.
3. Define the hyperparameter space to explore. 
4. Perform gridsearch using 10-fold cross validation.
5. Record best parameters found from gridsearch.
6. Evaluate fine-tuned SVM classifier on test set.
7. Record evaluation metrics (accuracy, F1 Score, confusion matrix).

_Hyperparameter Space:_
![](https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/svm-hyperparameter-space.png)

**Conclusion**
Our **SVM model obtained an accuracy of 72%**, whereas our **fine-tuned CNN model obtained an accuracy of 95%**. Thus, our SVM model obtained lower accuracy when compared to our CNN model. This can be attributed to the idea that deep learning models perform better for classification problems. However, if we discount the time it took to do gridsearch, the SVM classifier was faster to train. 

![](https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/SVM_Experiment_-_Image_2_-_CS_4664.png)

Upon performing gridsearch with cross validation, we obtained the following best hyperparameters, C=100, Gamma=auto, Kernel=rbf. Now, if we take a look at this bar graph, you can see that prior to tunning our SVM model we were obtaining quite low accuracy, and performing gridsearch with tuning made the SVM Model perform better. However, our fine tuned classifier obtained much better accuracy compared to the other models. 

Code for Experiment: 

### Exploring Bias in Our Model
### Adversarial Attack 
### Introducing Noise
**Motivation** Noisy images are actually more representative of real world data, which are normally not uniform and often contain many confounding details. Thus, our goal for this experiment was to evaluate our model’s performance on test images containing varying levels of noise.

This was achieved by applying Gaussian Noise with different levels of variance on our test set. We predict that if our model is robust,then performance should not decrease, unless a really large amount of noise is applied to our test set.

| Variance = 0.01 | Variance = 0.2 |
| ------ | ------ |
| <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/noise-experiment-1.png"  width="500" height="520"> | <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/noise-experiment-2.png"  width="500" height="520"> |


**Steps of SVM Experiment**
1. Apply Gaussian noise to test images.
2. Load model trained without noise.
3. Evaluate model performance on the noisy test set.
4. Repeat experiment steps 1-3 with different levels of variance (to change amount of noise).
5. Compare accuracies to model performance on test set without noise.

**Conclusion** 

| Variance | Test Accuracy |
| ------ | ------ |
| 0 (*original*) | 0.96 |
| 0.01 | 0.87 |
| 0.05 | 0.74 |
| 0.07 | 0.7 |
| 0.1 | 0.64 |
| 0.15 | 0.58 |
| 0.2 | 0.47 |

<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/noise-experiment-line-graph.png"  width="750" height="520">

Unfortunately **our model was not as resilient to noise** as much as we hoped. You can see that our model's accuracy decreases when we increase the amount of noise applied. This implies that our model would not perform very well on real world data except in the most ideal circumstances. In order to address this, there are multiple techniques we could apply. An obvious option is to retrain our model with a small random amount of noise added to our training images as a data augmentation. By training with noisy images, our model should be more agnostic to confounding details and perform better on real world images. Another option is to limit overfitting in our model using techniques such as _dropout, early stopping, and loss regularization_.

Code for Experiment: 

### Noise Experiment - Statistical Significance Study
We also performed a Statistical Significance Study on the noise experiment for a variance level of 0.1. We simply applied noise to our test set, and evaluated our model on the noisy test set 10 times, and plotted our test accuracies using a box plot. To summarize our findings, the median accuracy of our ten runs was 0.67, the minimum accuracy was 0.65, the maximum was 0.680, and that we have no outliers.

<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/noise-statistical-sig.png" width="520" height="450">
</div>

### t-SNE Feature Visualizations
### Saliency Maps 
**Using Vanilla Back Propagation**

**Using Guided Back Propagation**
**Motivation**



### Bonus: Image to Image Emotion Transfer






