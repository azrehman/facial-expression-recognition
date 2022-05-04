# Facial Expression Recognition

### CS 4664 - Capstone Project - Facial Expression Recognition

To accomplish our goal of facial expression recognition (FER), we apply ResNet-50, a pretrained deep-CNN, on two different datasets. In addition to classifying facial expressions, we explore many different aspects of computer vision by experimenting with our final trained classifier. We also intend to create different visualizations for our model to increase insight and understanding. This git repository has the code to the various investigative experiments that we conducted, as well as our final trained classifier.

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

Hyperparameter Space:

**Conclusion**
Our SVM model obtained an accuracy of 72%, whereas our fine-tuned CNN model obtained an accuracy of 95%. Thus, our SVM model obtained lower accuracy when compared to our CNN model. This can be attributed to the idea that deep learning models perform better for classification problems. However, if we discount the time it took to do gridsearch, the SVM classifier was faster to train. 

Upon performing gridsearch with cross validation, we obtained the following best hyperparameters, C=100, Gamma=auto, Kernel=rbf. Now, if we take a look at this bar graph, you can see that prior to tunning our SVM model we were obtaining quite low accuracy, and performing gridsearch with tuning made the SVM Model perform better. However, our fine tuned classifier obtained much better accuracy compared to the other models. 



### Exploring Bias in Our Model
### Adversarial Attack 
### Introducing Noise
### t-SNE Feature Visualizations
### Saliency Map using Guided Back Propagation
### Bonus: Image to Image Emotion Transfer






