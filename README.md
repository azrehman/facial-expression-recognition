# Facial Expression Recognition

### CS 4664 - Capstone Project - Facial Expression Recognition

To accomplish our goal of facial expression recognition (FER), we apply ResNet-50, a pretrained deep-CNN, on two different datasets. In addition to classifying facial expressions, we explore many different aspects of computer vision by experimenting with our final trained classifier. We also intend to create different visualizations for our model to increase insight and understanding. This git repository has the code to the various investigative experiments that we conducted, as well as our final trained classifier.

## How to Run Our Code
Abdullah

## Transfer Learning
Deepti

## Our Chosen Classifier Architecture 
Abdullah

## Baseline Model vs Our Classifier
Abdullah

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

## Experiments
### Experimenting Different Model Sizes
### Experimenting Different Dataset Sizes
**Motivation** Perform a sensitivity analysis to quantify the relationship between dataset size and model performance. We want to take fractions of the orginial dataset and observe how the model's ability to classify accurately changes. 

**Steps of Dataset Size Experiment**
1. Create a smaller training set from the original dataset.
2. Retrain main classifier under same settings for each dataset subset.
3. Record the test accuracy on the subset of the orginial dataset.
4. Repeat steps 1-3 for different dataset sizes.

**Conclusion**

| % of Dataset | 80% (original) |   70%   |   50%   |   20%   |   10%   | 
| ------ | ------ | ------ | ------ | ------ | ------ |
| *No. of Images* | 3920 | 3430 | 2450 | 980 | 490 |
| *Accuracy* | 0.96 | 0.94 | 0.94 | 0.89 | 0.83 |

<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/dataset_size_graph.png" width="550" height="400">

The results that we found were quite impressive given that the amount of data the model was trained with was significantly less. The graph above shows the downward trend in accuracy the dataset size used for training decreases. There is roughly a 10% decrease in accuracy from using 80% of the dataset to only 10%. We attribute this decent accuracy to the fact that we used transfer learning instead of training the model from scratch. Additionally, since data augmentations were applied, it artificially increased the number of training samples which could have contributed to the higher accuracy rates. 

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
### Introducing Noise
**Motivation** Noisy images are actually more representative of real world data, which are normally not uniform and often contain many confounding details. Thus, our goal for this experiment was to evaluate our model’s performance on test images containing varying levels of noise.

This was achieved by applying Gaussian Noise with different levels of variance on our test set. We predict that if our model is robust,then performance should not decrease, unless a really large amount of noise is applied to our test set.

| Variance = 0.01 | Variance = 0.2 |
| ------ | ------ |
| <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/noise-experiment-1.png"  width="500" height="520"> | <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/noise-experiment-2.png"  width="500" height="520"> |


**Steps of Noise Experiment**
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

### Adversarial Attack 
**Motivation** Adversarial machine learning, a technique that attempts to fool models with deceptive data, is a growing threat in the AI and machine learning research community. Therefore, to test our model's robustness, we used Fast Gradient Signed Method (FGSM). FGSM is a white-box attack as it leverages an internal component of the architecture which is its gradients. 
<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/adversarial_formula.png" width="1000" height="100">
</div>
The implementation follows the formula as seen above, where it takes the original image and then adjusts each pixel of the input image based on the gradient and a factor which is the pixel-wise perturbation amount known as epsilon. As per theory, as the epsilon value increases, the image becomes more perturbed causing the model to become more inaccurate

We used increased epsilon values to create more perturbed images and tested our model on these adversarial images to observe how well it could classify the images.

<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/faces_epsilon_values.png" width="520" height="430">
</div>

**Steps of Adversarial Attack**
1. Load the trained model.
2. Predict the label on the original test image.
3. Create perturbed image (for the current epsilon value) from original test image.
4. Let model predict on that perturbed image 
5. Record number of times model misclassified with perturbed images
6. Repeat steps 3-5 for different epsilon values.

**Conclusion** 

| Epsilon |0 (original)| 0.001 |  0.005 | 0.007 | 0.01 | 0.05 | 0.07 | 0.1 |
| ------ | ------ |
| Test Accuracy | 0.96 | 0.82 | 0.61 | 0.50 | 0.38 | 0.057 | 0.053 | 0.047|


<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/epsilon_graph.png" width="500" height="400">

The results show that even a small epsilon value can have quite a drastic impact on the performance of the model. An epsilon value of 0 gives us our original accuracy of 96% but an epsilon of 0.1 drops the accuracy significantly to 4.7%.

<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/epsilon_0.png" width="900" height="250">
</div>
<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/epsilon_0.01.png" width="900" height="250">
</div>

The figure above shows that with an epsilon of 0 the model is very confident and there are no incorrect predictions but an epsilon of 0.01 the model is very confident but on the wrong expression label since it believes that a happy image is actually afraid. Although, the epsilon value is very small and the perturbed image looks untampered, the accuracy is much lower.

<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/epsilon_0.3.png" width="900" height="300">
</div>

Moreover, When the epsilon value is too high, such as 0.3, it acts similarly to adding noise to the image.  This is because the model still misclassifies the image but it is no longer confident in one label and is more confused which causes it to have some confidence in numerous labels.

Overall, there is an inverse relationship where as the epsilon value increases, the test accuracy of our model decreases. This investigation proves to show that our model might not be the most robust against such white box attacks but it is particularly hard to make a model defend itself against
these types of attacks since the attacker has access to the model parameters. 

### t-SNE Feature Visualizations
### Saliency Maps 

**Motivation** One of the best ways to interpret and visualize the CNN model is through saliency maps. Saliency maps are a way to measure the spatial support of a particular class in each image. It is a visualization technique to gain better insight into the decision-making of the CNN and
helps to highlight what each layer of a convolutional layer focuses on.

**Using Vanilla Backpropagation**
The Vanilla Backpropagation technique creates a saliency map by forward passing the data and then passing the data backward to the input layer to get the gradient and then rendering the gadient as a normalized heatmap. 

**Steps for Vanilla Backprogation Saliency Map**
1. Apply custom transformations (pre-processing) on image
2. Retrieve model's output after passing image
3. Do backpropagation to get the derivative of the output based on the image
4. Retireve the saliency map and also pick the maximum value from channels on each pixel.
5. Plot saliency map

<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/vanilla_saliency_map.png" width="600" height="350">
</div>

The saliency map resulting from using the Vanilla Backpropagation approach shows a very noisy image since background features pass through the ReLu activation functions which causes it to be unclear. However, we can see that the brighter red spots are focused around the eyes and mouth area which provide us with an indication of what the model focuses on when trying to classify the image.

**Using Guided Backpropagation**
Guided Backpropagation combines the previously used Vanilla Backpropagation technique at ReLUs with DeconvNets. Guided backpropagation visualizes gradients with respect to the image where negative gradients are suppressed when backpropagating through ReLU layers. Essentially, this methodology aims to capture pixels detected by neurons, not the ones that suppress neurons. 

**Steps for Guided Backpropagation**

(STEPS)

**Conclusion**
|Guided Backpropagation Saliency|Colored Guided Backpropagation|Guided Backpropagation Negative Saliency|Guided Backpropagation Positive Saliency|
| ------ | ------ | ------ | ------ |
| <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/AF01HAHR_Guided_BP_gray.png"> | <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/AF01HAHR_Guided_BP_color.png"> | <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/AF01HAHR_neg_sal.png"> | <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/AF01HAHR_pos_sal.png"> |

Based on the above results we can see that our fine-tuned classifier tends to focus on specific regions of the human face to help correctly classify the human emotion. Our model mainly focuses on the center region of the face (the eyes, eyebrows, and mouth area), since the pixels seem to be the most highlighted in those parts in the Guided Backpropagation Saliency map. In addition to this, upon analyzing the Colored Guided Backpropagation map, the main features that the model is focusing on further well defined and outlined.  




### Bonus: Image to Image Emotion Transfer






