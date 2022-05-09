# Facial Expression Recognition

### CS 4664 - Capstone Project - Facial Expression Recognition

To accomplish our goal of facial expression recognition (FER), we apply ResNet-50, a pretrained deep-CNN, on two different datasets. In addition to classifying facial expressions, we explore many different aspects of computer vision by experimenting with our final trained classifier. We also intend to create different visualizations for our model to increase insight and understanding. This git repository has the code to the various investigative experiments that we conducted, as well as our final trained classifier.

## How to Run Our Code

1. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (python enviroment manager)
2. Clone this repository\
`git clone https://git.cs.vt.edu/sdeepti/facial-expression-recognition.git`
3. Create and activate conda enviroment\
`conda create --clone base --name face_env`\
`conda activate face_env`
4. Install [pytorch](https://pytorch.org/get-started/locally/)\
`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
4. Run our code :)\
`python main_resnet50/train_classifier.py`\
(note you may need to modify image_dir path to `../data/<dataset name>`)

## Datasets We Used
We chose datasets that ensure a sufficient amount of images for each label (e.g. approximately at least 1000 images for each expression) and ensure that our dataset contains clear visibility of the face and the images are of good quality to increase accuracy of our model. We also wanted to select a second dataset with data that is evenly distributed across races and genders so that we can investigate F.E.R model bias for emotions across these demographics and make our model more universal. 

Our primary dataset is the [Karolinska Directed Emotional Faces (KDEF)](https://www.kdef.se/) dataset which is a publicly available dataset consisting of 4900 facial expression images. It contains 70 individuals, 45 women and 45 men, each displaying 7 different emotional expressions, each expression being photographed (twice) from 5 different angles. We will be using all the images within KDEF. 

The second dataset we used was [The RADIATE Emotional Face Stimulus Set](http://fablab.yale.edu/page/assays-tools) which is another public available dataset that meets the requirement. The RADIATE face stimulus set contains over 1,700 unique photographs of over 100 racially and ethnically diverse models (25\% non-Hispanic White and 75\% minority or ethnic group). Each model posed 16 different facial expressions. 

## Transfer Learning
Transfer learning is a machine learning technique in which a pre-trained model is being repurposed for a similar task of interest. Applying transfer learning to our project was seen to be advantageous as it would reduce the complexity of our task and thereby increase efficiency. 
The image below displays a schematic architecture of the pre-trained ResNet Model plus the added Dense layers for facial expression recognition. This pre-trained model has been trained with ImageNet which is a large dataset for the purposes of image classification which is suitable for FER. Repurposing a pre-trained deep CNN involves two steps, replacing the original classifier with a new one and fine-tuning the model. The added classifier is achieved by combining some number of dense layers, and this gives us the opportunity to fine-tune the dense layers and a selected few layers of the pre-trained model with our data, and this is possible because they are all in the same pipeline.
Post training, we then feed our image into the model and the output of the final dense layer would be of length 7 each referring to a specific expression and their respective probabilities.

<figure>
<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/transfer_learning.png" width="850" height="400">
</div>
<figcaption align = "center"><b>Fig 1. Transfer Learning</b></figcaption>
</figure>

## Preprocessing Steps
The preprocessing steps we implemented were to add rotation, horizontal flips, brightness, contrast and saturation modifications to expand the scope of the images we have in our dataset with the goal of improving our model’s performance. A sample of the images created from these steps can be seen below in Fig 2. We then applied data normalization according to the ImageNet dataset standards and image resizing for faster training. Although grayscaling was a plausible preprocessing step that could be used to reduce noise, we did no proceed with this step as the pre-trained model we chose was trained on color images.

<figure>
<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/preprocessing.png" width="820" height="200">
</div>
<figcaption align = "center"><b>Fig 2. Preprocessing Steps</b></figcaption>
</figure>


[Link to Preprocessing Code.](https://gitlab.cs.vt.edu/sdeepti/facial-expression-recognition/-/blob/main/main_resnet50/preprocessing_visualization.ipynb)

## Our Chosen Classifier Architecture 

ResNet is the abbreviated name for a Residual Network. This deep CNN architecture can contain a numerous number of convolution layers ranging from 18 to 152. When performing our literature review, we saw that ResNet50 was quite popular and had higher accuracy rates compared to other models like VGG-16 and Inception-v3 etc. Fig 3. shows that ResNet 50 specifically has 48 convolution layers, 1 max pool layer and 1 average pool layer.
Another reason we chose ResNet was because we learned that the ResNet architecture overcame the “vanishing gradient” problem. A vanishing gradient occurs during backpropagation. When the training algorithm tries to find weights that bring the loss function to a minimal value, if there are too many layers, the gradient becomes very small until it disappears, and optimization cannot continue. Since, ResNet was capable of overcoming this challenge, it can be built with multiple layers and therefore outperform shallower networks.

<figure>
<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/resnet_arch.png" width="820" height="200">
</div>
<figcaption align = "center"><b>Fig 3. ResNet50 Architecture</b></figcaption>
</figure>

## Baseline Model vs Our Classifier

We selected the paper [_Facial Emotion Recognition Using Transfer Learning in the Deep CNN_](https://mdpi-res.com/d_attachment/electronics/electronics-10-01036/article_deploy/electronics-10-01036-v2.pdf?version=1619665750) as our baseline for the project.\
The baseline paper achieves **97.5%** accuracy on the KDEF dataset using transfer learning with ResNet-50.\
In contrast, we achieved **95.7%** accuracy with our model. The slight decrease in our model accuracy is likely because some hyperparameters the baseline optimized were not discussed in the paper.


[Code for Main Classifier](https://gitlab.cs.vt.edu/sdeepti/facial-expression-recognition/-/blob/main/main_resnet50/main_resnet50.ipynb)

## Investigative Experiments
- [Experimenting Different Model Sizes](https://git.cs.vt.edu/sdeepti/facial-expression-recognition/#experimenting-different-model-sizes)
- [Experimenting Different Dataset Sizes](https://gitlab.cs.vt.edu/sdeepti/facial-expression-recognition#experimenting-different-dataset-sizes)
- [Support Vector Machine Classifier Comparison](https://gitlab.cs.vt.edu/sdeepti/facial-expression-recognition#support-vector-machine-classifier-vs-final-trained-classifier)
- [Exploring Bias in Our Model](https://gitlab.cs.vt.edu/sdeepti/facial-expression-recognition#exploring-bias-in-our-model)
- [Introducing Noise](https://gitlab.cs.vt.edu/sdeepti/facial-expression-recognition#introducing-noise)
- [Adversarial Attack](https://gitlab.cs.vt.edu/sdeepti/facial-expression-recognition#adversarial-attack) 
- [t-SNE Feature Visualizations](https://gitlab.cs.vt.edu/sdeepti/facial-expression-recognition#t-sne-feature-visualizations)
- [Saliency Map Visualization using Guided Back Propagation](https://gitlab.cs.vt.edu/sdeepti/facial-expression-recognition#saliency-maps)
- [Bonus: Image to Image Emotion Transfer](https://gitlab.cs.vt.edu/sdeepti/facial-expression-recognition#bonus-image-to-image-emotion-transfer)

### Experimenting Different Model Sizes
**Motivation** Investigate shallower model architecture sizes trained from scratch performance compared to ResNet-50 transfer learning model.

**Steps of Model Size Experiment**
1. Build three smaller CNN models with different number of convolutional layers
2. Train each model under same settings until loss stops decreasing (~100 epochs)
3. Evaluate trained models on test set and compare accuracies

**Conclusion**

| Model | ResNet50 (48 Layers) |   CNN (6 Layers)   |   CNN (2 Layers)   |   CNN (1 Layer)|
| ------ | ------ | ------ | ------ | ------ | 
| *No. of Trainable Parameters* | 161 | 16 | 8 | 6 |
| *Accuracy* | 0.96 | 0.80 | 0.82 | 0.77 | 

<b>Table X. Model accuracy on varying Architecture sizes </b>

<figure>
<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/last-model-viz.png" width="550" height="400">
</div>
<figcaption align = "center"><b>Fig X. Number of CNN layers vs Accuracy</b></figcaption>
</figure>

According to our findings, the accuracy of all three models was significantly lower than that of our main ResNet50 classifier. We can see that raising the number of layers increases the amount of trainable parameters, which causes the accuracy rate to improve. This is expected because the CNN models took longer to converge because they were trained from scratch.

[Code for Model Size Experiment](https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/blob/main/model_size_experiment/Model_Size_Notebook.ipynb)

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

<b> Table X. Model accuracies against varying dataset sizes </b>
<figure>
<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/dataset_size_graph.png" width="550" height="400">
</div>
<figcaption align = "center"><b>Fig X. Graph of model accuracy vs dataset size</b></figcaption>
</figure>

The results that we found were quite impressive given that the amount of data the model was trained with was significantly less. The graph above shows the upward trend in accuracy the dataset size used for training increases. There is roughly a 10% decrease in accuracy from using 80% of the dataset to only 10%. We attribute this decent accuracy to the fact that we used transfer learning instead of training the model from scratch. Additionally, since data augmentations were applied, it artificially increased the number of training samples which could have contributed to the higher accuracy rates. 

[Code for Dataset Size Experiment](https://gitlab.cs.vt.edu/sdeepti/facial-expression-recognition/-/blob/main/dataset_size_experiment/dataset_size.ipynb)

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
```python
# parameters for gridsearchCV
parameter_space = [
    {
        'C': [1, 10, 100],
        'kernel': ['rbf', 'poly'],
        'gamma': ['scale', 'auto']
    },
    {
        'C': [1, 10, 100],
        'kernel': ['linear']
    }
]
```


**Conclusion**
Our **SVM model obtained an accuracy of 72%**, whereas our **fine-tuned CNN model obtained an accuracy of 95%**. Thus, our SVM model obtained lower accuracy when compared to our CNN model. This can be attributed to the idea that deep learning models perform better for classification problems. However, if we discount the time it took to do gridsearch, the SVM classifier was faster to train. 

<figure align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/SVM_Experiment_-_Image_2_-_CS_4664.png" width="500" height="300">
<figcaption align = "center"><b>Fig X. A graph displaying the trend of model accuracy
across different models (Deep CNN, Tuned SVM, Untuned SVM)</b></figcaption>
</figure>

Upon performing gridsearch with cross validation, we obtained the following best hyperparameters, C=100, Gamma=auto, Kernel=rbf. Now, if we take a look at this bar graph, you can see that prior to tunning our SVM model we were obtaining quite low accuracy, and performing gridsearch with tuning made the SVM Model perform better. However, our fine tuned classifier obtained much better accuracy compared to the other models. 
 
[Code for SVM Experiment](https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/blob/main/svm_experiment/svm_experiment.ipynb)


### Exploring Bias in Our Model
### Introducing Noise
**Motivation** Noisy images are actually more representative of real world data, which are normally not uniform and often contain many confounding details. Thus, our goal for this experiment was to evaluate our model’s performance on test images containing varying levels of noise.

This was achieved by applying Gaussian Noise with different levels of variance on our test set. We predict that if our model is robust,then performance should not decrease, unless a really large amount of noise is applied to our test set.

| Variance = 0.01 | Variance = 0.2 |
| ------ | ------ |
| <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/noise-experiment-1.png"  width="300" height="320"> | <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/noise-experiment-2.png"  width="300" height="320"> |

<b>Table X. Model accuracies against test set with varying values of variance</b>

**Steps of Noise Experiment**
1. Apply Gaussian noise to test images.
2. Load model trained without noise.
3. Evaluate model performance on the noisy test set.
4. Repeat experiment steps 1-3 with different levels of variance (to change amount of noise).
5. Compare accuracies to model performance on test set without noise.

**Conclusion** 

<figure>
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/noise_conclusion.png">
<figcaption align = "center"><b>Fig X. A graph displaying the trend of model accuracy
for varying levels of noise added to the test set</b></figcaption>
</figure>

Unfortunately **our model was not as resilient to noise** as much as we hoped. You can see that our model's accuracy decreases when we increase the amount of noise applied. This implies that our model would not perform very well on real world data except in the most ideal circumstances. In order to address this, there are multiple techniques we could apply. An obvious option is to retrain our model with a small random amount of noise added to our training images as a data augmentation. By training with noisy images, our model should be more agnostic to confounding details and perform better on real world images. Another option is to limit overfitting in our model using techniques such as _dropout, early stopping, and loss regularization_.

[Code for Noise Experiment](https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/blob/main/noise_experiment/noise_experiment.ipynb)

### Noise Experiment - Statistical Significance Study
We also performed a Statistical Significance Study on the noise experiment for a variance level of 0.1. We simply applied noise to our test set, and evaluated our model on the noisy test set 10 times, and plotted our test accuracies using a box plot. To summarize our findings, the median accuracy of our ten runs was 0.67, the minimum accuracy was 0.65, the maximum was 0.680, and that we have no outliers.

<figure>
<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/noise-statistical-sig.png" width="420" height="350">
</div>
<figcaption align = "center"><b>Fig X. Boxplot depicting a statistical significance study
displaying the distribution of model accuracies for variance
level of 0.1 added to test set.</b></figcaption>
</figure>

[Code for Noise Experiment - Statistical Significance Study](https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/blob/main/noise_experiment/noise_experiment.ipynb)

### Adversarial Attack 
**Motivation** Adversarial machine learning, a technique that attempts to fool models with deceptive data, is a growing threat in the AI and machine learning research community. Therefore, to test our model's robustness, we used Fast Gradient Signed Method (FGSM). FGSM is a white-box attack as it leverages an internal component of the architecture which is its gradients. 

<figure>
<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/adversarial_formula.png" width="1000" height="100">
</div>
<figcaption align="center"><b>Fig X. Formula for creating adversarial images </b></figcaption>
</figure>
The implementation follows the formula as seen above, where it takes the original image and then adjusts each pixel of the input image based on the gradient and a factor which is the pixel-wise perturbation amount known as epsilon. As per theory, as the epsilon value increases, the image becomes more perturbed causing the model to become more inaccurate

We used increased epsilon values to create more perturbed images and tested our model on these adversarial images to observe how well it could classify the images.

<figure>
<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/faces_epsilon_values.png" width="520" height="430">
</div>
<figcaption align="center"><b>Fig X. Perturbations for images for different epsilon values </b></figcaption>
</figure>

**Steps of Adversarial Attack**
1. Load the trained model.
2. Predict the label on the original test image.
3. Create perturbed image (for the current epsilon value) from original test image.
4. Let model predict on that perturbed image 
5. Record number of times model misclassified with perturbed images
6. Repeat steps 3-5 for different epsilon values.

**Conclusion** 

| Epsilon |0 (original)| 0.001 |  0.005 | 0.007 | 0.01 | 0.05 | 0.07 | 0.1 |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| **Test Accuracy** | 0.96 | 0.82 | 0.61 | 0.50 | 0.38 | 0.057 | 0.053 | 0.047|

<b>Table X. Model accuracy for varying epsilon values </b>

<figure>
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/epsilon_graph.png" width="500" height="400">
<figcaption><b>Fig X. Model accuracies against varying epsilon values </b></figcaption>
</figure>

The results show that even a small epsilon value can have quite a drastic impact on the performance of the model. An epsilon value of 0 gives us our original accuracy of 96% but an epsilon of 0.1 drops the accuracy significantly to 4.7%.

<figure>
<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/epsilon_0.png" width="900" height="250">
</div>
<figcaption align="center"><b>Fig X. Model confidenece for image with no perturbation (epsilon = 0) </b></figcaption>
</figure>

<figure>
<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/epsilon_0.01.png" width="900" height="250">
</div>
<figcaption align="center"><b>Fig X. Model confidence for image with some perturbation (epsilon = 0.01) </b></figcaption>
</figure>

The figure above shows that with an epsilon of 0 the model is very confident and there are no incorrect predictions but an epsilon of 0.01 the model is very confident but on the wrong expression label since it believes that a happy image is actually afraid. Although, the epsilon value is very small and the perturbed image looks untampered, the accuracy is much lower.

<figure>
<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/epsilon_0.3.png" width="900" height="300">
</div>
<figcaption align="center"><b>Fig X. Model confidence for image for high level of perturbation (epsilon = 0.3) </b></figcaption>
</figure>

Moreover, When the epsilon value is too high, such as 0.3, it acts similarly to adding noise to the image.  This is because the model still misclassifies the image but it is no longer confident in one label and is more confused which causes it to have some confidence in numerous labels.

Overall, there is an inverse relationship where as the epsilon value increases, the test accuracy of our model decreases. This investigation proves to show that our model might not be the most robust against such white box attacks but it is particularly hard to make a model defend itself against
these types of attacks since the attacker has access to the model parameters. 

[Code for Adversarial Attack](https://gitlab.cs.vt.edu/sdeepti/facial-expression-recognition/-/blob/main/adversarial_experiment/adversarial_attack.ipynb)

### t-SNE Feature Visualizations
**Motivation**  Insight  is extremely important for machine learning models, especially when the number of features can be millions for that model. Since data, such as images or videos, is seen in high dimensions, we want humans to make sense of the data by visualizing high dimensional data.  We used t-SNE to achieve this, t-SNE stands for t-distributed stochastic neighbor embedding, and it essentially reduces dimensionality by mapping each data point to two or three dimensions. 

```python
tSNE_embeddings = TSNE(n_components=2, perplexity=40, n_iter=1000).fit_transform(outputs)
```

t-SNE works by first creating a probability distribution of datapoint distances in higher dimensional space and then creating a similar probability distribution for datapoint distances in the lower dimensional map. Then uses a divergence algorithm such as KL-divergence between the two different probability distributions.The t-SNE algorithm main hyperparameter is the perplexity parameter. The perplexity changes the amount of attention between local and global aspects of the data. Changes in the perplexity usually changes the performance of t-SNE, and typical values are between 5 and 50. So it’s important to experiment with different perplexity values to see different results.

**Steps For t-SNE Visualization**
1. Load trained model
2. Pass images into model
3. Extract model features from last layer in ResNet model
4. Use features to compute t-SNE embeddings to reduce dimensionality to 2D or 3D
5. Repeat with different perplexity values


|2D - 20 perplexity value|2D - 100 perplexity value|3D - 20 perplexity value|3D - 100 perplexity value|
| ------ | ------ | ------ | ------ |
| <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/2-d-1.png"> | <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/2-d-2.png"> | <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/3-d-1.png"> | <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/3-d-2.png"> |

<b>Table X. This table depicts the resulting t-SNE visualization in 2D and 3D space viewing with 20 perplexity value and 100 perplexity value.</b>


**Results**  Our t-SNE performed extremely well. We find that as the perplexity increases the clusters become tighter and more defined. We discovered seven distinct clusters for each emotion as we investigated seven different emotions for our FER exploration. This visualization also does a great job supporting the accuracy we 
achieved from our ResNet50 model.

[Code for t-SNE Visualization](https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/blob/main/tsne_visualization/tsne_visualization.ipynb)

### Saliency Maps 

**Motivation** One of the best ways to interpret and visualize the CNN model is through saliency maps. Saliency maps are a way to measure the spatial support of a particular class in each image. It is a visualization technique to gain better insight into the decision-making of the CNN and
helps to highlight what each layer of a convolutional layer focuses on.

**Using Vanilla Backpropagation**
The Vanilla Backpropagation technique creates a saliency map by forward passing the data and then passing the data backward to the input layer to get the gradient and then rendering the gadient as a normalized heatmap. 

**Steps for Vanilla Backpropagation Saliency Map**
1. Apply custom transformations (pre-processing) on image
2. Retrieve model's output after passing image
3. Do backpropagation to get the derivative of the output based on the image
4. Retireve the saliency map and also pick the maximum value from channels on each pixel.
5. Plot saliency map

<figure>
<div align="center">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/vanilla_saliency_map.png" width="600" height="350">
</div>
<figcaption align="center"><b>Fig X. Saliency map using Vanilla Backpropagation Approach </b></figcaption>
</figure>

** Result **
The saliency map resulting from using the Vanilla Backpropagation approach shows a very noisy image since background features pass through the ReLu activation functions which causes it to be unclear. However, we can see that the brighter red spots are focused around the eyes and mouth area which provide us with an indication of what the model focuses on when trying to classify the image.

[Code for Vanilla Backpropagation Saliency Experiment](https://gitlab.cs.vt.edu/sdeepti/facial-expression-recognition/-/blob/main/saliency_experiment/vanilla_bp_saliency.ipynb)

**Using Guided Backpropagation**
Guided Backpropagation combines the previously used Vanilla Backpropagation technique at ReLUs with DeconvNets. Guided backpropagation visualizes gradients with respect to the image where negative gradients are suppressed when backpropagating through ReLU layers. Essentially, this methodology aims to capture pixels detected by neurons, not the ones that suppress neurons. 

**Steps for Guided Backpropagation**
1. Apply custom transformations (pre-processing) on image.
2. Retrieve model's output after passing image.
3. Do backpropagation to get the derivative of the output based on the image.
4. Save the colored gradients. 
5. Convert the colored gradients to grayscale and save grayscale gradients. 
6. Plot the positive and negative saliency maps. 

**Conclusion**
|Guided Backpropagation Saliency|Colored Guided Backpropagation|Guided Backpropagation Negative Saliency|Guided Backpropagation Positive Saliency|
| ------ | ------ | ------ | ------ |
| <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/AF01HAHR_Guided_BP_gray.png"> | <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/AF01HAHR_Guided_BP_color.png"> | <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/AF01HAHR_neg_sal.png"> | <img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/AF01HAHR_pos_sal.png"> |

<b>Table X. This table depicts the resulting saliency maps using Guided Backprogation. We were able to plot 4 kinds of saliency maps.</b>

Based on the above results we can see that our fine-tuned classifier tends to focus on specific regions of the human face to help correctly classify the human emotion. Our model mainly focuses on the center region of the face (the eyes, eyebrows, and mouth area), since the pixels seem to be the most highlighted in those parts in the Guided Backpropagation Saliency map. In addition to this, upon analyzing the Colored Guided Backpropagation map, the main features that the model is focusing on is further well defined and outlined.  

[Code for Guided Backpropagation Saliency Experiment](https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/blob/main/saliency_experiment/guided_bp_saliency.ipynb)


### Bonus: Image to Image Emotion Transfer

For our bonus task, we wanted to explore beyond the scope of image classification. In particular, our goal was to explore unpaired image-to-image translation using GANs. Image-to-image translation is a class of computervision problems where the goal is to learn the mapping between an input image and an output image. In the context of our project, we ultimately wanted to see if we could use such image translation techniques to transform one expression into another. For example, if our input is an image of a person depicting the expression, “happy”, we hope to make our model output an image of the same person, except now depicting another expression, such as “sad” by using image-to-image translation.

The model we chose for image-to-image translation was [CUT-GAN](https://github.com/taesungp/contrastive-unpaired-translation) (Contrastive Unpaired Translation GAN). We chose this model because CUT-GAN is less computationally expensive than other methods such as Cycle-GAN. Furthermore, CUT-GAN is supposed to perform better with less images due to using patchwise contrastive learning. 

**Steps:**

1. Take all 140 front-facing angry images and split into trainA, testA folders (90/10 split = 126 train / 14 test images)
2. Similarly split front-facing happy images to trainB, testB folders
3. Run CUT-GAN ([link to experiment settings](https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/blob/main/image_translation_bonus/train_opt.txt))

For our experiment, we used all straight facing angles of angry images and applied CUT-GAN to translate them into happy images. Since we only used straight images, we did not have many images for CUT-GAN to learn the mapping. Below example results are shown for real angry images translated into fake happy images by CUT-GAN. From the results, we can empirically see that the generated images had a large amount of artifacts, and did not transform facial features well into the target expression. This is likely due to the fact of how complicated facial images are, as well as not using many images.

#### Angry to Happy Image-to-Image Transfer Results
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/bonus_img1.png" width="400" height="250">
<img src="https://git.cs.vt.edu/sdeepti/facial-expression-recognition/-/raw/main/Images/bonus_img2.png" width="400" height="250">



