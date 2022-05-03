# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#create-the-optimizer

import os
import time
import copy
import csv

from pprint import pprint

import torch
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# constant parameters

#data_dir = 'face_images_80_10_10'
data_dir = 'radiate_faces_80_10_10'
print(f'using {data_dir} as data folder')

model_save_path = 'FEC_bias' + data_dir + '.pt'

num_classes = 7

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


# expected input size for resnet
input_size = 224


# transformations to apply to images
# data augmentation and normalization for training
# just normalization for validation and testing
# https://pytorch.org/vision/stable/transforms.html
data_transforms = {
	'train': transforms.Compose([
		transforms.Resize(size=(input_size, input_size)),
		# transforms.Grayscale(), (cannot use greyscale with resnet)
		# rotation augmentation
		transforms.RandomRotation(10),
		# random flip augmentaion
		transforms.RandomHorizontalFlip(),
		# jitter brightness, contrast, saturation augmentaion
		transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0),
		# convert to tensor and normalize
		transforms.ToTensor(),
		# use ImageNet standard mean and std dev for transfer learning
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
		transforms.Resize(size=(input_size, input_size)),
		transforms.ToTensor(),
		# use ImageNet standard mean and std dev for transfer learning
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'test': transforms.Compose([
		transforms.Resize(size=(input_size, input_size)),
		transforms.ToTensor(),
		# use ImageNet standard mean and std dev for transfer learning
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
}



# hyperparameters
batch_size = 16

# create train and validation datasets and apply transforms
datasets_dict = {dataset_type: ImageFolder(os.path.join(data_dir, dataset_type),
	transform=data_transforms[dataset_type]) for dataset_type in ['train', 'val', 'test']}

# create train and validation dataloaders
dataloaders_dict = {dataset_type: DataLoader(datasets_dict[dataset_type], batch_size=batch_size, shuffle=True, num_workers=4) for dataset_type in ['train', 'val', 'test']}




# tests performance on test set and computes metrics
def test(model, test_loader):
	# list of predicted labels of all batches
	predicted_labels = torch.zeros(0, dtype=torch.long, device='cpu')
	# list of actual labels of all batches
	actual_labels = torch.zeros(0, dtype=torch.long, device='cpu')

	with torch.no_grad():
		model.eval()
		# get batch of inputs (image) and outputs (expression label) from test_loader
		for inputs, labels in test_loader:
			inputs = inputs.to(device)
			labels = labels.to(device)

			# use model to predict label
			outputs = model(inputs)
			_, preds = torch.max(outputs, dim=1)

			# append batch prediction labels and actual labels
			predicted_labels = torch.cat([predicted_labels, preds.view(-1).cpu()])
			actual_labels = torch.cat([actual_labels, labels.view(-1).cpu()])

	print('\nTest Metrics:')
	# print confusion matrix
	print('Confusion Matrix:')
	print(confusion_matrix(actual_labels.numpy(), predicted_labels.numpy()))

	print('Test Accuracy:', accuracy_score(actual_labels.numpy(), predicted_labels.numpy()))
	print('F1 score:', f1_score(actual_labels.numpy(), predicted_labels.numpy(), average='weighted'))
	# print classification report
	print('Classification Report:')
	print(classification_report(actual_labels.numpy(), predicted_labels.numpy()))

	return predicted_labels




# load model
model = models.resnet50(num_classes=num_classes)
model.load_state_dict(torch.load('dataset_size_experiment/dataset_size_70/FEC_resnet50_trained_face_images_70_10_20.pt'))
model.eval()

# transfer model to gpu if available
model = model.to(device)



# test model on radiate images
test(model, dataloaders_dict['test'])
