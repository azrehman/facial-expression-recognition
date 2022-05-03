import os
import numpy as np

import torch
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from skimage.util import random_noise


variance = 0.05
print(f'using variance of {variance}')

model_path = '../main_resnet50/FEC_resnet50_trained_face_images_80_10_10.pt'
# model_path = '../dataset_size_experiment/dataset_size_70/FEC_resnet50_trained_face_images_70_10_20.pt'
data_dir = '../data/face_images_80_10_10'
num_classes = 7

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# load the trained model
model = models.resnet50(num_classes=num_classes)
# transfer model to gpu if available
model = model.to(device)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
# set model to evaluation mode
model.eval()



# custom noise transformation
# https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
class GaussianNoise(object):
	def __init__(self, mean=0., var=1.):
		self.var = var
		self.mean = mean

	def __call__(self, tensor):
		return torch.tensor(random_noise(tensor, mode='gaussian', mean=self.mean, var=self.var, clip=True), dtype=torch.float)

	def __repr__(self):
		return self.__class__.__name__ + f'(mean={self.mean}, var={self.var})'

# test transformations with noise added
test_transforms = transforms.Compose([
	transforms.Resize(size=(224, 224)),
	transforms.ToTensor(),
	# use ImageNet standard mean and std dev for transfer learning
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    	# add noise after normalization
	GaussianNoise(mean=0, var=variance)
])

# load test dataset and create dataloader
batch_size = 16
test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transforms)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


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



test(model, test_loader)
