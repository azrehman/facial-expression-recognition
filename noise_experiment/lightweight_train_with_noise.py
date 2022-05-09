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

from skimage.util import random_noise

# constant parameters

data_dir = 'face_images_80_10_10'
print(f'using {data_dir} as data folder')

model_save_path = 'FEC_resnet50_trained_' + data_dir + '.pt'

num_classes = 7

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# flag for feature extracting, when true, finetune the entire model
# when false, only update reshaped layer parameters (last fully connected layer)
finetune_all_parmas = True

# expected input size for resnet
input_size = 224

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
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                # add noise as data augmentation
                GaussianNoise(mean=0, var=0.01)
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
#learning_rate = 0.0005
learning_rate = 0.005
num_epochs = 50 # todo early stopping maybe?

# create train and validation datasets and apply transforms
datasets_dict = {dataset_type: ImageFolder(os.path.join(data_dir, dataset_type),
	transform=data_transforms[dataset_type]) for dataset_type in ['train', 'val', 'test']}

# create train and validation dataloaders
dataloaders_dict = {dataset_type: DataLoader(datasets_dict[dataset_type], batch_size=batch_size, shuffle=True) for dataset_type in ['train', 'val', 'test']}



# initialize a pretrained resnet model
def init_model(num_classes, resnet_size, finetune_all_parmas,
				print_model=True, class_to_idx=None):

	model = None

	if resnet_size == 18:
		model = models.resnet18(pretrained=True)
	elif resnet_size == 34:
		model = models.resnet34(pretrained=True)
	elif resnet_size == 50:
		model = models.resnet50(pretrained=True)
	elif resnet_size == 101:
		model = models.resnet101(pretrained=True)
	elif resnet_size == 152:
		model = models.resnet152(pretrained=True)
	else:
		raise ValueError(f'Invalid size of {resnet_size} given for resnet size.')


	# sets requires_grad attribute of parameters in model to false if not finetuning all parameters
	if not finetune_all_parmas:
		# don't relearn weights when transfer learning
		for param in model.parameters():
			param.requires_grad = False

	# when transfer learning, set last layer to be fully connected with num_classes number of outputs
	model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

	# map classes to indexes
	if class_to_idx:
		model.class_to_idx = class_to_idx
		model.idx_to_class = {idx: class_ for class_, idx in model.class_to_idx.items()}


		# print model information
	if print_model:
		# print model summary
		print()
		print(f'Using resnet size: {resnet_size}')
		print('Model summary:')
		print(model)

		# print model parameters
		total_params = sum(p.numel() for p in model.parameters())
		trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
		total_trainable_params = len(trainable_names)
		print()
		print(f'Model parameters ({total_params} total, {total_trainable_params} trainable)')
		print('List of trainable parameters:')
		pprint(trainable_names, width=80, compact=True)
		print()

		# print mapping for class indicies
		if class_to_idx:
			print('Model index to class mappings:')
			print(model.idx_to_class)
			print()

	return model



# model training and validation
# input: a PyTorch model, a dictionary of dataloaders, a loss function (criterion), an optimizer, a specified number of epochs to train and validate for
def train(model, dataloaders, criterion, optimizer,
			num_epochs=25, save_path=None, save_history_to_csv=True):

	since = time.time()

	# save train and val loss/accuracy history for each epoch
	train_loss_history = []
	val_loss_history = []
	train_acc_history = []
	val_acc_history = []


	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print(f'Epoch {epoch + 1}/{num_epochs}')
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data.
			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					# Get model outputs and calculate loss
					outputs = model(inputs)
					loss = criterion(outputs, labels)

					_, preds = torch.max(outputs, 1)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

			print(f'{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}')

			# deep copy the model if best accuracy
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

			# save loss and accuracy to history
			if phase == 'train':
				train_loss_history.append(epoch_loss)
				train_acc_history.append(epoch_acc.item())
			elif phase == 'val':
				val_loss_history.append(epoch_loss)
				val_acc_history.append(epoch_acc.item())

		print()

	time_elapsed = time.time() - since
	print(f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
	print(f'Best val Acc: {best_acc:4f}')

	# load best model weights
	model.load_state_dict(best_model_wts)

	# write history csv file
	if save_history_to_csv:
		history_header = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
		history_filename = model_save_path.split('.')[0] + '_history.csv'
		history = zip(train_loss_history, val_loss_history, train_acc_history, val_acc_history)
		history = [list(row) for row in history]
		with open(history_filename, 'w') as csv_file:
			writer = csv.writer(csv_file)
			writer.writerow(history_header)
			for row in history:
				writer.writerow(row)

		# save trained model to disk
		if save_path:
			torch.save(model.state_dict(), save_path)

	return model

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




model = init_model(num_classes, 50, finetune_all_parmas,
					class_to_idx=datasets_dict['train'].class_to_idx)

# transfer model to gpu if available
model = model.to(device)


# set parameters needed to be optimized/updated
# either entire model parameters or just of last added layer(s)
params_to_update = None
if not finetune_all_parmas:
	params_to_update = [param for param in model.parameters() if param.requires_grad]
else:
	params_to_update = model.parameters()

# set optimizer
#optimizer = torch.optim.Adam(params_to_update, lr=learning_rate)
optimizer = torch.optim.SGD(params_to_update, lr=learning_rate)

# set loss function
criterion = torch.nn.CrossEntropyLoss()

# train model
trained_model = train(model, dataloaders_dict, criterion, optimizer,
		num_epochs, model_save_path)

# test model
test(trained_model, dataloaders_dict['test'])
