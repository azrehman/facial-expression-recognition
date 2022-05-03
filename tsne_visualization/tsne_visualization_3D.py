# different tsne library cuz other one didint support 3D
from tsne_torch import TorchTSNE as TSNE

import os
import numpy as np

import torch
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib
import matplotlib.pyplot as plt


model_path = '../main_resnet50/FEC_resnet50_trained_face_images_80_10_10.pt'
data_dir = '../face_images_unsplit'
num_classes = 7

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')


# load the trained model
model = models.resnet50(num_classes=num_classes)
# transfer model to gpu if available
model = model.to(device)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
# set model to evaluation mode
model.eval()


data_transforms = transforms.Compose([
	transforms.Resize(size=(224, 224)),
	transforms.ToTensor(),
	# use ImageNet standard mean and std dev for transfer learning
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# load test dataset and create dataloader
batch_size = 32
data_set = ImageFolder(data_dir, transform=data_transforms)
data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)


def get_representations(model, device, data_loader):
	outputs = []
	intermediates = []
	labels = []

	model.eval()
	with torch.no_grad():
		for images, batch_labels in data_loader:
			images = images.to(device)
			batch_labels = batch_labels.to(device)

			y_pred = model(images)

			outputs.append(y_pred.cpu())
			labels.append(batch_labels.cpu())
	outputs = torch.cat(outputs, dim=0)
	labels = torch.cat(labels, dim=0)

	return outputs, labels.numpy()

outputs, labels = get_representations(model, device, data_loader)

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
	# compute the distribution range
	value_range = (np.max(x) - np.min(x))

	# move the distribution so that it starts from zero
	# by extracting the minimal value from all its values
	starts_from_zero = x - np.min(x)

	# make the distribution fit [0; 1] by dividing by its range
	return starts_from_zero / value_range

for perplexity in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
	X_emb = TSNE(n_components=3, perplexity=perplexity, n_iter=1000, verbose=True).fit_transform(outputs)
        # returns shape (n_samples, 2)

	print(f'perplexity: {perplexity}')


        # extract x and y coordinates representing the positions of the images on T-SNE plot
	tx = X_emb[:, 0]
	ty = X_emb[:, 1]
	tz = X_emb[:, 2]

	tx = scale_to_01_range(tx)
	ty = scale_to_01_range(ty)
	tz = scale_to_01_range(tz)

        # Create the figure
	fig = plt.figure( figsize=(8,8) )
	ax = fig.add_subplot(1, 1, 1, projection='3d', title=f't-SNE (perplexity={perplexity}) Projection of ResNet Features')

	colors= ['darkorange','crimson','seagreen','lightpink', 'lightslategray', 'royalblue', 'rebeccapurple']
        # Create the scatter
	scatter = ax.scatter(
            tx,
            ty,
            tz,
            c=labels,
            cmap=matplotlib.colors.ListedColormap(colors),
            alpha=0.4,
            s=6.0)

	ax.legend(handles=scatter.legend_elements()[0], labels=list(data_set.class_to_idx.keys()), loc='best')

	ax.set_xlabel('Scaled t-SNE X component embedding value')
	ax.set_ylabel('Scaled t-SNE Y component embedding value')
	ax.set_zlabel('Scaled t-SNE Z component embedding value')


	plt.tight_layout()
	plt.savefig(f'tsne_3D_perp_{perplexity}.png')
	plt.show()
	plt.close()


