# https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial10/Adversarial_Attacks.html

import os
import numpy as np

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt



model_path = '../main_resnet50/FEC_resnet50_trained_face_images_80_10_10.pt'
data_dir = '../data/face_images_80_10_10'
num_classes = 7

device = 'cuda' if torch.cuda.is_available() else 'cpu'


'''
List of epsilon values to use for the run.
It is important to keep 0 in the list because it represents the model performance on the original test set.
Also, intuitively we would expect the larger the epsilon, the more noticeable the perturbations
but the more effective the attack in terms of degrading model accuracy.
'''
epsilons = [0, .05, .1, .15, .2, .25, .3]


# load the trained model
model = models.resnet50(num_classes=num_classes)
model = model.to(device)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
# set model to evaluation mode
model.eval()
# transfer model to gpu if available



# load test dataset and create dataloader
test_transforms = transforms.Compose([
	transforms.Resize(size=(224, 224)),
	transforms.ToTensor(),
	# use ImageNet standard mean and std dev for transfer learning
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

batch_size = 1

test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transforms)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

idx_to_class = {v: k for k, v in test_set.class_to_idx.items()}

def test_normal(model, device, test_loader):
	model.eval()
	correct = 0
	total = 0
	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)

		_, pred = torch.max(outputs.data, 1)

		total += 1
		correct += (pred == labels).sum()
	print('Accuracy of test (no attack): %f %%' % (100 * float(correct) / total))



NORM_MEAN = np.array([0.485, 0.456, 0.406])
NORM_STD = np.array([0.229, 0.224, 0.225])
TENSOR_MEANS, TENSOR_STD = torch.FloatTensor(NORM_MEAN)[:,None,None], torch.FloatTensor(NORM_STD)[:,None,None]
# FGSM attack code add markdown from tutorial link here
def fgsm_attack(image, image_grad, epsilon):
	# Collect the element-wise sign of the data gradient
	sign_data_grad = image_grad.sign()
	# Create the perturbed image by adjusting each pixel of the input image
	perturbed_image = image + epsilon*sign_data_grad
	# Adding clipping to maintain [0,1] range
	perturbed_image = (torch.tanh(perturbed_image.cpu()) + 1 - 2 * TENSOR_MEANS) / (2 * TENSOR_STD)
	perturbed_image = torch.clamp(perturbed_image, 0, 1)
	# Return the perturbed image
	return perturbed_image.to(device)


def test_attack(model, device, test_loader, epsilon):
	# Accuracy counter
	correct = 0
	adv_examples = []

	criterion = torch.nn.CrossEntropyLoss()

	# Loop over all examples in test set
	for image, label in test_loader:

		# Send the image and label to the device
		image, label = image.to(device), label.to(device)

		image.requires_grad = True

		output = model(image)
		# get the index of the max log-probability
		initial_pred = output.max(1, keepdim=True)[1]

		# If the initial prediction is wrong, dont bother attacking, just move on
		if initial_pred.item() != label.item():
			continue


		loss = torch.nn.CrossEntropyLoss()(output, label)
		# Zero all existing gradients
		model.zero_grad() # ?

		loss.backward()

		image_grad = image.grad.data

		perturbed_image = fgsm_attack(image, image_grad, epsilon)


		# Forward pass the image through the model
		output_perturbed = model(perturbed_image)

		# Check for success on perturbed image
		# get the index of the max log-probability
		perturbed_pred = output_perturbed.max(1, keepdim=True)[1]
		if perturbed_pred.item() == label.item():
			correct += 1
			# Special case for saving 0 epsilon examples
			if (epsilon == 0) and (len(adv_examples) < 5):
				adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
				adv_examples.append( (initial_pred.item(), perturbed_pred.item(), adv_ex) )
		else:
			# Save some adv examples for visualization later
			if len(adv_examples) < 5:
				adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
				adv_examples.append( (initial_pred.item(), perturbed_pred.item(), adv_ex) )

	# Calculate final accuracy for this epsilon
	final_acc = correct/float(len(test_loader))
	print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

	# Return the accuracy and an adversarial example
	return final_acc, adv_examples


# run the attack
accuracies = []
examples = []

test_normal(model, device, test_loader)

# Run test for each epsilon
for eps in epsilons:
	acc, ex = test_attack(model, device, test_loader, eps)
	accuracies.append(acc)
	examples.append(ex)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.savefig('adversarial_attck_graph.png', bbox_inches='tight')
plt.show()
plt.close()


# show examples of perturbed images at different epsilons

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
	for j in range(len(examples[i])):
		cnt += 1
		plt.subplot(len(epsilons),len(examples[0]),cnt)
		plt.xticks([], [])
		plt.yticks([], [])
		if j == 0:
			plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
		orig,adv,ex = examples[i][j]

		plt.title("{} -> {}".format(idx_to_class[orig][:2].upper(), idx_to_class[adv][:2].upper()))
		plt.imshow(ex.transpose(1,2,0), cmap="gray")
plt.tight_layout()
plt.savefig('adversarial_attck_images.png', bbox_inches='tight')
plt.show()
plt.close()

