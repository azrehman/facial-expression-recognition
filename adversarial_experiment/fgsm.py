# https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial10/Adversarial_Attacks.html
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial10/Adversarial_Attacks.html

import os
import numpy as np

import torch
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score



model_path = '../main_resnet50/FEC_resnet50_trained_face_images_80_10_10.pt'
# model_path = '../dataset_size_experiment/dataset_size_70/FEC_resnet50_trained_face_images_70_10_20.pt'
data_dir = '../face_images_80_10_10'
num_classes = 7

device = 'cuda' if torch.cuda.is_available() else 'cpu'

NORM_MEAN = np.array([0.485, 0.456, 0.406])
NORM_STD = np.array([0.229, 0.224, 0.225])


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

for p in model.parameters():
	p.requires_grad = False


# transfer model to gpu if available



# load test dataset and create dataloader
test_transforms = transforms.Compose([
	transforms.Resize(size=(224, 224)),
	transforms.ToTensor(),
	# use ImageNet standard mean and std dev for transfer learning
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

batch_size = 32

test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transforms)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


idx_to_class = {v: k for k, v in test_set.class_to_idx.items()}


def eval_model(model, device, test_loader, img_func=None):
    tp, tp_5 = 0.0, 0.0
    counter = 0.0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        if img_func is not None:
            images = img_func(images, labels)
        with torch.no_grad():
            preds = model(images)
        tp += (preds.argmax(dim=-1) == labels).sum()
        tp_5 += (preds.topk(5, dim=-1)[1] == labels[...,None]).any(dim=-1).sum()
        counter += preds.shape[0]
    acc = tp.float().item()/counter
    top5 = tp_5.float().item()/counter
    print(f'Top-1 error: {(100.0 * (1 - acc)):4.2f}%')
    print(f"Top-5 error: {(100.0 * (1 - top5)):4.2f}%")
    return acc, top5


print('performance with no attack:')
eval_model(model, device, test_loader)


# FGSM attack code add markdown from tutorial link here
def fgsm_attack(model, imgs, labels, epsilon):
	# Collect the element-wise sign of the data gradient
	inp_imgs = imgs.clone().requires_grad_()
	preds = model(inp_imgs.to(device))
	preds = F.log_softmax(preds, dim=-1)
         # Calculate loss by NLL
	loss = -torch.gather(preds, 1, labels.to(device).unsqueeze(dim=-1))
	loss.sum().backward()
	# Update image to adversarial example as written above
	noise_grad = torch.sign(inp_imgs.grad.to(imgs.device))
	fake_imgs = imgs + epsilon * noise_grad
	fake_imgs.detach_()
	return fake_imgs, noise_grad


def graph_fgsm_confidence(model, device, image_batch, label_batch, epsilon):
    print(f'epsilon is {epsilon}')
    # filename = f'pred_{epsilon}.png'
    # Accuracy counter


    adv_images, noise_grad = fgsm_attack(model, image_batch, label_batch, epsilon)

    with torch.no_grad():
        adv_preds = model(adv_images.to(device))
    for i in range(1, 489, 60):
        filename = f'adversarial_sample_{i}_{epsilon}.png'
        show_prediction(image_batch[i], label_batch[i], adv_preds[i], filename, epsilon, adv_img=adv_images[i], noise=noise_grad[i])


def show_prediction(img, label, pred, filename, epsilon, K=5, adv_img=None, noise=None):

	if isinstance(img, torch.Tensor):
		# Tensor image to numpy
		img = img.cpu().permute(1, 2, 0).numpy()
		img = (img * NORM_STD[None,None]) + NORM_MEAN[None,None]
		img = np.clip(img, a_min=0.0, a_max=1.0)
		label = label.item()

	# Plot on the left the image with the true label as title.
	# On the right, have a horizontal bar plot with the top k predictions including probabilities
	if noise is None or adv_img is None:
		fig, ax = plt.subplots(1, 2, figsize=(10,2), gridspec_kw={'width_ratios': [1, 1]})
	else:
		fig, ax = plt.subplots(1, 5, figsize=(12,2), gridspec_kw={'width_ratios': [1, 1, 1, 1, 2]})

	ax[0].imshow(img)
	ax[0].set_title(idx_to_class[label])
	ax[0].axis('off')

	if adv_img is not None and noise is not None:
		# Visualize adversarial images
		adv_img = adv_img.cpu().permute(1, 2, 0).numpy()
		adv_img = (adv_img * NORM_STD[None,None]) + NORM_MEAN[None,None]
		adv_img = np.clip(adv_img, a_min=0.0, a_max=1.0)
		ax[1].imshow(adv_img)
		ax[1].set_title(f'Adversarial (epsilon={epsilon})')
		ax[1].axis('off')
		# Visualize noise
		noise = noise.cpu().permute(1, 2, 0).numpy()
		noise = noise * 0.5 + 0.5 # Scale between 0 to 1
		ax[2].imshow(noise)
		ax[2].set_title('Noise')
		ax[2].axis('off')
		# buffer
		ax[3].axis('off')

	if abs(pred.sum().item() - 1.0) > 1e-4:
		pred = torch.softmax(pred, dim=-1)
	topk_vals, topk_idx = pred.topk(K, dim=-1)
	topk_vals, topk_idx = topk_vals.cpu().numpy(), topk_idx.cpu().numpy()
	ax[-1].barh(np.arange(K), topk_vals*100.0, align='center', color=["C0" if topk_idx[i]!=label else "C2" for i in range(K)])
	ax[-1].set_yticks(np.arange(K))
	ax[-1].set_yticklabels([idx_to_class[c].title() for c in topk_idx])
	ax[-1].invert_yaxis()
	ax[-1].set_xlabel('Confidence')
	ax[-1].set_title('Predictions')

	plt.tight_layout()
	#plt.savefig(filename, bbox_inches='tight')
	plt.show()
	#plt.close()


# exmp_batch, label_batch = next(iter(test_loader))
# for eps in epsilons:
    # graph_fgsm_confidence(model, device, exmp_batch, label_batch, eps)
for eps in epsilons:
    print(f'evaluating epsilon: {eps}')
    _ = eval_model(model, device, test_loader, img_func=lambda x, y: fgsm_attack(model, x, y, epsilon=eps)[0])
