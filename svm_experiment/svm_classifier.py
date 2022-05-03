import math
import io

import sklearn
from sklearn.datasets import load_files
from sklearn.svm import SVC
from matplotlib.cbook import flatten
from skimage.transform import resize
from skimage.io import imread
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix

# load KDEF face images
dataset = load_files('../face_images', shuffle=True)


# resize images to 224,224,3 for resnet
# then flatten images
flat_data_arr = []
for i in dataset.data:
  img_array = imread(io.BytesIO(i))
  img_resized = resize(img_array, (224,224,3))
  flat_data_arr.append(img_resized.flatten())


# split data 80/20 train/val
split_idx = math.trunc((len(flat_data_arr)) * 0.8)

ss = StandardScaler()
X_train = ss.fit_transform(flat_data_arr[:split_idx])
y_train = dataset.target[:split_idx]

X_test = ss.transform(flat_data_arr[split_idx:])
y_test = dataset.target[split_idx:]

# create SVM classifier and apply gridsearch
print('training...')
# model = SVC(C=100, kernel='rbf', gamma='auto', probability=False, random_state=0)
# Default parameters (untuned:)
model = SVC(C=1.0, kernel='rbf', gamma ='scale', random_state=0)
model.fit(X_train, y_train)

print('done training')

# predict on test set
y_pred = model.predict(X_test)

# compute accuracy
print('Model accuracy:', accuracy_score(y_test, y_pred))

# compute f1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print('F1 score:', f1)

print('confusion matrix:')
print(confusion_matrix(y_test, y_pred))

print('classification report:')
print(classification_report(y_test, y_pred))
