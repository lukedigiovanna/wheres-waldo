from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator
import pickle

DATADIR = "/Users/joshleichty/Documents/"
CATEGORIES = ['images']
training_data = []
not_waldo = []
waldo = []
def create_train():
	for category in CATEGORIES:
		path = os.path.join(DATADIR, category) #Path to waldo or not waldo
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			img_array = cv2.imread(os.path.join(path, img), cv2.COLOR_BGR2RGB) #Convert images to grayscale arr

			'''
			If we want to resize
			IMG_SIZE = new image size
			new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
			'''
			if class_num == 0:
				waldo.append(img_array)
				training_data.append([img_array, class_num])
			elif class_num == 1:
				not_waldo.append(img_array)
				training_data.append([img_array, class_num])

create_train()

waldo = np.asarray(waldo)

not_waldo = np.asarray(not_waldo[0:38])

random.shuffle(training_data)


X_train = []
y_train = []



for features, label in training_data:
	X_train.append(features)
	y_train.append(label)

pickle_out = open("original.pickle", "wb")
pickle.dump(waldo, pickle_out)
pickle_out.close()


exit()
X = np.array(X_train).reshape(-1, 64, 64, 1)
X = X.astype('float32') / 255
Y = np.asarray(y_train)
Y_copy = Y.copy()
Y = to_categorical(Y)

model = Sequential()

num_classes = 2
input_shape = (64,64,1)
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, Y, validation_data=(X, Y), epochs=1, batch_size = 128)

predicted_classes = model.predict_classes(X)

print(np.count_nonzero(predicted_classes - Y_copy))
confusion_matrix = pd.crosstab(Y_copy, predicted_classes, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True,fmt='.5g') 

plt.show()