from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import pickle 

DATADIR = "/Users/joshleichty/Documents/64-bw"
CATEGORIES = ['waldo', 'notwaldo']
training_data = []
not_waldo = []
waldo = []
def create_train():
	for category in CATEGORIES:
		path = os.path.join(DATADIR, category) #Path to waldo or not waldo
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) #Convert images to grayscale arr

			'''
			If we want to resize
			IMG_SIZE = new image size
			new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
			'''
			if class_num == 0:
				waldo.append([img_array, class_num])
				training_data.append([img_array, class_num])
			elif class_num == 1:
				not_waldo.append([img_array, class_num])
				training_data.append([img_array, class_num])

create_train()

random.shuffle(training_data)

X_train = []
y_train = []

for features, label in training_data:
	X_train.append(features)
	y_train.append(label)

X = np.array(X_train).reshape(-1, 64, 64, 1)
X = X.astype('float32') / 255
Y = np.asarray(y_train)
Y = to_categorical(Y)

model = Sequential()

model.add(Conv2D(12, kernel_size=3, activation='relu', input_shape = (64,64,1)))
model.add(Conv2D(6, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, Y, validation_data=(X, Y), epochs=3, batch_size = 128)

predicted_classes = model.predict_classes(X)
print(predicted_classes)
print(Y)