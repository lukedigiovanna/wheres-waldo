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
			training_data.append([img_array, class_num])
			if class_num == 0:
				waldo.append([img_array, class_num])
			elif class_num == 1:
				not_waldo.append([img_array, class_num])
create_train()

X_train = []
y_train = []
for features, label in training_data:
	X_train.append(features)
	y_train.append(label)

X = np.array(X_train).reshape(-1, 64, 64, 1)
X = X.astype('float32') / 255
Y = np.asarray(y_train).astype('float32')/255

'''
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000,28,28,1)
X_train = X_train.astype('float32')/255

X_test = X_test.reshape(10000,28,28,1)
X_test = X_test.astype('float32')/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()


model.add(Conv2D(12, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(6, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=1, batch_size = 64)
'''
