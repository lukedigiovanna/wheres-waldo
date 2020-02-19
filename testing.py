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
from skimage import color

pickle_in = open("waldo.pickle","rb")
waldo = pickle.load(pickle_in)

pickle_in2 = open("notwaldo.pickle","rb")
notwaldo = pickle.load(pickle_in2)


notwaldo = notwaldo[0:len(waldo)] #limits the amount of notwaldos to be the same as the number of waldos so that the data is a 50/50 split of waldos and not waldos

counter = 0
for i in notwaldo:
	if i[1] == 1:
		counter += 1
counter2 = 0
for i in waldo:
	if i[1] == 0:
		counter2 += 1


data = np.concatenate((waldo, notwaldo),axis=0)
print(len(data))


print("Before")
print(counter)
print(counter2)

print("")


X_train = []
Y_train = []
countone = 0
countzero = 0
for i in data:
	if i[1] == 0:
		countzero += 1
	elif i[1] == 1:
		countone += 1
	X_train.append(i[0])
	Y_train.append(i[1])
print("After")
print(countone)
print(countzero)
X_train = np.array(X_train)

X_train = np.array(X_train).reshape(-1, 64, 64, 3)
X_train = X_train.astype('float32') / 255 #normalize
Y_train = np.asarray(Y_train)

Y_copy = Y_train.copy()
Y_train = to_categorical(Y_train)


model = Sequential()

num_classes = 2
input_shape = (64,64,3)
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

print("created model")

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_train, Y_train), epochs=1, batch_size = 128)

print("ran model")

predicted_classes = model.predict_classes(X_train)

print(np.count_nonzero(predicted_classes - Y_copy))
confusion_matrix = pd.crosstab(Y_copy, predicted_classes, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True,fmt='.5g')

plt.show()
