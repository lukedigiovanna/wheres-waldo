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
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

pickle_in = open("waldo.pickle","rb")
waldo = pickle.load(pickle_in)


pickle_in2 = open("notwaldo.pickle","rb")
notwaldo = pickle.load(pickle_in2)


notwaldo = notwaldo[0:len(waldo)] 


data = waldo + notwaldo
data = np.concatenate((waldo, notwaldo),axis=0)
np.random.shuffle(data)

X_train = []
Y_train = []
for i in data:
	if(i[1] == 1):
		X_train.append(i[0][...,::-1])
	else:
		X_train.append(i[0] * 255) 
	Y_train.append(i[1])


X_train = np.array(X_train)


X_train = np.array(X_train).reshape(-1, 64, 64, 3)
X_train = X_train.astype('float32') / 255 
Y_train = np.asarray(Y_train)
X = X_train
Y = Y_train


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)


Y_copy = Y_test.copy()
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

model = Sequential()

num_classes = 2
input_shape = (64,64,3)

model = Sequential()
checkpoint = ModelCheckpoint("model.h5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(2, activation='softmax'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])



print("created model")


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size = 64, callbacks = [checkpoint] )

print("ran model")

predicted_classes = model.predict_classes(X_test)


model.save_weights("model.h5")
print("Saved model to disk")
 

confusion_matrix = pd.crosstab(Y_copy, predicted_classes, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True,fmt='.5g')

plt.show()
