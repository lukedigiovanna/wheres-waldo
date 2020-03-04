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
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, concatenate
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import plot_model

pickle_in = open("oldwaldo.pickle","rb")
oldwaldo = pickle.load(pickle_in)


pickle_in2 = open("waldo.pickle","rb")
waldo = pickle.load(pickle_in2)

for i in range(len(waldo)):
	waldo[i][0] = waldo[i][0] * 255

pickle_in3 = open("notwaldo.pickle","rb")
notwaldo = pickle.load(pickle_in3)

oldwaldo = np.concatenate((oldwaldo, waldo[0:len(oldwaldo)]), axis = 0)
notwaldo = notwaldo[0:len(oldwaldo)]

#data = waldo + notwaldo
data = np.concatenate((oldwaldo, notwaldo),axis=0)
np.random.shuffle(data)

X_train = []
Y_train = []
for i in data:
	if(i[1] == 1):
		X_train.append(i[0][...,::-1])
	else:
		X_train.append(i[0])
	if (i[1] == 0):
		Y_train.append(1)
	else:
		Y_train.append(0)


X_train = np.array(X_train)


X_train = np.array(X_train).reshape(-1, 64, 64, 3)
X_train = X_train.astype('float32') / 255
Y_train = np.asarray(Y_train)
X = X_train
Y = Y_train


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

Y_copy = Y_test.copy()


model = Sequential()

num_classes = 1
input_shape = (64,64,3)

model = Sequential()
checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

batch_size = 32
kernel_size = 3
dropout = 0.4
n_filters = 32

# left branch of Y network
left_inputs = Input(shape=input_shape)
x = left_inputs
filters = n_filters
# 3 layers of Conv2D-Dropout-MaxPooling2D
# number of filters doubles after each layer (32-64-128)
for i in range(3):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D()(x)
    filters *= 2

# right branch of Y network
right_inputs = Input(shape=input_shape)
y = right_inputs
filters = n_filters
# 3 layers of Conv2D-Dropout-MaxPooling2D
# number of filters doubles after each layer (32-64-128)
for i in range(3):
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu',
               dilation_rate=2)(y)
    y = Dropout(dropout)(y)
    y = MaxPooling2D()(y)
    filters *= 2

# merge left and right branches outputs
y = concatenate([x, y])
# feature maps to vector before connecting to Dense
y = Flatten()(y)
y = Dropout(dropout)(y)
outputs = Dense(1, activation='sigmoid')(y)

# build the modela in functional API
model = Model([left_inputs, right_inputs], outputs)
# verify the model using graph
plot_model(model, to_file='cnn-y-network.png', show_shapes=True)
# verify the model using layer text description
model.summary()

# classifier loss, Adam optimizer, classifier accuracy
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("created model")


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

print(Y_test)

history = model.fit([X_train, X_train], Y_train, validation_data=([X_test, X_test], Y_test), epochs=12, batch_size = 64, callbacks = [checkpoint] )

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.plot(range(len(accuracy)), accuracy, color="black")
ax.plot(range(len(val_accuracy)), val_accuracy, color="red")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")

plt.show()

print(accuracy)

print("ran model")

exit()

#predicted_classes = model.predict_classes([X_test, X_test])


model.save_weights("model.h5")
print("Saved model to disk")


confusion_matrix = pd.crosstab(Y_copy, predicted_classes, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True,fmt='.5g')

plt.show()
