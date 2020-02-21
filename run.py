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
from PIL import Image
import glob

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def gradCAM(orig, intensity=0.5, res=250):
  img = image.load_img(orig, target_size=(DIM, DIM))

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  preds = model.predict(x)
  print(decode_predictions(preds)[0][0][1]) # prints the class of image

  with tf.GradientTape() as tape:
    last_conv_layer = model.get_layer('conv2d')
    iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
    model_out, last_conv_layer = iterate(x)
    class_out = model_out[:, np.argmax(model_out[0])]
    grads = tape.gradient(class_out, last_conv_layer)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
  heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)
  heatmap = heatmap.reshape((8, 8))

  img = cv2.imread(orig)

  heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

  heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

  img = heatmap * intensity + img

  cv2_imshow(cv2.resize(cv2.imread(orig), (res, res)))
  cv2_imshow(cv2.resize(img, (res, res)))


image_list = []
for filename in glob.glob('/Users/joshleichty/Documents/Hey-Waldo-master/64/waldo/*.jpg'): #assuming gif
    im=Image.open(filename)
    im.load()
    data = np.asarray(im, dtype="int32" )
    image_list.append(data)

image_list = np.array(image_list)
image_list = image_list.reshape(-1, 64, 64, 3)
image_list = image_list.astype('float32') / 255 


Y = model.predict_classes(image_list)
for i in range(len(Y)):
	if Y[i] == 0:
		plt.imshow(image_list[i])
		plt.show()
print(Y)