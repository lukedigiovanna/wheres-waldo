import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
import image_slicer

IMAGE_SIZE = 64
STRIDE = 8

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

a = image_slicer.slice("/Users/joshleichty/Documents/images/12.jpg", 300)


confarr = []
for i in a:
    arr = np.array(i.image.convert('RGB')) / 255
    arr = cv2.resize(arr, (64,64)) 
    confidence = model.predict_proba(np.array([arr]))
    confarr.append([confidence[0][0], confidence[0][1]])
print(confarr)
exit()

for i, e in enumerate(confarr):
    if e > .5:
        arr = np.array(a[i].image.convert('RGB'))
        arr = cv2.resize(arr, (64,64))
        plt.imshow(arr)
        plt.show()


'''
image = np.array(image)
img = image[:,:,::-1]
import image_slicer
'''
'''

#height, width, channel

shape = image.shape
height = shape[0]
width = shape[1]
channels = shape[2]

height_of_hmap = height-IMAGE_SIZE
width_of_hmap = width-IMAGE_SIZE

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#contains 2D map of each image slice
#index 1 gives the row of the
hmap = []

counter = 0
print("Starting slicing")
for y in range(height_of_hmap):
    if y % STRIDE == 0:
        counter += 1
        print("{0} / {1}".format(counter, (int(height_of_hmap / STRIDE))))
    if (y % STRIDE == 0):
        hmap_row = []
        for x in range(width_of_hmap):
            if (x % STRIDE == 0):
                slice_x = image[y:y+IMAGE_SIZE,x:x+IMAGE_SIZE]
                X = []
                X.append(slice_x)
                X = np.array(X)
                confidence = model.predict_proba(X)[0][0]
                hmap_row.append(confidence)
        hmap.append(np.array(hmap_row))

hmap = np.array(hmap)

print("Done, starting model prediction")

print(hmap)

plt.imshow(hmap)
plt.show()'''