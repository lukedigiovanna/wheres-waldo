import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas 
import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


df = pandas.read_csv('annotations.csv')

data = df[["x1","y1","x2","y2"]]

DATADIR = "/Users/joshleichty/Documents/"
CATEGORIES = ['images']
myarr = [0 for i in range(36)]
print(len(myarr))
def create_train():
	for category in CATEGORIES:
		path = os.path.join(DATADIR, category) #Path to waldo or not waldo
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			num = 0
			try:
				num = int(''.join([i for i in img if i in "0123456789"]))
				img_array = cv2.imread(os.path.join(path, img), cv2.COLOR_BGR2RGB)
				if num != 0:
					print("Placing image {0} in location {1}.".format(num, num-1))
					myarr[num - 1] = img_array
			except:
				pass
			
create_train()

crop_img_arr = []

for i in range(0,28): 
	img_string = df["image"][i]

	num = int(''.join([i for i in img_string if i in "0123456789"]))

	img_size = 64
	x2 = data["x2"][i]
	x1 = data["x1"][i]
	y2 = data["y2"][i]
	y1 = data["y1"][i]


	deltax = (img_size - (x2 - x1)) / 2.0
	deltay = (img_size - (y2 - y1)) / 2.0


	if deltax.is_integer():
		x2 = x2 + deltax
		x1 = x1 - deltax

	else:
		x1 = x1 - int(deltax)
		x2 = x2 + round(deltax)

	if deltay.is_integer():
		y2 = y2 + deltay
		y1 = y1 - deltay

	else:
		y1 = y1 - int(deltay)
		y2 = y2 + round(deltay)


	x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

	x1 += (x2 - x1) - img_size
	y1 += (y2 - y1) - img_size
	try:
		if num == 8:
			img_copy = myarr[num - 1].copy()
			img_copy = np.fliplr(img_copy.reshape(-1,3)).reshape(img_copy.shape)
			img_copy = img_copy[y1-32:y2+32, x1-32:x2+32]
			crop_img_arr.append(img_copy)
		else:
			img_copy = myarr[num - 1].copy()
			img_copy = np.fliplr(img_copy.reshape(-1,3)).reshape(img_copy.shape)
			img_copy = img_copy[y1:y2, x1:x2]
			crop_img_arr.append(img_copy)
		#print("Saving cropped image {0} to array. Dimensions are {1} by {2}.".format(num, (x2 - x1), (y2 - y1)))
	except:
		pass

#Centered and zoomed waldo on image 3
crop_img_arr[2] = crop_img_arr[2][18:50, 18:50]
crop_img_arr[2] = cv2.resize(crop_img_arr[2], (64,64))

#Centered and zoomed waldo on image 5
crop_img_arr[4] = crop_img_arr[4][20:64, 13:50]
crop_img_arr[4] = cv2.resize(crop_img_arr[4], (64,64))

#Centered and zoomed waldo on image 7
crop_img_arr[6] = crop_img_arr[6][20:60, 13:50]
crop_img_arr[6] = cv2.resize(crop_img_arr[6], (64,64))

#Centered and zoomed waldo on image 8
crop_img_arr[7] = crop_img_arr[7][18:55, 18:50]
crop_img_arr[7] = cv2.resize(crop_img_arr[7], (64,64))

#Centered and zoomed waldo on image 10
crop_img_arr[9] = crop_img_arr[9][20:64, 0:64]
crop_img_arr[9] = cv2.resize(crop_img_arr[9], (64,64))

#Zoomed in and resized image 16
crop_img_arr[15] = cv2.resize(crop_img_arr[15], (64,64))

#Zoomed in and resized image 19
crop_img_arr[19] = crop_img_arr[19][18:50, 14:50]
crop_img_arr[19] = cv2.resize(crop_img_arr[19], (64,64))

#Zoomed waldo and resized
crop_img_arr[20] = crop_img_arr[20][18:50, 18:50]
crop_img_arr[20] = cv2.resize(crop_img_arr[20], (64,64))

#Zoomed, resized, on image 23
crop_img_arr[22] = crop_img_arr[22][18:50, 18:50]
crop_img_arr[22] = cv2.resize(crop_img_arr[22], (64,64))

#Zoomed and shifted on image 24
crop_img_arr[23] = crop_img_arr[23][18:50, 20:50]
crop_img_arr[23] = cv2.resize(crop_img_arr[23], (64,64))

#Zoomed image 25
crop_img_arr[25] = crop_img_arr[25][18:50, 20:50]
crop_img_arr[25] = cv2.resize(crop_img_arr[25], (64,64))


indexes = [1, 18, 21, 26]
for index in sorted(indexes, reverse=True):
    del crop_img_arr[index]


datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img_list = []
for test_image in crop_img_arr:
	x = test_image.reshape((1,) + test_image.shape)
	counter = 0
	for batch in datagen.flow(x, batch_size=1):
		counter += 1
		if batch[0].shape == (64, 64, 3):
			img_list.append([batch[0], 0])
		if counter > 220:
			break


with open('waldo.pickle','wb') as f:
	pickle.dump(img_list, f)