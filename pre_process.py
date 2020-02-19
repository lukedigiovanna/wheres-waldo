import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas 
import os
from PIL import Image


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
			except:
				pass
			img_array = cv2.imread(os.path.join(path, img), cv2.COLOR_BGR2RGB)
			if num != 0:
				print("Placing image {0} in location {1}.".format(num, num-1))
				myarr[num - 1] = img_array
create_train()

crop_img_arr = []

print("Starting")
for shiftx in range(8):
	print("Loop {0}".format(shiftx))
	for shifty in range(8): 
		for i in range(0,28): #Will generate 8 * 8 * 28 images (1729 images) and save them to crop_img_arr
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

			x1 += shiftx * 3
			x2 += shiftx * 3

			y1 += shifty * 3
			y2 += shifty * 3
			try:
				img_copy = myarr[num - 1].copy()
				img_copy = np.fliplr(img_copy.reshape(-1,3)).reshape(img_copy.shape)
				img_copy = img_copy[y1:y2, x1:x2]
				crop_img_arr.append(img_copy)
				#print("Saving cropped image {0} to array. Dimensions are {1} by {2}.".format(num, (x2 - x1), (y2 - y1)))
			except:
				pass


new_arr = []
for i in range(len(crop_img_arr)):
	if np.array(crop_img_arr[i]).shape == (64,64,3):
		new_arr.append([crop_img_arr[i],0])
		

new_arr = np.array(new_arr)
pickle_out = open("waldo.pickle","wb")
pickle.dump(new_arr, pickle_out)
pickle_out.close()
	
'''
fig = plt.figure(figsize=(8, 8))
columns = 9
rows = 3
for i in range(1, columns*rows + 1):
    img = crop_img_arr[i - 1]
    fig.add_subplot(rows, columns, i)
    try:
    	plt.imshow(img)
    except:
    	pass

plt.show()'''