import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas 
import os
df = pandas.read_csv('annotations.csv')
pickle_in = open("original.pickle", "rb")

data = df[["x1","y1","x2","y2"]]

X = pickle.load(pickle_in)
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
'''

for i in range(0,38):
	img_string = df["image"][i]
	print(img_string)
	num = int([i for i in img_string if i in "0123456789"][0])

	img_size = 128
	x2 = data["x2"][i]
	x1 = data["x1"][i]
	y2 = data["y2"][i]
	y1 = data["y1"][i]
	print(x1,y1,x2,y2)
	deltax = (img_size - (x2 - x1)) / 2
	deltay = (img_size - (y2 - y1)) / 2
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

	X[num] = np.fliplr(X[num].reshape(-1,3)).reshape(X[num].shape)

	print("Showing image {0}".format(num))
	plt.imshow(X[0])
	plt.show()
	exit()

	X_copy = X[num].copy()
	X[num] = X[num][y1:y2, x1:x2]

	X[num] = np.fliplr(X[num].reshape(-1,3)).reshape(X[num].shape)
	plt.imshow(X[num])
	plt.show()
	exit()
	img_arr.append(X[-1 * i])


fig = plt.figure(figsize=(8, 8))
columns = 6
rows = 6
for i in range(1, columns*rows + 1):
    img = X[i - 1]
    fig.add_subplot(rows, columns, i)
    try:
    	plt.imshow(img)
    except:
    	pass

plt.show()'''