from scipy.stats import mode
import numpy as np
import numpy.random as npr
import cv2 as cv
import pandas as pd
inp = input("Enter name: ")
df = pd.read_csv(inp,compression = 'zip') #reading the zipped csv

data = np.array(df)		# converting dataframe to numpy array
data = data.T[0]		# converting into single vector 

a,b,k = data[0],data[1],data[2]		# information about length breath and cluster number 

data = data[3:]						# trimming the used part 
r = data[0:3*k]						# taking the rgb values of cluster mean 
r = r.reshape(int(len(r)/3),3)		# shapping as needed 

data = data[3*k:]					# trimming 

img = np.zeros((len(data),3))		# a base array to load the image 

for i in range(len(data)):			# loading each pixel value
	img[i] = r[int(data[i])]

img = img.reshape(a,b,3)			# reshaping into original

inp = inp[:-3]+'png'

cv.imwrite(inp,img)				# saving the image 
