from scipy.stats import mode
import numpy as np
import numpy.random as npr
import cv2 as cv
import pandas as pd

##### finding clusters using mean 

def cl(data,z,rep):
	for i in range(len(z)):			# for each data point 
		dis = np.inf				# let default distance be inf 

		for j in range(len(rep)):	# finding the closest representative 
			if(np.linalg.norm(d[i]-rep[j])<dis):
				z[i] = j
				dis = np.linalg.norm(d[i]-rep[j])

##### finding mean using clusters 


def mean(data,z,rep):				# finding mean of the clusters 
	t = np.zeros_like(rep)			# temporary varaible to save sum 

	c= [0]*len(rep)					# number of occurance of each cluster 

	for i in range(len(z)):
		c[int(z[i])] += 1
		t[int(z[i])] += data[i]
	for i in range(len(rep)):		# finding mean 
		if(c[i]!=0):
			t[i] = t[i]/c[i]	
	return t

name = input("Enter complete name of image: ")

img = cv.imread(name)		# reading image file 

a = len(img)						# saving initial dimention of 
b = len(img[0])						# the image 
print(img.shape)

#cv.imshow('Z',img)		# showing intial image

img = img.reshape((a*b,3))			# reshaping image as an array of 
									# pixel values
d = [[0,0,0]]*a*b 					# similer array with int64 values 
d = np.array(d)

d += img


z = np.zeros_like(d.T[0])


k = int(input("Enter number of cluster"))		# number of required clusters 
r = []

### K-Mean driver function 

def driver(d,r,z):					# k mean finding function 
	
	rand = npr.randint(0,high= len(d),size=k)	# start with k randomly 
												# selected point in data set 
	for i in rand:		
		r.append(d[i])
	r = np.array(r)								# finding the initail representer set

	for i in range(50):							# run till the upper limit or till convergence
		cl(d,z,r)
		cur = mean(d,z,r)

		if((r-cur).any()==0):					# checking convergence 
			break 								# if new and last representer set 
		else:									# is same or not 
			r = cur		
	return z,r 									# returning representer and latent variable 

R = []								# to save result of multiple runs 

for i in range(1):					# running the k mean n-times 
	z = np.zeros_like(d.T[0])
	r = []
	z,r = driver(d,r,z)
	#print(r)
	R.append(r)						# saving the set of represeters 

R = np.array(R)						

K = []								# to save the sorted set of representers 

for i in range(len(R)):
	K = K + [sorted(R[i],key = lambda x: x[0])]

K = np.array(K)

r = mode(K)[0][0]					# taking the set with have maximux occurance 
z = np.zeros_like(d.T[0])			

cl(d,z,r)							# running clustering again to get the 
									# corresponding latent variable 
print(r)

#for i in range(len(z)):				# changing the data with representer values 


df = pd.DataFrame([a,b,k])
#print(r.reshape(len(r)*3))
df = df.append(list(r.reshape(len(r)*3)))
df = df.append(list(z))

df.to_csv(index=False)
compression_opts = dict(method='zip',archive_name='out.csv')  
df.to_csv('out.zip', index=False,compression=compression_opts)  

print("Succesfully converted")