import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.misc
from skimage.color import rgb2gray
from skimage.transform import resize
import sys

#root directory
root_directory=sys.argv[1]
copy_root_directory="Pkl/"
#make Pkl direcory
os.makedirs(copy_root_directory,exist_ok=True)
#get all the user_directory in the directory
users_directory=os.listdir(root_directory)

#userd directory
print(users_directory)

#image characterist
image_size=100
pixel_depth=255.0
num_characters=247

def hotfixLabel(n):
	#print(n,num_characters)
	label=[0]*num_characters
	label[n]=1
	return label


total_captured=0;
flag=True
for user in users_directory:
	print("Processing Directory:"+str(user))
	# define images and labels
	images=[]
	labels=[]

	#extract label and image from the files in uset_direcotory
	for file in os.listdir(root_directory+"/"+user+"/"):
		#check for other types of files excluding .tiff images
		#print(file)
		try:
			#do a dummy operation that would exec error on notOkayfiles
			label=int(file[:3])
		except:
			continue
		file_path=root_directory+"/"+user+"/"+file
		print("File Name:"+str(file_path))
		image=(plt.imread(file_path)-pixel_depth/2)/pixel_depth
		if(not(file[-3:]=="png")):
			image=image[:,:,:3] # remove alpha channel
		image=rgb2gray(image) # remove rgb traces
		#no need as normalixation depends on the whole dataset and it can be done in trainer side
		#image=(image-pixel_depth/2)/pixel_depth # do normalization
		image=resize(image,(image_size,image_size)) #resize image to image_size,image_size
		#print(np.array(image).shape)
		images.append(image)
		labels.append(hotfixLabel(label))
		total_captured+=1
		print("Count:",total_captured)
		#plt.imshow(image)
		#plt.show()
	images=np.array(images)
	labels=np.array(labels)
	save={
	"images":images,
	"labels":labels
	}
	pickel_file=copy_root_directory+user+".pkl"
	file_p=open(pickel_file,'wb')
	pickle.dump(save,file_p)
	file_p.close()
	print("Saved in ",pickel_file)
		
print("Whole User Dataset Dumped Suceesfully")
