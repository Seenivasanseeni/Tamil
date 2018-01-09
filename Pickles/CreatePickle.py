import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.misc
from skimage.color import rgb2gray
from skimage.transform import resize

#root directory
root_directory="tamil_dataset_offline"
#get all the user_directory in the directory
users_directory=os.listdir(root_directory)

# define images and labels
images=[]
labels=[]

#image characterist
image_size=100
pixel_depth=255.0
num_characters=247

def hotfixLabel(n):
	print(n,num_characters)
	label=[0]*num_characters
	label[n]=1
	return label


total_captured=0;
for user in users_directory:
	print("Processing Directory:"+str(user))
	#extract label and image from the files in usrt_direcotory
	for file in os.listdir(root_directory+"/"+user+"/"):
		#check for other types of files excluding .tiff images
		print(file)
		if(not (file[-5:]==".tiff")):
			continue
		file_path=root_directory+"/"+user+"/"+file
		print("File Name:"+str(file_path))
		image=(plt.imread(file_path)-pixel_depth/2)/pixel_depth
		image=image[:,:,:3] # remove alpha channel
		image=rgb2gray(image) # remove rgb traces
		image=(image-pixel_depth/2)/pixel_depth # do normalization
		image=resize(image,(image_size,image_size)) #resize image to image_size,image_size
		print(np.array(image).shape)
		images.append(image)
		labels.append(hotfixLabel(int(file[:3])))
		total_captured+=1
	if(total_captured>500):
		break
		
images=np.array(images)
labels=np.array(labels)
print("Images shape ",images.shape)
print("Labels shape ",labels.shape)
#let us print the mean and deviation
print("No of images:"+str(len(images)))
print("mean:"+str(np.mean(images)))
print("Standard deviation:"+str(np.std(images)))

#store the dataset in pickelfile
pickel_file="Dataset.pkl"
save={
	"images":images,
	"labels":labels
}

file_p=open(pickel_file,'wb')
pickle.dump(save,file_p)

print("Dataset Dumped Suceesfully")
