import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.misc
from skimage.color import rgb2gray
from skimage.transform import resize
import sys
from tools import get

image_size=int(get("image_size"))
num_characters=int(get("num_characters"))

def hotfixLabel(n):
	#print(n,num_characters)
	label=[0]*num_characters
	label[n]=1
	return label


def load_test_data():
    user_directory="test_data"
    print("Test Directory:"+str(user_directory))
    # define images and labels
    images=[]
    labels=[]
    total_captured=0
    #extract label and image from the files in uset_direcotory
    for file in os.listdir(user_directory):
        #check for other types of files excluding .tiff images
        file_path=user_directory+"/"+file
        #print("File Name:"+str(file_path))
        try:
            #do a dummy operation that would exec error on notOkayfiles
            label=int(file[:3])
            if(label>=num_characters):
                #print("Ignoring File")
                continue
            #print("File can be used")
        except:
            continue
        try:
            image=plt.imread(file_path)
        except:
            #print("Invalid Image")
            continue

        if(not(file[-3:]=="png")):
            image=image[:,:,:3] # remove alpha channel
            image=rgb2gray(image) # remove rgb traces

        image=resize(image,(image_size,image_size)) #resize image to image_size,image_size

        images.append(image)
        labels.append(hotfixLabel(label))
        total_captured+=1
        #print("Count:",total_captured)
    images=np.array(images)
    labels=np.array(labels)
    images=images.reshape([-1,image_size*image_size])
    return images,labels

def process_image(file_path):
	print("File Name:"+str(file_path))
	try:
		image=plt.imread(file_path)
	except:
		raise Exception("Invalid Image")

	if(not(file_path[-3:]=="png")):
		image=image[:,:,:3] # remove alpha channel
		image=rgb2gray(image) # remove rgb traces

	image=resize(image,(image_size,image_size)) #resize image to image_size,image_size

	return image

def predict_data():
	input_file_path=input("\nEnter absolute  File Path:")
	image=[process_image(input_file_path)]
	image=np.array(image)
	image=image.reshape([-1,image_size*image_size])
	return image
