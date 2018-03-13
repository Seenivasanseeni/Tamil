import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.misc
from skimage.color import rgb2gray
from skimage.transform import resize
import sys
from tools import get

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
image_size=int(get("image_size"))
num_characters=int(get("num_characters"))

def hotfixLabel(n):
#print(n,num_characters)
    label=[0]*num_characters
    label[n]=1
return label

def isimageneeded(label):
    if(label<=num_characters):
        return True
    return False

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
			label=int(file[:3]);
            if(not isimageneeded(label)):
                continue #change
		except:
			continue
		file_path=root_directory+"/"+user+"/"+file
		print("File Name:"+str(file_path))
		try:
			image=plt.imread(file_path)
		except:
			print("Invalid Image")
			continue

		if(not(file[-3:]=="png")):
			image=image[:,:,:3] # remove alpha channel
		image=rgb2gray(image) # remove rgb traces

		image=resize(image,(image_size,image_size)) #resize image to image_size,image_size

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
