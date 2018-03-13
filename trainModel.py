import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from collections import deque
import model
from tools import get
import Loader

#Specify which user is writing
user=int(sys.argv[1])

#image parameters
image_size=int(get("image_size"))
num_characters=int(get("num_characters"))

# prepare code to fetch datasets
users=deque(os.listdir("Pickles/Pkl"))
def load_next_batch():
    #return image
    file_c=users.popleft()
    print("Batch Name:",file_c)
    users.append(file_c)
    pickle_file="Pickles/Pkl/"+file_c
    pickle_file=open(pickle_file,"rb")
    save=pickle.load(pickle_file)
    image_r=save["images"].reshape([-1,image_size*image_size])
    label_r=save["labels"]
    if(len(label_r)==0):
        print("Error in loading .. Continuing for the nest iteration")
        return load_next_batch()
    return image_r,label_r

def make_test_data():
	test_pickle_file="Pickles/Pkl/usr_"+str(user)+".pkl"
	test_pickle_file=open(test_pickle_file,"rb")
	save=pickle.load(test_pickle_file)
	images=save["images"].reshape([-1,image_size*image_size])
	labels=save["labels"]
	return images,labels

def load_user_batch(user):
    user_pickle_file="Pickles/Pkl/usr_"+str(user)+".pkl"
    user_pickle_file=open(user_pickle_file,"rb")
    save=pickle.load(user_pickle_file)
    images=save["images"].reshape([-1,image_size*image_size])
    labels=save["labels"]

    if(len(images)==0):
        raise Exception("User "+user+" is empty. Please Specify another user")
    return images,labels

Mod=model.Model()
Mod.construct(image_size,num_characters)

step=[]
loss=[]
accuracy=[]

test_images,test_labels=Loader.load_test_data()

num_iterations=int(get("num_iterations"))

for iter in range(num_iterations):
    train_images,train_labels=load_user_batch(user)

    print("Training with ",len(train_labels))
    l,acc=Mod.train(train_images,train_labels)
    print("Testing it")
    print("Testing with ",len(test_labels))
    l,acc=Mod.test(test_images,test_labels)
    step.append(iter)
    loss.append(l)
    accuracy.append(acc)
    print("===============================")

plt.plot(step,loss,color="red")
plt.plot(step,accuracy,color="blue")
plt.savefig("Train")
