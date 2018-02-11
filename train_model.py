import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import os
import sys
from collections import deque

user=sys.argv[1]

#image parameters
image_size=100
num_characters=247

#variables
weights=tf.Variable(np.zeros(shape=(image_size*image_size,num_characters)),dtype=tf.float32)
biases=tf.Variable(np.zeros(shape=[num_characters]),dtype=tf.float32)

# placehoders
image=tf.placeholder(dtype=tf.float32,shape=[None,image_size*image_size],name="image_input")
label=tf.placeholder(dtype=tf.float32,shape=[None,num_characters])

#input layer
input_layer=image

#a single output layer y=w*x+b
logits=tf.nn.softmax(tf.matmul(input_layer,weights)+biases)


loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=logits))

accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label,1),tf.argmax(logits,1)),tf.float32))
learning_rate=0.5

optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

print("Shapes")
print("weights",weights.get_shape())
print("biases",biases.get_shape())
print("image",image.get_shape())
print("label",label.get_shape())
print("logits",logits.get_shape())


init_g=tf.global_variables_initializer()
init_l=tf.local_variables_initializer()


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
	return image_r,label_r

def make_test_data():
	test_pickle_file="Pickles/Pkl/usr_"+user+".pkl"
	test_pickle_file=open(test_pickle_file,"rb")
	save=pickle.load(test_pickle_file)
	images=save["images"].reshape([-1,image_size*image_size])
	labels=save["labels"]
	return images,labels

sess=tf.InteractiveSession()
sess.run(init_g)
sess.run(init_l)
x=[]
y=[]

num_iterations=10
for iter in range(num_iterations):
	train_images,train_labels=load_next_batch()
	for inner_iter in range(10):
		_,loss_c,acc,log=sess.run([optimizer,loss,accuracy,logits],feed_dict={image:train_images,label:train_labels})
		print("loss:",loss_c)
		print("accuracy:",acc)
		x.append(iter*inner_iter)
		y.append(loss_c)
	print("===============================")
		
plt.plot(x,y)
plt.savefig("Train")
	