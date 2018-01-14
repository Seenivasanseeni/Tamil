import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import os
from collections import deque
#image parameters
image_size=100
num_characters=247

#variables
W=tf.Variable(np.zeros(shape=(image_size*image_size,num_characters)),dtype=tf.float32)
b=tf.Variable(np.zeros(shape=[num_characters]),dtype=tf.float32)

# placehoders
image=tf.placeholder(dtype=tf.float32,shape=[None,image_size*image_size],name="image_input")
label=tf.placeholder(dtype=tf.float32,shape=[None,num_characters])

#input layer
input_layer=image

#a single output layer y=w*x+b
output_layer=tf.matmul(input_layer,W)+b

logits=output_layer

prediction=tf.nn.softmax(logits)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=logits))

learning_rate=0.03

optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)
accuracy=tf.reduce_sum(tf.cast(tf.argmax(label)==tf.argmax(prediction),tf.float32))*100
init=tf.global_variables_initializer()

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


with tf.Session() as sess:
	sess.run(init)
	iters=[]
	loss_s=[]
	num_iterations=10
	for iter in range(num_iterations):
		train_images,train_labels=load_next_batch()
		for inner_iter in range(10):
			_,loss_c=sess.run([optimizer,loss],feed_dict={image:train_images,label:train_labels})
			print("loss:",loss_c)
			iters.append(iter*inner_iter)
			loss_s.append(loss_c)
		print("===============================")
		
	plt.plot(iters,loss_s)
	plt.savefig("Train")
	plt.show()
