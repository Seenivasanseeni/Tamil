import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import pickle


#image parameters
image_size=100
num_characters=156

#variables
W=tf.Variable(np.zeros(shape=(num_characters,image_size*image_size)),dtype=tf.float32)
b=tf.Variable(np.zeros(shape=[156,1]),dtype=tf.float32)

# placehoders
images=tf.placeholder(dtype=tf.float32,shape=[image_size*image_size,1],name="image_input")
labels=tf.placeholder(dtype=tf.float32,shape=[num_characters,1])

#input layer
input_layer=images

#a single output layer y=w*x+b
output_layer=tf.matmul(W,input_layer)+b

logits=output_layer

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))

learning_rate=0.03

optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)

init=tf.global_variables_initializer()

# prepare code to fetch datasets
pickle_file=

def load_next_batch():
	#return image
	image_r=np.zeros(shape=(image_size*image_size,1))
	label_r=np.zeros(shape=(num_characters,1))
	return image_r,label_r

with tf.Session() as sess:
	sess.run(init)
	num_iterations=100
	for iter in range(num_iterations):
		train_image,train_labels=load_next_batch()
		_,loss_c=sess.run([optimizer,loss],feed_dict={images:train_image,labels:train_labels})
		print("Loss",loss_c)


