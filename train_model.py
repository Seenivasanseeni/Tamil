import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import Pickle

tf.graph(name="Graph for Tamil Character Recognition")

#image parameters
image_size=100
num_characters=100

#variables
W=tf.Variable(tf.float32,shape=[tnum_characters,image_size*image_size])
b=tf.Variable(tf.float32,shape=[num_characters])

# placehoders
images=tf.placeholder(tf.float32,shape=[None,image_size*image_size],name="image input")
labels=tf.placeholder(tf.float32,shape=[None,num_characters])

#input layer
input_layer=images

#a single output layer y=w*x+b
output_layer=tf.add(tf.multiply(W,image),b)

logits=tf.nn.softmax(output_layer)

loss=tf.cross_entropy_with_logits(labels=labels,logits=logits)

learning_rate=0.03

optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)



init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	num_iterations=100
	for iter in range(num_iterations):
		train_image,train_labels=load_next_batch()
		_,loss=sess.run([optimizer,loss],feed_dict={images:train_image,labels:train_labels})
		print("Loss",loss)


