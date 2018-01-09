import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import pickle


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

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label,logits=logits))

learning_rate=0.03

optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)

init=tf.global_variables_initializer()

# prepare code to fetch datasets
pickle_file=open("Pickles/Dataset.pkl","rb")
save=pickle.load(pickle_file)
images=save["images"]
images=(images-images.mean())/images.std()
labels=save["labels"]
size=50
index=0
def load_next_batch():
	#return image
	global index
	image_r=images[index:index+size].reshape((-1,image_size*image_size))
	label_r=labels[index:index+size]
	index+=size
	if(index>len(images)):
		index=0
	return image_r,label_r

with tf.Session() as sess:
	sess.run(init)
	num_iterations=100
	for iter in range(num_iterations):
		train_images,train_labels=load_next_batch()
		_,loss_c=sess.run([optimizer,loss],feed_dict={image:train_images,label:train_labels})
		print("Loss",loss_c)


