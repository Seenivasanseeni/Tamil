import tensorflow as tf
import numpy as np
class Model(object):
    def __init__(self):
        pass
    def construct(self,image_size,num_characters):
        ''' This method constructs the model based on the parametrs based '''
        
        #paraemeters for image and labels creation
        self.image_size=image_size
        self.num_characters=num_characters

        #variables
        self.weights=tf.Variable(np.zeros(shape=(self.image_size*self.image_size,num_characters)),dtype=tf.float32)
        self.biases=tf.Variable(np.zeros(shape=[num_characters]),dtype=tf.float32)

        # placehoders
        self.image=tf.placeholder(dtype=tf.float32,shape=[None,self.image_size*self.image_size],name="self.image_input")
        self.label=tf.placeholder(dtype=tf.float32,shape=[None,num_characters])

        #input layer
        input_layer=self.image

        #a single output layer y=w*x+b
        logits=tf.nn.softmax(tf.matmul(input_layer,self.weights)+self.biases)


        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=logits))

        self.accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.label,1),tf.argmax(logits,1)),tf.float32))
        self.learning_rate=0.5

        self.optimizer=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.sess=tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        
    def train(self,images,labels):
        ''' This method trains the model that is constructed using the cosntruct method'''
    
        _,l,acc=self.sess.run([self.optimizer,self.loss,self.accuracy],feed_dict={self.image:images,self.label:labels})
        print("Loss {} Accuaracy {}".format(l,acc))
        return l,acc
    def test(self,images,labels):
        ''' This method test the trained model using passed data'''
        acc=self.sess.run([self.accuracy],feed_dict={self.image:images,self.label:labels})
        print(" Accuaracy {}".format(acc))
        return acc
