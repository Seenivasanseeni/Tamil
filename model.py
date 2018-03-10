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

        
        # placehoders
        self.image=tf.placeholder(dtype=tf.float32,shape=[None,self.image_size*self.image_size],name="self.image_input")
        self.label=tf.placeholder(dtype=tf.float32,shape=[None,num_characters])
        
        #layer1
        layer0=tf.layers.dense(self.image,units=num_characters,activation=tf.nn.relu)
        
        layer1=tf.layers.dense(layer0,units=1024,activation=tf.nn.relu)
        
        layer2=tf.layers.dense(layer1,units=512,activation=tf.nn.relu)
        
        layer3=tf.layers.dense(layer2,units=self.num_characters,activation=tf.nn.relu)
        
        logits=tf.nn.softmax(layer3)

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
