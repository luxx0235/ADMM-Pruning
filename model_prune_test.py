import tensorflow as tf 
import numpy as np 

class model():
    def __init__(self):
        self.layers = ['fc1/weights1','fc2/weights2','fc3/weights3']
        self.x = tf.placeholder(tf.float32,[None,9])
        self.y = tf.placeholder(tf.float32,[None,2])

        with tf.name_scope('fc1'):
            self.weights1 = tf.Variable(tf.random_normal([9,15],stddev=0.1),name = 'weights1')
            bias1 = tf.Variable(tf.constant(0.1,shape = [15]),name = 'bias1')

            fc1 = tf.nn.sigmoid(tf.matmul(self.x,self.weights1)+bias1)
        
        with tf.name_scope('fc2'):
            self.weights2 = tf.Variable(tf.random_normal([15,21],stddev=0.1),name = 'weights2')
            bias2 = tf.Variable(tf.constant(0.1,shape = [21]),name = 'bias2')

            fc2 = tf.nn.sigmoid(tf.matmul(fc1,self.weights2)+bias2)
        
        with tf.name_scope('fc3'):
            self.weights3 = tf.Variable(tf.random_normal([21,2],stddev = 0.1),name = 'weights3')
            bias3 = tf.Variable(tf.constant(0.1,shape = [1]),name = 'bias3')

            fc3 = tf.matmul(fc2,self.weights3)+bias3
        
        self.y_pred = fc3
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y,logits = self.y_pred))
        #vs = tf.trainable_variables()
        #for v in vs:
            #print(v)

def create_model():
    return model()

