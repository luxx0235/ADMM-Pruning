import tensorflow as tf 
import numpy as np 

class AdmmSolver():
    def __init__(self,model):
        A = self.A = tf.placeholder(tf.float32, shape = [9,15])
        B = self.B = tf.placeholder(tf.float32, shape = [9,15])
        C = self.C = tf.placeholder(tf.float32, shape = [15,21])
        D = self.D = tf.placeholder(tf.float32, shape = [15,21])
        E = self.E = tf.placeholder(tf.float32, shape = [21,2])
        F = self.F = tf.placeholder(tf.float32, shape = [21,2])

        weights1 = model.weights1
        weights2 = model.weights2
        weights3 = model.weights3

        loss = model.loss

        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(0.001).minimize(loss+0.00005*(tf.nn.l2_loss(weights1)+
                                                                            tf.nn.l2_loss(weights2)+tf.nn.l2_loss(weights3)))
            
            train_step1 = tf.train.AdamOptimizer(0.001).minimize(loss+0.00005*(tf.nn.l2_loss(weights1)+
                                                                            tf.nn.l2_loss(weights2)+tf.nn.l2_loss(weights3)) +
                                                                            0.0001*(tf.nn.l2_loss(weights1-A+B)+tf.nn.l2_loss(weights2-C+D)
                                                                            +tf.nn.l2_loss(weights3-E+F)))
            self.train_step = train_step
            self.train_step1 = train_step1


def create_admm_solver(model):
    return AdmmSolver(model)