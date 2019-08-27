from model_prune_test import create_model
from prune_solver_test import create_admm_solver
from prune_utility_test import apply_prune_on_grads,apply_prune,get_configuration,projection
import tensorflow as tf
import numpy as np
from numpy import linalg as LA
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


prune_configuration = get_configuration()

dense_w = {}
P1 = prune_configuration.P1
P2 = prune_configuration.P2
P3 = prune_configuration.P3


#prepare data
df = pd.read_csv('cancer.csv',header=None)

#replace the y: 2->0, 4->1
df.iloc[:,10].replace(2, 0,inplace=True)
df.iloc[:,10].replace(4, 1,inplace=True)

#eliminate all rows that hold missing values
df = df[~df[6].isin(['?'])]
df = df.astype(float)
#data normalization
names = df.columns[0:10]
scaler = MinMaxScaler() 
scaled_df = scaler.fit_transform(df.iloc[:,0:10]) 
scaled_df = pd.DataFrame(scaled_df, columns=names)

scaled_df[10]= df[10]
x=scaled_df.iloc[0:500,1:10].values
y=df.iloc[0:500,10:].values

y = np.array(y)
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoded = one_hot_encoder.fit_transform(y)


x = np.array(x)
y = one_hot_encoded


xval=scaled_df.iloc[501:683,1:10].values.transpose()
yval=df.iloc[501:683,10:].values.transpose()




def main(X_,Y_):

    model = create_model()
    x = model.x
    y = model.y
    loss = model.loss
    layers = model.layers
    y_pred = model.y_pred
    solver = create_admm_solver(model)
    train_step = solver.train_step
    train_step1 = solver.train_step1

    weights1 = model.weights1
    weights2 = model.weights2
    weights3 = model.weights3

    A = solver.A
    B = solver.B
    C = solver.C
    D = solver.D
    E = solver.E
    F = solver.F

    my_trainer = tf.train.AdamOptimizer(0.001)
    grads = my_trainer.compute_gradients(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(10000):
            feed_dict = {x:X_,y:Y_}
            train_step.run(feed_dict=feed_dict)
            l = sess.run(loss,feed_dict=feed_dict)
            print(l)
        

        Z1 = sess.run(weights1)
        Z1 = projection(Z1,percent=P1)

        U1 = np.zeros_like(Z1)

        Z2 = sess.run(weights2)
        Z2 = projection(Z2,percent=P2)

        U2 = np.zeros_like(Z2)

        Z3 = sess.run(weights3)
        Z3 = projection(Z3,percent=P3)
        
        U3 = np.zeros_like(Z3)

        for j in range(30):
            for i in range(3000):
                feed_dict = {x:X_,y:Y_,A:Z1,B:U1,C:Z2,D:U2,E:Z3,F:U3}
                train_step1.run(feed_dict=feed_dict)
                l = sess.run(loss,feed_dict=feed_dict)
                print(l)

            Z1 = sess.run(weights1) + U1
            Z1 = projection(Z1,percent=P1)

            U1 = U1 + sess.run(weights1) - Z1

            Z2 = sess.run(weights2) + U2
            Z2 = projection(Z2,percent=P2)

            U2 = U2 + sess.run(weights2) - Z2

            Z3 = sess.run(weights3) + U3
            Z3 = projection(Z3,percent=P3)

            U3 = U3 + sess.run(weights3) - Z3

            print(LA.norm(sess.run(weights1) - Z1))
            print(LA.norm(sess.run(weights2) - Z2))
            print(LA.norm(sess.run(weights3) - Z3))
        

        sess.run(weights1.assign(Z1))
        sess.run(weights2.assign(Z2))
        sess.run(weights3.assign(Z3))

        dense_w['fc1/weights1'] = weights1
        dense_w['fc2/weights2'] = weights2
        dense_w['fc3/weights3'] = weights3

        dict_nzidx = apply_prune(dense_w,sess)
        print (dict_nzidx.keys())
        grads = apply_prune_on_grads(grads,dict_nzidx)
        apply_gradient_op = my_trainer.apply_gradients(grads)

        for var in tf.global_variables():
            if tf.is_variable_initialized(var).eval() == False:
                sess.run(tf.variables_initializer([var]))
        
        for i in range(10000):
            feed_dict = {x:X_,y:Y_}
            apply_gradient_op.run(feed_dict=feed_dict)
            l = sess.run(loss,feed_dict=feed_dict)
            print(l)
        
        
        saver = tf.train.Saver()
        saver.save(sess,"3fc_pruned_model.ckpt")
        print('saved')
        print(sess.run(weights3))
        

main(x,y)








            

