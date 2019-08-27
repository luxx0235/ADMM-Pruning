import numpy as np 
import tensorflow as tf 

class PruneConfiguration():
    P1 = 80
    P2 = 92
    P3 = 50

configuration = PruneConfiguration()

target_w = ['fc1/weights1','fc2/weights2','fc3/weights3']
prune_percent = {'fc1/weights1':configuration.P1,'fc2/weights2':configuration.P2,'fc3/weights3':configuration.P3}

def get_configuration():
    return configuration

def projection(weight_arr, percent = 10):
    pcen = np.percentile(abs(weight_arr),percent)
    print('percentile'+str(pcen))
    under_threshold = abs(weight_arr) < pcen
    weight_arr[under_threshold] = 0

    return weight_arr


def prune_weight(weight_arr,weight_name):

    #percent = prune_percent[weight_name]
    #pcen = np.percentile(abs(weight_arr),percent)
    #print('percentile'+str(pcen))
    #under_threshold = abs(weight_arr) < pcen
    #weight_arr[under_threshold] = 0
    above_threshold = weight_arr != 0

    #return [above_threshold,weight_arr]
    return above_threshold


def apply_prune(dense_w,sess):
    dict_nzidx = {}
    for target_name in target_w:
        weight_arr = sess.run(dense_w[target_name])
        print('before pruning #non zero parameters'+str(np.sum(weight_arr!=0)))
        #before = np.sum(weight_arr!=0)
        #mask, weight_arr_pruned = prune_weight(weight_arr,target_name)
        mask = prune_weight(weight_arr,target_name)
        #after = np.sum(weight_arr_pruned != 0)
        #print ("pruned "+ str(before-after))

        #print ("after prunning #non zero parameters " + str(np.sum(weight_arr_pruned!=0)))
        #sess.run(dense_w[target_name].assign(weight_arr_pruned))
        dict_nzidx[target_name] = mask
    
    return dict_nzidx


def apply_prune_on_grads(grads_and_vars,dict_nzidx):

    for key,nzidx in dict_nzidx.items():
        count = 0 
        for grad, var in grads_and_vars:
            if var.name == key+':0':
                nzidx_obj = tf.cast(tf.constant(nzidx),tf.float32)
                grads_and_vars[count] = (tf.multiply(nzidx_obj,grad),var)
            
            count = count + 1
    
    return grads_and_vars

