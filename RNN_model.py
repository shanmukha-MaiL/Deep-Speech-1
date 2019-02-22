"""REFERENCES : 1)https://arxiv.org/abs/1412.5567 ----- DEEP_SPEECH_1
                2)https://github.com/chagge/DeepSpeech-1
"""

import os
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import ctc_ops
from math import ceil
from collections import OrderedDict
from xdg import BaseDirectory as xdg
import importlib
from Text_preprocess import wer,sparse_tensor_value_to_texts

"""Dataset path and importer"""
ds_importer_module = importlib.import_module('Dataset_importer')
data_set_path = '/home/shanmukha/AnacondaProjects/Spyder_projects/DEEP_SPEECH_1'

"""VARIABLES"""
epochs = 75
dropout_rate = 0.05
dropout_1 = dropout_2 = dropout_3 = dropout_5 = dropout_rate
dropout_4_fw = dropout_4_bw = 0.00
dropout_rates = [dropout_1,dropout_2,dropout_3,dropout_4_fw,dropout_4_bw,dropout_5]
no_dropout = [0.0]*6
#fixing the upper bound for Relu
relu_bound = 20
#for optimizers
learning_rate = 0.001
momentum = 0.99
#batch sizes
train_batch_size = 10
validation_batch_size = 10
test_batch_size = 10
#phrases to print in the WER report
phrase_count = 10
#random initializers for all variables and default standard deviation
random_seed = 4567
std_dev = 0.046875
#no. of features
n_inputs = 26
#no. of contexts
n_contexts = 9
#no. of units in hidden layers
cell_dim = 2048
hidden_layer_1 = 2048
hidden_layer_2 = 2048
hidden_layer_3 = 2*cell_dim
"""hidden layer 4 is a BiRNN"""
hidden_layer_5 = 2048
#no. of characters in english including blank space 
n_characters = 29
output_layer = n_characters
#displaying,validating and checkpointing steps
display_step = 1
validation_step = 0
ckpt_step = 5
#checkpoints
ckpt_dir = xdg.save_data_path('DEEP_SPEECH_1')
#whether to restore ckpt while training
restore_ckpt = False
#session
session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

"""using cpu for creating variables"""
def cpu_variable(name,shape,initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name,shape,initializer=initializer)
    return var

"""RNN model"""
def rnn_model(x_batch,seq_length,dropout):
    #input = [batch_size,n_steps,n_inputs+2*n_inputs*n_contexts]
    x_batch_shape = tf.shape(x_batch)
    #reshaping input to [batch_size*n_steps,n_inputs+2*n_inputs*n_contexts]
    x_batch = tf.transpose(x_batch,[1,0,2])
    x_batch = tf.reshape(x_batch,[-1,n_inputs+2*n_inputs*n_contexts])
    #hidden layers
    b1 = cpu_variable('b1',[hidden_layer_1],tf.random_normal_initializer(stddev=std_dev))
    w1 = cpu_variable('w1',[n_inputs+2*n_inputs*n_contexts,hidden_layer_1],tf.random_normal_initializer(stddev=std_dev))
    l1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(x_batch,w1),b1)),relu_bound)
    l1 = tf.nn.dropout(l1,1-dropout_rates[0])
    
    b2 = cpu_variable('b2',[hidden_layer_2],tf.random_normal_initializer(stddev=std_dev))
    w2 = cpu_variable('w2',[hidden_layer_1,hidden_layer_2],tf.random_normal_initializer(stddev=std_dev))
    l2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(l1,w2),b2)),relu_bound)
    l2 = tf.nn.dropout(l2,1-dropout_rates[1])
    
    b3 = cpu_variable('b3',[hidden_layer_3],tf.random_normal_initializer(stddev=std_dev))
    w3 = cpu_variable('w3',[hidden_layer_2,hidden_layer_3],tf.random_normal_initializer(stddev=std_dev))
    l3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(l2,w3),b3)),relu_bound)
    l3 = tf.nn.dropout(l3,1-dropout_rates[2])
    #forward and backward direction LSTM cell
    fw_lstm = rnn_cell.BasicLSTMCell(cell_dim)
    fw_lstm = rnn_cell.DropoutWrapper(fw_lstm,1-dropout_rates[3],1-dropout_rates[3],seed=random_seed)
    bw_lstm = rnn_cell.BasicLSTMCell(cell_dim)
    bw_lstm = rnn_cell.DropoutWrapper(bw_lstm,1-dropout_rates[4],1-dropout_rates[4],seed=random_seed)
    #reshaping l3 to [max_time,batch_size,n_inputs]
    l3 = tf.reshape(l3,[-1,x_batch_shape[0],hidden_layer_3])
    l4,l4_states = tf.nn.bidirectional_dynamic_rnn(fw_lstm,bw_lstm,l3,seq_length,dtype=tf.float32)
    #concatenate forward and backward outputs
    l4 = tf.concat(l4,2)
    #reshape l4 into [max_time*batch_size,inputs]
    l4 = tf.reshape(l4,[-1,hidden_layer_3])
    #hidden layers 
    b5 = cpu_variable('b5',[hidden_layer_5],tf.random_normal_initializer(stddev=std_dev))
    w5 = cpu_variable('w5',[hidden_layer_3,hidden_layer_5],tf.random_normal_initializer(stddev=std_dev))
    l5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(l4,w5),b5)),relu_bound)
    l5 = tf.nn.dropout(l5,1-dropout_rates[5])
    
    b6 = cpu_variable('b6',[output_layer],tf.random_normal_initializer(stddev=std_dev))
    w6 = cpu_variable('w6',[hidden_layer_5,output_layer],tf.random_normal_initializer(stddev=std_dev))
    output = tf.minimum(tf.nn.relu(tf.add(tf.matmul(l5,w6),b6)),relu_bound)
    #reshaping output to [n_steps,batch_size,n_inputs]
    output = tf.reshape(output,[-1,x_batch_shape[0],output_layer])
    return output

"""Loss metrics and beam search"""
def calc_metrics(batch_set,dropout):
    batch_x,seq_length,batch_y = batch_set.next_batch()
    logits = rnn_model(batch_x,tf.to_int64(seq_length),dropout)
    #CTC loss
    total_loss = ctc_ops.ctc_loss(batch_y,logits,seq_length)
    avg_loss = tf.reduce_mean(total_loss)
    #beam search decoder
    decoded,log_prob = ctc_ops.ctc_beam_search_decoder(logits,seq_length,merge_repeated=False)
    #compute edit distance
    distance = tf.edit_distance(tf.cast(decoded[0],tf.int32),batch_y)
    accuracy = tf.reduce_mean(distance)
    return total_loss,avg_loss,distance,accuracy,decoded,batch_y

"""Adam optimizer"""
def create_optimizer():
    optimizer = tf.train.AdamOptimizer(learning_rate)
    return optimizer

available_devices = ['/cpu:0']

"""Calculating gradients and grouping required results"""
def calc_results(batch_set,optimizer=None):
    labels = []
    decodings = []
    total_losses = []
    avg_losses = []
    gradients = []
    distances = []
    accuracies = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(len(available_devices)):
            with tf.device(available_devices[i]):
                with tf.name_scope('cpu') :
                    total_loss,avg_loss,distance,accuracy,decoded,label = calc_metrics(batch_set,no_dropout if optimizer is None else dropout_rates)
                    tf.get_variable_scope().reuse_variables()
                    labels.append(label)
                    decodings.append(decoded)
                    total_losses.append(total_loss)
                    avg_losses.append(avg_loss)
                    distances.append(distance)
                    accuracies.append(accuracy)
                    gradient = optimizer.compute_gradients(avg_loss)
                    gradients.append(gradient)
    return (labels,decodings,distances,total_losses),gradients,tf.reduce_mean(accuracies,0),tf.reduce_mean(avg_losses,0)

"""Calculating average gradients"""
def calc_avg_gradient(gradients):
    avg_grads = []
    for grad_and_var in zip(*gradients):
        grads = []
        for g,_ in grad_and_var:
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)
        grad = tf.concat(grads,0)
        grad = tf.reduce_mean(grad,0)
        grad_and_var = (grad,grad_and_var[0][1])
        avg_grads.append(grad_and_var)
    return avg_grads

"""Applying gradients"""
def apply_gradient(avg_grads,optimizer):
    applied_grad = optimizer.apply_gradients(avg_grads)
    return applied_grad

    
                                     
"""WER report"""
def calc_wer(caption,result_tuple):
    #results_tuple = ([labels],[decodings],[distances],[losses])
    items = zip(*result_tuple)
    count = len(items)
    mean_wer = 0.0
    for i in range(count):
        item = items[i]
        #checking the distance
        if item[2]>0:
            item = (item[0],item[1],item[2],item[3])
            #insert wer
            item = (item[0],item[1],wer(item[0],item[1]),item[3])
            items[i] = item
            mean_wer += item[2]
    mean_wer = mean_wer/count
    #removing outputs whose wer=0
    items = [a for a in items if a[2]>0]
    #sorting in ascending order of losses
    items.sort(key=lambda a: a[3])
    #first 'phrase_count' phrases to be appeared in the report
    items = items[:phrase_count]
    #sort them in ascending order of wer
    items.sort(key=lambda a:a[2])
    #printing report
    print("WER report :")
    print(caption)
    for a in items:
        print("-"*80)
        print("WER = ",a[2])
        print("Loss = ",a[3])
        print("Source = ",a[0])
        print("Result = ",a[1])
    return mean_wer

"""converting tensors to texts"""    
def collecting_results(results_tuple,returns):
    for i in range(len(available_devices)):
        results_tuple[0].extend(sparse_tensor_value_to_texts(returns[0][i]))
        results_tuple[1].extend(sparse_tensor_value_to_texts(returns[1][i][0]))
        results_tuple[2].extend(returns[2][i])
        results_tuple[3].extend(returns[3][i])

"""Reading datasets"""
def read_data_sets(data_set_names):
    data_sets = ds_importer_module.read_data_sets(data_set_path,train_batch_size,validation_batch_size,test_batch_size,n_inputs,n_contexts,sets=data_set_names)
    return data_sets

def read_particular_set(data_set_name):
    data_sets = read_data_sets([data_set_name])
    data_set = getattr(data_sets,data_set_name)
    return data_set

"""Creating execution context"""
def create_context(data_set_name):
    #context is (graphs,results,data_set,saver)
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(random_seed)
        data_set = read_particular_set(data_set_name)
        is_train = data_set_name == 'train'
        optimizer = create_optimizer() if is_train else None
        results = calc_results(data_set,optimizer)
        if is_train:
            avg_grads = calc_avg_gradient(results[1])
            applied_grad = apply_gradient(avg_grads,optimizer)
        saver = tf.train.Saver(tf.global_variables())
        
        if is_train:
            return (graph,data_set,results,saver,applied_grad)
        else:
            return (graph,data_set,results,saver)
        
"""Starting context or restoring context"""
def start_context(context,model_path=None):
    graph = context[0]
    session = tf.Session(graph=graph,config=session_config)
    with graph.as_default():
        if model_path is None:
            session.run(tf.global_variables_initializer())
        else:
            context[3].restore(session,model_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session,coord)
        threads = threads + context[1].start_queue_threads(session,coord)
    return session,coord,threads

"""Saving model incase of interruption"""
def save_model(context,session,ckpt_path,global_step):
    return context[3].save(session,ckpt_path,global_step=global_step)

"""Stopping context"""
def stop_context(context,session,coord,threads,ckpt_path=None,global_step=None):
    #set hibernation path as None
    hibernation_path = None
    if ckpt_path is not None and global_step is not None:
        hibernation_path = save_model(context,session,ckpt_path,global_step)
    #stop and join all queue threads
    context[1].close_queue(session)
    coord.request_stop()
    coord.join(threads)
    session.close()
    return hibernation_path
    
"""Calculating batch loss and report"""
def calc_loss_and_report(context,session,epoch=-1,to_report=False):
    #negative epoch means no training
    is_train = epoch >= 0
    if is_train:
        graph,data_set,results,saver,applied_grad = context
    else:
        graph,data_set,results,saver = context
    result_tuple,gradients,avg_accuracy,avg_loss = results
    batches_per_device = ceil(float(data_set.total_batches)/len(available_devices))
    print('Total Batches = ',data_set.total_batches)
    parameters = OrderedDict()
    total_loss = 0.0
    parameters['avg_loss'] = avg_loss
    if is_train:
        parameters['applied_grad'] = applied_grad
    if to_report:
        total_accuracy = 0.0
        report_results = ([],[],[],[])
        parameters['sample_results'] = [result_tuple,avg_accuracy]
    parameter_id = dict(zip(parameters.keys(),range(len(parameters))))
    parameters = list(parameters.values())
    for batch in range(int(batches_per_device)):
        extra_params = { }
        if is_train :
            loss_run_metadata            = tf.RunMetadata()
            #extra_params['options']      = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            extra_params['run_metadata'] = loss_run_metadata

        compute = session.run(parameters,**extra_params)
        total_loss += compute[parameter_id['avg_loss']]
        if to_report:
            sample_results = compute[parameter_id['sample_results']]
            collecting_results(report_results,sample_results[0])
            total_accuracy += sample_results[1]
    loss = total_loss/batches_per_device
    if to_report:
        return (loss,total_accuracy/batches_per_device,report_results)
    else:
        return (loss,None,None)
    
"""printing report"""
def print_report(caption,loss_report):
    loss,accuracy,result_tuple = loss_report
    title = caption + 'Loss = ' + loss
    mean_wer = None
    if accuracy is not None and result_tuple is not None:
        title += 'Accuracy = ' + accuracy
        mean_wer = calc_wer(title,result_tuple)
    else:
        print(title)
    return mean_wer

"""running sets based on context"""
def run_set(data_set_name,model_path=None,to_report=False):
    context = create_context(data_set_name)
    session,coord,threads = start_context(context,model_path)
    loss_report = calc_loss_and_report(context,session,to_report=to_report)
    stop_context(context,session,coord,threads)
    return loss_report

"""Training function"""
def train():
    train_context = create_context('train')
    train_wer = 0.0
    validation_wer = 0.0
    start_epoch = 0
    hibernation_path = None
    if restore_ckpt:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            hibernation_path = ckpt.model_checkpoint_path
            start_epoch = int(ckpt.model_checkpoint_path.split('-')[-1])
            print('Resuming from epoch ',start_epoch+1)
            
    for epoch in range(start_epoch,epochs):
        print('Starting epoch: ',epoch)
        if epoch == 0 or hibernation_path is not None:
            if hibernation_path is not None:
                print('Resuming training from ',hibernation_path)    
            session,coord,threads = start_context(train_context,hibernation_path)
        hibernation_path = None    
        is_validation_step = validation_step > 0 and (epoch + 1) % validation_step == 0
        is_checkpoint_step = (ckpt_step > 0 and (epoch + 1) % ckpt_step == 0) or epoch == epochs - 1
        print('Training starts ------')
        result = calc_loss_and_report(train_context,session,epoch)
        result = print_report('Training metrics ',result)
        if result is not None:
            train_wer = result
        if is_checkpoint_step and not is_validation_step:
            print('Hibernating training session into directory',ckpt_dir)
            ckpt_path = os.path.join(ckpt_dir, 'model.ckpt')
            save_model(train_context, session, ckpt_path, epoch)
        # Validation step
        if is_validation_step:
            print('Hibernating training session into directory',ckpt_dir)
            ckpt_path = os.path.join(ckpt_dir, 'model.ckpt')
            hibernation_path = stop_context(train_context, session, coord,threads, ckpt_path, global_step=epoch)
            print('Validation starts ------' ) 
            result = run_set('validation', model_path=hibernation_path, to_report=True)
            result = print_report("Validation metrics ", result)
            if result is not None:
                 validation_wer = result
        print('Epoch ',epoch,' finished.')
        if hibernation_path is None:
            hibernation_path = stop_context(train_context,session,coord,threads,ckpt_path,epoch)
    return train_wer,validation_wer,hibernation_path
    
if __name__ == '__main__':
    train_wer,validation_wer,hibernation_path = train()
    result = run_set('test',hibernation_path,to_report=True)
    test_wer = print_report('Testing metrics ',result)
    
        
        
    
    
            
