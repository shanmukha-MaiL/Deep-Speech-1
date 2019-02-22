"""REFERENCES : 1)https://arxiv.org/abs/1412.5567 ----- DEEP_SPEECH_1
                2)https://github.com/chagge/DeepSpeech-1
"""

import numpy as np
import tensorflow as tf
from functools import reduce

SPACE_TOKEN = '<space>'
SPACE_ID = 0
REF_ID = ord('a') - 1

"""function to convert strings to corresponding id arrays"""
def text_to_id(string):
    output = string.replace("'","")
    output = output.replace(" '","")
    output = output.replace(" ","  ")
    output = output.split(" ")
    #stacking all the characters
    output = np.hstack([SPACE_TOKEN if char == '' else list(char) for char in output])
    output = np.asarray([SPACE_ID if char == SPACE_TOKEN else ord(char)-REF_ID for char in output])
    return output

"""converting tuples back to text"""
def sparse_tensor_value_to_texts(value):
    return sparse_tuple_to_texts((value.indices, value.values, value.dense_shape))

def sparse_tuple_to_texts(tuple):
    indices = tuple[0]
    values = tuple[1]
    results = [''] * tuple[2][0]
    for i in range(len(indices)):
        index = indices[i][0]
        c = values[i]
        c = ' ' if c == SPACE_ID else chr(c + REF_ID)
        results[index] = results[index] + c
    return results

"""Calculating WER"""
def wer(original, result):
    original = original.split()
    result = result.split()
    return levenshtein(original, result) / float(len(original))

def levenshtein(a,b):
    n, m = len(a), len(b)
    if n > m:
        a,b = b,a
        n,m = m,n
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
    return current[n]

def gather_nd(params, indices, shape):
    rank = len(shape)
    flat_params = tf.reshape(params, [-1])
    multipliers = [reduce(lambda x, y: x*y, shape[i+1:], 1) for i in range(0, rank)]
    l = [j for j in range(0,rank-1)]
    indices_unpacked = tf.unstack(tf.transpose(indices, [rank - 1] + l ))
    flat_indices = sum([a*b for a,b in zip(multipliers, indices_unpacked)])
    return tf.gather(flat_params, flat_indices)

def ctc_label_dense_to_sparse(labels, label_lengths, batch_size):
    correct_shape_assert = tf.assert_equal(tf.shape(labels)[1], tf.reduce_max(label_lengths))
    with tf.control_dependencies([correct_shape_assert]):
        labels = tf.identity(labels)

    label_shape = tf.shape(labels)
    num_batches_tns = tf.stack([label_shape[0]])
    max_num_labels_tns = tf.stack([label_shape[1]])
    def range_less_than(previous_state, current_input):
        return tf.expand_dims(tf.range(label_shape[1]), 0) < current_input

    init = tf.cast(tf.fill(max_num_labels_tns, 0), tf.bool)
    init = tf.expand_dims(init, 0)
    dense_mask = tf.scan(range_less_than, label_lengths, initializer=init, parallel_iterations=1)
    dense_mask = dense_mask[:, 0, :]
    label_array = tf.reshape(tf.tile(tf.range(0, label_shape[1]), num_batches_tns),
          label_shape)
    label_ind = tf.boolean_mask(label_array, dense_mask)
    batch_array = tf.transpose(tf.reshape(tf.tile(tf.range(0, label_shape[0]), max_num_labels_tns), tf.reverse(label_shape, [0])))
    batch_ind = tf.boolean_mask(batch_array, dense_mask)
    indices = tf.transpose(tf.reshape(tf.concat([batch_ind, label_ind], 0), [2, -1]))
    shape = [batch_size, tf.reduce_max(label_lengths)]
    vals_sparse = gather_nd(labels, indices, shape)
    return tf.SparseTensor(tf.to_int64(indices), vals_sparse, tf.to_int64(label_shape))

"""Function for cleaning labels"""
def validate_label(label):
#    
    label = label.replace('-','')
    label = label.replace('_','')
    label = label.replace('.','')
    label = label.replace(',','')
    label = label.strip()
    return label
