"""REFERENCES : 1)https://arxiv.org/abs/1412.5567 ----- DEEP_SPEECH_1
                2)https://github.com/chagge/DeepSpeech-1
"""

import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav

"""Function to convert .wav files to vectors(mfcc features)"""
def wav_to_vector(file_name,num_ceps,num_context):
    sample_rate,audio = wav.read(file_name)
    inputs = mfcc(audio,sample_rate,numcep=num_ceps)
    #stride = 2
    inputs = inputs[::2]
    #reshaping of input features
    train_inputs = np.array([],np.float32)
    train_inputs.resize([inputs.shape[0],num_ceps + 2*num_ceps*num_context])
    #empty feature
    empty_feature = np.array([])
    empty_feature.resize((num_ceps))
    #time slices
    time_slices = range(train_inputs.shape[0])
    min_past_context = time_slices[0] + num_context
    max_future_context = time_slices[-1] - num_context
    #preparing past and future contexts
    for time_slice in time_slices:
        #past
        empty_past_l = max(0,min_past_context - time_slice)
        empty_past = list(empty_feature for empty in range(empty_past_l))
        data_past = inputs[max(0,time_slice - num_context):time_slice]
        assert(len(empty_past) + len(data_past) == num_context)
        #future
        empty_future_l = max(0,time_slice - max_future_context)
        empty_future = list(empty_feature for empty in range(empty_future_l))
        data_future = inputs[time_slice +1:time_slice + num_context + 1]
        assert(len(empty_future) + len(data_future) == num_context)
        
        if empty_past_l:
            past = np.concatenate((empty_past,data_past))
        else:
            past = data_past
        if empty_future_l:
            future = np.concatenate((data_future,empty_future))
        else:
            future = data_future
        
        past = np.reshape(past,[num_ceps*num_context])
        present = inputs[time_slice]
        future = np.reshape(future,[num_ceps*num_context])
        train_inputs[time_slice] = np.concatenate((past,present,future))
        assert(len(train_inputs[time_slice]) == num_ceps + 2*num_ceps*num_context)
        #widen inputs
    train_inputs = (train_inputs-np.mean(train_inputs))/np.std(train_inputs)
    return train_inputs
    
    