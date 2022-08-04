#Copyright 2022 Hamidreza Sadeghi. All rights reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import pickle
from sklearn.utils import shuffle

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                 X,
                 Y,
                 bpet,
                 model,
                 save_path,
                 batch_size = 32,
                 shuffle_data = False):
        
        'Initialization'
        self.X = X
        self.Y = Y
        self.bpet = bpet
        self.model = model
        self.save_path = save_path
        self.shuffle_data = shuffle_data
        self.batch_size = batch_size
        
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        len__ = int(len(self.X) // self.batch_size)
        if len__*self.batch_size < len(self.X):
            len__ += 1
        return len__

    def __getitem__(self, index):
        'Generate one batch of data'
        if (index+1)*self.batch_size > len(self.X):
            start = index*self.batch_size
            end = len(self.X)
            batch_len = len(self.X) - index*self.batch_size
        else:
            start = index*self.batch_size
            end = (index+1)*self.batch_size
            batch_len = self.batch_size

        X_batch = self.X[start:end]
        if self.Y is not None:
            Y_batch = self.Y[start:end]

        
        max_seq_len1 = max([len(x) for x in X_batch]) + 1
        if self.Y is not None:
            max_seq_len2 = max([len(x) for x in Y_batch]) + 1

        X_in1 = np.full((batch_len,
                     max_seq_len1 #self.n_ctx
                     ), self.bpet.bpe2idx["__pad"])

        if self.Y is not None:
            X_in2 = np.full((batch_len,
                        max_seq_len2 #self.n_ctx
                        ), self.bpet.bpe2idx["__pad"])
            
            Y_out = np.zeros((batch_len, max_seq_len2 #self.n_ctx
                        , len(self.bpet.bpe2idx)))
        

        for i,x in enumerate(X_batch):
            for t,j in enumerate(x):
                X_in1[i,t] = j

        if self.Y is not None:
            for i,x in enumerate(Y_batch):
                for t,j in enumerate([self.bpet.bpe2idx["__she"]] + x):
                    X_in2[i,t] = j
                    if t > 0:
                        Y_out[i,t-1,j] = 1

            return [X_in1, X_in2], Y_out
        else:
            return X_in1

    def on_epoch_end(self):
        if self.Y is not None:
            if self.shuffle_data:
                self.X, self.Y = shuffle(self.X, self.Y)
        else:
            if self.shuffle_data:
                self.X = shuffle(self.X)
            
        if self.save_path is not None:
            self.model.save_weights(self.save_path + 'model_weights.h5')
            symbolic_weights = getattr(self.model.optimizer, 'weights')
            weight_values = K.batch_get_value(symbolic_weights)
            with open(self.save_path + 'optimizer.pkl', 'wb') as f:
                pickle.dump(weight_values, f)
