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



import tensorflow as tf
from tensorflow.keras.layers import  Conv1D, BatchNormalization, Dropout, MaxPool1D, Add, LeakyReLU

class Cnn_residual_block(tf.keras.layers.Layer):
    def __init__(self, filters, conv_num = 5, pooling = True, dropout_rate = 0.0, name = None):
        super(Cnn_residual_block, self).__init__(name = name)

        self.conv_num = conv_num
        self.pooling = pooling
        self.dropout_rate = dropout_rate

        self.conv1 = Conv1D(filters, 1, padding="same")
        self.conv2 = Conv1D(filters, 3, padding="same")
        self.conv_list = []
        self.batchNormalization_list = []
        self.activation_list = []
        self.dropout_lists = []
        for i in range(conv_num - 1):
            self.conv_list.append(Conv1D(filters, 3, padding="same"))
            self.batchNormalization_list.append(BatchNormalization())
            self.activation_list.append(LeakyReLU())
            if dropout_rate != 0.0:
                self.dropout_lists.append(Dropout(dropout_rate))

        self.add = Add()
        self.bn = BatchNormalization()
        self.mp = MaxPool1D(pool_size=2, strides=2)
        self.ler = LeakyReLU()

    def call(self, x):
        # Shortcut
        s = self.conv1(x)
        for i in range(self.conv_num - 1):
            x = self.conv_list[i](x)
            x = self.batchNormalization_list[i](x)
            x = self.activation_list[i](x)
            if self.dropout_rate != 0.0:
                x = self.dropout_lists(x)
        x = self.conv2(x)
        x = self.add([x, s])
        x = self.bn(x)
        x = self.ler(x)
        if self.pooling:
            return self.mp(x)
        else:
            return x



def main():
    crb = Cnn_residual_block(50)(tf.random.uniform((64, 43, 2048)))
    print(crb.shape)
    

if __name__ == '__main__':
    main()   