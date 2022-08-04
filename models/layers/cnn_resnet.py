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
from tensorflow.keras.layers import AveragePooling1D
from .residual_blocks import Cnn_residual_block

class Cnn_resnet(tf.keras.layers.Layer):
    def __init__(self, pooling = True, name = None, architecture = [(16, 2, False, 0.0),
                                                        (32, 2, False, 0.0),
                                                        (32, 2, False, 0.0),
                                                        (32, 2, True, 0.0),
                                                        (32, 2, True, 0.0),
                                                        (64, 3, True, 0.0),
                                                        (128, 4, True, 0.0),
                                                        (128, 4, True, 0.0)]):
        super(Cnn_resnet, self).__init__(name = name)
        
        self.num_layers = len(architecture)
        self.residuals = []
        for a in architecture:
            self.residuals.append(Cnn_residual_block(filters=a[0], conv_num=a[1], pooling=a[2], dropout_rate = a[3]))

        self.pooling = pooling
        if pooling:
            self.ap = AveragePooling1D(pool_size=3, strides=3)

    def call(self, x):
        for i in range(self.num_layers):
            x = self.residuals[i](x)
        if self.pooling:
            x = self.ap(x)
        return x

def main():
    cr = Cnn_resnet()(tf.random.uniform((64, 900, 2048)))
    print(cr.shape)
    

if __name__ == '__main__':
    main()   