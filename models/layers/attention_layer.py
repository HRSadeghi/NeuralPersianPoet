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
from tensorflow.keras.layers import Concatenate, Activation, dot

class Attention_layer(tf.keras.layers.Layer):
    def __init__(self, name = None):
        super(Attention_layer, self).__init__(name = name)

    def call(self, encoder_out, decoder_out):
        attention = dot([decoder_out, encoder_out], axes=[2, 2])
        attention = Activation('softmax')(attention)
        context = dot([attention, encoder_out], axes=[2,1])
        decoder_combined_context = Concatenate()([context, decoder_out])
        return decoder_combined_context


def main():
    attention = Attention_layer()(tf.random.uniform((64, 900, 2048)),
                                tf.random.uniform((64, 300, 2048)))
    print(attention.shape)
    

if __name__ == '__main__':
    main() 