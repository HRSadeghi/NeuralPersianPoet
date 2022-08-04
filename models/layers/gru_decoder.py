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
from tensorflow.keras.layers import GRU


class Gru_decoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, name = None):
        super(Gru_decoder, self).__init__(name = name)
        self.decoder_gru = GRU(latent_dim, return_sequences=True, return_state=True)

    def call(self, decoder_input, encoder_states):
        decoder_out, state = self.decoder_gru(decoder_input, initial_state=encoder_states)
        return decoder_out, state



def main():
    gru_decoder = Gru_decoder(1024)(tf.random.uniform((64, 900, 1024)),
                              tf.random.uniform((64, 1024)))
    print(gru_decoder[0].shape)
    print(gru_decoder[1].shape)
    

if __name__ == '__main__':
    main() 