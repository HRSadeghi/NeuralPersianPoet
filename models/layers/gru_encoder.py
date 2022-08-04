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
from tensorflow.keras.layers import GRU, Dense, Bidirectional, TimeDistributed


class Gru_encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, bidirectional = False, state_latent_same_size = True, name = None):
        super(Gru_encoder, self).__init__(name = name)
        self.bidirectional = bidirectional
        self.state_latent_same_size = state_latent_same_size
        if bidirectional:
            self.encoder_gru = Bidirectional(GRU(latent_dim, return_sequences=True, return_state=True))
        else:
            self.encoder_gru = GRU(latent_dim, return_sequences=True, return_state=True)
        if self.state_latent_same_size:
            self.dense = Dense(latent_dim)
            self.tdd = TimeDistributed(Dense(latent_dim))

    def call(self, encoder_input):
        if self.bidirectional:
            encoder_out, state1, state2 = self.encoder_gru(encoder_input)
            states = Concatenate()([state1, state2])
            if self.state_latent_same_size:
                encoder_out = self.tdd(encoder_out) 
                state = self.dense(states)
            else:
                state = states
        else:
            encoder_out, state = self.encoder_gru(encoder_input)
        return encoder_out, state


def main():
    gru_encoder = Gru_encoder(1024, bidirectional=False, state_latent_same_size=True)(tf.random.uniform((64, 900, 1024)))
    print(gru_encoder[0].shape)
    print(gru_encoder[1].shape)
    

if __name__ == '__main__':
    main()   