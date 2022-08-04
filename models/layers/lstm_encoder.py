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
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, Concatenate


class LSTM_encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim, bidirectional = False, state_latent_same_size = True, name = None):
        super(LSTM_encoder, self).__init__(name = name)
        self.bidirectional = bidirectional
        self.state_latent_same_size = state_latent_same_size
        if bidirectional:
            self.encoder_lstm = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))
        else:
            self.encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)

        if self.state_latent_same_size:
            self.dense1 = Dense(latent_dim)
            self.dense2 = Dense(latent_dim)
            self.tdd = TimeDistributed(Dense(latent_dim))

    def call(self, encoder_input):
        if self.bidirectional:
            encoder_out, state1, state2, state3, state4 = self.encoder_lstm(encoder_input)
            states = [Concatenate()([state1, state2]), Concatenate()([state3, state4])]
            if self.state_latent_same_size:
                encoder_out = self.tdd(encoder_out) 
                state = [self.dense1(states[0]), self.dense2(states[1])]
            else:
                state = states
        else:
            encoder_out, state1, state2 = self.encoder_lstm(encoder_input)
            state = [state1, state2]
        return encoder_out, state[0], state[1]


def main():
    lstm_encoder = LSTM_encoder(1024, bidirectional=False, state_latent_same_size=True)(tf.random.uniform((64, 900, 1024)))
    print(lstm_encoder[0].shape)
    print(lstm_encoder[1].shape)
    print(lstm_encoder[2].shape)
    

if __name__ == '__main__':
    main()   