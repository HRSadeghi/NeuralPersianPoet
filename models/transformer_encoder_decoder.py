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
from .layers.transformer_decoder import Decoder
from .layers.transformer_encoder import Encoder
from tensorflow.keras.layers import Dense, Input, TimeDistributed
from tensorflow.keras.models import Model


def Transformer_gru_encoder_decoder(encoder_dim,
                                    decoder_dim,
                                    d_model=512,
                                    num_heads=8,
                                    num_layers=8,
                                    dff=2048,
                                    pe_input=400,
                                    pe_target=300,
                                    rate = 0.1,
                                    training=False,
                                    enc_padding_mask=None,
                                    look_ahead_mask=None,
                                    dec_padding_mask=None):

    ####### Inputs #######
    encoder_input = Input(shape=(None, ), name='encoder_input')
    decoder_input = Input(shape=(None,), name="decoder_input")
    ######################


    ############## Encoder ###############
    #Transformer encoder
    encoder_out = Encoder(num_layers, d_model, num_heads, dff,
                        encoder_dim, pe_input, rate, name = 'transformer_encoder_layer')(encoder_input, training, enc_padding_mask)
    ######################################

    decoder_out, attention_weights = Decoder(num_layers, d_model, num_heads, dff, 
                            decoder_dim, pe_target, rate, name = 'transformer_decoder_layer')(decoder_input, encoder_out, training, look_ahead_mask, dec_padding_mask)

    decoder_outputs = TimeDistributed(Dense(decoder_dim, activation='softmax'),name='decoder_outputs')(decoder_out)

    return Model([encoder_input, decoder_input], decoder_outputs)



