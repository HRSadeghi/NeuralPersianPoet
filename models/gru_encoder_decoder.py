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

from .layers.gru_encoder import Gru_encoder
from .layers.gru_decoder import Gru_decoder
from .layers.attention_layer import Attention_layer
import tensorflow as tf
from tensorflow.keras.layers import Input, TimeDistributed, Dense, Embedding, Dropout
from tensorflow.keras.models import Model


def Gru_encoder_decoder(encoder_dim, decoder_dim, latent_dim, use_att = True, dropout_rate = 0.0):

    ####### Inputs #######
    encoder_input = Input(shape=(None, ), name='encoder_input')
    decoder_input = Input(shape=(None,), name="decoder_input")
    ######################


    ####### Embeddings #######
    encoder_embedding = Embedding(encoder_dim, latent_dim,name='encoder_embedding')(encoder_input)
    decoder_embedding = Embedding(decoder_dim, latent_dim,name='decoder_embedding')(decoder_input)
    ##########################

    ############## Encoder ###############
    #GRU decoder
    encoder_out, encoder_states = Gru_encoder(latent_dim, bidirectional=False, state_latent_same_size = False, name = 'gru_encoder_layer')(encoder_embedding) 
    #Dropout
    if dropout_rate != 0:
        encoder_states = Dropout(dropout_rate)(encoder_states)
    ######################################

    ####### Decoder with attention #######
    decoder_out, h = Gru_decoder(latent_dim, name = 'gru_decoder_layer')(decoder_embedding, encoder_states)

    #Attention
    if use_att:
        decoder_out = Attention_layer(name = 'attention_layer')(encoder_out, decoder_out)

    #Dropout
    if dropout_rate != 0:
        decoder_out = Dropout(dropout_rate)(decoder_out)
    decoder_outputs = TimeDistributed(Dense(decoder_dim, activation='softmax'),name='decoder_outputs')(decoder_out)
    ######################################


    model = Model([encoder_input, decoder_input], decoder_outputs)

    return model


def get_encoder_decoder(complete_model, latent_dim, decoder_dim, use_att = True):

    #########################  encoder ############################
    encoder_model = Model(complete_model.input[0],
                          complete_model.get_layer('gru_encoder_layer').output)
    ###############################################################

    #######################  decoder inputs  ######################
    decoder_input = Input(shape=(None,), name="decoder_input")
    encoder_states = Input(shape=(latent_dim,), name="encoder_states")
    encoder_out = Input(shape=(None, latent_dim), name="encoder_out")
    ###############################################################

    ############################ embedding ########################
    decoder_embedding = Embedding(decoder_dim, latent_dim,name='decoder_embedding')(decoder_input)
    ###############################################################

    #########################  decoder gru ########################
    decoder_out, h = Gru_decoder(latent_dim, name = 'gru_decoder_layer')(decoder_embedding, encoder_states)
    ###############################################################
    
    ############################ attnetion #######################
    if use_att:
        decoder_out = Attention_layer(name = 'attention_layer')(encoder_out, decoder_out)
    ##############################################################

    ############################## dense ###########################
    decoder_outputs = TimeDistributed(Dense(decoder_dim, activation='softmax'),name='decoder_outputs')(decoder_out)
    ################################################################

    decoder_model = Model([decoder_input, encoder_out, encoder_states], [decoder_outputs, h])

    ##################### load weights to decoder ##################
    if use_att:
        for x in ['decoder_embedding', 'gru_decoder_layer', 'attention_layer', 'decoder_outputs']:
            decoder_model.get_layer(x).set_weights(complete_model.get_layer(x).get_weights())
    else:
        for x in ['decoder_embedding', 'gru_decoder_layer', 'decoder_outputs']:
            decoder_model.get_layer(x).set_weights(complete_model.get_layer(x).get_weights())
    ################################################################
    
    return encoder_model, decoder_model


def main():
    print(Attention_gru_encoder_decoder(100, 100, 1024).summary())
    

if __name__ == '__main__':
    main()   