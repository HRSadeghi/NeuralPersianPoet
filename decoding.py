import numpy as np


def decode_sequence_lstm_encoder_decoder(input_seq, encoder_model, decoder_model, bpet):
    # Encode the input as state vectors.
    ouput_values, states_values = encoder_model.predict(input_seq)
    #print(states_value)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1),dtype='float32')
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = bpet.bpe2idx['__she']
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    decoded_sentence = []
    while 1:
        output_tokens,h1= decoder_model.predict(
            [target_seq, ouput_values, states_values])
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = bpet.idx2bpe[sampled_token_index]
        

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == '__ehe2' or sampled_word == '__ehe1' or
           len(decoded_sentence) > 100):
            break
        decoded_sentence += [sampled_token_index]
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1),dtype='float32')
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_values = h1

    return decoded_sentence



def decode_batch_lstm_encoder_decoder(X_input, encoder_model, decoder_model, bpet):
    # Encode the input as state vectors.
    ouput_values, states_values = encoder_model.predict(X_input)
    #print(states_value)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((len(X_input), 1),dtype='float32')
    # Populate the first character of target sequence with the start character.
    target_seq[:, 0] = bpet.bpe2idx['__she']
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    decoded_sentences = [[] for i in range(len(X_input))]
    counter = 0
    while 1:
        output_tokens,h1= decoder_model.predict(
            [target_seq, ouput_values, states_values])
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[:,:, :], axis=2)
        sampled_words = [(x[-1], bpet.idx2bpe[x[-1]]) for x in sampled_token_index]
        
        acceptable_tokens = [i for i,x in enumerate(sampled_words) if x[1] != '__ehe2' and x[1] != '__ehe1']

        # Exit condition: either hit max length
        # or find stop character.
        if (len(acceptable_tokens) == 0 or counter > 40):
            break
        
        for i in range(len(decoded_sentences)):
            if sampled_words[i][1] != '__ehe2' and sampled_words[i][1] != '__ehe1':
                decoded_sentences[i].append(sampled_words[i][0])

        target_seq = sampled_token_index[:,:]

        # Update states
        states_values = h1
        counter += 1

    return decoded_sentences