import numpy as np
from decoding import decode_sequence_lstm_encoder_decoder, decode_batch_lstm_encoder_decoder
from tqdm import tqdm_notebook, tqdm
from dataLoader.generator import DataGenerator
from decoding import decode_batch_lstm_encoder_decoder
from dataLoader.dataCleaner import remove_extra_tokens
import pandas as pd
from dataLoader.utils import get_training, separate
from tqdm import tqdm_notebook




def predict_and_save(poem_list, 
                     encoder_model, 
                     decoder_model, 
                     bpet,
                     path, 
                     max_num_he = 8):
    for h in tqdm_notebook(range(2, max_num_he, 1)):
        temp__ = get_hemisstich(poem_list, bpet, num_he = h, include_shorter = False)
        X__, Y__ = separate(temp__, bpet)

        gen_temp = DataGenerator(X__,
                                Y__,
                                bpet,
                                None,
                                save_path = None,
                                batch_size = 512)

        out_preds = []
        for i in range(gen_temp.__len__()):
            [X,Y1], Y2 = gen_temp.__getitem__(i)
            prediction = decode_batch_lstm_encoder_decoder(X, encoder_model, decoder_model, bpet)
            for j in range(len(prediction)):
                dic_temp = dict()
                #context
                dic_temp['context'] = remove_extra_tokens(bpet.to_sentence(X[j]).replace('__ehe1', '\t').replace('__ehe2', '\n')).strip()
                #ground_truth
                dic_temp['ground_truth'] = remove_extra_tokens(bpet.to_sentence(Y1[j])).strip()
                #predicted
                dic_temp['predicted'] = remove_extra_tokens(bpet.to_sentence(prediction[j])).strip()

                out_preds.append(dic_temp)

        df_ = pd.DataFrame(out_preds)
        df_.to_csv(path + 'prediction_he_{}.csv'.format(h), index = False)


def generate_batch(contexts, Y_out, encoder_model, decoder_model, bpet, max_num_he = 8, save_path = None):
    X__ = []
    for i in range(len(contexts)):
        prefix = []
        for j in range(len(contexts[i])):
            if j % 2 == 0:
                prefix += bpet.to_index(contexts[i][j]) + [bpet.bpe2idx['__ehe1']]
            else:
                prefix += bpet.to_index(contexts[i][j]) + [bpet.bpe2idx['__ehe2']]
        X__.append(prefix)

    new_X = X__[:][:]

    for i in tqdm_notebook(range(max_num_he)):
        gen_temp = DataGenerator(new_X,
                                None,
                                bpet,
                                None,
                                save_path = None,
                                batch_size = 512)

        out = []
        for i in tqdm(range(gen_temp.__len__())):
            X = gen_temp.__getitem__(i)
            prediction = decode_batch_lstm_encoder_decoder(X, encoder_model, decoder_model, bpet)
            out += prediction
        out__ = []
        for i,x in enumerate(out):
            if new_X[i][-1] == bpet.bpe2idx['__ehe1']:
                temp__ = new_X[i] + [y for y in x if y != bpet.bpe2idx['__pad']] + [bpet.bpe2idx['__ehe2']]
            elif new_X[i][-1] == bpet.bpe2idx['__ehe2']:
                temp__ = new_X[i] + [y for y in x if y != bpet.bpe2idx['__pad']] + [bpet.bpe2idx['__ehe1']]
            out__.append(temp__)

        new_X = out__[:][:]

    final_out = [bpet.to_sentence(x[len(X__[i]):]).replace('__ehe2', '\n').replace('__ehe1', '\n').split('\n') for i,x in enumerate(new_X)]
    final_out = [[remove_extra_tokens(y).strip() for y in x[:-1]] for x in final_out]
    final_out = [x[:len(Y_out[i])] for i,x in enumerate(final_out)]


    if save_path is not None:
        df = pd.DataFrame(list(zip(contexts, Y_out, final_out)), 
                columns =['context', 'references', 'hypotheses'])
        
        df.to_csv(save_path, index = False)

    return final_out



def inference(input_sentence, encoder_model, decoder_model, bpet, num_pads = 0):
    sen = []
    temp = input_sentence.split('\n')
    for x in temp:
        temp2 = x.split('\t')
        sen += bpet.to_index(temp2[0]) + [bpet.bpe2idx["__ehe1"]]
        if len(temp2) > 1:
            sen += bpet.to_index(temp2[1]) + [bpet.bpe2idx["__ehe2"]]
    sen += [bpet.bpe2idx["__pad"]]*num_pads
    out = decode_sequence_lstm_encoder_decoder([np.array([sen])],encoder_model, decoder_model, bpet)
    return bpet.to_sentence(out)

def inference2(tokens, encoder_model, decoder_model, bpet):
    out = decode_sequence_lstm_encoder_decoder([np.array([tokens])],encoder_model, decoder_model, bpet)
    return bpet.to_sentence(out)


def generate_new_sample(prefix, num_he, encoder_model, decoder_model, bpet, num_pads = 0):
    sen = []
    temp = input_sentence.split(';')
    first = True
    for x in temp:
        temp2 = x.split('/')
        sen += bpet.to_index(temp2[0]) + [bpet.bpe2idx["__ehe1"]]
        first = False
        if len(temp2) > 1:
            sen += bpet.to_index(temp2[1]) + [bpet.bpe2idx["__ehe2"]]
            first = True
    
    for i in range(num_he):
        sen1 = sen[:] + [bpet.bpe2idx["__pad"]]*num_pads
        out = decode_sequence_lstm_encoder_decoder([np.array([sen1])],encoder_model, decoder_model, bpet)
        if first:
            sen += bpet.to_index(out) + [bpet.bpe2idx["__ehe1"]]
            first = False
        else:
            sen += bpet.to_index(out) + [bpet.bpe2idx["__ehe2"]]
            first = True



def generate_new_sample(prefix, encoder_model, decoder_model, bpet, num_he = 1, num_pads = 0):
    sen = []
    full = ''
    temp = prefix.split('\n')
    first = True
    for x in temp:
        temp2 = x.split('\t')
        sen += bpet.to_index(temp2[0]) + [bpet.bpe2idx["__ehe1"]]
        full += temp2[0] + '\t\t'
        first = False
        if len(temp2) > 1:
            sen += bpet.to_index(temp2[1]) + [bpet.bpe2idx["__ehe2"]]
            full += temp2[1] + '\n'
            first = True
    
    for i in range(num_he):
        sen1 = sen[:] + [bpet.bpe2idx["__pad"]]*num_pads
        out = decode_sequence_lstm_encoder_decoder([np.array([sen1])],encoder_model, decoder_model, bpet)
        if first:
            sen += out + [bpet.bpe2idx["__ehe1"]]
            first = False
            full += bpet.to_sentence(out) + '\t\t'
        else:
            sen += out + [bpet.bpe2idx["__ehe2"]]
            first = True
            full += bpet.to_sentence(out) + '\n'
    return full