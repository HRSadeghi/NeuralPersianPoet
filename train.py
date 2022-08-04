from dataLoader.utils import load_file, save_file
from dataLoader.utils import get_poems_and_poets
from dataLoader.utils import get_training, separate
from sklearn.model_selection import train_test_split
from tokenization.bpe_tokenization import BPE_Tokenization
from dataLoader.generator import DataGenerator
from models.gru_encoder_decoder import get_encoder_decoder
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Create a train command.')
    parser.add_argument('--poem_train_pkl',
                        type=str, 
                        required=True,
                        default='dataset/poem_list_train.pickle',
                        help='path to the poem_list_train.pickle file')
    parser.add_argument('--poem_test_pkl', 
                        type=str,
                        required=True,
                        default='dataset/poem_list_test.pickle',
                        help='path to the poem_list_test.pickle file')
    parser.add_argument('--poet_train_pkl', 
                        type=str, 
                        required=True,
                        default='dataset/poet_list_train.pickle',
                        help='path to the poet_list_train.pickle file')
    parser.add_argument('--poet_test_pkl', 
                        type=str, 
                        required=True,
                        default='dataset/poet_list_test.pickle',
                        help='path to the poet_list_test.pickle file')
    parser.add_argument('--tokenizer', 
                        type=str, 
                        required=True,
                        default='tokenization/bpe_tokenization.pkl',
                        help='path to the tokenizer file')
    args = parser.parse_args()

    return

if __name__ == "__main__":
    main()


    