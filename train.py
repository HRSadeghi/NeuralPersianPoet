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


    