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

import pandas as pd
import pickle
from .dataCleaner import clean


def read_data(path = "dataset/ganjoor.csv"):
    df = pd.read_csv(path, sep="\t",encoding='utf-8')
    df = df[df["position"] != -1]
    df = df[df["position"] < 2]
    df = df.reset_index(drop=True)
    return df

def get_poems_and_poets(path, cleaning = True, constrained_poets = []):
    poem_list = []
    poet_list = []
    print('reading data...')
    df = read_data(path = path)
    poems = df.groupby(by=["poem_id"]).groups
    for i, x in enumerate(poems.keys()):
        poet = list(df["cat"].iloc[poems[x]])[0].split("____")[0]
        if (poet in constrained_poets) or len(constrained_poets) == 0:
            poem_list.append(list(df["text"].iloc[poems[x]])) 
            poet_list.append(poet)
    print('cleaning data...')
    if cleaning:
        poem_list_ = []
        for p in poem_list:
            temp = []
            for x in p:
                temp.append(clean(x))
            poem_list_.append(temp)
        poem_list = poem_list_

    return poem_list, poet_list


def get_hemisstich(poem_list, bpet, num_he = 2, include_shorter = True):
    final_poems = []
    for x in poem_list:
        if len(x)%2 == 0:
            if len(x) > num_he:
                for i in range(len(x)//num_he):
                    pm = []
                    len__ = 0
                    start = num_he*i if num_he*i%2 == 0 else num_he*i + 1
                    end = start + num_he
                    if end > len(x):
                        continue
                    for j in range(start, end, 1):
                        len__ += 1
                        if j%2 == 0:
                            pm +=  [bpet.bpe2idx[z] for z in bpet.bpe_encoder.tokenize(x[j])] + [bpet.bpe2idx["__ehe1"]]
                        if j%2 == 1:
                            pm += [bpet.bpe2idx[z] for z in bpet.bpe_encoder.tokenize(x[j])] + [bpet.bpe2idx["__ehe2"]]
                    if len(pm) > 0:
                        final_poems.append(pm)
            else:
                if include_shorter:
                    pm = []
                    len__ = 0
                    for i in range(len(x)):
                        len__ += 1
                        if i%2 == 0:
                            pm +=  [bpet.bpe2idx[z] for z in bpet.bpe_encoder.tokenize(x[i])] + [bpet.bpe2idx["__ehe1"]]
                        if i%2 == 1:
                            pm += [bpet.bpe2idx[z] for z in bpet.bpe_encoder.tokenize(x[i])] + [bpet.bpe2idx["__ehe2"]]
                    if len(pm) > 0:
                        final_poems.append(pm)
    return final_poems



def get_training(poem_list, bpet, max_num_he = 9):
    out = []
    for i in range(2, max_num_he + 1, 1):
        out += get_hemisstich(poem_list, bpet, i)
    return out


def separate(poems, bpet):
    X, Y = [], []
    for p in poems:
        if p[-1] == bpet.bpe2idx['__ehe2']:
            index = len(p) - p[::-1].index(bpet.bpe2idx["__ehe1"]) - 1
            X.append(p[:index+1])
            Y.append(p[index+1:])
        elif p[-1] == bpet.bpe2idx['__ehe1']:
            index = len(p) - p[::-1].index(bpet.bpe2idx["__ehe2"]) - 1
            X.append(p[:index+1])
            Y.append(p[index+1:])
    return X, Y


def split_context_out(poem_list, num_contexts, max_num_generated):
    X, Y = [], []
    for p in poem_list:
        start = 0
        end = len(p)
        while end >= start + num_contexts + 1:
            X.append(p[start:start+num_contexts])
            if end >= start+num_contexts+max_num_generated:
                Y.append(p[start+num_contexts:start+num_contexts+max_num_generated])
                start = start+num_contexts+max_num_generated
            else:
                Y.append(p[start+num_contexts:end])
                break
    
    return X, Y

def save_file(file, path):
    with open(path, 'wb') as f:
        pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)

def load_file(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
    return file

