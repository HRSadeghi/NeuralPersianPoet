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


from bpe import Encoder
import pickle

class BPE_Tokenization():
    def __init__(self, texts, extra_tokens, num_tokens = 30000,ngram_max = 10):
        self.texts = texts
        self.extra_tokens = extra_tokens
        self.bpe_vocab = set()
        self.bpe2idx = dict()
        self.idx2bpe = dict()
        self.num_tokens = num_tokens
        self.ngram_max = ngram_max
        self.bpe_encoder = self.load_bpe_encoder()
        self.prepare()
    
    def load_bpe_encoder(self):
        bpe_encoder = Encoder(self.num_tokens, ngram_max=self.ngram_max) 
        bpe_encoder.fit([" ".join(x) for x in self.texts])
        return bpe_encoder
    
    def prepare(self):
        tokn_poem = []
        for x in self.texts:
            for y in x:
                tokn_poem.append(self.bpe_encoder.tokenize(y))

        for x in tokn_poem:
            for y in x:
                self.bpe_vocab.add(y)

        for x in self.extra_tokens:
            self.bpe_vocab.add(x)

        self.bpe_vocab = sorted(list(self.bpe_vocab))
    
        for i,x in enumerate(self.bpe_vocab):
            self.bpe2idx[x] = i
            self.idx2bpe[i] = x

    def to_index(self, sentence):
        return [self.bpe2idx[x] for x in self.bpe_encoder.tokenize(sentence)]

    def to_sentence(self, indices):
        temp = [self.idx2bpe[x] for x in indices]
        out = ""
        flag = False
        for x in temp:
            if x == "__eow":
                out += " "
                flag = False
            if flag:
                out += x
            if x == "__sow":
                flag = True
            if x != "__sow" and x != "__eow" and not flag:
                out += x + " "
        return out.strip()


