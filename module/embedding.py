#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
from tools.logger import *

class Word_Embedding(object):
    def __init__(self, path, vocab):
        """
        :param path: string; the path of word embedding
        :param vocab: object;
        """
        logger.info("[INFO] Loading external word embedding...")
        self._path = path
        self._vocablist = vocab.word_list()
        self._vocab = vocab

    def load_my_vecs(self, k=200):
        """Load word embedding"""
        word_vecs = {}
        with open(self._path, encoding="utf-8") as f:
            count = 0
            lines = f.readlines()[1:]
            for line in lines:
                values = line.split(" ")
                word = values[0]
                count += 1
                if word in self._vocablist:  # whether to judge if in vocab
                    vector = []
                    for count, val in enumerate(values):
                        if count == 0:
                            continue
                        if count <= k:
                            vector.append(float(val))
                    word_vecs[word] = vector
        return word_vecs

    def add_unknown_words_by_zero(self, word_vecs, k=200):
        """Solve unknown by zeros"""
        zero = [0.0] * k
        list_word2vec = []
        oov = 0
        iov = 0
        for i in range(self._vocab.size()):
            word = self._vocab.id2word(i)
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = zero
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])
        logger.info("[INFO] oov count %d, iov count %d", oov, iov)
        return list_word2vec

    def add_unknown_words_by_avg(self, word_vecs, k=200):
        """Solve unknown by avg word embedding"""
        # solve unknown words inplaced by zero list
        word_vecs_numpy = []
        for word in self._vocablist:
            if word in word_vecs:
                word_vecs_numpy.append(word_vecs[word])
        col = []
        for i in range(k):
            sum = 0.0
            for j in range(int(len(word_vecs_numpy))):
                sum += word_vecs_numpy[j][i]
                sum = round(sum, 6)
            col.append(sum)
        zero = []
        for m in range(k):
            avg = col[m] / int(len(word_vecs_numpy))
            avg = round(avg, 6)
            zero.append(float(avg))

        list_word2vec = []
        oov = 0
        iov = 0
        for i in range(self._vocab.size()):
            word = self._vocab.id2word(i)
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = zero
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])
        logger.info("[INFO] External Word Embedding iov count: %d, oov count: %d", iov, oov)
        return list_word2vec

    def add_unknown_words_by_uniform(self, word_vecs, uniform=0.25, k=200):
        """Solve unknown word by uniform(-0.25,0.25)"""
        list_word2vec = []
        oov = 0
        iov = 0
        for i in range(self._vocab.size()):
            word = self._vocab.id2word(i)
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = np.random.uniform(-1 * uniform, uniform, k).round(6).tolist()
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])
        logger.info("[INFO] oov count %d, iov count %d", oov, iov)
        return list_word2vec

    # load word embedding
    def load_my_vecs_freq1(self, freqs, pro):
        word_vecs = {}
        with open(self._path, encoding="utf-8") as f:
            freq = 0
            lines = f.readlines()[1:]
            for line in lines:
                values = line.split(" ")
                word = values[0]
                if word in self._vocablist:  # whehter to judge if in vocab
                    if freqs[word] == 1:
                        a = np.random.uniform(0, 1, 1).round(2)
                        if pro < a:
                            continue
                    vector = []
                    for count, val in enumerate(values):
                        if count == 0:
                            continue
                        vector.append(float(val))
                    word_vecs[word] = vector
        return word_vecs