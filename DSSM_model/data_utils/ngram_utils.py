#encoding:utf-8
import numpy as np


import codecs
from gensim.models.utils_any2vec import _compute_ngrams
import numpy as np

import os
from tensorflow.contrib import learn
import numpy.linalg as LA




def word2ngram(sentences,n):
    """
   将句子表示成 ngram 形式

    """
    for sentence in sentences:
        ngram = []
        for word in sentence:
            l1 = _compute_ngrams(word, n, n)
            ngram.extend(l1)
        yield ngram

































































