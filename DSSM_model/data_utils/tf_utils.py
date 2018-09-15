#encoding:utf-8
import numpy as np
import os

import codecs
from gensim.models.utils_any2vec import _compute_ngrams, _ft_hash
import numpy as np

import os
from tensorflow.contrib import learn
import numpy.linalg as LA



"""
tf 一些常用操作

"""




def update_embedding(sess,W_variable,vocab_processor,emb_dim,W_word2vec):
    """
    这个是使用预训练好的词向量来进行初始化embedding层
    :param sess:
    :param vocab_processor: 训练对应的字典
    :param embedding_dim: 50
    :param W:  charagram 对应的 embedding 层
    :param ngram_w:  预训练好的 ngram 向量 它是一个字典 dict
    :return:
    """
    print 'init the embedding W '
    initW = np.random.uniform(-0.25, 0.25,[vocab_processor.vocab_size,emb_dim])


    fname = 'embedding_initW.txt'




    if os.path.exists(fname):
        initW = np.loadtxt(fname)
    else:
        count = 0
        for w in vocab_processor.vocabluarity:

            # print w
            arr = []
            if w in W_word2vec:
                arr = W_word2vec[w]
                if len(arr) != emb_dim:
                    print 'the embbeding dim != the vector embedding, and init failed !!!'
                    return
                count += 1
            else:
                continue
            if len(arr) > 0:
                idx = vocab_processor.mapping[w]
                # word -- id
                initW[idx] = np.asarray(arr).astype(np.float32)
        print 'the proportion: {}'.format(float(count) / len(vocab_processor.vocabulary))
        np.savetxt(fname,initW)

    sess.run(W_variable.assign(initW))
    del initW

    print 'init the embedding W finished '



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


































