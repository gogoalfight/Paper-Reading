#encoding:utf-8


"""

建立字典
"""
import numpy as np
from collections import Counter

import pickle
import codecs


class Vocabprocessor(object):
    def __init__(self):
        self.vocab_size = 0
        self.max_len = -1
        self.word2id = {}
        self.id2word = {}
        self.vocabulary = {}
        self.mapping = {}
        pass

    def build(self,sentences,max_len,max_size = -1 ,min_freq = -1,encoding = 'utf-8'):

        self.max_len = max_len

        s_concat = np.concatenate(sentences)

        counter = Counter(s_concat)

        counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        amount = len(counts)


        if min_freq != -1:
            counts_copy = []
            for item in counts:
                if item[1] < min_freq:
                    break
                counts_copy.append(item)

            counts = counts_copy

        if min_freq == -1 and max_size != -1 and amount > max_size:
            counts = counts[:max_size]



        # 0 -- u'pad'

        self.id2word = {}
        if encoding == 'utf-8':
            self.id2word[0] = u'pad'
        else:
            self.id2word[0] = 'pad'


        self.word2id = {}

        if encoding == 'utf-8':
            self.word2id[u'pad'] = 0
        else:
            self.word2id['pad'] = 0

        item_num = 1
        for item in counts[:-1]:
            self.id2word[item_num] = item[0]
            self.word2id[item[0]]  = item_num
            item_num += 1


        self.vocab_size = len(self.word2id)
        self.pad = self.id2word[0]

        self.vocabulary = self.id2word
        self.mapping = self.word2id

    def fit(self,raw_documents):
        # 将文本转化为对应的id
        sentences = raw_documents
        pads = [self.pad] * self.max_len
        for s in sentences:
            if len(s) < self.max_len:
                s = np.concatenate([s,pads[:self.max_len-len(s)]])
            if len(s) > self.max_len:
                s = s[:self.max_len]

            id = map(lambda x: self.get_id(x),s)
            yield id

    def revesre(self,documents):

        for item in documents:
            output = []
            for class_id in item:
                output.append(self.id2word[class_id])
            yield ' '.join(output)


    def batch_for_sparse(self,batch_x,size1,repeat = False):
    #    输入的形状是 batch * seq_len
    #    indices, values, shape= [batch,vocab_size]
        len1 = len(batch_x[0])

        batch_x = np.asarray(batch_x)

        if repeat == True:
            batch_x = np.tile(batch_x,[1,size1])
            batch_x = np.reshape(batch_x,[-1,len1])

        batch_size = len(batch_x)


        shape = [batch_size,self.vocab_size]
        indices = []
        values  = []

        for i in xrange(batch_size):
            for j in xrange(len1):
                if batch_x[i][j] != 0:
                    indice1 = [i,batch_x[i][j]]
                    if indice1 not in indices:
                        indices.append(indice1)
                        values.append(1)
                    else:
                        ind1 = indices.index(indice1)
                        values[ind1] += 1

        return indices,values,shape






    def get_id(self,x):
        if x in self.word2id:
            return self.word2id[x]
        else:
            return 0 # denote pad

    def save(self,file):
        # self.word2id
        with codecs.open(file, 'w',encoding='utf-8') as f:
            # max_len
            f.write(str(self.max_len))
            f.write('\n')
            for item in self.word2id:
                f.write(item)
                f.write('\t')
                f.write(str(self.word2id[item]))
                f.write('\n')







    @staticmethod
    def restore(file):
        vocab = Vocabprocessor()
        f1 = codecs.open(file,'r',encoding='utf-8')

        line = f1.readline()

        vocab.max_len = int(line.strip())

        line = f1.readline()
        while line:
            linelist = line.strip().split('\t')
            vocab.word2id[linelist[0]]  = int(linelist[1])
            vocab.id2word[int(linelist[1])] = linelist[0]
            line = f1.readline()

        f1.close()

        vocab.vocabulary = vocab.id2word
        vocab.mapping = vocab.word2id

        vocab.vocab_size = len(vocab.word2id)

        vocab.pad = vocab.id2word[0]

        return vocab




















































