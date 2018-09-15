#encoding:utf-8

import codecs
from DSSM_model.data_utils.vocabulary import Vocabprocessor
from DSSM_model.data_utils.ngram_utils import word2ngram
import os

import numpy as np
"""
unigram 45

"""

def readfile(file, encoding= 'utf-8'):


    f1 = codecs.open(file, 'r', encoding=encoding)
    line = f1.readline()
    lines = []
    while line:
        linelist = line.strip().split('\t')
        lines.append(linelist)
        line = f1.readline()
    f1.close()
    return lines


def build_vocab(sentences, max_d, vocab_file):
    """
    :param sentences:
    :param max_d: the max length of the sentence
    :param vocab_file:
    :return:
    """
    print('build the vocabulary !!!')
    if os.path.exists(vocab_file):
        print('there is a vocabfile saved ....  load the vocab ')
        vocab = Vocabprocessor.restore(vocab_file)
    else:
        vocab = Vocabprocessor()
        vocab.build(sentences, max_d)
        print('save the vocabulary !!!')
        vocab.save(vocab_file)

    print('the length of vocabulary is : {}'.format(vocab.vocab_size))

    return vocab




def batch_for_sparsetensor(batch_x,vocab_size):

    batch_size = len(batch_x)
    len1 = len(batch_x[0])
    shape = [batch_size,vocab_size]
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

    return (indices,values,shape)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.

    data 是一个 list


    data  tuple (data,label)
    """
    data = list(data)
    data = map(lambda x: np.asarray(x), data)

    data_size = len(data[0])
    num_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = []
            for item in data:
                shuffled_data.append(item[shuffle_indices])
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size

            if (batch_num + 1) * batch_size > data_size:
                end_index = data_size
            else:
                end_index = (batch_num + 1) * batch_size


            batch_data= map(lambda x: x[start_index: end_index], shuffled_data)

            yield batch_data

def buildneg_test(batch_label, labels_texts):
    # self.labels_ids_ngram
    # self.labels_texts_ngram
    # self.labels_texts
    batch_label = list(batch_label)
    num_labels = len(labels_texts)

    num_samples = 1
    neg_labels = []
    for label in batch_label:
        ys = []

        x = np.arange(num_labels)
        np.random.shuffle(x)
        sample_indexes = x[: num_samples + 1]
        for index in sample_indexes:
            if index == label:
                continue
            ys.append(labels_texts[index])

        ys = ys[:num_samples]
        neg_labels.extend(ys)

    return np.asarray(neg_labels)



class InputHelper(object):

    def __init__(self):
        pass

    def loaddata(self,trainfile,labelfile,vocab_file,ngram=2,max_len = 35):
        """
        :param trainfile: 训练数据文件 --- 划分成训练子集和验证集
        :param labelfile: 用来进行负采样用的，负采样的只是 标准问题
        :param n:  使用的是 n-gram 来进行训练 dssm

        """


        print 'read the data from the trainfile .'
        train_data = readfile(trainfile)
        querys = [x[0].split() for x in train_data]
        docs = [x[2].split() for x in train_data]
        qdlabels = map(lambda x: int(x[1]), train_data)

        print('word to ngram .')
        querys_ngram = np.asarray(list(word2ngram(querys,ngram)))
        docs_ngram   = np.asarray(list(word2ngram(docs,ngram)))
        # qdlabels
        qdlabels = np.asarray(qdlabels)


        np.random.seed(123)
        shuffle_indices = np.random.permutation(np.arange(len(qdlabels)))
        test_l1 = int(len(shuffle_indices) * 0.05)


        print('shuffle the data ... ')
        querys_ngram = querys_ngram[shuffle_indices]
        docs_ngram = docs_ngram[shuffle_indices]
        qdlabels = qdlabels[shuffle_indices]


        texts_ngram=np.concatenate([querys_ngram, docs_ngram])

        print(' get the vocabulary ... ')
        self.vocab = build_vocab(texts_ngram,max_len,vocab_file)

        print(' deal the label files ... ')
        self.labels_texts, self.labels_texts_ngram, self.labels_ids_ngram = \
            self.deal_labelfile(labelfile, ngram, self.vocab)



        print('text to id .')
        docs_ngram = list(self.vocab.fit(docs_ngram))
        querys_ngram = list(self.vocab.fit(querys_ngram))

        test_query,train_query = querys_ngram[:test_l1],querys_ngram[test_l1:]
        test_doc  , train_doc  = docs_ngram[:test_l1],docs_ngram[test_l1:]
        test_label , train_label = qdlabels[:test_l1],qdlabels[test_l1:]


        train_set = [train_query,train_doc,train_label]
        val_set   = [test_query,test_doc,test_label]

        return train_set,val_set


    def deal_labelfile(self, labelfile, ngram, vocab):

        print('read the label file ... ')
        all_labels = readfile(labelfile)

        all_labels_copy = []
        for label in all_labels:
            label[1] = int(label[1])
            all_labels_copy.append(label)

        all_labels = all_labels_copy

        all_labels = sorted(all_labels, key=lambda x: x[1])

        labels_texts = []
        for item in all_labels:
            labels_texts.append(item[0].split())

        print('deal with label text, and convert words to ngram ...')
        labels_texts_ngram = list(word2ngram(labels_texts, ngram))
        #   word --- ngram

        labels_ids_ngram = list(vocab.fit(labels_texts_ngram))




        return labels_texts,labels_texts_ngram,labels_ids_ngram





    def split_data(self,train_set,val_set):
        print('split the train/val data, randomly')
        querys = list(train_set[0]) + list(val_set[0])
        docs   = list(train_set[1]) + list(val_set[1])
        labels = list(train_set[2]) + list(val_set[2])
        querys = np.asarray(querys)
        docs  = np.asarray(docs)
        labels = np.asarray(labels)


        shuffle_indices = np.random.permutation(np.arange(len(labels)))

        querys = querys[shuffle_indices]
        docs   = docs[shuffle_indices]
        labels = labels[shuffle_indices]

        test_l1 = int(len(shuffle_indices) * 0.05)
        test_query, train_query = querys[:test_l1], querys[test_l1:]
        test_doc, train_doc = docs[:test_l1], docs[test_l1:]
        test_label, train_label = labels[:test_l1], labels[test_l1:]

        train_set = [train_query, train_doc, train_label]
        val_set = [test_query, test_doc, test_label]

        return train_set, val_set

    def batch_for_sparse(self, batch_x, batch_y, batch_label,
                         repeat_times , label_ids_ngram):
        """

        输入的形状是 batch * seq_len

        indices, values, shape= [batch,vocab_size]

        repeat_times: num_neg + 1

        """

        batch_x = np.asarray(batch_x)

        rows = len(batch_x)

        cols = len(batch_x[0])

        num_labels = len(label_ids_ngram)

        # self.labels_ids_ngram

        batch_x = np.tile(batch_x, [1, repeat_times])
        batch_x = np.reshape(batch_x, [-1, cols])

        batch_y = list(batch_y)
        batch_label = list(batch_label)

        # np.random.shuffle(np.arange(num_labels))
        #  从 label 里面进行sample负例
        batch_y1 = []
        for y, label in zip(batch_y, batch_label):
            ys=[]
            ys.append(y)
            x = np.arange(num_labels)
            np.random.shuffle(x)
            sample_indexes = x[:repeat_times]
            for index in sample_indexes:
                if index == label:
                    continue
                ys.append(label_ids_ngram[index])

            ys = ys[:repeat_times]
            batch_y1.extend(ys)

        batch_y = np.asarray(batch_y1)

        batch_x_sparse = batch_for_sparsetensor(batch_x, self.vocab.vocab_size)
        batch_y_sparse = batch_for_sparsetensor(batch_y, self.vocab.vocab_size)
        label_tensor = np.zeros(dtype=np.int32, shape=rows)

        return batch_x_sparse, batch_y_sparse, label_tensor

    def batch_for_sparse_x(self, batch_x,vocab):
        #    输入的形状是 batch * seq_len
        #    repeat true表示的是 输入的是 query,要进行复制；docs 则要进行随机采样，构造反例
        #    indices, values, shape= [batch,vocab_size]
        # size1 表示的是 neg_num + 1

        batch_x = np.asarray(batch_x)

        batch_x_sparse = batch_for_sparsetensor(batch_x,vocab.vocab_size)

        return batch_x_sparse

























