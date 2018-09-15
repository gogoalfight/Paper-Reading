# encoding:utf-8
import os
from DSSM_model.data_utils.utils import InputHelper
import numpy as np
from DSSM_model.models.DSSM_model import DSSM_model

import tensorflow as tf
from DSSM_model.data_utils.vocabulary import Vocabprocessor

from DSSM_model.data_utils.ngram_utils import word2ngram
import sys



curdir = os.path.dirname(os.path.realpath(__file__))


class DSSM(object):
    def __init__(self):

        model_file = curdir + "/model_save/checkpoints/model-32000"
        self.vocab = Vocabprocessor.restore(curdir + '/data/vocab.txt')
        self.sess = tf.Session()
        self.inputhelper = InputHelper()

        with self.sess.as_default():
            self.dssm = DSSM_model(vocab_size=self.vocab.vocab_size, num_neg=0)
            self.sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(self.dssm.params, max_to_keep=1000)


            saver.restore(self.sess, model_file)

    def get_dssm_score(self,sen1,sen2):
        sens1 = [sen1]
        sens2 = [sen2]

        sens1 = word2ngram(sens1, 2)
        sens2 = word2ngram(sens2, 2)

        sens1 = list(self.vocab.fit(sens1))
        sens2 = list(self.vocab.fit(sens2))
        s1 = self.inputhelper.batch_for_sparse_x(sens1, self.vocab)
        s2 = self.inputhelper.batch_for_sparse_x(sens2, self.vocab)
        score = self.dssm.get_cosine(self.sess, s1, s2)
        score = score[0]
        score = (score + 1.0)/2.0

        return score


# for s1,s2 in zip(sens1,sens2):
#     print dssm.get_dssm_score(s1,s2)
#
# time2 = time.time()
#
# print '\n'
# print(time1)
# print(time2)
# print (time2 - time1) / len(sens1)
# # 0.00100909942275