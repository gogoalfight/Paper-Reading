# encoding:utf-8

import tensorflow as tf

# query doc

import numpy as np


class DSSM_model(object):
    def __init__(self, num_neg, vocab_size, r=1, learning_rate=0.001):
        """
        query 表示的是 拓展问题  --- 输入之前，每个query，进行复制 （num_neg+1）次
        doc   表示的是 标准问题  --- 里面正例和负例的比例是 1:num_neg
        hiddens = [300,50]
        实际的 batch的大小为 batch_size * (num_neg+1)

        :param num_neg:  一个正例，同时多个反例 这里我们选取的是 4；
         保持的是和 dssm论文中的超参数一样; num_neg 表示的是
         每个样本进行训练的时候，构造的反例的数目

        :param vocab_size: 表示的是 字典的大小
        r 表示的是 最后进行softmax的时候，使用的超参数，起平滑作用的
        """
        self.vocab_size = vocab_size

        with tf.name_scope('input'):
            query_in_shape = [None, self.vocab_size]
            # None 表示的是 batch_size * (num_neg+1)
            # 因为每次进行训练的时候，每个正例对应了num_neg个负例
            self.query_batch = tf.sparse_placeholder(tf.float32, shape=query_in_shape, name='QueryBatch')

            doc_in_shape = [None, self.vocab_size]
            self.doc_batch = tf.sparse_placeholder(tf.float32, shape=doc_in_shape, name='DocBatch')

            self.labels = tf.placeholder(tf.int32, [None], name='labels')
            # batch   标签是全 0

        l1_par_range = np.sqrt(6.0 / (vocab_size + 300))

        with tf.name_scope('L1_layers'):
            W_l1 = tf.Variable(tf.random_uniform([vocab_size, 300], -l1_par_range, l1_par_range))
            b_l1 = tf.Variable(tf.random_uniform([300], -l1_par_range, l1_par_range))
            query_vec1 = tf.sparse_tensor_dense_matmul(self.query_batch, W_l1) + b_l1

            query_vec1 = tf.nn.tanh(query_vec1)

            doc_vec1 = tf.sparse_tensor_dense_matmul(self.doc_batch, W_l1) + b_l1
            doc_vec1 = tf.nn.tanh(doc_vec1)

        l2_par_range = np.sqrt(6.0 / (300 + 50))
        with tf.name_scope('L2_layers'):
            W_l2 = tf.Variable(tf.random_uniform([300, 50], -l2_par_range, l2_par_range))
            b_l2 = tf.Variable(tf.random_uniform([50], -l2_par_range, l2_par_range))

            query_vec2 = tf.nn.tanh(tf.matmul(query_vec1, W_l2) + b_l2)
            doc_vec2 = tf.nn.tanh(tf.matmul(doc_vec1, W_l2) + b_l2)

        # query_vec2 = tf.tile(query_vec2,[1,num_neg + 1])
        # query_vec2 = tf.reshape(query_vec2,[-1,num_neg+1,50])
        # query_vec2 = tf.reshape(query_vec2,[-1,50])
        # batch*(num_neg+1) ,  50

        query_vec2_norm = tf.nn.l2_normalize(query_vec2, axis=1)
        doc_vec2_norm = tf.nn.l2_normalize(doc_vec2, axis=1)

        batch_cosine = tf.reduce_sum(tf.multiply(query_vec2_norm, doc_vec2_norm), axis=-1)
        #  batch * (neg_num + 1)
        self.batch_cosine = tf.reshape(batch_cosine, [-1, num_neg + 1])

        batch_cosine1 = r * self.batch_cosine

        # self.labels 全部是 0，因为正例每次都是放在第一个位置，负例是放在剩下的4个位置
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                   logits=batch_cosine1)

        self.loss = tf.reduce_mean(self.loss)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.params = tf.trainable_variables()

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.params), 5)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        self.grads_and_vars = optimizer.compute_gradients(self.loss, self.params)
        self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
        # 调用一次上面这个函数，就会将 global_step 自动加 1

    def get_cosine(self, sess, querys, docs):
        feed_dict = {
            self.query_batch: querys,
            self.doc_batch: docs
        }

        batch_cosine = sess.run(self.batch_cosine, feed_dict)
        # batch_size , (num_neg+1)  在进行测试的时候，num_neg = 0,
        batch_cosine = np.reshape(batch_cosine, [-1])

        return batch_cosine



















