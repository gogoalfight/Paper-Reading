import sys
import tensorflow as tf
import numpy as np

from .import data_utils
from . import seq2seq_model
import  os

class NNTrans:
    def __init__(self):
        self.buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
        self.dtype = tf.float32
        self.from_vocab_size = 6550
        self.to_vocab_size = 6550
        self.size = 256
        self.num_layers = 3
        self.max_gradient_norm = 5.0
        self.batch_size = 1
        self.learning_rate = 0.5
        self.learning_rate_decay_factor = 0.99
        # self.modelPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),"model")
        # self.src_vocab_path = os.path.join(self.modelPath,"vocab6550.from")
        # self.tgt_vocab_path = os.path.join(self.modelPath,"vocab520.to")
        self.modelPath = "/tmp/QL_translate"
        self.src_vocab_path = "/tmp/QL_translate/vocab6550.from"
        self.tgt_vocab_path = "/tmp/QL_translate/vocab6550.to"
        self.src_vocab, _ = data_utils.initialize_vocabulary(self.src_vocab_path)
        self.tgt_vocab,_ = data_utils.initialize_vocabulary(self.tgt_vocab_path)
        self.sess = tf.Session()
        self.dtype = tf.float32
        self.model = seq2seq_model.Seq2SeqModel(
          self.from_vocab_size,
          self.to_vocab_size,
          self.buckets,
          self.size,
          self.num_layers,
          self.max_gradient_norm,
          self.batch_size,
          self.learning_rate,
          self.learning_rate_decay_factor,
          forward_only=True,
          dtype=self.dtype)
        ckpt = tf.train.get_checkpoint_state(self.modelPath)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
  
    def getTransScore(self,src,tgt):
        self.model.batch_size = 1
        srcWords = src
        strSrcSeg = src
        tgtWords = tgt
        strTgtSeg = tgt
        src_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(strSrcSeg), self.src_vocab)
        target_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(strTgtSeg), self.tgt_vocab)
        bucket_id = len(self.buckets) - 1
        for i, bucket in enumerate(self.buckets):
            if bucket[0] >= len(src_ids):
                bucket_id = i
                break
        encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
            {bucket_id: [(src_ids, target_ids)]}, bucket_id)
        _, loss, _ = self.model.step(self.sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, True)
        alpha = 1.0/(1.0 + abs(len(srcWords) - len(tgtWords)))
        return -1*loss