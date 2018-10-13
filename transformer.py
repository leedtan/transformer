# Standard PyTorch imports
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layer_norm
import nn_utils
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math, copy
# from torch.autograd import Variable

# # For plots
# #%matplotlib inline
# import matplotlib.pyplot as plt

#

# #!conda install torchtext spacy
# # !python -m spacy download en
# # !python -m spacy download de

# from torchtext import data
# from torchtext import datasets

# import re
# import spacy

# spacy_de = spacy.load('de')
# spacy_en = spacy.load('en')

# url = re.compile('(<url>.*</url>)')

# def tokenize_de(text):
#     return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

# def tokenize_en(text):
#     return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

# # Testing IWSLT
# DE = data.Field(
#     tokenize=tokenize_de,
#     init_token='<bos>',
#     eos_token='<eos>',
#     include_lengths=True)
# EN = data.Field(
#     tokenize=tokenize_en,
#     init_token='<bos>',
#     eos_token='<eos>',
#     include_lengths=True)

# train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN))

# train_it = data.Iterator(
#     train,
#     batch_size=4,
#     sort_within_batch=True,
#     train=True,
#     repeat=False,
#     shuffle=True)
# MIN_WORD_FREQ = 10
# MAX_NUM_WORDS = 1000
# DE.build_vocab(train.src, min_freq=MIN_WORD_FREQ, max_size=MAX_NUM_WORDS)
# EN.build_vocab(train.trg, min_freq=MIN_WORD_FREQ, max_size=MAX_NUM_WORDS)

# num_wds_input = len(DE.vocab.itos)
# num_wds_output = len(EN.vocab.itos)

# num_wds_input, num_wds_output
num_wds_input = 1004

src_tensor = np.random.randint(low=0, high=num_wds_input, size=(4, 81))
src_len = np.array([81, 12, 12, 4])
trg_tensor = np.random.randint(low=0, high=num_wds_input, size=(4, 84))
trg_len = np.array([84, 21, 12, 5])


def masked_softmax(v, mask, dim=2):
  #bs, query dimension, key dimension
  v_mask = v * mask
  v_max = tf.reduce_max(v_mask, dim, keep_dims=True)
  v_stable = v_mask - v_max
  v_exp = tf.exp(v_stable) * mask
  v_exp_sum = tf.reduce_sum(v_exp, dim, keep_dims=True)
  return v_exp / v_exp_sum


class Encoder:
  def __init__(self, num_wds, wd_ind, mask, ndims=20, n_layers=2):
    self.num_wds = num_wds
    self.wd_ind = wd_ind
    self.mask = mask
    self.length = tf.shape(self.wd_ind)[1]
    self.wd_emb = tf.Variable(
        tf.random_uniform([self.num_wds, ndims], minval=-1, maxval=1.))
    self.wd_vec = tf.nn.embedding_lookup(self.wd_emb, wd_ind)
    self.position = tf.reshape(
        tf.range(tf.cast(self.length, tf.float32), dtype=tf.float32) / 10000,
        (1, -1, 1))
    self.w_tilde = embedding = self.wd_vec + self.position
    self.encoding = []
    self.attentionLayers = []
    for _ in range(n_layers):
      attentionLayer = AttentionLayer(embedding, mask)
      embedding = attentionLayer.output
      self.encoding.append(embedding)
      self.attentionLayers.append(attentionLayer)


class AttentionLayer:
  def __init__(self, X, mask, X_decode=None, decode_mask=None, ff_layer=True):
    bs, length, ndim = [v.value for v in X.shape]
    if X_decode is None:
      self.q, self.k, self.v = [
          tf.tanh(tf.layers.dense(X, ndim)) for _ in range(3)
      ]
      decode_mask = mask
    else:
      self.q = tf.tanh(tf.layers.dense(X_decode, ndim))
      self.k, self.v = [tf.tanh(tf.layers.dense(X, ndim)) for _ in range(2)]
    #batch, attention queries, attention keys, embeddings
    self.q_expanded = tf.expand_dims(self.q, 2)
    self.k_expanded = tf.expand_dims(self.k, 1)
    self.v_expanded = tf.expand_dims(self.v, 1)
    self.s_raw = tf.reduce_sum(self.q_expanded * self.k_expanded, -1)
    self.mask = tf.expand_dims(decode_mask, 2) * tf.expand_dims(mask, 1)
    self.s = masked_softmax(self.s_raw, self.mask)
    self.a = tf.expand_dims(self.s, -1) * self.v_expanded
    #A is shape bs, query, key, emb
    self.a_compressed = tf.reduce_sum(self.a, 2)
    self.e = layer_norm(self.a_compressed + X)
    if ff_layer:
      self.output = layer_norm(tf.layers.dense(self.e, ndim) + self.e)
    else:
      self.output = self.e


class Decoder:
  def __init__(self, num_wds, wd_ind, mask, encoder, ndims=20, n_layers=2):
    self.num_wds = num_wds
    self.wd_ind = wd_ind
    self.mask = mask
    self.encoder = encoder
    self.length = tf.shape(self.wd_ind)[1]
    self.wd_emb = tf.Variable(
        tf.random_uniform([self.num_wds, ndims], minval=-1, maxval=1.))
    self.wd_vec = tf.nn.embedding_lookup(self.wd_emb, wd_ind)
    self.position = tf.reshape(
        tf.range(tf.cast(self.length, tf.float32), dtype=tf.float32) / 10000,
        (1, -1, 1))
    self.w_tilde = embedding = self.wd_vec + self.position
    self.decoding = []
    self.self_attentions = []
    self.encoder_attentions = []
    for l_idx in range(n_layers):
      attn = AttentionLayer(embedding, mask, ff_layer=False)
      self.self_attentions.append(attn)
      encode_attn = AttentionLayer(encoder.encoding[l_idx], encoder.mask,
                                   attn.output, mask)
      self.encoder_attentions.append(encode_attn)
      embedding = encode_attn.output

    self.output_raw = tf.layers.dense(embedding, num_wds)
    self.output = masked_softmax(self.output_raw, mask, dim=2)


class Transformer:
  def __init__(self, num_wds):
    self.num_wds = num_wds
    self.learning_rate = tf.placeholder(tf.float32, None)
    self.wd_ind_src = wd_ind_src = tf.placeholder(tf.int32, (None, None))
    self.wd_ind_trg = wd_ind_trg = tf.placeholder(tf.int32, (None, None))
    self.input_lengths = tf.placeholder(tf.int32, [None])
    self.output_lengths = tf.placeholder(tf.int32, [None])
    self.input_mask = tf.sequence_mask(
        self.input_lengths,
        maxlen=tf.shape(self.wd_ind_src)[-1],
        dtype=tf.float32)
    self.output_mask = tf.sequence_mask(
        self.output_lengths,
        maxlen=tf.shape(self.wd_ind_trg)[-1],
        dtype=tf.float32)
    self.encoder = Encoder(num_wds, wd_ind_src, self.input_mask)
    self.decoder = Decoder(num_wds, wd_ind_trg, self.output_mask, self.encoder)
    opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.wd_ind_trg, logits=self.decoder.output_raw)
    self.optimizer, self.grad_norm_total = nn_utils.apply_clipped_optimizer(
        opt, self.loss)


transformer = Transformer(num_wds_input)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

fd = {
    transformer.wd_ind_src: src_tensor,
    transformer.wd_ind_trg: trg_tensor,
    transformer.input_lengths: src_len,
    transformer.output_lengths: trg_len
}
sess.run(transformer.loss, fd)

print(src_tensor.shape, src_len.shape, trg_tensor.shape, trg_len.shape)

#for train_batch in train_it:
# src_tensor = train_batch.src[0].data.cpu().numpy().transpose()
# src_len = train_batch.src[1].cpu().numpy()
# trg_tensor = train_batch.trg[0].data.cpu().numpy().transpose()
# trg_len = train_batch.trg[1].cpu().numpy()
#     print(src_tensor.shape, src_len.shape, trg_tensor.shape, trg_len.shape)
#     print(src_tensor, src_len, trg_tensor, trg_len)
#print(loss)
