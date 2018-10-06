# Standard PyTorch imports
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layer_norm
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
            tf.range(tf.cast(self.length, tf.float32), dtype=tf.float32) /
            10000, (1, -1, 1))
        self.w_tilde = embedding = self.wd_vec + self.position
        self.encoding = []
        self.attentionLayers = []
        for _ in range(n_layers):
            attentionLayer = AttentionLayer(embedding, mask)
            embedding = attentionLayer.output
            self.encoding.append(embedding)
            self.attentionLayers.append(attentionLayer)


class AttentionLayer:
    def __init__(self, X, mask, X_decode=None, decode_mask=None,
                 ff_layer=True):
        bs, length, ndim = [v.value for v in X.shape]
        if X_decode is None:
            self.q, self.k, self.v = [
                tf.tanh(tf.layers.dense(X, ndim)) for _ in range(3)
            ]
        else:
            pass
        self.q_expanded = tf.expand_dims(self.q, 2)
        self.k_expanded = tf.expand_dims(self.k, 1)
        self.s_raw = tf.reduce_sum(self.q_expanded * self.k_expanded, -1)
        self.mask = tf.expand_dims(mask, 1) * tf.expand_dims(mask, 2)
        self.s = masked_softmax(self.s_raw, self.mask)
        self.a = self.s * self.v
        self.e = layer_norm(self.a + X)
        if ff_layer:
            self.output = layer_norm(tf.layers.dense(self.e, ndim) + self.e)
        else:
            self.output = self.e


class Decoder:
    def __init__(self, num_wds, wd_ind, mask, encoder, ndims=20, n_layers=2):
        pass


class Transformer:
    def __init__(self, num_wds):
        self.num_wds = num_wds
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
        self.decoder = Decoder(num_wds, wd_ind_trg, self.output_mask,
                               self.encoder)

        pass


transformer = Transformer(num_wds_input)

print(src_tensor.shape, src_len.shape, trg_tensor.shape, trg_len.shape)

#for train_batch in train_it:
# src_tensor = train_batch.src[0].data.cpu().numpy().transpose()
# src_len = train_batch.src[1].cpu().numpy()
# trg_tensor = train_batch.trg[0].data.cpu().numpy().transpose()
# trg_len = train_batch.trg[1].cpu().numpy()
#     print(src_tensor.shape, src_len.shape, trg_tensor.shape, trg_len.shape)
#     print(src_tensor, src_len, trg_tensor, trg_len)
#print(loss)
