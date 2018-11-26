#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class Model():
    def __init__(self, learning_rate=0.001, batch_size=16, num_steps=32, num_words=5000, dim_embedding=128, rnn_layers=3):
        r"""初始化函数

        Parameters
        ----------
        learning_rate : float
            学习率.
        batch_size : int
            batch_size,一个batch有多少输入序列.
        num_steps : int
            RNN有多少个time step，也就是输入数据每个seq的长度.
        num_words : int
            字典里有多少个字，用作embeding变量的第一个维度的确定和onehot编码.
        dim_embedding : int
            embding中，编码后的字向量的维度
        rnn_layers : int
            有多少个RNN层，在这个模型里，一个RNN层就是一个RNN Cell，各个Cell之间通过TensorFlow提供的多层RNNAPI（MultiRNNCell等）组织到一起

        batch大小 = batch_size * num_steps

        """
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_words = num_words
        self.dim_embedding = dim_embedding
        self.rnn_layers = rnn_layers
        self.learning_rate = learning_rate

    def build(self, embedding_file=None):

        # 创建一个LSTM Cell
        def get_a_cell(state_size=128, keep_prob=1):
            # LSTM Cell
            lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=state_size)
            # Dropout防止过拟合
            drop = tf.nn.rnn_cell.DropoutWrapper(cell=lstm, output_keep_prob=keep_prob)
            return drop


        # global step
        self.global_step = tf.Variable(
            0, trainable=False, name='self.global_step', dtype=tf.int64)

        self.X = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='input')
        self.Y = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='label')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.variable_scope('embedding'):
            if embedding_file:
                # if embedding file provided, use it.
                embedding = np.load(embedding_file)
                embed = tf.constant(embedding, name='embedding')
            else:
                # if not, initialize an embedding and train it.
                embed = tf.get_variable(
                    'embedding', [self.num_words, self.dim_embedding])
                tf.summary.histogram('embed', embed)

            self.rnn_inputs = tf.nn.embedding_lookup(embed, self.X)

        with tf.variable_scope('rnn'):
            # 用tf.nn.rnn_cell MultiRNNCell创建self.rnn_layers层RNN
            # state_size是(self.dim_embedding,self.dim_embedding,self.dim_embedding)也就是每个隐层大小均为128
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.dim_embedding, self.keep_prob) for _ in range(self.rnn_layers)])
            # 初始状态，通过zero_state得到一个全0的初始状态
            self.initial_state = cell.zero_state(self.batch_size, tf.float32)
            # 通过dynamic_rnn对cell在时间维度进行展开 lstm_outputs 64*32*128
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.rnn_inputs,
                                                                    initial_state=self.initial_state)

        # 通过lstm_outputs得到概率
        seq_output = tf.concat(self.lstm_outputs, 1)

        # flatten it  展开
        seq_output_final = tf.reshape(seq_output, [-1, self.dim_embedding])


        with tf.variable_scope('softmax'):
            # softmax_w = tf.Variable(tf.truncated_normal([self.dim_embedding, self.num_words], stddev=0.1))
            # softmax_b = tf.Variable(tf.zeros(self.num_words))
            softmax_w = tf.get_variable('softmax_W', [self.dim_embedding, self.num_words],
                                  initializer=tf.random_normal_initializer(stddev=0.01))
            softmax_b = tf.get_variable('softmax_b', [self.num_words], initializer=tf.constant_initializer(0.0))

        self.logits = tf.matmul(seq_output_final, softmax_w) + softmax_b
        tf.summary.histogram('logits', self.logits)

        self.predictions = tf.nn.softmax(self.logits, name='predictions')

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.Y, [-1]))
        mean, var = tf.nn.moments(self.logits, -1)
        self.loss = tf.reduce_mean(loss)
        tf.summary.scalar('logits_loss', self.loss)

        var_loss = tf.divide(10.0, 1.0+tf.reduce_mean(var))
        tf.summary.scalar('var_loss', var_loss)
        # 把标准差作为loss添加到最终的loss里面，避免网络每次输出的语句都是机械的重复
        self.loss = self.loss + var_loss
        tf.summary.scalar('total_loss', self.loss)

        # gradient clip
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(
            zip(grads, tvars), global_step=self.global_step)

        tf.summary.scalar('loss', self.loss)

        self.merged_summary_op = tf.summary.merge_all()
