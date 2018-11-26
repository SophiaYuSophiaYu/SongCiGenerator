#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import copy

import numpy as np


def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data


def index_data(sentences, dictionary):
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)

# 数据处理中，data为文本中一段随机截取的文字，label为data对应的下一个标号的文字。
# 以苏轼的江神子（江城子）为例：输入为 “老夫聊发少年”，则对应的label为"夫聊发少年狂"
def get_train_data(vocabulary, batch_size, num_steps):
    """
    获取数据
    :param vocabulary: word列表
    :param batch_size: batch大小
    :param num_steps: time step 数量，即每个seq长度
    :return:
    """
    vocabulary = np.array(copy.copy(vocabulary))
    vocabulary = vocabulary.flatten()

    # batch数量
    n_batches = int(len(vocabulary) / (batch_size*num_steps))
    vocabulary = vocabulary[: batch_size*num_steps*n_batches]
    # 变为batch_size行，num_steps*n_batches列
    vocabulary = vocabulary.reshape((batch_size, -1))

    # 为vocabulary每行最后添加下一个元素（用于获取lable）
    temp = vocabulary[:, 0:1]
    temp[:-1, :] = temp[1:, :]
    vocabulary = np.concatenate((vocabulary, temp), axis=1)

    while True:
        # 打乱vocabulary行
        np.random.shuffle(vocabulary)
        is_over = False
        for n in range(0, vocabulary.shape[1], num_steps):
            if n + num_steps + 1 >= vocabulary.shape[1]:
                is_over = True
                break
            data = vocabulary[:, n:n+num_steps]
            # label的0到倒数第二列与input的1到最后一列相同，
            # 例：输入为 “老夫聊发少年”，则对应的label为"夫聊发少年狂"
            lable = vocabulary[:, n+1:n+num_steps+1]

            # 把一个函数变成一个 generator,调用函数不会执行函数，而是返回一个 iterable 对象
            # 下次迭代时，代码从 yield 的下一条语句继续执行
            yield data, lable
        if is_over:
            print('Have trained all of the data once!')
            break

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
