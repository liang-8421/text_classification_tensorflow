# -*- coding: utf-8 -*-
# @Time    : 2021/6/6 19:10
# @Author  : miliang
# @FileName: config.py.py
# @Software: PyCharm

from pretrain_model_utils.bert import tokenization
import numpy as np
import tensorflow as tf
import os
import datetime
from total_utils.common import get_logger


class Config(object):
    def __init__(self):
        self.gpu_id = -1
        self.use_pooling = ["max", "avg", "None"][2]

        # 基本参数
        self.train_epoch = 10
        self.random_seed = 2021
        self.batch_size = 32
        self.sequence_length = 128

        self.dropout_rate = 0.2
        self.decay_rate = 0.25
        self.decay_step = int(2800 / self.batch_size)
        self.down_learning_rate = 1e-4  # 下接结构的学习率
        self.pretrain_learning_rate = 5e-5  # BERT的微调学习率

        # 继续训练
        self.num_checkpoints = 10
        self.continue_training = False
        self.checkpoint_path = None

        self.origin_data_dir = "/home/wangzhili/LiangZ/text_classification/1_classification_tf/datasets/origin_data/"
        self.source_data_dir = "/home/wangzhili/LiangZ/text_classification/1_classification_tf/datasets/source_data/"
        self.model_save_path = "/home/wangzhili/LiangZ/text_classification/1_classification_tf/model_save/"
        self.config_file_path = "/home/wangzhili/LiangZ/text_classification/1_classification_tf/config.py"
        self.read_checkpoint_path = "/home/wangzhili/LiangZ/text_classification/1_classification_tf/model_save/bert_use_pooling_None_2021-07-02_00_26_07/model_0.9998_0.9998_0.9998-3130"

        self.class_list = ['finance', 'realty', 'stocks', 'education', 'science', 'society',
                           'politics', 'sports', 'game', 'entertainment']
        self.class2id = {'finance': 0, 'realty': 1, 'stocks': 2, 'education': 3, 'science': 4, 'society': 5,
                         'politics': 6, 'sports': 7, 'game': 8, 'entertainment': 9}
        self.id2class = {0: 'finance', 1: 'realty', 2: 'stocks', 3: 'education', 4: 'science', 5: 'society',
                         6: 'politics', 7: 'sports', 8: 'game', 9: 'entertainment'}
        self.label_num = len(self.class2id)

        # bert 相关路径
        self.pretrain_file = "/home/wangzhili/pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt"
        self.pretrain_config_file = "/home/wangzhili/pretrained_model/chinese_L-12_H-768_A-12/bert_config.json"
        self.vocab_file = "/home/wangzhili/pretrained_model/chinese_L-12_H-768_A-12/vocab.txt"
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=True)

    def train_init(self):
        np.random.seed(self.random_seed)
        tf.set_random_seed(self.random_seed)
        self.get_save_path()

    def get_save_path(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

        self.model_save_path = self.model_save_path + "bert_use_pooling_{}_{}".format(self.use_pooling, timestamp)

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # 将config文件写入文件夹中
        with open(self.model_save_path + "/config.txt", "w", encoding="utf8") as fw:
            with open(self.config_file_path, "r", encoding="utf8") as fr:
                content = fr.read()
                fw.write(content)

        self.logger = get_logger(self.model_save_path + "/log.log")
