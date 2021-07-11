# -*- coding: utf-8 -*-
# @Time    : 2021/6/20 19:39
# @Author  : miliang
# @FileName: model_base.py
# @Software: PyCharm

import tensorflow as tf
from pretrain_model_utils.bert.modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint
from total_utils.common import  fenge

class Model(object):
    def __init__(self, config):
        self.input_ids = tf.placeholder(tf.int32, shape=[None, config.sequence_length], name="input_ids")
        self.input_mask = tf.placeholder(tf.int32, shape=[None, config.sequence_length], name="input_mask")
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, config.sequence_length], name="segment_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None], name="label")

        self.dropout_rate = tf.placeholder(tf.float32, shape=None, name="dropout_rate")
        self.is_training = tf.placeholder(tf.bool, shape=None, name="is_training")

        # 用到的参数从config里面参数过来
        self.bert_config_file = config.pretrain_config_file
        self.init_checkpoint = config.pretrain_file
        self.use_pooling = config.use_pooling
        self.label_num = config.label_num

        sequence_output = self.get_bert_embedding()

        # 池化
        if self.use_pooling == "max":
            pass
        elif self.use_pooling == "avg":
            pass
        else:
            avpooled_out = sequence_output[:, 0:1, :]  # pooled_output
            avpooled_out = tf.squeeze(avpooled_out, axis=1)

        avpooled_out = tf.nn.dropout(avpooled_out, keep_prob=(1 - self.dropout_rate))  # delete dropout
        logits = tf.layers.dense(avpooled_out, self.label_num, name="logits")
        self.prob = tf.nn.softmax(logits, axis=-1, name="prob")
        self.predcit = tf.argmax(logits, axis=-1)

        fenge()
        print(self.prob)
        print(self.predcit)
        fenge()


        labels_one_hot = tf.one_hot(self.labels, depth=self.label_num, dtype=tf.float32)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_one_hot)

    def get_bert_embedding(self, bert_init=True):
        model = BertModel(
            config=BertConfig.from_json_file(self.bert_config_file),
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False,
        )

        sequence_output = model.get_sequence_output()

        if bert_init:
            tvars = tf.trainable_variables()
            assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
            tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                print("  name = {}, shape = {}{}".format(var.name, var.shape, init_string))
            print('init bert from checkpoint: {}'.format(self.init_checkpoint))

        return sequence_output
