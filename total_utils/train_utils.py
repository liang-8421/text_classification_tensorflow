# -*- coding: utf-8 -*-
# @Time    : 2021/6/14 18:50
# @Author  : miliang
# @FileName: train_utils.py.py
# @Software: PyCharm

from sklearn.metrics import classification_report
import tensorflow as tf
from models.model_base import Model
from pretrain_model_utils.bert.optimization import create_optimizer
from tqdm import tqdm
import os
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score


def train(config, train_iter, dev_iter):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)
        with session.as_default():
            model = Model(config)  # config.sequence_length,

            global_step = tf.Variable(0, name='step', trainable=False)
            learning_rate = tf.train.exponential_decay(config.down_learning_rate, global_step, config.decay_step,
                                                       config.decay_rate, staircase=True)
            normal_optimizer = tf.train.AdamOptimizer(learning_rate)
            all_variables = graph.get_collection('trainable_variables')
            bert_var_list = [x for x in all_variables if 'bert' in x.name]
            normal_var_list = [x for x in all_variables if 'bert' not in x.name]
            normal_op = normal_optimizer.minimize(model.loss, global_step=global_step, var_list=normal_var_list)
            num_batch = int(train_iter.num_records / config.batch_size * config.train_epoch)
            embed_step = tf.Variable(0, name='step', trainable=False)
            if bert_var_list:  # 对bert微调
                print('bert trainable!!')
                word2vec_op, pretrain_learning_rate, embed_step = create_optimizer(
                    model.loss, config.pretrain_learning_rate, num_train_steps=num_batch,
                    num_warmup_steps=int(num_batch * 0.05), use_tpu=False, variable_list=bert_var_list
                )
                train_op = tf.group(normal_op, word2vec_op)
            else:
                train_op = normal_op

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)
            if config.continue_training:
                print('recover from: {}'.format(config.checkpoint_path))
                saver.restore(session, config.checkpoint_path)
            else:
                session.run(tf.global_variables_initializer())

            cum_step = 0
            for i in range(config.train_epoch):
                for input_ids, input_mask, segment_ids, labels, tokens_list in tqdm(train_iter, position=0, ncols=80,
                                                                                    desc='训练中'):
                    feed_dict = {
                        model.input_ids: input_ids,
                        model.input_mask: input_mask,
                        model.segment_ids: segment_ids,
                        model.labels: labels,

                        model.dropout_rate: config.dropout_rate,
                        model.is_training: True,
                    }

                    _, step, _, loss, lr = session.run(
                        fetches=[train_op,
                                 global_step,
                                 embed_step,
                                 model.loss,
                                 learning_rate
                                 ],
                        feed_dict=feed_dict)

                P, R, F1 = set_test(config, model, dev_iter, session)
                print('dev set : step_{},precision_{},recall_{},F1_{}'.format(cum_step, P, R, F1))

                saver.save(session, os.path.join(config.model_save_path, 'model_{:.4f}_{:.4f}_{:.4f}'.format(P, R, F1)),
                           global_step=step)


def set_test(config, model, test_iter, session):
    if not test_iter.is_test:
        test_iter.is_test = True

    true_doc_label_list = []
    pred_doc_label_list = []
    for input_ids, input_mask, segment_ids, labels, tokens_list in tqdm(test_iter, position=0, ncols=80, desc='验证中'):
        feed_dict = {
            model.input_ids: input_ids,
            model.input_mask: input_mask,
            model.segment_ids: segment_ids,
            model.labels: labels,

            model.dropout_rate: 0,
            model.is_training: False,
        }

        prob, pred = session.run(
            fetches=[model.prob, model.predcit],
            feed_dict=feed_dict
        )

        true_doc_label_list.extend(pred)
        pred_doc_label_list.extend(labels)

    report = classification_report(y_true=true_doc_label_list, y_pred=pred_doc_label_list,
                                   target_names=config.class_list)
    f1 = f1_score(y_true=true_doc_label_list, y_pred=pred_doc_label_list, average='macro')
    p = precision_score(y_true=true_doc_label_list, y_pred=pred_doc_label_list, average='macro')
    r = recall_score(y_true=true_doc_label_list, y_pred=pred_doc_label_list, average='macro')
    config.logger.info(report)
    config.logger.info('precision: {}, recall {}, f1 {}'.format(p, r, f1))

    return f1, p, r
