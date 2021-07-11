# -*- coding: utf-8 -*-
# @Time    : 2021/7/1 22:44
# @Author  : miliang
# @FileName: predict_utils.py.py
# @Software: PyCharm

import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from tqdm import tqdm


def get_session(checkpoint_path):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)
        with session.as_default():
            # Load the saved meta graph and restore variables
            try:
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_path))
            except OSError:
                saver = tf.train.import_meta_graph("{}.ckpt.meta".format(checkpoint_path))
            saver.restore(session, checkpoint_path)

            input_ids = graph.get_operation_by_name("input_ids").outputs[0]
            input_mask = graph.get_operation_by_name("input_mask").outputs[0]
            segment_ids = graph.get_operation_by_name("segment_ids").outputs[0]
            dropout_rate = graph.get_operation_by_name("dropout_rate").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            prob = graph.get_tensor_by_name('prob:0')
            preb = graph.get_tensor_by_name('ArgMax:0')

            def run_predict(feed_dict):
                return session.run([prob, preb], feed_dict)

    print('recover from: {}'.format(checkpoint_path))
    return run_predict, (input_ids, input_mask, segment_ids, dropout_rate, is_training)


def predict(config, test_iter):
    if not test_iter.is_test:
        test_iter.is_test = True

    true_doc_label_list = []
    pred_doc_label_list = []
    predict_fun, feed_keys = get_session(config.read_checkpoint_path)
    for input_ids, input_mask, segment_ids, labels, tokens_list in tqdm(test_iter, position=0, ncols=80, desc='测试中'):
        # 对每一个batch的数据进行预测
        prob, pred = predict_fun(
            dict(
                zip(feed_keys,
                    (input_ids, input_mask, segment_ids, 0, False))
            )
        )

        true_doc_label_list.extend(pred)
        pred_doc_label_list.extend(labels)

    report = classification_report(y_true=true_doc_label_list, y_pred=pred_doc_label_list,
                                   target_names=config.class_list)
    f1 = f1_score(y_true=true_doc_label_list, y_pred=pred_doc_label_list, average='macro')
    p = precision_score(y_true=true_doc_label_list, y_pred=pred_doc_label_list, average='macro')
    r = recall_score(y_true=true_doc_label_list, y_pred=pred_doc_label_list, average='macro')
    print(report)
    print('precision: {}, recall {}, f1 {}'.format(p, r, f1))

    return f1, p, r
