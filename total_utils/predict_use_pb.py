# -*- coding: utf-8 -*-
# @Time    : 2021/7/2 9:13
# @Author  : miliang
# @FileName: predict_use_pb.py.py
# @Software: PyCharm
import sys
sys.path.append("/home/wangzhili/LiangZ/text_classification/1_classification_tf")
import tensorflow as tf
from config import Config
import os
from total_utils.dataiter import DataIterator
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from tqdm import tqdm


def get_session(pb_path):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)
        with session.as_default():
            # Load the saved meta graph and restore variables
            output_graph_def = tf.GraphDef()
            with open(pb_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")

            input_ids = graph.get_operation_by_name("input_ids").outputs[0]
            input_mask = graph.get_operation_by_name("input_mask").outputs[0]
            segment_ids = graph.get_operation_by_name("segment_ids").outputs[0]
            dropout_rate = graph.get_operation_by_name("dropout_rate").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            prob = graph.get_tensor_by_name('prob:0')
            preb = graph.get_tensor_by_name('ArgMax:0')

            def run_predict(feed_dict):
                return session.run([prob, preb], feed_dict)

    print('recover from: {}'.format(pb_path))
    return run_predict, (input_ids, input_mask, segment_ids, dropout_rate, is_training)

def predict(config, test_iter):
    if not test_iter.is_test:
        test_iter.is_test = True

    true_doc_label_list = []
    pred_doc_label_list = []
    predict_fun, feed_keys = get_session(config.pb_file)
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

if __name__ == '__main__':
    config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    test_iter = DataIterator(config, config.source_data_dir + "test.csv", is_test=True)
    config.pb_file = "/home/wangzhili/LiangZ/text_classification/1_classification_tf/model_save/bert_use_pooling_None_2021-07-02_00_26_07/model_0.9998_0.9998_0.9998-3130_pb.pb"
    predict(config, test_iter)




