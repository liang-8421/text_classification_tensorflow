# -*- coding: utf-8 -*-
# @Time    : 2021/7/2 8:46
# @Author  : miliang
# @FileName: covert_ckpt_to_pb.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow import graph_util
from config import Config
import os


def freeze_graph(input_checkpoint, output_graph, output_node_names):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :param output_node_names: 输出的节点名称,该节点名称必须是原模型中存在的节点
    :return:
    '''
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


if __name__ == '__main__':
    config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    input_checkpoint = "/home/wangzhili/LiangZ/text_classification/1_classification_tf/model_save/bert_use_pooling_None_2021-07-02_00_26_07/model_0.9998_0.9998_0.9998-3130"
    output_graph = "/home/wangzhili/LiangZ/text_classification/1_classification_tf/model_save/bert_use_pooling_None_2021-07-02_00_26_07/model_0.9998_0.9998_0.9998-3130_pb.pb"
    output_node_names = "prob,ArgMax"
    freeze_graph(input_checkpoint, output_graph, output_node_names)


