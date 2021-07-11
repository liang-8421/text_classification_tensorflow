# -*- coding: utf-8 -*-
# @Time    : 2021/6/6 19:47
# @Author  : miliang
# @FileName: data_preprocess.py.py
# @Software: PyCharm

from config import Config
import pandas as pd

def Get_class2id(file_path):
    class_list, class2id, id2class = [], dict(), dict()
    with open(file_path, "r", encoding="utf-8") as fr:
        for i in fr:
            i = i.strip()
            class_list.append(i)
    for num, value in enumerate(class_list):
        id2class[num] = value
        class2id[value] = num

    return class2id, id2class

config = Config()
class2id, id2class = Get_class2id(config.origin_data_dir + "class.txt")




def get_csv(file_path):
    text_list, label_list = [], []
    with open(file_path, "r", encoding="utf-8") as fr:
        for i in fr:
            item = i.strip().split("\t")
            text_list.append(item[0])
            label_list.append(item[1])

    df = pd.DataFrame({
        "text": text_list,
        "label": label_list
    })
    df["label"] = df["label"].apply(restore_label)
    return df


def restore_label(label_id):
    return id2class[int(label_id)]




if __name__ == '__main__':

    train_df = get_csv(config.origin_data_dir+"train.txt")
    dev_df = get_csv(config.origin_data_dir+"dev.txt")
    test_df = get_csv(config.origin_data_dir+"test.txt")
    #
    # #生成csv文件
    train_df.to_csv(config.source_data_dir+"train.csv", index=False, encoding="utf-8")
    dev_df.to_csv(config.source_data_dir+"dev.csv", index=False, encoding="utf-8")
    test_df.to_csv(config.source_data_dir+"test.csv",  index=False, encoding="utf-8")






