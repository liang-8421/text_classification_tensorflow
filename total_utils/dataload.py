# -*- coding: utf-8 -*-
# @Time    : 2021/6/14 18:50
# @Author  : miliang
# @FileName: dataload.py.py
# @Software: PyCharm

from config import Config
import pandas as pd

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text, label):
        self.guid = guid
        self.text = text
        self.label = label


def load_data(file_path):
    """读取数据"""
    df = pd.read_csv(file_path)
    lines = []
    for _, item in df.iterrows():
        lines.append((item['text'], item['label']))

    return lines



def create_example(file_path):
    """put data into example """
    example = []
    lines = load_data(file_path)
    file_type = file_path.split('/')[-1].split('.')[0]
    for index, content in enumerate(lines):
        guid = "{0}_{1}".format(file_type, str(index))
        text = content[0]
        label = content[1]
        example.append(InputExample(guid=guid, text=text, label=label))

    return example


if __name__ == '__main__':
    config = Config()
    a = create_example(config.source_data_dir+"train.csv")
    for i in a:
        # print(i.text)
        print(list(i.text))
        print(i.label)
        # break
    print("sdf")
