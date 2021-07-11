# -*- coding: utf-8 -*-
# @Time    : 2021/6/14 18:49
# @Author  : miliang
# @FileName: train_finetune.py.py
# @Software: PyCharm

import os
from config import Config
from total_utils.dataiter import DataIterator
from total_utils.train_utils import train

if __name__ == '__main__':
    config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    config.train_init()
    train_iter = DataIterator(config, config.source_data_dir + "dev.csv", is_test=False)
    dev_iter = DataIterator(config, config.source_data_dir + "dev.csv", is_test=True)

    train(config, train_iter, dev_iter)
