# -*- coding: utf-8 -*-
# @Time    : 2021/7/2 9:29
# @Author  : miliang
# @FileName: predict.py
# @Software: PyCharm

import os
from total_utils.dataiter import DataIterator
from total_utils.predict_utils import predict
from config import Config

if __name__ == '__main__':
    config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    test_iter = DataIterator(config, config.source_data_dir + "test.csv", is_test=True)
    predict(config, test_iter)
