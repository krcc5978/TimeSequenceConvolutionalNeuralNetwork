import numpy as np

from main.TSCNN.TSCNN_Config import TSCNN_Config
from main.const_value import *
from main.TSCNN.utils import make_teacher_data, data_generator


def train():
    tscnn_config = TSCNN_Config('Concat_CRNN')
    train_data, val_data = make_teacher_data(base_path, teacher_directory_list, timesteps)

    # generatorの作成
    train_generator = data_generator(train_data, batch_size, timesteps)
    val_generator = data_generator(val_data, batch_size, timesteps)

    # 学習の開始
    tscnn_config.trainning_start(train_generator, val_generator, len(train_data), len(val_data), batch_size)

def predict():
    pass


if __name__ == '__main__':
    """
    学習 → train
    認証 → predict
    """
    train()
    # predict()
