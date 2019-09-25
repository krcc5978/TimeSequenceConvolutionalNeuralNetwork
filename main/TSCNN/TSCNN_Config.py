import traceback
from main.const_value import *
from keras import optimizers
from main.TSCNN.callback_method import model_checkpoint


class TSCNN_Config:

    def __init__(self, use_model_name='CLSTM'):
        if use_model_name == 'CLSTM':
            from main.TSCNN.models.CLSTM_model import CLSTM as use_model
            self.model = use_model((timesteps, w, h, c), len(teacher_directory_list)).get_model()
        elif use_model_name == '3DCNN':
            from main.TSCNN.models.CNN_3D_model import CNN_3D as use_model
            self.model = use_model((timesteps, w, h, c), len(teacher_directory_list)).get_model()
        elif use_model_name == 'CRNN':
            from main.TSCNN.models.CRNN import CRNN as use_model
            self.model = use_model((timesteps, w, h, c), len(teacher_directory_list)).get_model()
        elif use_model_name == 'Concat_CLSTM':
            from main.TSCNN.models.Concat_CLSTM import Concat_CLSTM as use_model
            self.model = use_model((timesteps, w, h, c), (timesteps, axis), len(teacher_directory_list)).get_model()
        elif use_model_name == 'Concat_CRNN':
            from main.TSCNN.models.Concat_CRNN import Concat_CRNN as use_model
            self.model = use_model((timesteps, w, h, c), (timesteps, axis), len(teacher_directory_list)).get_model()
        else:
            from main.CNN.models.load_model import load_model as use_model
            self.model = use_model(input_model_path, input_weight_path).get_model()

    def load_weight(self, weight_path):
        """
        :param weight_path: 使用する重みファイルのパス
        :return:
        """
        self.model.load_weights(weight_path)

    def trainning_start(self, train_data, vali_data, num_train, num_val, batch_size, model_save_path='./'):
        """
        :param train_data: 学習データのgenerator
        :param vali_data: 検証データのgenerator
        :param num_train: 学習データの数
        :param num_val: 検証データの数
        :param batch_size: バッチサイズ
        :param model_save_path: 学習モデルの保存場所
        :return:
        """

        # オプティマイザーの設定
        optimizer = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=1e-4)

        # 使用するモデルのコンパイル
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        # モデルの保存
        try:
            model_json_str = self.model.to_json()
            open(model_save_path + 'face_model.json', 'w').write(model_json_str)
        except:
            traceback.print_exc()

        # コールバック関数の宣言
        checkpoint = model_checkpoint('./logs/000/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                      'val_loss',
                                      True,
                                      True,
                                      3)

        # 学習
        self.model.fit_generator(train_data,
                                 steps_per_epoch=max(1, num_train // batch_size),
                                 validation_data=vali_data,
                                 validation_steps=max(1, num_val // batch_size),
                                 epochs=10000,
                                 initial_epoch=0,
                                 callbacks=[checkpoint]
                                 )

        # 結果の出力
        self.model.save_weights('./main/face_model_weights.h5')
