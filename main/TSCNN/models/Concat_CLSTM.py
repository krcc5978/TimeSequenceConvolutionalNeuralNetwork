from keras.layers import ConvLSTM2D, Dense, GlobalAveragePooling2D, Input, Concatenate, LSTM
from keras.models import Model


class Concat_CLSTM:

    def __init__(self, input_shape1, input_shape2, label):
        model_input1 = Input(shape=input_shape1)
        model_input2 = Input(shape=input_shape2)

        CLSTM_layer = ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(model_input1)
        global_ave_pool = GlobalAveragePooling2D()(CLSTM_layer)
        mid = Dense(10, activation='sigmoid')(global_ave_pool)

        LSTM_layer = LSTM(300, activation='tanh', batch_input_shape=(None, input_shape2[0], input_shape2[1]))(model_input2)
        mid2 = Dense(10)(LSTM_layer)

        concat_layer = Concatenate(axis=1)([mid, mid2])

        additional_dense = Dense(10, activation='sigmoid')(concat_layer)
        output1 = Dense(label, activation='sigmoid')(additional_dense)
        self.model = Model(inputs=[model_input1, model_input2], outputs=[output1])

    def get_model(self):
        return self.model
