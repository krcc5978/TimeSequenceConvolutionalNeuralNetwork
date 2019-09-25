from keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, SimpleRNN, Dense, Dropout, \
    BatchNormalization, Concatenate
from keras.models import Model


class Concat_CRNN:

    def __init__(self, input_shape1, input_shape2, label):
        model_input1 = Input(shape=input_shape1)
        model_input2 = Input(shape=input_shape2)

        time_distribute_layer = TimeDistributed(Conv2D(96, (7, 7),
                                                       strides=(2, 2),
                                                       kernel_initializer="he_normal",
                                                       activation='relu'),
                                                input_shape=input_shape1)(model_input1)
        time_distribute_layer = TimeDistributed(MaxPooling2D())(time_distribute_layer)
        time_distribute_layer = TimeDistributed(BatchNormalization())(time_distribute_layer)
        time_distribute_layer = TimeDistributed(Conv2D(256, (7, 7),
                                                       strides=(2, 2),
                                                       kernel_initializer="he_normal",
                                                       activation='relu'))(time_distribute_layer)
        time_distribute_layer = TimeDistributed(MaxPooling2D())(time_distribute_layer)

        time_distribute_layer = TimeDistributed(Conv2D(512, (7, 7),
                                                       strides=(2, 2),
                                                       kernel_initializer="he_normal",
                                                       activation='relu',
                                                       padding='same'))(time_distribute_layer)

        time_distribute_layer = TimeDistributed(Conv2D(512, (7, 7),
                                                       strides=(2, 2),
                                                       kernel_initializer="he_normal",
                                                       activation='relu',
                                                       padding='same'))(time_distribute_layer)

        time_distribute_layer = TimeDistributed(MaxPooling2D())(time_distribute_layer)
        flatten_layer = TimeDistributed(Flatten())(time_distribute_layer)

        crnn_layer = SimpleRNN(300, batch_input_shape=flatten_layer.shape, return_sequences=False)(flatten_layer)
        crnn_layer = Dense(4096)(crnn_layer)

        rnn_layer = SimpleRNN(100, batch_input_shape=(None, input_shape2[0], input_shape2[1]), return_sequences=False)(
            model_input2)
        rnn_layer = Dense(4096)(rnn_layer)

        concat_layer = Concatenate(axis=1)([crnn_layer, rnn_layer])

        dropout_layer = Dropout(0.5)(concat_layer)

        # layer = Flatten()(concat_layer)
        layer = Dense(2048)(dropout_layer)
        last_layer = Dense(label, activation='softmax')(layer)
        self.model = Model(inputs=[model_input1, model_input2], outputs=[last_layer])

    def get_model(self):
        return self.model
