from keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, SimpleRNN, Dense, Dropout, BatchNormalization


class CRNN:

    def __init__(self, input_shape, label):
        self.model.add(TimeDistributed(Conv2D(96, (7, 7),
                                              strides=(2, 2),
                                              kernel_initializer="he_normal",
                                              activation='relu'),
                                       input_shape=input_shape))
        self.model.add(TimeDistributed(MaxPooling2D()))
        self.model.add(TimeDistributed(BatchNormalization()))
        self.model.add(TimeDistributed(Conv2D(256, (5, 5),
                                              strides=(2, 2),
                                              kernel_initializer="he_normal",
                                              activation='relu')))
        self.model.add(TimeDistributed(MaxPooling2D()))
        self.model.add(TimeDistributed(Conv2D(512, (3, 3),
                                              kernel_initializer="he_normal",
                                              activation='relu')))
        self.model.add(TimeDistributed(Conv2D(512, (3, 3),
                                              kernel_initializer="he_normal",
                                              activation='relu')))
        self.model.add(TimeDistributed(MaxPooling2D()))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(SimpleRNN(300, batch_input_shape=self.model.output_shape, return_sequences=False))
        self.model.add(Dense(4096))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2048))
        self.model.add(Dense(label, activation='softmax'))

    def get_model(self):
        return self.model
