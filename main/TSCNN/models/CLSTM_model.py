from keras.layers import ConvLSTM2D, Dense, GlobalAveragePooling2D
from keras.models import Sequential


class CLSTM:

    def __init__(self, input_shape, label):
        self.model = Sequential()
        self.model.add(
            ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(16, activation='sigmoid'))
        self.model.add(Dense(label, activation='softmax'))

    def get_model(self):
        return self.model
