from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout


class CNN_3D:

    def __init__(self, input_shape, label):
        self.model.add(
            Conv3D(input_shape=input_shape, filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                   padding='same'))
        self.model.add(
            Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                   padding='same'))
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        self.model.add(
            Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                   padding='same'))
        self.model.add(
            Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                   padding='same'))
        self.model.add(
            Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                   padding='same'))
        self.model.add(
            Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                   padding='same'))
        self.model.add(
            Conv3D(filters=8, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                   padding='same'))
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(50))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(label, activation='softmax'))

    def get_model(self):
        return self.model
