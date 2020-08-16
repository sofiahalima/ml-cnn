from keras import Sequential
from keras.backend import categorical_crossentropy
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.losses import sparse_categorical_crossentropy
from keras import backend as K



class LeNet:
    @staticmethod
    def build(inputShape, numClasses,
              activation="relu", weightsPath=None):
        # initialize the model
        model = Sequential()
        model.add(Conv2D(6, 5, padding="valid", strides=(1, 1), input_shape=inputShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        model.add(Conv2D(16, 5, padding="valid", strides=(1, 1)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())

        # full-connected
        model.add(Dense(120))
        model.add(Activation(activation))

        # second FC
        model.add(Dense(84))
        model.add(Activation(activation))

        model.add(Dense(numClasses))
        # at last softmax classifier
        model.add(Activation("softmax"))
        model.compile(optimizer='adam',
                    loss=sparse_categorical_crossentropy,
                    metrics=['accuracy'])

        return model

class AlexNet:

    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses,
              activation="relu", weightsPath=None):
        model = Sequential()

        inputShape = (imgRows, imgCols, numChannels)

        if K.image_data_format() == "channels_first":
            inputShape = (numChannels, imgRows, imgCols)

        # 1st CONV Layer
        model.add(Conv2D(filters=96, input_shape=(28, 28, 1), kernel_size=(2, 2),
                         strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        # max pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # 2nd CONV Layer
        model.add(Conv2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        # max pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # 3rd CONV Layer
        model.add(Conv2D(filters=384, kernel_size=(2, 2), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))

        # 4th CONV Layer
        model.add(Conv2D(filters=384, kernel_size=(2, 2), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))

        # 5th CONV Layer
        model.add(Conv2D(filters=256, kernel_size=(2, 2), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        # max pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        #  passing to a Fully Connected Layer
        model.add(Flatten())
        # 1st Fully Connected Layer
        model.add(Dense(4096, input_shape=(28 * 28 * 1,)))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))

        # 2nd Fully Connected Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # 3rd Fully Connected Layer
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # Output Layer
        model.add(Dense(numClasses))
        model.add(Activation('softmax'))

        # model.compile(loss="categorical_crossentropy", optimizer='adam',
        #               metrics=["accuracy"])
        return model
