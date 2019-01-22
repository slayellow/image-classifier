from keras.layers import Input, Dense, Activation, Flatten, Conv2D
from keras.layers import ZeroPadding2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.models import Sequential

class Alexnet():

    def __init__(self, input_shape=(224, 224, 3), num_classes=5005):
        self.model = self.get_alexnet(input_shape, num_classes)

    def get_alexnet(self, input_shape, num_classes):
        model = Sequential()
        # Conv1
        model.add(Conv2D(96, (11, 11), strides=(4, 4), name='conv1', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((3, 3), strides=(2,2)))
        # Conv2
        model.add(ZeroPadding2D((2,2)))
        model.add(Conv2D(256, (5, 5), name='conv2'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((3, 3), strides=(2,2)))
        # Conv3
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(384, (3, 3), name='conv3'))
        model.add(Activation('relu'))
        # Conv4
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(384, (3, 3), name='conv4'))
        model.add(Activation('relu'))
        # Conv5
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3), name='conv5'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((3, 3), strides=(2, 2)))
        # Dense1
        model.add(Flatten(name='flatten') )
        model.add(Dense(6000, name='dense1'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # Dense2
        model.add(Dense(6000, name='dense2'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # Dense3
        model.add(Dense(num_classes, name='dense3'))
        model.add(Activation('softmax', name='softmax'))
        # Loss Func, Optimizer
        optim = Adam(lr=0.001, decay=0.0005)
        model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
        model.summary()
        return model