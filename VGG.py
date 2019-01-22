from keras.layers import Input, Dense, Activation, Flatten, Conv2D
from keras.layers import ZeroPadding2D, MaxPooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential

class VGG():

    def __init__(self, cfg, input_shape=(224, 224, 3), num_classes=5005):
        self.cfg = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }
        self.model = self.get_VGG(self.cfg[cfg], input_shape, num_classes)


    def get_VGG(self, cfg, input_shape, num_classes):
        model = Sequential()
        count = 1
        for v in cfg:
            if v == 'M':
                model.add(MaxPooling2D((2,2), strides=(2,2)))
                count += 1
            else:
                if count == 1:
                    print('first')
                    model.add(Conv2D(v, (3, 3), input_shape=input_shape, padding='SAME'))
                    model.add(BatchNormalization())
                    model.add(Activation('relu'))
                    count += 1
                else:
                    model.add(Conv2D(v, (3, 3), padding='SAME'))
                    model.add(BatchNormalization())
                    model.add(Activation('relu'))
                    count += 1
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