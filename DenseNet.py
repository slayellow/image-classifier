from keras.layers import Dense, Activation, Conv2D, multiply, Reshape, Dropout
from keras.layers.merge import add, concatenate
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.models import Input, Model
from keras.regularizers import l2
from keras import backend as K


class DenseNet():

    def __init__(self, input_shape=(224, 224, 3), num_classes=5005):
        self.model = self.build_DenseNet_121(input_shape, num_classes)

    # 121 Layer
    def build_DenseNet_121(self, input_shape, num_outputs):
        return self.get_DenseNet(input_shape, num_outputs, [6, 12, 24, 16])

    # 169 Layer
    def build_DenseNet_169(self, input_shape, num_outputs):
        return self.get_DenseNet(input_shape, num_outputs, [6, 12, 32, 32])

    # 201 Layer
    def build_DenseNet_201(self, input_shape, num_outputs):
        return self.get_DenseNet(input_shape, num_outputs, [6, 12, 48, 32])

    # Bottleneck Layer
    def conv_block(self, filters, dropout_rate=None):

        def func(input):
            bn_1 = BatchNormalization()(input)
            relu_1 = Activation('relu')(bn_1)
            conv_1 = Conv2D(filters=filters * 4, kernel_size=(1, 1), padding="same", kernel_initializer="he_normal",
                            kernel_regularizer=l2(0.0001))(relu_1)
            # Dropout 유무
            if dropout_rate:
                conv_1 = Dropout(dropout_rate)(conv_1)
            bn_2 = BatchNormalization()(conv_1)
            relu_2 = Activation('relu')(bn_2)
            conv_3 = Conv2D(filters=filters, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal",
                            kernel_regularizer=l2(0.0001))(relu_2)
            # Dropout 유무
            if dropout_rate:
                conv_3 = Dropout(dropout_rate)(conv_3)
            return conv_3
        return func

    # Transition Layer
    def transition(self, filters, compression=0.5, dropout_rate=None):
        # BN --> 1x1 Conv --> 2x2 Average Pooling
        def func(input):
            bn = BatchNormalization()(input)
            relu = Activation('relu')(bn)
            conv1 = Conv2D(kernel_size=(1, 1), filters=int(filters * compression), strides=(1, 1), padding='same', kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(relu)
            # Dropout 유무
            if dropout_rate:
                conv1 = Dropout(dropout_rate)(conv1)

            pool = AveragePooling2D((2, 2), strides=(2, 2))(conv1)
            return pool
        return func

    # Dense Block
    def dense_block(self, filter, repetitions, growth_rate, dropout_rate=None):

        def func(input):
            filters = filter
            prev_input_list = [input]
            concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

            for i in range(repetitions):
                next = self.conv_block(growth_rate, dropout_rate=dropout_rate)(input)
                prev_input_list.append(next)
                input = concatenate([input, next], axis=concat_axis)
                filters += growth_rate
            return input, filters
        return func

    # Block 형성을 위한 틀
    def residual_block(self, block_function, filter, repetitions, first=False):

        def func(input):
            for i in range(repetitions):
                strides = (1, 1)
                if i == 0 and not first:
                    strides = (2, 2)
                input = block_function(filters=filter, strides=strides, is_first=(first and i == 0))(input)
            return input

        return func

    def get_DenseNet(self, input_shape, num_classes, dense):
        # Default
        # growth rate: 16
        # compression: 0.5
        # dropout_rate: 0.5
        compression = 0.5
        dropout_rate = 0.5
        growth_rate = 32
        filter = growth_rate * 2
        #  First Conv Layer
        input = Input(shape=input_shape)
        conv1 = Conv2D(filters=filter, kernel_size=(7, 7), strides=(2, 2), padding='same',
                       kernel_initializer='he_normal')(input)
        bn1 = BatchNormalization()(conv1)
        relu1 = Activation('relu')(bn1)
        pool1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(relu1)

        # Dense Block
        block = pool1
        for i in range(len(dense) - 1):
            block, filter = self.dense_block(filter=filter, repetitions=dense[i], growth_rate=growth_rate, dropout_rate=dropout_rate)(block)
            block = self.transition(filter, compression=compression, dropout_rate=dropout_rate)(block)
            filter = int(filter * compression)
        block, filter = self.dense_block(filter=filter, repetitions=dense[-1], growth_rate=growth_rate, dropout_rate=dropout_rate)(block)
        block = BatchNormalization()(block)
        block = Activation('relu')(block)

        # Classification Layer
        pool2 = GlobalAveragePooling2D()(block)
        dense = Dense(units=num_classes, kernel_initializer="he_normal", activation='softmax')(pool2)
        model = Model(inputs=input, outputs=dense)

        optim = Adam(lr=0.001, decay=0.0005)
        model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
        model.summary()
        return model
