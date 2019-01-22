import six
from keras.layers import Input, Dense, Activation, Flatten, Conv2D
from keras.layers.merge import add
from keras.layers import AveragePooling2D, ZeroPadding2D, MaxPooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential, Input, Model
from keras.regularizers import l2
from keras import backend as K
class Resnet():

    def __init__(self, input_shape=(224, 224, 3), num_classes=5005):
        self.model = self.build_resnet_50(input_shape, num_classes)

    # 34 Layer
    def build_resnet_34(self, input_shape, num_outputs):
        return self.get_Resnet(input_shape, num_outputs, self.basic_block, [3, 4, 6, 3])

    # 50 Layer
    def build_resnet_50(self, input_shape, num_outputs):
        return self.get_Resnet(input_shape, num_outputs, self.bottleneck, [3, 4, 6, 3])

    # 101 Layer
    def build_resnet_101(self, input_shape, num_outputs):
        return self.get_Resnet(input_shape, num_outputs, self.bottleneck, [3, 4, 23, 3])

    # 기본적인 Block 형태 34 Layer 미만
    def basic_block(self, filters, strides=(1,1)):
        # 2016버전 Resnet Block: pre_activation 형태
        def func(input):
            conv1 = self.bn_relu_conv(kernel=(3,3), filter=filters, stride=strides)(input)
            residual = self.bn_relu_conv(kernel=(3, 3), filter=filters, stride=strides)(conv1)
            return self.shortcut(input, residual)
        return func

    # 34 Layer 이상 사용하는 Block형태
    def bottleneck(self, filters, strides=(1,1)):

        def func(input):
            conv1 = self.bn_relu_conv(filter=filters, kernel=(1,1), stride=strides)(input)
            conv3 = self.bn_relu_conv(filter=filters, kernel=(3,3), stride=strides)(conv1)
            residual = self.bn_relu_conv(filter=filters*4, kernel=(1,1), stride=strides)(conv3)
            return self.shortcut(input, residual)
        return func

    # F(x) + x 형태
    def shortcut(self, input, residual):
        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(residual)
        stride_width = int(round(input_shape[1] / residual_shape[2]))
        stride_height = int(round(input_shape[1] / residual_shape[2]))
        equal_channels = input_shape[3] == residual_shape[3]
        shortcut = input
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)
        return add([shortcut, residual])

    # BN --> ReLU
    def bn_relu(self, input):
        norm = BatchNormalization()(input)
        return Activation('relu')(norm)

    # CNN --> Bn --> ReLU
    def conv_bn_relu(self, kernel, filter, stride=(1,1)):

        def func(input):
            conv = Conv2D(kernel_size=kernel, filters=filter, strides=stride,kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4), padding='same')(input)
            return self.bn_relu(conv)
        return func

    # BN --> ReLU --> CNN
    def bn_relu_conv(self, kernel, filter, stride):

        def func(input):
            activation = self.bn_relu(input)
            return Conv2D(filters=filter, kernel_size=kernel, strides=stride, kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4), padding='same')(activation)
        return func

    # Block 형성을 위한 틀
    def residual_block(self, block_function, filter, repetitions):

        def func(input):
            for i in range(repetitions):
                strides = (1,1)
                input = block_function(filters=filter, strides=strides)(input)
            return input
        return func

    def get_Resnet(self, input_shape, num_classes, block_fn, residual):
        # First Conv Layer
        input = Input(shape=input_shape)
        conv1 = self.conv_bn_relu(filter=64, kernel=(7,7), stride=(2,2))(input)
        pool1 = MaxPooling2D((3,3), strides=(2,2), padding='same')(conv1)
        # Residual Layer
        block = pool1
        filter=64
        for i, r in enumerate(residual):
            block = self.residual_block(block_function=block_fn, filter=filter,repetitions=r)(block)
            filter *= 2

        block = self.bn_relu(block)

        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[1], block_shape[2]), strides=(1,1))(block)
        flatten = Flatten()(pool2)
        dense = Dense(units=num_classes, kernel_initializer="he_normal", activation='softmax')(flatten)

        model = Model(inputs=input, outputs=dense)

        optim = Adam(lr=0.001, decay=0.0005)
        model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
        model.summary()
        return model
