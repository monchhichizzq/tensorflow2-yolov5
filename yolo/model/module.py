from numpy import concatenate
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, MaxPool2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda
# from tensorflow.keras.layers import DepthwiseConv2D
# from tensorflow.keras.layers.experimental import SyncBatchNormalization


# class Mish(object):
#     def __call__(self, x):
#         return x * tf.math.tanh(tf.math.softplus(x))


# class Swish(object):
#     def __call__(self, x):
#         return tf.nn.swish(x)  # tf.nn.leaky_relu(x, alpha=0.1)

class Mish(Layer):
    def __init__(self) -> None:
        super(Mish, self).__init__()
        pass 

    def call(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))
    

class Swish(Layer):
    def __init__(self) -> None:
        super(Swish, self).__init__()
        pass

    def call(self, x):
        return tf.nn.swish(x)  # tf.nn.leaky_relu(x, alpha=0.1)

class Conv(Layer):
    def __init__(self, filters, kernel_size, strides, padding='SAME', groups=1):
        super(Conv, self).__init__()
        self.conv = Conv2D(filters, kernel_size, strides, padding, groups=groups, use_bias=False,
                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                           kernel_regularizer=tf.keras.regularizers.L2(5e-4))
        self.bn = BatchNormalization()
        self.activation = Mish()

    def call(self, x):
        return self.activation(self.bn(self.conv(x)))
    
    def get_config(self):
        config = super(Conv, self).get_config()
        config.update({'conv': self.conv,
                        'bn': self.bn,
                        'activation': self.activation,
                      })
        return config


class DWConv(Layer):
    def __init__(self, filters, kernel_size, strides):
        super(DWConv, self).__init__()
        self.conv = Conv(filters, kernel_size, strides, groups=1)  # Todo

    def call(self, x):
        return self.conv(x)
    
    def get_config(self):
        config = super(DWConv, self).get_config()
        config.update({'conv': self.conv
                      })
        return config


class Focus(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='SAME'):
        super(Focus, self).__init__()
        self.conv = Conv(filters, kernel_size, strides, padding)

    def call(self, x):
        return self.conv(Concatenate(axis=-1)([x[..., ::2, ::2, :],
                                    x[..., 1::2, ::2, :],
                                    x[..., ::2, 1::2, :],
                                    x[..., 1::2, 1::2, :]]))
    def get_config(self):
        config = super(Focus, self).get_config()
        config.update({'conv': self.conv
                      })
        return config

class CrossConv(Layer):
    def __init__(self, filters, kernel_size, strides=1, groups=1, expansion=1, shortcut=False):
        super(CrossConv, self).__init__()
        units_e = int(filters * expansion)
        self.conv1 = Conv(units_e, (1, kernel_size), (1, strides))
        self.conv2 = Conv(filters, (kernel_size, 1), (strides, 1), groups=groups)
        self.shortcut = shortcut

    def call(self, x):
        if self.shortcut:
            return x + self.conv2(self.conv1(x))
        return self.conv2(self.conv1(x))
    
    def get_config(self):
        config = super(CrossConv, self).get_config()
        config.update({'conv1': self.conv1,
                       'conv2': self.conv2,
                       'shortcut': self.shortcut
                      })
        return config


class MP(Layer):
    # Spatial pyramid pooling layer
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = MaxPool2D(pool_size=k, strides=k)

    def forward(self, x):
        return self.m(x)
    
    def get_config(self):
        config = super(MP, self).get_config()
        config.update({'m': self.m,
                      })
        return config


class Bottleneck(Layer):
    def __init__(self, units, shortcut=True, expansion=0.5):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv(int(units * expansion), 1, 1)
        self.conv2 = Conv(units, 3, 1)
        self.shortcut = shortcut

    def call(self, x):
        if self.shortcut:
            return x + self.conv2(self.conv1(x))
        return self.conv2(self.conv1(x))
    
    def get_config(self):
        config = super(Bottleneck, self).get_config()
        config.update({'conv1': self.conv1,
                        'conv2': self.conv2,
                        'shortcut': self.shortcut
                        })
        return config


class BottleneckCSP(Layer):
    def __init__(self, units, n_layer=1, shortcut=True, expansion=0.5):
        super(BottleneckCSP, self).__init__()
        units_e = int(units * expansion)
        self.conv1 = Conv(units_e, 1, 1)
        self.conv2 = Conv2D(units_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv3 = Conv2D(units_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv4 = Conv(units, 1, 1)
        self.bn = BatchNormalization(momentum=0.03)
        self.activation = Mish()
        self.modules = tf.keras.Sequential([Bottleneck(units_e, shortcut, expansion=1.0) for _ in range(n_layer)])

    def call(self, x):
        y1 = self.conv3(self.modules(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.activation(self.bn(Concatenate(axis=-1)([y1, y2]))))

    def get_config(self):
        config = super(BottleneckCSP, self).get_config()
        config.update({'conv1': self.conv1,
                        'conv2': self.conv2,
                        'conv3': self.conv3,
                        'conv4': self.conv4,
                        'bn': self.bn, 
                        'activation': self.activation,
                        'modules': self.modules,
                        })
        return config

class BottleneckCSP2(Layer):
    def __init__(self, units, n_layer=1, shortcut=False, expansion=0.5):
        super(BottleneckCSP2, self).__init__()
        units_e = int(units)  # hidden channels
        self.conv1 = Conv(units_e, 1, 1)
        self.conv2 = Conv2D(units_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv3 = Conv(units, 1, 1)
        self.bn = BatchNormalization()
        self.activation = Mish()
        self.modules = tf.keras.Sequential([Bottleneck(units_e, shortcut, expansion=1.0) for _ in range(n_layer)])

    def call(self, x):
        x1 = self.conv1(x)
        y1 = self.modules(x1)
        y2 = self.conv2(x1)
        return self.conv3(self.activation(self.bn(Concatenate(axis=-1)([y1, y2]))))

    def get_config(self):
        config = super(VoVCSP, self).get_config()
        config.update({'conv1': self.conv1,
                        'conv2': self.conv2,
                        'conv3': self.conv3,
                        'bn': self.bn, 
                        'activation': self.activation,
                        'modules': self.modules,
                        })
        return config

class VoVCSP(Layer):
    def __init__(self, units, expansion=0.5):
        super(VoVCSP, self).__init__()
        units_e = int(units * expansion)
        self.conv1 = Conv(units_e // 2, 3, 1)
        self.conv2 = Conv(units_e // 2, 3, 1)
        self.conv3 = Conv(units_e, 1, 1)

    def call(self, x):
        _, x1 = tf.split(x, 2, axis=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x1)
        return self.conv3(tf.concat([x1, x2], axis=-1))
    
    def get_config(self):
        config = super(VoVCSP, self).get_config()
        config.update({'conv1': self.conv1,
                        'conv2': self.conv2,
                        'conv3': self.conv3
                        })
        return config


class SPP(Layer):
    def __init__(self, units, kernels=(5, 9, 13)):
        super(SPP, self).__init__()
        units_e = units // 2  # Todo:
        self.conv1 = Conv(units_e, 1, 1)
        self.conv2 = Conv(units, 1, 1)
        self.modules = [MaxPool2D(pool_size=x, strides=1, padding='SAME') for x in kernels]  # Todo: padding check

    def call(self, x):
        x = self.conv1(x)
        return self.conv2(Concatenate(axis=-1)([x] + [module(x) for module in self.modules]))

    def get_config(self):
        config = super(SPP, self).get_config()
        config.update({'conv1': self.conv1,
                        'conv2': self.conv2,
                        'modules': self.modules
                        })
        return config

class SPPCSP(Layer):
    # Cross Stage Partial Networks
    def __init__(self, units, n=1, shortcut=False, expansion=0.5, kernels=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        units_e = int(2 * units * expansion)
        self.conv1 = Conv(units_e, 1, 1)
        self.conv2 = Conv2D(units_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv3 = Conv(units_e, 3, 1)
        self.conv4 = Conv(units_e, 1, 1)
        self.modules = [MaxPool2D(pool_size=x, strides=1, padding='same') for x in kernels]
        self.conv5 = Conv(units_e, 1, 1)
        self.conv6 = Conv(units_e, 3, 1)
        self.bn = BatchNormalization()
        self.act = Mish()
        self.conv7 = Conv(units, 1, 1)

    def call(self, x):
        x1 = self.conv4(self.conv3(self.conv1(x)))
        y1 = self.conv6(self.conv5(Concatenate(axis=-1)([x1] + [module(x1) for module in self.modules])))
        y2 = self.conv2(x)
        return self.conv7(self.act(self.bn(Concatenate(axis=-1)([y1, y2]))))
    
    def get_config(self):
        config = super(SPPCSP, self).get_config()
        config.update({'conv1': self.conv1,
                        'conv2': self.conv2,
                        'conv3': self.conv3,
                        'conv4': self.conv4,
                        'modules': self.modules,
                        'conv5': self.conv5,
                        'conv6': self.conv6,
                        'bn': self.bn,
                        'act': self.act,
                        'conv7':self.conv7
                        })
        return config


class Upsample(Layer):
    def __init__(self, i=None, ratio=2, method='bilinear'):
        super(Upsample, self).__init__()
        self.ratio = ratio
        self.method = method

    def call(self, x):
        IM_H = tf.shape(x)[1] * self.ratio
        IM_W = tf.shape(x)[2] * self.ratio
        x = Lambda(lambda x: tf.image.resize(x, (IM_H, IM_W), method=self.method))(x)
        return x
        # return tf.image.resize(x, (tf.shape(x)[1] * self.ratio, tf.shape(x)[2] * self.ratio), method=self.method)

    def get_config(self):
        config = super(Upsample, self).get_config()
        config.update({'ratio': self.ratio})
        config.update({'method': self.method})
        return config

class Concat(Layer):
    def __init__(self, dims=-1):
        super(Concat, self).__init__()
        self.dims = dims

    def call(self, x):
        return Concatenate(axis=self.dims)(x)
    
    def get_config(self):
        config = super(Concat, self).get_config()
        config.update({'dims': self.dims})
        return config

