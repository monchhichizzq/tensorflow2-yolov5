#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@auther    : zzhu
@contact   : zeqi.z.cn@gmail.com
@time      : 22/07/21 1:48 PM
@fileName  : module.py
'''

from numpy import concatenate
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, MaxPool2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Add

class Mish(Layer):
    def call(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))
    

class Swish(Layer):
    def call(self, x):
        return tf.nn.swish(x)  # tf.nn.leaky_relu(x, alpha=0.1)


def CBM(x, filters, kernel_size, strides, groups, use_bias, add_bn, add_mish, block_id):
    conv_name = 'CBM_Conv_{}'.format(block_id)
    bn_name = 'CBM_Conv_{}_bn'.format(block_id)
    act_name = 'CBM_Conv_{}_mish'.format(block_id)
    
    x = Conv2D(filters, kernel_size, strides, 'SAME', groups=groups, use_bias=use_bias,
                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                           kernel_regularizer=tf.keras.regularizers.L2(5e-4),
                           name = conv_name)(x)

    if add_bn:
        x = BatchNormalization(name=bn_name)(x)
    
    if add_mish:
        x =  Mish()(x)
    else:
        x = ReLU()(x)
    return x


def Focus(x, filters, kernel_size, strides, use_bias, add_bn, add_mish):
    '''
    SpaceToDepth TResNet: High Performance GPU-Dedicated Architecture
    
    To replace the traditional convolution-baseddownscaling unit by a fast and seamless layer, 
    with little in-formation loss as possible

    auther's explanation:
    https://github.com/ultralytics/yolov5/issues/847

    1. Focus() module is designed for FLOPS reduction and speed increase, not mAP increase
    2. Also designed for layer count reduction. 1 Focus module replaces 3 yolov3/4 layers
    '''
    # 640x640x3 -> 320x320x12 
    # 
    downscale_x = [x[..., ::2, ::2, :],
                    x[..., 1::2, ::2, :],
                    x[..., ::2, 1::2, :],
                    x[..., 1::2, 1::2, :]]
    x = Concatenate(axis=-1)(downscale_x)

    x = CBM(x, filters, kernel_size, strides, 1, use_bias, add_bn, add_mish, 'focus')
    return x


def CrossConv(x, filters, kernel_size, strides=1, 
            groups=1, expansion=1, shortcut=False, block_id=None,
            use_bias=False, add_bn=True, add_mish=True):
    '''
    3x3 conv can be replaced by 1x3 and 3x1

    '''
    conv_1_name = 'cross_conv1_{}'.format(block_id)
    conv_2_name = 'cross_conv2_{}'.format(block_id)
    units_e = int(filters * expansion)

    inputs = x
    x_ =  CBM(inputs, units_e, (1, kernel_size), (1, strides), 1, use_bias, add_bn, add_mish, conv_1_name)
    m_out =  CBM(x_, filters, (kernel_size, 1), (strides, 1), groups, use_bias, add_bn, add_mish, conv_2_name)

    if shortcut:
        out = inputs + m_out
        return out
    else:
        return m_out

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


def Bottleneck(x, units, use_bias, add_bn, add_mish, 
                shortcut=True, expansion=0.5, block_id=None):
    inputs = x
    # main
    cbm1_name = 'bottleneck_cbm1_{}'.format(block_id) 
    cbm2_name = 'bottleneck_cbm2_{}'.format(block_id) 
    shortcut_name = 'bottlenect_shortcut_{}'.format(block_id)
    x = CBM(inputs, int(units * expansion), 1, 1, 1, use_bias, add_bn, add_mish, cbm1_name)
    m_out = CBM(x, units, 3, 1, 1, use_bias, add_bn, add_mish, cbm2_name)

    if shortcut:
        out = Add()([inputs, m_out])
        return out
    else:
        return m_out


def BottleneckCSP(x, units, use_bias, add_bn, add_mish, 
                  n_layer=1, shortcut=True, expansion=0.5, block_id=None):
    
    units_e = int(units * expansion)
    inputs = x

    # branch 1
    b11 = 'CSP_{}_1'.format(block_id)
    x1 =  CBM(inputs, units_e, 1, 1, 1, use_bias, add_bn, add_mish, b11)

    for i in range(n_layer):
        b1_subname='CSP_{}{}'.format(block_id, i)
        x1 = Bottleneck(x1, units_e, use_bias, add_bn, add_mish, shortcut, expansion=1.0, block_id=b1_subname)

    y1 = Conv2D(units_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))(x1)

    # branch 2
    y2 = Conv2D(units_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))(inputs)

    # concatenation
    y3 = Concatenate(axis=-1)([y1, y2])
    y3 = BatchNormalization(momentum=0.03)(y3)

    if add_mish:
        y3 = Mish()(y3)
    else:
        y3 =ReLU()(y3)

    c_name = 'CSP_{}_out'.format(block_id)
    y3 =  CBM(y3, units, 1, 1, 1, use_bias, add_bn, add_mish, c_name)
    return y3

def BottleneckCSP2(x, units, use_bias, add_bn, add_mish, 
                  n_layer=1, shortcut=False, expansion=0.5, block_id=None):

    units_e = int(units)
    inputs = x

    # branch 1
    b11 = 'CSP2_{}_1'.format(block_id)
    x1 =  CBM(inputs, units_e, 1, 1, 1, use_bias, add_bn, add_mish, b11)
    for i in range(n_layer):
        b1_subname='CSP2_{}{}'.format(block_id, i)
        x1 = Bottleneck(x1, units_e, use_bias, add_bn, add_mish, shortcut, expansion=1.0, block_id=b1_subname)
    y1 = x1

    # branch 2
    y2 = Conv2D(units_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))(inputs)

    # concatenation
    y3 = Concatenate(axis=-1)([y1, y2])
    y3 = BatchNormalization()(y3)

    if add_mish:
        y3 = Mish()(y3)
    else:
        y3 =ReLU()(y3)

    c_name = 'CSP2_{}_out'.format(block_id)
    y3 =  CBM(y3, units, 1, 1, 1, use_bias, add_bn, add_mish, c_name)
    return y3


def VoVCSP(x, units, use_bias, add_bn, add_mish, expansion=0.5, block_id=None):
    units_e = int(units * expansion)
    CBM_name1 = 'VoVCSP_{}_{}'.format(block_id, 1)
    CBM_name2 = 'VoVCSP_{}_{}'.format(block_id, 2)
    CBM_name3 = 'VoVCSP_{}_{}'.format(block_id, 3)

    _, x_in = tf.split(x, 2, axis=1)
    x1 = CBM(x_in, units_e//2, 3, 1, 1, use_bias, add_bn, add_mish, CBM_name1)
    x2 = CBM(x_in, units_e//2, 3, 1, 1, use_bias, add_bn, add_mish, CBM_name2)
    x3 = Concatenate(axis=-1)([x1, x2])
    x3 = CBM(x_in, units_e, 1, 1, 1, use_bias, add_bn, add_mish, CBM_name3)
    return x3


def SPP(x, units, use_bias, add_bn, add_mish, kernels=(5, 9, 13), block_id=None):
    units_e = units // 2 
    
    SPP_name1 = 'SPP_{}_{}'.format(block_id, 1)
    SPP_name2 = 'SPP_{}_{}'.format(block_id, 2)
    x1 = CBM(x, units_e, 1, 1, 1, use_bias, add_bn, add_mish, SPP_name1)

    spp_maxpool_out = []
    for k_size in kernels:
        mp_out = MaxPool2D(pool_size=k_size, strides=1, padding='SAME')(x1)
        spp_maxpool_out.append(mp_out)
    
    concate_out = Concatenate(axis=-1)([x1] + spp_maxpool_out)

    concate_out = CBM(concate_out, units, 1, 1, 1, use_bias, add_bn, add_mish, SPP_name2)
    return concate_out

def SPPCSP(x, units, use_bias, add_bn, add_mish, 
            n=1, shortcut=False, expansion=0.5, kernels=(5, 9, 13), block_id=None):

    units_e = int(2 * units * expansion)

    SPPCSP_name1 = 'SPPCSP_{}_{}'.format(block_id, 1)
    SPPCSP_name2 = 'SPPCSP_{}_{}'.format(block_id, 2)
    SPPCSP_name3 = 'SPPCSP_{}_{}'.format(block_id, 3)
    SPPCSP_name4 = 'SPPCSP_{}_{}'.format(block_id, 4)
    SPPCSP_name5 = 'SPPCSP_{}_{}'.format(block_id, 5)
    SPPCSP_name6 = 'SPPCSP_{}_{}'.format(block_id, 6)
    SPPCSP_name7 = 'SPPCSP_{}_{}'.format(block_id, 7)

    # branch 1
    x1 = CBM(x, units_e, 1, 1, 1, use_bias, add_bn, add_mish, SPPCSP_name1)
    x1 = CBM(x1, units_e, 3, 1, 1, use_bias, add_bn, add_mish, SPPCSP_name3)
    x1 = CBM(x1, units_e, 1, 1, 1, use_bias, add_bn, add_mish, SPPCSP_name4)

    # spp concatenate
    spp_maxpool_out = []
    for k_size in kernels:
        mp_out = MaxPool2D(pool_size=k_size, strides=1, padding='SAME')(x1)
        spp_maxpool_out.append(mp_out)
    
    y1 = Concatenate(axis=-1)([x1] + spp_maxpool_out)
    y1 = CBM(y1, units_e, 1, 1, 1, use_bias, add_bn, add_mish, SPPCSP_name5)
    y1 = CBM(y1, units_e, 3, 1, 1, use_bias, add_bn, add_mish, SPPCSP_name6)

    # branch 2
    y2 = Conv2D(units_e, 1, 1, use_bias=use_bias, name=SPPCSP_name2,
               kernel_initializer=tf.random_normal_initializer(stddev=0.01))(x)
    
    # concatenate
    y3 = Concatenate(axis=-1)([y1, y2])
    y3 = BatchNormalization()(y3)

    if add_mish:
        y3 = Mish()(y3)
    else:
        y3 =ReLU()(y3)
    
    y3 = CBM(y3, units, 1, 1, 1, use_bias, add_bn, add_mish, SPPCSP_name7)

    return y3


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

