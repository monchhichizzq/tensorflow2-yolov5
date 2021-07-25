

from sys import modules
import yaml
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import UpSampling2D
# from .module import *
from module import Mish
from module import Swish
from module import CBM
from module import Focus
from module import CrossConv
from module import MP
from module import Bottleneck
from module import BottleneckCSP
from module import BottleneckCSP2
from module import VoVCSP
from module import SPP
from module import SPPCSP
from module import Upsample
from module import Concat

class Yolo(object):
    def __init__(self, yaml_dir):
        print('Loading yaml file: {} ....'.format(yaml_dir))
        with open(yaml_dir) as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.module_list = self.parse_model(yaml_dict)
        module = self.module_list[-1]

    def parse_model(self, yaml_dict):
        # anchors and num_classes
        anchors, nc = yaml_dict['anchors'], yaml_dict['nc']
        depth_multiple, width_multiple = yaml_dict['depth_multiple'], yaml_dict['width_multiple']
        num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
        output_dims = num_anchors * (nc + 5)

        print('depth multiple: {}'.format(depth_multiple))
        print('width multiple: {}'.format(width_multiple))
        print('num classes: {}'.format(nc))
        print('num anchors: {}'.format(num_anchors))
        print('output dims: {}'.format(output_dims))

        layers = yaml_dict['backbone'] + yaml_dict['head']
        print('layers: ', np.shape(layers))
        # from which nodes, number of blocks, module, args
        # example: -1, 1, Focus, [64, 3]
        # Read backbone and head - from, number, module, args
        # turn string into class
        for i, (f, number, module, args) in enumerate(yaml_dict['backbone'] + yaml_dict['head']):
            # all component is a Class, initialize here, call in self.forward
            module = 'CBM' if module == 'Conv' else module
            module = eval(module) if isinstance(module, str) else module

            for j, arg in enumerate(args): # args 数列 string
                try:
                    args[j] = eval(arg) if isinstance(arg, str) else arg  # eval strings, like Detect(nc, anchors)
                except:
                    pass
            
            print(i, f, number, module, args)
        
            number = max(round(number * depth_multiple), 1) if number > 1 else number  # control the model scale
            print('scaled depth: {}'.format(number))

            if module in [Conv2D, CBM, Bottleneck, SPP, Focus, BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP]:
                c2 = args[0] # number of filters
                # scale layer width
                c2 = math.ceil(c2 * width_multiple / 8) * 8 if c2 != output_dims else c2
                args = [c2, *args[1:]]

                if module in [BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP]:
                    '''
                    BottleneckCSP: [num_filters] -> [num_filters, number(depth)] 
                    BottleneckCSP2: [num_filters] -> [num_filters, number(depth)] 
                    SPPCSP: ?
                    VoVCSP: ?
                    '''
                    args.insert(1, number)
                    number = 1
                    # print(args)
            
            '''
            CBM: x, filters, kernel_size, strides, groups, use_bias, add_bn, add_mish, block_id
            Focus: x, filters, kernel_size, strides, use_bias, add_bn, add_mish
            Bottleneck: x, units, use_bias, add_bn, add_mish, shortcut=True, expansion=0.5, block_id=None
            BottleneckCSP: x, units, use_bias, add_bn, add_mish, n_layer=1, shortcut=True, expansion=0.5, block_id=None
            BottleneckCSP2: x, units, use_bias, add_bn, add_mish, n_layer=1, shortcut=False, expansion=0.5, block_id=None
            SPP: x, units, use_bias, add_bn, add_mish, kernels=(5, 9, 13), block_id=None
            SPPCSP: x, units, use_bias, add_bn, add_mish, n=1, shortcut=False, expansion=0.5, kernels=(5, 9, 13), block_id=None
            Upsample(Layer): i=None, ratio=2, method='bilinear'
            Concat(Layer): dims=-1
            '''

            # if f!=-1: # if not from previous layer
            #     if isinstance(module.f, int):
            #         x = y[module.f]

        #     modules = tf.keras.Sequential(*[module(*args) for _ in range(number)]) if number > 1 else module(*args)
        #     modules.i, modules.f = i, f
        #     layers.append(modules)
        # return layers

            

class Detect(Layer):
    def __init__(self, num_classes, anchors=()):
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.num_scale = len(anchors)
        self.output_dims = self.num_classes + 5
        self.num_anchors = len(anchors[0])//2
        self.stride = np.array([8, 16, 32], np.float32)  # fixed here, modify if structure changes
        self.anchors = tf.cast(tf.reshape(anchors, [self.num_anchors, -1, 2]), tf.float32)
        self.modules = [Conv2D(self.output_dims * self.num_anchors, 1, use_bias=False) for _ in range(self.num_scale)]

    def call(self, x, training=True):
        res = []       
        for i in range(self.num_scale):  # number of scale layer, default=3
            y = self.modules[i](x[i])
            _, grid1, grid2, _ = y.shape
            y = tf.reshape(y, (-1, grid1, grid2, self.num_scale, self.output_dims))               
          
            grid_xy = tf.meshgrid(tf.range(grid1), tf.range(grid2))  # grid[x][y]==(y,x)
            grid_xy = tf.cast(tf.expand_dims(tf.stack(grid_xy, axis=-1), axis=2),tf.float32)  

            y_norm = tf.sigmoid(y)  # sigmoid for all dims
            xy, wh, conf, classes = tf.split(y_norm, (2, 2, 1, self.num_classes), axis=-1)

            pred_xy = (xy * 2. - 0.5 + grid_xy) * self.stride[i]  # decode pred to xywh
            pred_wh = (wh * 2) ** 2 * self.anchors[i] * self.stride[i]
            
            out = Concatenate(axis=-1)([pred_xy, pred_wh, conf, classes])
            res.append(out)
        return res
    
    def get_config(self):
        config = super(Detect, self).get_config()
        config.update({'num_classes': self.num_classes,
                       'num_scale': self.num_scale,
                       'output_dims': self.output_dims,
                       'num_anchors': self.num_anchors,
                       'stride': self.stride,
                       'anchors': self.anchors,
                       'modules': self.modules,
                      })
        return config





if __name__ == '__main__':
    yaml_path = '../models/configs/yolo-l-mish.yaml'
    yolo = Yolo(yaml_path)

'''
knowledge of eval(string)
eval(str)函数很强大，官方解释为：将字符串str当成有效的表达式来求值并返回计算结果。所以，结合math当成一个计算器很好用。

example1:
>>> eval('2 + 2')
4
>>> a = "{1:'xx',2:'yy'}"
>>> c = eval(a)
>>> c
{1: 'xx', 2: 'yy'}
'''