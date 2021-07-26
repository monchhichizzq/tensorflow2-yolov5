from os import terminal_size
from sys import modules
import yaml
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import UpSampling2D
from models.module import *
from models.detector import Detect
# from module import Mish
# from module import Swish
# from module import CBM
# from module import Focus
# from module import CrossConv
# from module import MP
# from module import Bottleneck
# from module import BottleneckCSP
# from module import BottleneckCSP2
# from module import VoVCSP
# from module import SPP
# from module import SPPCSP
# from module import Upsample
# from module import Concat
# from detector import Detect

'''
Conv2D, Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP
filters = math.ceil(c2 * width_multiple / 8) * 8 if c2 != output_dims else c2

[BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP]
number = max(round(number * depth_multiple), 1) if number > 1 else number
'''


class Yolo(object):
    def __init__(self, use_bias, add_bn, add_mish, yaml_dir):
        self.use_bias = use_bias
        self.add_bn = add_bn
        self.add_mish = add_mish
        print('Loading yaml file: {} ....'.format(yaml_dir))
        with open(yaml_dir) as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        
        # anchors and num_classes
        self.anchors, self.nc = yaml_dict['anchors'], yaml_dict['nc']
        self.depth_multiple, self.width_multiple = yaml_dict['depth_multiple'], yaml_dict['width_multiple']
        self.num_anchors = (len(self.anchors[0]) // 2) if isinstance(self.anchors, list) else self.anchors
        self.output_dims = self.num_anchors * (self.nc + 5)

        # print('depth multiple: {}'.format(self.depth_multiple))
        # print('width multiple: {}'.format(self.width_multiple))
        # print('num classes: {}'.format(self.nc))
        # print('num anchors: {}'.format(self.num_anchors))
        # print('output dims: {}'.format(self.output_dims))

    def __call__(self, img_size, name='yolo_l'):
        x = tf.keras.Input([img_size, img_size, 3])
        output = self.forward(x)
        return tf.keras.Model(inputs=x, outputs=output, name=name)
    
    def get_filter_num(self, n_filters):
        return math.ceil(n_filters * self.width_multiple / 8) * 8
    
    def get_layer_num(self, layer_number):
        return max(round(layer_number * self.depth_multiple), 1) 


    def build_backbone(self, x):
        backbone_logs = {}
        # 0-P1/2
        x = Focus(x, 64, 3, 1, self.use_bias, self.add_bn, self.add_mish)

        # 1-P2/4
        n_filters = self.get_filter_num(128)
        x = CBM(x, n_filters, 3, 2, 1, self.use_bias, self.add_bn, self.add_mish, 'P2_1')

        # 2-P2/4
        n_filters = self.get_filter_num(128)
        layer_number = self.get_layer_num(3)
        x = BottleneckCSP(x, n_filters, self.use_bias, self.add_bn, self.add_mish, 
                n_layer=layer_number, shortcut=True, expansion=0.5, block_id='P2_2')

        # 3-P3/8
        n_filters = self.get_filter_num(256)
        x = CBM(x, n_filters, 3, 2, 1, self.use_bias, self.add_bn, self.add_mish, 'P3_1')

        # 4-P3/8
        n_filters = self.get_filter_num(256)
        layer_number = self.get_layer_num(9)
        x = BottleneckCSP(x, n_filters, self.use_bias, self.add_bn, self.add_mish, 
                n_layer=layer_number, shortcut=True, expansion=0.5, block_id='P3_2')
        backbone_logs['P3'] = x

        # 5-P4/16
        n_filters = self.get_filter_num(512)
        x = CBM(x, n_filters, 3, 2, 1, self.use_bias, self.add_bn, self.add_mish, 'P4_1')
        
        # 6-P4/16
        n_filters = self.get_filter_num(512)
        layer_number = self.get_layer_num(9)
        x = BottleneckCSP(x, n_filters, self.use_bias, self.add_bn, self.add_mish, 
                    n_layer=layer_number, shortcut=True, expansion=0.5, block_id='P4_2')
        backbone_logs['P4'] = x

        # 7-P5/32
        n_filters = self.get_filter_num(1024)
        x = CBM(x, n_filters, 3, 2, 1, self.use_bias, self.add_bn, self.add_mish, 'P5_1')

        # 8-P5/32
        n_filters = self.get_filter_num(1024)
        x = SPP(x, n_filters, self.use_bias, self.add_bn, self.add_mish, kernels=(5, 9, 13), block_id='P5_SPP')

        # 9-P5/32
        n_filters = self.get_filter_num(1024)
        layer_number = self.get_layer_num(3)
        x = BottleneckCSP(x, 1024, self.use_bias, self.add_bn, self.add_mish, 
                    n_layer=layer_number, shortcut=False, expansion=0.5, block_id='P5_2')
        backbone_logs['backbone_out'] = x
        return backbone_logs
        
    def build_head(self, backbone_logs):
        head_logs = {}

        # 10-P5/32
        x = backbone_logs['backbone_out']
        n_filters = self.get_filter_num(512)
        x = CBM(x, n_filters, 1, 1, 1, self.use_bias, self.add_bn, self.add_mish, 'head_10')
        head_logs['head_P5'] = x

        # 11-P4/16
        x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)

        # 12-P4/16
        x = Concatenate(axis=-1)([x, backbone_logs['P4']])

        # 13-P4/16
        n_filters = self.get_filter_num(512)
        layer_number = self.get_layer_num(3)
        x = BottleneckCSP(x, n_filters, self.use_bias, self.add_bn, self.add_mish, 
                    n_layer=layer_number, shortcut=False, expansion=0.5, block_id='head_13')

        # 14-P4/16
        n_filters = self.get_filter_num(256)
        x = CBM(x, n_filters, 1, 1, 1, self.use_bias, self.add_bn, self.add_mish, 'head_14')
        head_logs['head_P4'] = x

        # 15-P3/8
        x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)

        # 16-P3/8 cat backbone P3
        x = Concatenate(axis=-1)([x, backbone_logs['P3']])

        # 17-P3/8 -small
        n_filters = self.get_filter_num(256)
        layer_number = self.get_layer_num(3)
        x = BottleneckCSP(x, n_filters, self.use_bias, self.add_bn, self.add_mish, 
                    n_layer=layer_number, shortcut=False, expansion=0.5, block_id='head_17')
        head_logs['detect_P3'] = x

        # 18-P4/16
        n_filters = self.get_filter_num(256)
        x = CBM(x, n_filters, 3, 2, 1, self.use_bias, self.add_bn, self.add_mish, 'head_18')

        # 19-P4/16 cat head P4
        x = Concatenate(axis=-1)([x, head_logs['head_P4']])

        # 20 (P4/16-medium)
        n_filters = self.get_filter_num(512)
        layer_number = self.get_layer_num(3)
        x = BottleneckCSP(x, n_filters, self.use_bias, self.add_bn, self.add_mish, 
                    n_layer=layer_number, shortcut=False, expansion=0.5, block_id='head_20')
        head_logs['detect_P4'] = x

        # 21-P5/8
        n_filters = self.get_filter_num(512)
        x = CBM(x, n_filters, 3, 2, 1, self.use_bias, self.add_bn, self.add_mish, 'head_21')

        # 22-P5/8 cat head P5
        x = Concatenate(axis=-1)([x, head_logs['head_P5']])

        # 23 (P5/32-large)
        n_filters = self.get_filter_num(1024)
        layer_number = self.get_layer_num(3)
        x = BottleneckCSP(x, n_filters, self.use_bias, self.add_bn, self.add_mish, 
                    n_layer=layer_number, shortcut=False, expansion=0.5, block_id='head_23')
        head_logs['detect_P5'] = x
        return head_logs

    def forward(self, x):
        backbone_logs = self.build_backbone(x)
        head_logs = self.build_head(backbone_logs)
        out = Detect(self.nc, anchors=self.anchors)([head_logs['detect_P3'], head_logs['detect_P4'], head_logs['detect_P5']])
        return out


if __name__ == '__main__':
    yaml_path = '../models/configs/yolo-s-mish.yaml'
    yolo = YoloL(use_bias=False, add_bn=True, add_mish=True, yaml_dir=yaml_path)
    model = yolo(img_size=640, name='yolo_s')
    model.summary()

    for layer in model.layers:
        print(layer.name)

    net = tf.keras.Model(inputs=model.input, outputs=model.get_layer('mish_111').output)
    net.summary()

    net.save('yolo-s-mish.h5', save_format='tf')