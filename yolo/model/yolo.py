import yaml
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D
from tensorflow.keras.layers import Concatenate
from .module import *
# from module import Mish
# from module import Swish
# from module import Conv
# from module import DWConv
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


class Yolo(object):
    def __init__(self, yaml_dir):
        with open(yaml_dir) as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        self.module_list = self.parse_model(yaml_dict)
        module = self.module_list[-1]
        if isinstance(module, Detect):
            # transfer the anchors to grid coordinator, 3 * 3 * 2
            module.anchors /= tf.reshape(module.stride, [-1, 1, 1])
            print('module anchors', module.anchors)
            print('module stride', module.stride)

    def __call__(self, img_size, name='yolo'):
        x = tf.keras.Input([img_size, img_size, 3])
        output = self.forward(x)
        return tf.keras.Model(inputs=x, outputs=output, name=name)

    def forward(self, x):
        y = []
        for module in self.module_list:
            if module.f != -1:  # if not from previous layer
                if isinstance(module.f, int):
                    x = y[module.f]
                else:
                    x = [x if j == -1 else y[j] for j in module.f]
            x = module(x)
            y.append(x)
        return x

    def parse_model(self, yaml_dict):
        anchors, nc = yaml_dict['anchors'], yaml_dict['nc']
        depth_multiple, width_multiple = yaml_dict['depth_multiple'], yaml_dict['width_multiple']
        num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
        output_dims = num_anchors * (nc + 5)

        layers = []
        # from, number, module, args
        for i, (f, number, module, args) in enumerate(yaml_dict['backbone'] + yaml_dict['head']):
            # all component is a Class, initialize here, call in self.forward
            module = eval(module) if isinstance(module, str) else module
            
            for j, arg in enumerate(args): # args 数列 string
                try:
                    args[j] = eval(arg) if isinstance(arg, str) else arg  # eval strings, like Detect(nc, anchors)
                except:
                    pass
            
            print(i, module, args)
            number = max(round(number * depth_multiple), 1) if number > 1 else number  # control the model scale

            if module in [Conv2D, Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP]:
                c2 = args[0]
                c2 = math.ceil(c2 * width_multiple / 8) * 8 if c2 != output_dims else c2
                args = [c2, *args[1:]]

                if module in [BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP]:
                    args.insert(1, number)
                    number = 1

            modules = tf.keras.Sequential(*[module(*args) for _ in range(number)]) if number > 1 else module(*args)
            modules.i, modules.f = i, f
            layers.append(modules)
        return layers


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
    yaml_dir = '../configs/yolo-l-mish.yaml'
    model_obj = Yolo(yaml_dir=yaml_dir)
    model = model_obj(img_size=640, name='yolo')
    model.summary()
    for layer in model.layers:
        print(layer.name)

    net = tf.keras.Model(inputs=model.input, outputs=model.get_layer('bottleneck_csp_7').output)
    net.summary()

    net.save('yolo-l-mish.h5', save_format='tf')