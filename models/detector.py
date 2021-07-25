import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import UpSampling2D

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
        '''
        out 0 (None, 80, 80, 256)
        out 1 (None, 40, 40, 512)
        out 2 (None, 20, 20, 1024)
        '''
        for i in range(self.num_scale):  # number of scale layer, default=3
            y = self.modules[i](x[i])
            _, grid1, grid2, _ = y.shape
            # shape (None, x, y, 3, 25)
            y = tf.reshape(y, (-1, grid1, grid2, self.num_scale, self.output_dims))             
          
            grid_xy = tf.meshgrid(tf.range(grid1), tf.range(grid2))  # grid[x][y]==(y,x), (2, )
            # (2, ) -> (x, y, 1, 2)
            grid_xy = tf.cast(tf.expand_dims(tf.stack(grid_xy, axis=-1), axis=2),tf.float32)  

            y_norm = tf.sigmoid(y)  # sigmoid for all dims
            xy, wh, conf, classes = tf.split(y_norm, (2, 2, 1, self.num_classes), axis=-1)

            pred_xy = (xy * 2. - 0.5 + grid_xy) * self.stride[i]  # decode pred to xywh
            pred_wh = (wh * 2) ** 2 * self.anchors[i] * self.stride[i]
            
            # 2+2+1+20
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
