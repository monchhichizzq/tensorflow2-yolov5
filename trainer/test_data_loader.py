import os
import sys
filePath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(filePath)[0])

import cv2
import yaml
import time
import shutil
import numpy as np
import tensorflow as tf
from preprocess.old.read_data import DataReader, transforms
from preprocess.old.load_data import DataLoader

params = {'train_annotations_dir': '../preparation/txt_files/voc/voc_train.txt',
          'val_annotations_dir': '../preparation/txt_files/voc/voc_test.txt',

          'img_size': 640,
          'mosaic_data': True,
          'augment_data': True,
          'model_stride': [8, 16, 32],
          'anchor_assign_method': 'wh',
          'anchor_positive_augment': True,

          'batch_size': 1,

          'yaml_dir': '../models/configs/yolo-l-mish.yaml',
}


with open(params['yaml_dir']) as f:
    yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

yaml_anchors = yaml_dict['anchors']
# yaml_stride = yaml_dict['stride']
print(yaml_anchors)
print(yaml_dict)

# anchors and num_classes
anchors, nc = yaml_dict['anchors'], yaml_dict['nc']
depth_multiple, width_multiple = yaml_dict['depth_multiple'], yaml_dict['width_multiple']
num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
output_dims = num_anchors * (nc + 5)
stride = np.array(params['model_stride'], np.float32) 
anchors = tf.cast(tf.reshape(anchors, [num_anchors, -1, 2]), tf.float32)
processed_anchors = anchors/tf.reshape(stride, [-1, 1, 1])

print('depth multiple: {}'.format(depth_multiple))
print('width multiple: {}'.format(width_multiple))
print('num classes: {}'.format(nc))
print('num anchors: {}'.format(num_anchors))
print('output dims: {}'.format(output_dims))
print('model stride: {}'.format(stride))
print('model anchors: {}'.format(processed_anchors))


DataReader = DataReader(params['train_annotations_dir'], img_size=params['img_size'], transforms=transforms,
                        mosaic=params['mosaic_data'], augment=params['augment_data'], filter_idx=None)
data_loader = DataLoader(DataReader,
                         processed_anchors,
                         stride,
                         params['img_size'],
                         params['anchor_assign_method'],
                         params['anchor_positive_augment'])
train_dataset = data_loader(batch_size=params['batch_size'], anchor_label=True) # True

for step, (image, target) in enumerate(train_dataset):   
    print('image', image.shape, np.min(image), np.max(image))
    
    for i, t in enumerate(target):
        print('target', i, np.shape(t))
        # target 0 (1, 80, 80, 3, 6)
        # target 1 (1, 40, 40, 3, 6)
        # target 2 (1, 20, 20, 3, 6)
    
    print('')

    # image = np.array(image[0]*255, dtype=np.uint8)
    # cv2.imshow('image', image)
    # cv2.waitKey(1000)
