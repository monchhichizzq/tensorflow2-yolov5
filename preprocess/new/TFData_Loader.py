#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@auther    : zzhu
@contact   : zeqi.z.cn@gmail.com
@time      : 15/04/21 1:48 PM
@fileName  : TFData_Loader.py
'''

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
sys.path.append('../../Kitti-Object-Detection')
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from Preprocess.anchors import BBoxUtility, get_mobilenet_anchors, get_pruned_mobilenetv2_anchors
from tensorflow.keras.applications.imagenet_utils import preprocess_input

num_classes = 9
input_shape = (160, 480, 3)
priors = get_pruned_mobilenetv2_anchors(img_size=(input_shape[1], input_shape[0]))
bbox_utils = BBoxUtility(num_classes, priors)
label_shape = np.shape(bbox_utils.assign_boxes([]))

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def process_box(labels, bbox_utils):
    # If objects are labeled in the input image
    if len(labels) != 0:
        height, width, channel = input_shape
        # Get the box coordinates, left, top, right, bottom

        boxes = np.array(labels[:, :4], dtype=np.float32)
        boxes[:, 0] = boxes[:, 0] / width
        boxes[:, 1] = boxes[:, 1] / height
        boxes[:, 2] = boxes[:, 2] / width
        boxes[:, 3] = boxes[:, 3] / height
        one_hot_label = np.eye((num_classes-1))[np.array(labels[:, 4], np.int32)]
        if ((boxes[:, 3] - boxes[:, 1]) <= 0).any() and ((boxes[:, 2] - boxes[:, 0]) <= 0).any():
            pass

        labels = np.concatenate([boxes, one_hot_label], axis=-1)
        # print('labels:', tf.shape(labels))
    train_labels = bbox_utils.assign_boxes(labels)
    return train_labels


def get_val_data(zip_path):

    '''No data augmentation'''
    # logger.info('Preprocess on Validation data without Data Augmentation')
    # Get image
    img_path = zip_path[0]
    bboxes = zip_path[1]
    # print(bboxes)

    img_path = bytes.decode(img_path.numpy())
    bboxes_path = bytes.decode(bboxes.numpy())
 
    # Read image
    image = Image.open(img_path)
    # Original image size
    iw, ih = image.size
    # Model input shape
    h, w, channel = input_shape
    bboxes = np.load(bboxes_path)
    box = np.array([np.array(list(map(int, box.split(',')))) for box in bboxes])

    # Resize image
    # Scale factor
    scale = min(w / iw, h / ih)
    # New scaled size
    nw = int(iw * scale)
    nh = int(ih * scale)
    # New scaled image
    image = image.resize((nw, nh), Image.BICUBIC)

    # Create a new RGB image
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    # Crop the image
    dx, dy = (w - nw) // 2, (h - nh) // 2
    new_image.paste(image, (dx, dy))
    
    # Tensorflow process
    image_data = np.array(new_image, dtype=np.float32)
    image_data = preprocess_input(image_data)

    # correct boxes
    box_data = np.zeros((len(box), 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        box_data = np.zeros((len(box), 5))
        box_data[:len(box)] = box

    box_data = process_box(box_data, bbox_utils)

    if len(box) == 0:
        # return image_data, []
        return image_data, np.zeros(label_shape)

    if (box_data[:, :4] > 0).any():
        return image_data, box_data
    else:
        # return image_data, []
        return image_data, np.zeros(label_shape)


def get_random_data(zip_path, jitter=.3, hue=.1, sat=1.5, val=1.5):
   
    '''Data augmentation'''
    # logger.info('Preprocess on Validation data with Data Augmentation')
    # Image path + class and box
    
    # Get image
    img_path = zip_path[0]
    bboxes = zip_path[1]
    # print(bboxes)

    img_path = bytes.decode(img_path.numpy())
    bboxes_path = bytes.decode(bboxes.numpy())
 
    # Read image
    image = Image.open(img_path)
    # Original image size
    iw, ih = image.size
    # Model input shape
    h, w, channel = input_shape
    bboxes = np.load(bboxes_path)
    box = np.array([np.array(list(map(int, box.split(',')))) for box in bboxes])
 
    # Resize image
    # Choose a random value between 0.7. and 1.3
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    # Choose a random value between .5 and 1.5
    scale = rand(.5, 1.5)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # Place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    # image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue) # -0.1, 0.1
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat) # 1, 1.5
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val) # 1, 1.5
    x = cv2.cvtColor(np.array(new_image, np.float32) / 255, cv2.COLOR_RGB2HSV)
    x[..., 0] += hue * 360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

    # Tensorflow process
    image_data = preprocess_input(image_data)
    
    image_data = np.array(image_data, dtype=np.float32)

    # correct boxes
    box_data = np.zeros((len(box), 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        box_data = np.zeros((len(box), 5))
        box_data[:len(box)] = box

    box_data = process_box(box_data, bbox_utils)

    if len(box) == 0:
        # return image_data, []
        return image_data, np.zeros(label_shape)

    if (box_data[:, :4] > 0).any():
        return image_data, box_data
    else:
        # return image_data, []
        return image_data, np.zeros(label_shape)

@tf.autograph.experimental.do_not_convert
def preprocess_data(annotation_line, is_train):

    if is_train:
        result_tensors = tf.py_function(get_random_data, [annotation_line], [tf.float32, tf.float32])
    else:
        result_tensors = tf.py_function(get_val_data, [annotation_line], [tf.float32, tf.float32])
    result_tensors[0].set_shape(input_shape)
    result_tensors[1].set_shape(label_shape)
    return result_tensors

def get_filenames_list(file):
    with open(file, 'r') as f:
        filenames = f.readlines()
    
    filenames_list = []
    for annotation_line in filenames:
        line = annotation_line.split()  # Image path + class and box
        img_path = line[0]
        box = str(line[1:])
        filenames_list.append([img_path, box])
    return filenames_list

def build_input(file, batch_size, is_train, num_parallel=8):
    # num_parallel
    filenames_list = save_bbox_txt(file)
    dataset = tf.data.Dataset.from_tensor_slices(filenames_list)
    dataset = dataset.map(lambda x: preprocess_data(x, is_train), 
                          num_parallel_calls=num_parallel)
    # dataset = dataset.interleave(dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.map(lambda x: preprocess_data(x, is_train), 
    #                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    if is_train:
        dataset = dataset.shuffle(5)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = dataset.prefetch(batch_size)
    return dataset

def save_bbox_txt(file):
    with open(file, 'r') as f:
        filenames = f.readlines()
    
    filenames_list = []
    for annotation_line in filenames:
        line = annotation_line.split()  # Image path + class and box
        img_path = line[0]
        bbox_path = img_path.replace('JPEGImages', 'bbox_labels')
        bbox_path = bbox_path.replace('png', 'npy')
        box = line[1:]
        # print(bbox_path)
        # np.save(bbox_path, box)

        filenames_list.append([img_path, bbox_path])
    return filenames_list

if __name__ == '__main__':
    batch_size = 4
    # num_classes = 9
    # input_shape = (160, 480, 3)
    # priors = get_mobilenet_anchors(img_size=(input_shape[1], input_shape[0]))
    # bbox_utils = BBoxUtility(num_classes, priors)
 
    file = os.path.join('../preparation/data_txt', 'kitti_obj_trainval.txt')
    train_dataset = build_input(file, batch_size, is_train=True)

    file = os.path.join('../preparation/data_txt', 'kitti_obj_test.txt')
    val_dataset = build_input(file, batch_size, is_train=False)
    
    for data in val_dataset:
        img, bbox = data
        print(img.shape, bbox.shape)
