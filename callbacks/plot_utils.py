#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@auther    : zzhu
@time      : 21/7/21 7:14 PM
@fileName  : plot_utils.py
'''

import os
import cv2
import numpy as np
import colorsys


def get_class(path):
    classes_path = os.path.expanduser(path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    num_classes = len(class_names) + 1
    return class_names, num_classes

def draw_box(image, label, class_names, classes_map=None):
    # label: xyxy
    box = label[:, 0:4].copy()
    classes = label[:, -1]    

    if np.max(box) <= 1:
        box[:, [0, 2]] = box[:, [0, 2]] * image.shape[1]
        box[:, [1, 3]] = box[:, [1, 3]] * image.shape[0]

    if not isinstance(box, int):
        box = box.astype(np.int16)

    image_h, image_w, _ = image.shape
    num_classes = len(classes_map) if classes_map is not None else len(range(int(np.max(classes)) + 1))
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    bbox_thick = int(0.6 * (image_h + image_w) / 600)   
    font_scale = 0.5

    for i in range(label.shape[0]):
        x1y1 = tuple(box[i, 0:2])
        x2y2 = tuple(box[i, 2:4])
        class_ind = int(classes[i])
        bbox_color = colors[class_ind]
        image = cv2.rectangle(image, x1y1, x2y2, bbox_color, bbox_thick)
        class_name = class_names[class_ind]


        # show labels
        if classes_map is not None:
            class_ind = classes_map[class_ind]
        else:
            class_ind = str(class_ind)
        
        

        if label.shape[-1] == 6:
            score = ': ' + str(round(label[i, -2], 2))
        else:
            score = ''

        bbox_text = '%s %s' % (class_name, score)
        t_size = cv2.getTextSize(bbox_text, 0, font_scale, thickness=bbox_thick//2)[0]
        cv2.rectangle(image, x1y1, (x1y1[0] + t_size[0], x1y1[1] - t_size[1] - 3), bbox_color, -1)  # filled
        cv2.putText(image, bbox_text, (x1y1[0], x1y1[1]-2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image
