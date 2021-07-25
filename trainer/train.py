import os
import sys
filePath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(filePath)[0])

import cv2
import yaml
import time
import shutil
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from preprocess.old.read_data import DataReader, transforms
from preprocess.old.load_data import DataLoader
from models.yolo_l_mish import YoloL
from tools.optimizer import Optimizer, LrScheduler
from loss.loss_fn import YoloLoss

from tensorflow.keras.callbacks import ModelCheckpoint



params = {# dataset
          'train_annotations_dir': '../preparation/txt_files/voc/voc_train.txt',
          'val_annotations_dir': '../preparation/txt_files/voc/voc_test.txt',

          'img_size': 640,
          'mosaic_data': True,
          'augment_data': True,
          'model_stride': [8, 16, 32],
          'anchor_assign_method': 'wh',
          'anchor_positive_augment': True,

          # lr
          'n_epochs': 30,
          'init_learning_rate': 3e-4,
          'warmup_learning_rate': 1e-6,
          'warmup_epochs': 2, 

          'label_smoothing': 0.02,

          'batch_size': 4,

          'class_path': '../preparation/voc_names.txt',
          'yaml_dir': '../models/configs/yolo-l-mish.yaml',
          'model_save_dir': '../model_save/yolo_l',
          'tf_log_dir': '../logs/yolo_l',
}

@tf.function
def dist_train_step(model, loss_fn, image, target):
    loss = train_step(model, loss_fn, image, target)
    return loss

def train_step(model, loss_fn, image, target):
    with tf.GradientTape() as tape:
        logit = model(image, training=True)
        iou_loss, conf_loss, prob_loss = loss_fn(target, logit)
        total_loss = iou_loss + conf_loss + prob_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    lr = lr_scheduler.step()
    optimizer.lr.assign(lr)
    # self.global_step.assign_add(1)    
    return total_loss


if __name__ == '__main__':
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

    # build data loader
    DataReader = DataReader(params['train_annotations_dir'], img_size=params['img_size'], transforms=transforms,
                            mosaic=params['mosaic_data'], augment=params['augment_data'], filter_idx=None)
    data_loader = DataLoader(DataReader,
                            processed_anchors,
                            stride,
                            params['img_size'],
                            params['anchor_assign_method'],
                            params['anchor_positive_augment'])
    train_dataset = data_loader(batch_size=params['batch_size'], anchor_label=True) # True
    train_dataset.len = len(DataReader)
    

    # build model
    yolo = YoloL(use_bias=False, add_bn=True, add_mish=True, yaml_dir=params['yaml_dir'])
    model = yolo(img_size=params['img_size'], name='yolo_l')

    # set loss
    loss_fn = YoloLoss(processed_anchors,
                       ignore_iou_threshold=0.3,
                       num_classes=nc,
                       label_smoothing=params['label_smoothing'],
                       img_size=params['img_size'])
    optimizer = Optimizer('adam')()   
    
    # set callbacks
    # learning rate
    steps_per_epoch = train_dataset.len / params['batch_size']
    total_steps = int(params['n_epochs'] * steps_per_epoch)
    params['warmup_steps'] = params['warmup_epochs'] * steps_per_epoch
    lr_scheduler = LrScheduler(total_steps, params, scheduler_method='cosine')

    # callbacks 
    log_dir = params['model_save_dir']
    os.makedirs(log_dir, exist_ok=True)       
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                 monitor='val_loss', save_weights_only=False, save_best_only=True, verbose=1)


    # tf log writer
    if os.path.exists(params['tf_log_dir']):
        shutil.rmtree(params['tf_log_dir'])
    log_writer = tf.summary.create_file_writer(params['tf_log_dir'])

    # train
    loss_history = []
    min_loss = 0
    for epoch in range(1, params['n_epochs'] + 1):
        epoch_losses = []
        for step, (image, target) in enumerate(tqdm(train_dataset)):          
            loss = dist_train_step(model, loss_fn, image, target)
            np_loss = loss.numpy()
            # print('=> Epoch {}, Step {}, Loss {:.5f}'.format(epoch, step, loss.numpy()))
            with log_writer.as_default():
                tf.summary.scalar('loss', loss, step=step)
                tf.summary.scalar('lr', optimizer.lr, step=step)
            log_writer.flush()
            
            epoch_losses.append(np_loss)
        
        ep_mean_loss = np.mean(epoch_losses) 
        # save history
        history = 'train_history.npy'
        loss_history.append([epoch, ep_mean_loss])
        np.save(history, loss_history)
        print('=> Epoch {}, Loss {:.5f}'.format(epoch, ep_mean_loss))

        if epoch == 1 or ep_mean_loss <= min_loss:
            min_loss = ep_mean_loss
            ckpt_save_path=os.path.join(log_dir, 'ep_{}-loss_{}.h5'.format(epoch, ep_mean_loss))
            model.save(ckpt_save_path)
            print('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))

