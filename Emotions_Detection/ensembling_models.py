from tensorflow.keras.applications.vgg16 import VGG16 #type: ignore
from tensorflow.keras.applications.efficientnet import EfficientNetB4 #type: ignore
from tensorflow.keras import Model, Input #type: ignore
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, TopKCategoricalAccuracy #type: ignore
from tensorflow.keras.losses import CategoricalCrossentropy #type: ignore
from tensorflow.keras.optimizers import Adam  #type: ignore
import tensorflow as tf
import matplotlib.pyplot as plt
from set_model import ModelConfigs, make_sequential_transfer_learning
from em_det_model_plt_metrics import get_dataset, show_model_metrics, test_model
import numpy as np
import os
import cv2

def config_values():   
    image_size = (200, 200)
    learning_rate = 0.001
    epochs = 30
    batch_size = 32
    dense_01 = 512
    dense_02 = 128
    
    configs = ModelConfigs (image_size = image_size, learning_rate = learning_rate, epochs = epochs,
                            batch_size = batch_size, dense_01 = dense_01, dense_02 = dense_02)

    return configs

def set_vgg_backbone(configs: ModelConfigs):
    shape = (configs['image_size'][0], configs['image_size'][1], 3)
    vgg_backbone = VGG16(include_top = False, weights = 'imagenet', input_shape = shape)
    vgg_backbone.trainable = False
    return vgg_backbone

def set_efnet_backbone(configs: ModelConfigs):
    shape = (configs['image_size'][0], configs['image_size'][1], 3)
    backbone = EfficientNetB4(include_top = False, weights = 'imagenet', input_shape = shape)
    backbone.trainable = False
    return backbone

def make_model(train_ds, val_ds, configs: ModelConfigs):
    metrics = [CategoricalAccuracy(name = 'accuracy'), TopKCategoricalAccuracy(k = 2, name = 'top_k_acc'), Precision(name = 'precision'),
               Recall(name = 'recall')]
    
    vgg_backbone = set_vgg_backbone(configs)
    efnet_backbone = set_efnet_backbone(configs)
    
    vgg_model = make_sequential_transfer_learning(configs, vgg_backbone)
    efnet_model = make_sequential_transfer_learning(configs, efnet_backbone)
    
    loss = CategoricalCrossentropy()
    
    vgg_model.compile(optimizer = Adam(learning_rate = configs['learning_rate']), loss = loss, metrics = metrics)
    vgg_model.fit(train_ds, validation_data = val_ds, epochs = configs['epochs'], verbose = 1)
    
    efnet_model.compile(optimizer = Adam(learning_rate = configs['learning_rate']), loss = loss, metrics = metrics)
    efnet_model.fit(train_ds, validation_data = val_ds, epochs = configs['epochs'], verbose = 1)
    
    return vgg_model, efnet_model

def ensembling_models(train_ds, val_ds, configs: ModelConfigs):
    metrics = [CategoricalAccuracy(name = 'accuracy'), TopKCategoricalAccuracy(k = 2, name = 'top_k_acc'), Precision(name = 'precision'),
            Recall(name = 'recall')]
    
    vgg_model, efnet_model = make_model(train_ds, val_ds, configs)
    inputs = Input(shape = (configs['image_size'][0], configs['image_size'][1], 3))
    y_1 = vgg_model(inputs)
    y_2 = efnet_model(inputs)
    output = 0.5 * y_1 + 0.5 * y_2
    ensemble_model = Model(inputs = inputs, outputs = output)
    return ensemble_model

if __name__ == '__main__':
    configs = config_values()
    train_ds, val_ds, test_ds = get_dataset(configs)
    vgg_model, efnet_model = make_model(train_ds, val_ds,configs)
    show_model_metrics(vgg_model, test_ds, configs)
    show_model_metrics(efnet_model, test_ds, configs)
    test_model(vgg_model, test_ds)
    test_model(efnet_model, test_ds)
    
    plt.show()