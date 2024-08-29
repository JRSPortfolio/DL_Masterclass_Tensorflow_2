from tensorflow.keras.applications.vgg16 import VGG16 #type: ignore
from tensorflow.keras import Model #type: ignore
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


def set_vgg_backbone():
    shape = (200, 200, 3)
    vgg_backbone = VGG16(include_top = False, weights = 'imagenet', input_shape = shape)
    vgg_backbone.trainable = False
    # print(vgg_backbone.summary())
    return vgg_backbone

def layer_is_conv(layer_name):
    if 'conv' in layer_name:
        return True
    else:
        return False
    
def set_feature_maps_model():
    vgg = set_vgg_backbone()
    feature_maps = [layer.output for layer in vgg.layers[1:] if layer_is_conv(layer.name)]
    features_map_model = Model(inputs = vgg.input, outputs = feature_maps)
    print(features_map_model.summary())
    return features_map_model

def get_random_image_path(num_images: int):
    images_path = []
    folders = ['angry', 'happy', 'sad']
    
    def random_images():
        img_folder = np.random.randint(0, (len(folders) - 1))
        folder_path = f'Emotions_Detection/dataset/{folders[img_folder]}'
        images = os.listdir(folder_path)
        rand_image = np.random.randint(0, (len(images) - 1))
        image_path = f'{folder_path}/{images[rand_image]}'
        return image_path
    
    for _ in range(num_images):
        image_path = random_images()
        while image_path in images_path:
            image_path = random_images()
        images_path.append(image_path)            

    return images_path

def set_array_images(images_path):
    images_list = []
    for i in range(len(images_path)):
        image = cv2.imread(images_path[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (200, 200))
        images_list.append(image)
        
    arr_images = np.array(images_list)
    return arr_images

def visualize_images(images):
    x_val = int(np.ceil(len(images) / 6))
    fig, axes = plt.subplots(x_val, 6, figsize = (36, 18))
    plt.subplots_adjust(left = 0.04, right = 0.96, top = 0.95, bottom = 0.1, wspace = 0.6, hspace = 0.6)

    ax_x = int(0)
    ax_y = int(0)
    for i in range(len(images)):
        image = tf.constant(images[i], dtype = tf.float32)
        if x_val == 1:
            axes[ax_y].imshow(image / 255)
        else:
            axes[ax_x, ax_y].imshow(image / 255)

        ax_y += 1
        if ax_y == 6:
            ax_y = 0
            ax_x += 1
    
def visualize_features(model, images):        
    f_maps = []
    for i in images:
        fm_image = model.predict(i.reshape(1, 200, 200, 3))
        f_maps.append(fm_image)
        
    fig, axes = plt.subplots(len(f_maps[0]), len(f_maps), figsize = (48, 28))
    plt.subplots_adjust(left = 0.04, right = 0.96, top = 0.95, bottom = 0.1, wspace = 0.05, hspace = 0.05)
    ax_x = int(0)
    ax_y = int(0)
    
    for i in range(len(f_maps)):        
        for k in range(len(f_maps[i])):
            f_size = f_maps[i][k].shape[1]
            n_channels = f_maps[i][k].shape[3]
            joint_maps = np.ones((f_size, f_size * n_channels))
            
            for n in range(n_channels):
                joint_maps[:, f_size * n : f_size * (n + 1)] = f_maps[i][k][..., n]
            
            if len(f_maps) == 1:
                axes[ax_x].imshow(joint_maps[:, 0:600])
            else:
                axes[ax_x, ax_y].imshow(joint_maps[:, 0:600])
            ax_x += 1
        ax_y += 1
        ax_x = 0
  
        
if __name__ == '__main__':
    fm_model = set_feature_maps_model()
    im_paths = get_random_image_path(1)
    images = set_array_images(im_paths)
    visualize_images(images)
    visualize_features(fm_model, images)
    plt.show()