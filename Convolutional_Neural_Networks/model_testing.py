'''
Loading previous created model and printing random samples from the dataset
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model #type: ignore

dataset, dataset_info = tfds.load('malaria', with_info = True, as_supervised = True, shuffle_files = True, data_dir = 'Convolutional_Neural_Networks/mds')

model = load_model('Convolutional_Neural_Networks/keras_models/malaria_diagnosis_da_01.keras')

def resize_rescalae_img(image, label):
    image = tf.expand_dims(image, axis = 0)
    return tf.image.resize(image, (180, 180)) / 255, label

def assign_labels(result):
    if result == 1:
        return 'uninfected'
    else:
        return 'infected'

random_list = np.random.randint(0, dataset['train'].reduce(0, lambda x, _: x + 1).numpy(), 60).tolist()
fig, axes = plt.subplots(6, 10, figsize = (36, 18))
plt.subplots_adjust(left = 0.04, right = 0.96, top = 0.95, bottom = 0.1, wspace = 0.6, hspace = 0.6)

tf.experimental.numpy.experimental_enable_numpy_behavior()
ax_x = int(0)
ax_y = int(0)
for i in random_list:
    for image, label in dataset['train'].skip(i).take(1):
        img, lbl = resize_rescalae_img(image, label)
        pred = model(img, training = False)
        pred = (pred > 0.5).astype('int32')
        img = tf.squeeze(img, axis = 0)
        axes[ax_x, ax_y].imshow(img)
        axes[ax_x, ax_y].set_title(f"Real: {assign_labels(lbl)} | Pred: {assign_labels(pred)}")
        ax_y += 1
        if ax_y == 10:
            ax_y = 0
            ax_x += 1
                
plt.show()