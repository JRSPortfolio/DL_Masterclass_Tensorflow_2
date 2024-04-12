'''
Convolutional Neural Network model for malaria diagnosis based on cell images
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
# from keras import Sequential
# from keras.layers import InputLayer, Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from sklearn.metrics import classification_report, confusion_matrix
from model_maker import ModelArgs, make_sequential


##
## Dataset load and data shuffle and split
##
dataset, dataset_info = tfds.load('malaria', with_info = True, as_supervised = True, shuffle_files = True, data_dir = 'Convolutional_Neural_Networks/mds')

print(dataset_info)

def resize_rescalae_img(image, label):
    return tf.image.resize(image, (224, 224)) / 255, label

dataset = dataset['train'].map(resize_rescalae_img)

DS_SIZE = len(dataset)
TRAIN_R = 0.8
VAL_R = 0.1
TEST_R = 0.1

dataset = dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True)

train_ds = dataset.take(int(DS_SIZE * TRAIN_R))
val_ds = dataset.skip(int(DS_SIZE * TRAIN_R)).take(int(DS_SIZE * VAL_R))
test_ds = dataset.skip(int(DS_SIZE * TRAIN_R)).skip(int(DS_SIZE * VAL_R)).take(int(DS_SIZE * TEST_R))

train_ds = train_ds.batch(64).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(64).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(1).prefetch(tf.data.AUTOTUNE)

##
## Model creation with Sequential
##
# model = Sequential([InputLayer(shape = (224, 224, 3)),
#                     Conv2D(filters = 4, kernel_size = 3, strides = 1, padding = 'valid', activation = 'relu'),
#                     BatchNormalization(),
#                     MaxPool2D(pool_size = 2, strides = 2),
#                     Conv2D(filters = 12, kernel_size = 4, strides = 1, padding = 'valid', activation = 'relu'),
#                     BatchNormalization(),
#                     MaxPool2D(pool_size = 2, strides = 2),
#                     Conv2D(filters = 36, kernel_size = 6, strides = 1, padding = 'valid', activation = 'relu'),
#                     BatchNormalization(),
#                     MaxPool2D(pool_size = 2, strides = 2),
#                     Flatten(),
#                     Dense(60, activation = 'relu'),
#                     BatchNormalization(),
#                     Dense(10, activation = 'relu'),
#                     BatchNormalization(),
#                     Dense(1, activation = 'sigmoid')])

# print(model.summary())

##
## Model creation with module
##
filters = [4, 12, 36]
ker_size = [3, 4, 6]
strides = [1, 1, 1]
padding = ['valid', 'valid', 'valid']
img_activation = ['relu', 'relu', 'relu']
units = [60, 10, 1]
feat_activation = ['relu', 'relu', 'sigmoid']

shape = (224, 224, 3)
model_args = ModelArgs(shape, filters, ker_size, strides, padding, img_activation, units, feat_activation)
model = make_sequential(model_args)

print(model.summary())

model.compile(optimizer = Adam(learning_rate = 0.01), loss = BinaryCrossentropy(), metrics = ['accuracy', 'precision'])
model.fit(train_ds, validation_data = val_ds, epochs = 6, verbose = 1)

##
##Model visualization and evaluation
##
metrics =  pd.DataFrame(model.history.history)
sns.lineplot(metrics)
plt.xticks(range(6))
plt.yticks([i / 10 for i in range(20)])
plt.grid(True)

print(f'Model eval: {model.evaluate(test_ds)}')

##
## Transform test data and print reports
##
test_labels = []
test_data = []
for i, (data, label) in enumerate(test_ds):
    test_labels.append(label)
    test_data.append(data)

test_labels = np.array(test_labels)
test_data = np.array(test_data)
test_data = np.array(test_data).reshape(2755, 224, 224, 3)


preds = model.predict(test_data)
pred_classes = (preds > 0.5).astype('int32')

print(f'Classification Report:\n{classification_report(test_labels, pred_classes)}')
print(f'Confusion Matrix:\n{confusion_matrix(test_labels, pred_classes)}')


##
## Model saved to file
##
# model.save('Convolutional_Neural_Networks/malaria_diagnosis_model.keras')
model.save('Convolutional_Neural_Networks/malaria_diagnosis_model_02.keras')

##
##     
plt.show()