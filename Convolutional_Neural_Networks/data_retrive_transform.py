import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf


##
## Define dataset slit ratios for train, test and validation
##
class SplitRatios(dict):
    def __init__(self, train_ratio: float, val_ratio: float, test_ratio):
        super(SplitRatios, self).__init__()
        self['train'] = train_ratio
        self['val'] = val_ratio
        self['test'] = test_ratio
    

##
## Donwload/load the malaria dataset, resize and resclae the images, augment the dataset and return it
##

# def get_dataset():
#     def resize_rescalae_img(image, label):
#         return tf.image.resize(image, (180, 180)) / 255, label
    
#     def rotate_image(image, label):
#         return tf.image.rot90(image), label
    
#     def flip_up_down_image(image, label):
#         return tf.image.flip_up_down(image), label
    
#     def flip_left_right_image(image, label):
#         return tf. image.flip_left_right(image), label
    
#     def augment_dataset(dataset):
#         ds = dataset.map(rotate_image)
#         return_ds = dataset.concatenate(ds)
#         ds = dataset.map(flip_up_down_image)
#         return_ds = return_ds.concatenate(ds)
#         # ds = dataset.map(flip_left_right_image)
#         # return_ds = return_ds.concatenate(ds)
#         return return_ds
            
#     dataset = tfds.load('malaria', as_supervised = True, shuffle_files = True, data_dir = 'Convolutional_Neural_Networks/mds')
#     ds = dataset['train'].map(resize_rescalae_img)
#     ds = augment_dataset(ds)
#     return ds

def get_dataset():
    def resize_rescalae_img(image, label):
        return tf.image.resize(image, (90, 90)) / 255, label
                
    dataset = tfds.load('malaria', as_supervised = True, shuffle_files = True, data_dir = 'Convolutional_Neural_Networks/mds')
    ds = dataset['train'].map(resize_rescalae_img)
    return ds


##
## Shuffle the dataset and return the train, validation and test splits
##
def ds_shuffle_split(ds, ratios: SplitRatios, batch_size: int):
    ds = ds.shuffle(buffer_size = 8, reshuffle_each_iteration = True)
    ds_size = len(ds)

    train_ds = ds.take(int(ds_size * ratios['train']))
    val_ds = ds.skip(int(ds_size * ratios['train'])).take(int(ds_size * ratios['val']))
    test_ds = ds.skip(int(ds_size * ratios['train'])).skip(int(ds_size * ratios['val'])).take(int(ds_size * ratios['test']))

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(1).prefetch(tf.data.AUTOTUNE)
        
    return train_ds, val_ds, test_ds

def ds_shuffle_split_unbatched(ds, ratios: SplitRatios):
    ds = ds.shuffle(buffer_size = 8, reshuffle_each_iteration = True)
    ds_size = len(ds)

    train_ds = ds.take(int(ds_size * ratios['train']))
    val_ds = ds.skip(int(ds_size * ratios['train'])).take(int(ds_size * ratios['val']))
    test_ds = ds.skip(int(ds_size * ratios['train'])).skip(int(ds_size * ratios['val'])).take(int(ds_size * ratios['test']))
        
    print(train_ds.element_spec)
    print('-------------------------')
    print(val_ds.element_spec)
    print('-------------------------')
    print(test_ds.element_spec)
    print('-------------------------')
    
    return train_ds, val_ds, test_ds

##
## Transform test data for reports
##
def transform_test(test_ds):
    test_labels = []
    test_data = []
    size = len(test_ds)
    
    for data, label in test_ds.as_numpy_iterator():
        test_labels.append(label)
        test_data.append(data)

    test_labels = np.array(test_labels)
    test_data = np.array(test_data)
    test_data = np.array(test_data).reshape(size, 180, 180, 3)
    
    return test_data, test_labels
