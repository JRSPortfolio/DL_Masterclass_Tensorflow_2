import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory #type: ignore
from set_model import ModelConfigs

class SplitRatios(dict):
    def __init__(self, train_ratio: float, val_ratio: float, test_ratio: float):
        super(SplitRatios, self).__init__()
        self['train'] = train_ratio
        self['val'] = val_ratio
        self['test'] = test_ratio
  
def create_dataset(configs: ModelConfigs):
    dataset = image_dataset_from_directory('Emotions_Detection/dataset',
                                           labels = 'inferred',
                                           label_mode = 'categorical',
                                           class_names = ['angry', 'happy', 'sad'],
                                           color_mode = 'rgb',
                                           image_size = configs['image_size'],
                                           batch_size = configs['batch_size'])
    
    dataset = data_augmentation(dataset)
    
    return dataset

def data_augmentation(dataset):
    def flip_horizontal(image, label):
        return tf.image.flip_left_right(image), label
    
    def random_brigth(image, label):
        return tf.image.random_brightness(image, 0.4), label
    
    def random_contrast(image, label):
        return tf.image.random_contrast(image, 0.1, 0.6), label

    ds = dataset.map(flip_horizontal)
    return_ds = dataset.concatenate(ds)
    ds = dataset.map(random_brigth)
    return_ds = return_ds.concatenate(ds)
    ds = dataset.map(random_contrast)
    return_ds = return_ds.concatenate(ds)
    
    return return_ds
    

def ds_shuffle_split(dataset, ratios: SplitRatios):
    dataset = dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True)
    ds_size = len(dataset)

    train_ds = dataset.take(int(ds_size * ratios['train']))
    val_ds = dataset.skip(int(ds_size * ratios['train'])).take(int(ds_size * ratios['val']))
    test_ds = dataset.skip(int(ds_size * ratios['train'])).skip(int(ds_size * ratios['val'])).take(int(ds_size * ratios['test']))
    
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds



