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
    return dataset

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



