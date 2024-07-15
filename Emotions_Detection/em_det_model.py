import tensorflow as tf
from data_preparation import SplitRatios, create_dataset, ds_shuffle_split
from set_model import ModelConfigs, make_sequential_model
from tensorflow.keras import Sequential #type:ignore







if __name__ == '__main__':
    configs = ModelConfigs(batch_size = 1, image_size = (200, 200))
    ds = create_dataset(configs)

    ratios = SplitRatios(0.7, 0.15, 0.15)

    train_ds, val_ds, test_ds = ds_shuffle_split(ds, ratios)
