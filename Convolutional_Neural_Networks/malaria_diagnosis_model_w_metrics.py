'''
Convolutional Neural Network model for malaria diagnosis based on cell images
'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from model_maker import ModelArgs, make_sequential
from data_retrive_transform import SplitRatios, get_dataset, ds_shuffle_split, transform_test
from keras.metrics import BinaryAccuracy, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall, AUC


def get_splits_from_dataset():
    dataset = get_dataset()
    ratios = SplitRatios(0.8, 0.1, 0.1)
    train_ds, val_ds, test_ds = ds_shuffle_split(dataset, ratios, 64)
    return train_ds, val_ds, test_ds


##
## Model creation with module
##
def create_module(train, validation, model_args: ModelArgs):
    model = make_sequential(model_args)

    print(model.summary())

    metrics = [TruePositives(name = 'tp'), FalsePositives(name = 'fp'), TrueNegatives(name = 'tn'), FalseNegatives(name = 'fn'),
            BinaryAccuracy(name = 'accuracy'), Precision(name = 'precision'), Recall(name = 'recall'), AUC(name = 'auc')]
    model.compile(optimizer = Adam(learning_rate = 0.01), loss = BinaryCrossentropy(), metrics = metrics)
    model.fit(train, validation_data = validation, epochs = 7, verbose = 1)
    
    return model

##
##Model visualization and evaluation
##
def evaluate_model(model, test):
    metrics =  pd.DataFrame(model.history.history)

    fig01, axes01 = plt.subplots(2, 1, figsize = (18, 8))
    plt.subplots_adjust(left = 0.04, right = 0.99, top = 0.99, bottom = 0.08)
    fig02, axes02 = plt.subplots(1, 1, figsize = (18, 8))
    plt.subplots_adjust(left = 0.04, right = 0.99, top = 0.99, bottom = 0.08)

    sns.lineplot(metrics[['accuracy', 'auc', 'loss', 'val_accuracy', 'val_auc', 'val_loss']], ax = axes01[0])
    axes01[0].set_xticks(range(7))
    axes01[0].set_yticks([i / 10 for i in range(20)])
    axes01[0].grid(True)
    
    sns.lineplot(metrics[['precision', 'recall', 'val_precision', 'val_recall']], ax = axes01[1])
    axes01[1].set_xticks(range(7))
    axes01[1].set_yticks([i / 10 for i in range(20)])
    axes01[1].grid(True)
    
    print(f'Model eval: {model.evaluate(test)}')

    ##
    ## Print test prediction reports
    ##
    test_data, test_labels = transform_test(test)

    preds = model.predict(test_data)
    pred_classes = (preds > 0.1).astype('int32')

    print(f'Classification Report:\n{classification_report(test_labels, pred_classes)}')
    print(f'Confusion Matrix:\n{confusion_matrix(test_labels, pred_classes)}')

    ##
    ## Plot ROC curve
    ##
    fp, tp, ths = roc_curve(test_labels, preds)
    sns.lineplot(x = fp, y = tp, ax = axes02)
    for k in range(0, len(ths), 30):
        plt.text(fp[k], tp[k], ths[k])
    plt.grid(True)

    plt.show()
##
## Model saved to file
##
def save_model(model, name: str):
    model.save(f'Convolutional_Neural_Networks/{name}.keras')


##
## main
##

if __name__ == '__main__':
    train_ds, val_ds, test_ds = get_splits_from_dataset()
    
    model_args = ModelArgs(shape = (224, 224, 3),
                           filters = [4, 12, 36],
                           kernel_size = [3, 4, 6],
                           strides = [1, 1, 1],
                           padding = ['valid', 'valid', 'valid'],
                           img_activation = ['relu', 'relu', 'relu'],
                           units = [60, 10, 1],
                           d_activation = ['relu', 'relu', 'sigmoid'])

    
    model = create_module(train_ds, val_ds, model_args)
    evaluate_model(model, test_ds)
    save_model(model, 'malaria_diagnosis_03')
