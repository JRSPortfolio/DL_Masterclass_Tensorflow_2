'''
Convolutional Neural Network model for malaria diagnosis based on cell images
'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from sklearn.metrics import classification_report, confusion_matrix
from model_maker import ModelArgs, make_sequential
from data_retrive_transform import SplitRatios, get_dataset, ds_shuffle_split, transform_test

dataset = get_dataset()
ratios = SplitRatios(0.8, 0.1, 0.1)
train_ds, val_ds, test_ds = ds_shuffle_split(dataset, ratios, 64)

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
## Print test prediction reports
##
test_data, test_labels = transform_test(test_ds)

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