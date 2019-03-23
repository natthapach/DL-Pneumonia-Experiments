
# coding: utf-8

# In[ ]:


from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')

import glob
import imageio
import pandas as pd
from os import listdir
from datetime import timezone, datetime


# # Initialize
# 
# ## Define constant variable
# - data_dir : root directory for date
# - label_file : location of label file
# - history_csv : location of history output file
# - model_file : location of model output file
# - weight_file : locaiton of weight output file
# 
# ## Checking processor
# Checking CPU and GPU available

# In[ ]:


# input
data_dir = '../../DATASET/Black/data/'
label_file = '../../DATASET/Black/label.csv'

# output
timestamp = int(datetime.now().timestamp()*1000)
history_csv = './results/history_{}.csv'.format(timestamp)
confusion_csv = './results/confusion_{}.csv'.format(timestamp)
model_file = './models/model_{}.json'.format(timestamp)
weight_file = './models/weight_{}.h5'.format(timestamp)
log_tb = './logs/tensorboard/{}/'.format(timestamp)

# config training
batch_size = 2
num_epochs = 1
learning_rate = 0.0005

print(device_lib.list_local_devices())


# # Prepare Model
# create simple model for CNN

# In[ ]:


def create_model() :
    model = Sequential()

    # Convolution
    ## 5x5 convolution with 2x2 stride and 32 filters
    model.add(Conv2D(32, (5, 5), strides = (4, 4), padding='same',
                     input_shape=(3000, 3000, 1)))
    model.add(Activation('relu'))

    ## 2x2 max pooling reduces to 3 x 3 x 32
    model.add(MaxPooling2D(pool_size=(4, 4)))

    ## Another 5x5 convolution with 2x2 stride and 32 filters
    model.add(Conv2D(32, (5, 5), strides = (4, 4)))
    model.add(Activation('relu'))

    ## 2x2 max pooling reduces to 3 x 3 x 32
    model.add(MaxPooling2D(pool_size=(4, 4)))

    ## Flatten turns 3x3x32 into 288x1
    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    return model


# In[ ]:


model = create_model()
model.summary()


# # Prepare data
# ## Prepare label and input file name

# In[ ]:


df  = pd.read_csv(label_file, index_col=0)
df.image = data_dir + df.image
df.head()


# ## Split data set
# split data to train(70%), val(15%) and test(15%) set

# In[ ]:


# Split - train:70, val:15, test:15
x_train, x_test, y_train, y_test = train_test_split(df.image, df.label, test_size=0.30)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.50)


# ## Create data generator
# create data generator for query data from data set

# In[ ]:


class DataGenerator(Sequence) :
    def __init__(self, x, y, batch_size) :
        self.x = x
        self.y = keras.utils.to_categorical(y, 2)
        self.batch_size = batch_size
        
    def __len__(self) :
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    
    def __getitem__(self, idx) :
        # get all data in batch number idx
        start = idx * self.batch_size
        end = (idx+1) * self.batch_size
        names = self.x.iloc[start : end]
        
        batch_x = np.array([ np.array(imageio.imread(f, pilmode='L')).reshape((3000, 3000, 1)) for f in names])
        batch_x = batch_x.astype('float16')
        batch_x /= 255
        batch_y = np.array(self.y[start: end])
        return batch_x, batch_y


# In[ ]:


train_generator = DataGenerator(x_train, y_train, batch_size)
val_generator = DataGenerator(x_val, y_val, batch_size)
test_generator = DataGenerator(x_test, y_test, batch_size)


# # Training model
# traing model with training set and validate set

# In[ ]:


opt = keras.optimizers.rmsprop(lr=learning_rate, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=log_tb)

history = model.fit_generator(generator=train_generator,
                                          steps_per_epoch=(len(train_generator) // batch_size),
                                          epochs=num_epochs,
                                          verbose=1,
                                          validation_data=val_generator,
                                          validation_steps=(len(val_generator) // batch_size),
                                          callbacks=[tensorboard])


# ## plot loss and acc graph

# In[ ]:


def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)


    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["acc"],'r-x', label="Train Accuracy")
    ax.plot(history.history["val_acc"],'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)
    

plot_loss_accuracy(history)


# # Save model
# - save model to .json file
# - save weight to .h5 file
# - save history to .csv file

# In[ ]:


model_json = model.to_json()
with open(model_file, "w+") as json_file:
    json_file.write(model_json)
model.save_weights(weight_file)


# In[ ]:


history_df = pd.DataFrame(history.history)
history_df.to_csv(history_csv)
history_df


# # Eavalute
# ## predict Y
# predict Y with test set

# In[ ]:


# predict
_y_pred = model.predict_generator(test_generator, verbose=1)
y_pred = _y_pred.argmax(axis=1)


# ## Confusion matrix
# calculate confusion matrix and accuracy

# In[ ]:


result_df = pd.concat([y_test.reset_index(), pd.Series(y_pred, name='predict')], axis=1)
false_positive_df = result_df[(result_df.predict - result_df.label) == 1]
false_negative_df = result_df[(result_df.predict - result_df.label) == -1]
true_positive_df = result_df[((result_df.predict - result_df.label) == 0) & (result_df.label == 1)]
true_negative_df = result_df[((result_df.predict - result_df.label) == 0) & (result_df.label == 0)]

fp = len(false_positive_df)
fn = len(false_negative_df)
tp = len(true_positive_df)
tn = len(true_negative_df)
tr = tp+tn
fp_percent = fp / len(result_df)
fn_percent = fn / len(result_df)
tp_percent = tp / len(result_df)
tn_percent = tn / len(result_df)
tr_percent = tr / len(result_df)
total = len(result_df)

confusion_mat = pd.DataFrame({'false_positive': [fp, fp_percent],
                                              'false_negative': [fn, fn_percent],
                                              'true_positive': [tp, tp_percent],
                                              'true_negative': [tn, tn_percent],
                                              'true_result':[tr, tr_percent],
                                              'total': [total, 1.0]}, index=['amount', 'percent'])
confusion_mat.to_csv(confusion_csv)
confusion_mat


# In[ ]:


# load json and create model

# json_file = open(model_file, 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights(weight_file)
# print("Loaded model from disk")

# opt = keras.optimizers.rmsprop(lr=0.0005, decay=1e-6)
# loaded_model.compile(loss='categorical_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])
# result = loaded_model.evaluate_generator(test_generator, verbose=1)

