from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import numpy as np
import pickle
import argparse
from tensorflow.python.lib.io import file_io
import tensorflow as tf

def train (job_dir='./', **args):
    training_images = pickle.load(file_io.FileIO('gs://kerastraindata/X.pickle', mode='r'))
    testing_images = pickle.load('gs://kerastraindata/Y.pickle', mode='r')
    tr_img_data = np.array([i[0] for i in training_images]).reshape(-1, 64,64,1)
    tr_lbl_data = np.array([i[1] for i in training_images])
    tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1, 64, 64, 1)
    tst_lbl_data = np.array([i[1] for i in testing_images])

    model = Sequential()
    model.add(InputLayer(input_shape=[64,64,1]))
    model.add(Conv2D(filters=32,kernel_size=5,strides=1,padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))
    model.add(Conv2D(filters=50, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))
    model.add(Conv2D(filters=80, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(2,activation='softmax'))
    optimizer = Adam(lr=1e-3)

    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x=tr_img_data, y=tr_lbl_data, epochs=100, batch_size=100)
    model.summary()
    loss_and_metrics = model.evaluate(tst_img_data, tst_lbl_data, batch_size=100)
    model.save('cloud_ml_model.h5')
    with file_io.FileIO('cloud_ml_model.h5', mode='rb') as input_f:
        with file_io.FileIO(job_dir + '/cloud_ml_model.h5', mode='wb+') as output_f:
            output_f.write(input_f.read())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        help='write the final model',
        required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    train(**arguments)