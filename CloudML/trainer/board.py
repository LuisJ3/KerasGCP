__author__ = 'Luis'

import keras

def createTensorboardConfig(pathToLogs):
    return keras.callbacks.TensorBoard(
        log_dir=pathToLogs,
        histogram_freq=0,
        write_graph=True,
        write_images=False)