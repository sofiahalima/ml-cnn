import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.2):
            print("\nLoss is low so cancelling training !")
            self.model.stop_training=True


