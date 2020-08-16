import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow import keras

import glob
import os
import cv2
import numpy as np

from util import data_helper, interceptor

training_img = []
training_label = []
test_img = []
test_label = []
training_path = '../data_set/training'
test_path = '../data_set/test'
#test_data = data_helper.fruits_dataset('../data_set/test')

for dir_path in glob.glob(training_path+'/*'):
    img_label = dir_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 100))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        training_img.append(img)
        training_label.append(img_label)
training_img = np.array(training_img)
training_label = np.array(training_label)

for dir_path in glob.glob(test_path+'/*'):
    img_label = dir_path.split("/")[-1]
    for img_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 100))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_img.append(img)
        test_label.append(img_label)
test_img = np.array(test_img)
test_label = np.array(test_label)

label_to_id = {v : k for k, v in enumerate(np.unique(training_label))}
training_label_id = np.array([label_to_id[i] for i in training_label])
test_label_id = np.array([label_to_id[i] for i in test_label])

training_img = training_img / 255.0
test_img = test_img / 255.0

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

model.summary()
callbacks = interceptor.myCallback()
model.fit(training_img, training_label_id, epochs=5, verbose=1, callbacks=[callbacks])

# f, axarr = plt.subplots(3, 4)

