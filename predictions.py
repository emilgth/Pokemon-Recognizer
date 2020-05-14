import pathlib

from keras import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

import cv2
import numpy as np

PATH = './pkmn'
data_dir = pathlib.Path(PATH)
# CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
CLASS_NAMES = ['bilasba', 'charmmand', 'pciak', 'squite']
# print(CLASS_NAMES)

IDG = ImageDataGenerator(rescale=1. / 255, validation_split=0.2, )

train_data = IDG.flow_from_directory(PATH, target_size=(224, 224), classes=list(CLASS_NAMES),
                                     subset='training')

validation_data = IDG.flow_from_directory(PATH, target_size=(224, 224), classes=list(CLASS_NAMES),
                                          subset='validation')

model = load_model('./saved_model2_val_loss_0.1268_val_accuracy097')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


def predict(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [1, 224, 224, 3])
    classes = model.predict_proba(img, batch_size=None, verbose=True)
    # return CLASS_NAMES[np.argmax(classes)]
    return classes[0]
