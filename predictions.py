import pathlib

from keras import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
import base64
import tempfile


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

model = load_model('./saved_model_400epoch')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.clf()
    plt.grid(False)
    x = range(4)
    plt.xticks(x, CLASS_NAMES)
    plt.yticks([])
    thisplot = plt.bar(range(4), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    # thisplot[true_label].set_color('blue')

def img_base64():
    #Gemmer fil i base64 på et temporary sted
    imgFile = tempfile.TemporaryFile()
    plt.savefig(imgFile)
    imgFile.seek(0)
    returnImg = base64.b64encode(imgFile.read()).decode('UTF-8')
    imgFile.close()
    return returnImg

def predict(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgShow = cv2.resize(img, (224, 224))
    img = np.reshape(imgShow, [1, 224, 224, 3])

    plt.clf()
    plt.imshow(imgShow)
    savedImg = img_base64()

    prediction = model.predict_proba(img, batch_size=None, verbose=True)
    # return CLASS_NAMES[np.argmax(classes)]
    i = np.argmax(prediction[0])
    plot_value_array(i, prediction[0], CLASS_NAMES)

    #Gemmer fil i base64 på et temporary sted
    # imgFile = tempfile.TemporaryFile()
    # plt.savefig(imgFile)
    # imgFile.seek(0)
    # savedplot = base64.b64encode(imgFile.read()).decode('UTF-8')
    # imgFile.close()
    savedplot = img_base64()
    pct = round(100*np.max(prediction[0]),2)
    values = [CLASS_NAMES[i], savedplot, pct, savedImg]
    return values

