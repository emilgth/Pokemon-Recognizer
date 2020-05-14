import pathlib
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

import cv2

import numpy as np

PATH = 'Starters'
data_dir = pathlib.Path(PATH)

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
# print(CLASS_NAMES)

IDG = ImageDataGenerator(rescale=1. / 255, validation_split=0.2, )

train_data = IDG.flow_from_directory(PATH, target_size=(224, 224), classes=list(CLASS_NAMES),
                                     subset='training')

validation_data = IDG.flow_from_directory(PATH, target_size=(224, 224), classes=list(CLASS_NAMES),
                                          subset='validation')

model = load_model('./saved_model')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

img = cv2.imread('./bulb.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = np.reshape(img, [1, 224, 224, 3])
# classes = model.predict_classes(train_data, batch_size=None)
classes = model.predict_classes(img)
prediction = model.predict_proba(img)
print(prediction[0])

print(classes[0])

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(4))
    plt.yticks([])
    thisplot = plt.bar(range(4), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    # thisplot[true_label].set_color('blue')

plot_value_array(0, prediction[0], CLASS_NAMES)




# for guess, real in zip(classes, train_data.filenames):
#     print(CLASS_NAMES[guess], real)

print(CLASS_NAMES[classes[0]])

plt.show()

# print(train_data.filenames)



