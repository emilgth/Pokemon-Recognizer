import pathlib
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

import cv2

import numpy as np

PATH = './PokemonData'
data_dir = pathlib.Path(PATH)

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
print(CLASS_NAMES)

IDG = ImageDataGenerator(rescale=1. / 255, validation_split=0.2, )

train_data = IDG.flow_from_directory(PATH, target_size=(224, 224), classes=list(CLASS_NAMES),
                                     subset='training')

validation_data = IDG.flow_from_directory(PATH, target_size=(224, 224), classes=list(CLASS_NAMES),
                                          subset='validation')

model = load_model('./saved_model')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

img = cv2.imread('./PokemonData/Voltorb/aabfc0156f674322ba27b2e977d9d787.jpg')
img = cv2.resize(img, (224, 224))
img = np.reshape(img, [1, 224, 224, 3])
classes = model.predict_classes(img)

print(train_data.filenames)



