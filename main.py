import keras
import numpy as np
import pathlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import layers
from keras import regularizers

PATH = './PokemonData'
data_dir = pathlib.Path(PATH)

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])

IDG = ImageDataGenerator(rescale=1. / 255, validation_split=0.2, )

train_data = IDG.flow_from_directory(PATH, target_size=(224, 224), classes=list(CLASS_NAMES),
                                     subset='training')

validation_data = IDG.flow_from_directory(PATH, target_size=(224, 224), classes=list(CLASS_NAMES),
                                          subset='validation')

model = Sequential()
model.add(layers.Conv2D(8, (4, 4), input_shape=(224, 224, 3)))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(.25))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(16, (4, 4)))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(.25))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (4, 4)))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(.25))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (4, 4)))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(.25))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (4, 4)))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(.25))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(layers.Dropout(.5))
model.add(layers.Dense(len(CLASS_NAMES), activation='softmax'))

model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model.fit(train_data, epochs=10, validation_data=validation_data)