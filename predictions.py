from keras.models import load_model
import matplotlib.pyplot as plt
import base64
import tempfile
import cv2
import numpy as np
import settings

CLASS_NAMES = ['Bulbasaur', 'Charmander', 'Pikachu', 'Squirtle']

model = load_model(settings.settings['MODEL_PATH'])
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.clf()
    plt.grid(False)
    x = range(4)
    plt.xticks(x, CLASS_NAMES)
    plt.yticks(range(2), ['0%', '100%'])
    thisplot = plt.bar(range(4), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    # thisplot[true_label].set_color('blue')


def img_base64():
    # Gemmer fil i base64 på et temporary sted
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
    plt.yticks([])
    plt.xticks([])
    savedImg = img_base64()

    prediction = model.predict_proba(img, batch_size=None, verbose=True)
    i = np.argmax(prediction[0])
    plot_value_array(i, prediction[0], CLASS_NAMES)
    savedplot = img_base64()
    pct = round(100 * np.max(prediction[0]), 2)
    values = [CLASS_NAMES[i], savedplot, pct, savedImg]
    return values
