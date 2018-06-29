import matplotlib.pyplot as plt
from keras.models import load_model
import json


def show_losses(history):
  acc = history['acc']
  val_acc = history['val_acc']
  loss = history['loss']
  val_loss = history['val_loss']
  epochs = range(1, len(acc) + 1)
  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.legend()
  plt.figure()
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()
  plt.show()

def load_models():
    model = load_model('cats_and_dogs_small_1.h5')
    return model

def load_json():
    with open('history.json', 'r') as fp:
        history = json.load(fp)
    return history

def visualise_losses():
    model=load_models()
    show_losses(load_json())
