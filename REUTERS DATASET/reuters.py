from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models, layers
import matplotlib.pyplot as plt
import numpy as np

(train_data, train_labels), (test_data, test_labels)=reuters.load_data(num_words=10000)
print('{}, {}, {}'.format(len(train_data), len(test_data), train_data[10]))

def convert_to_word(arr):
        word_index=reuters.get_word_index()
        reverse_word_index=dict([(value,key) for (key, value) in word_index.items()])
        decoded_newsWire= ' ' .join([reverse_word_index.get(i-3, '?') for i in arr])
        print(decoded_newsWire)

#convert_to_word(train_data[0])

def vectorise_dataset(sequences, dimension=10000):
    results= np.zeros((len(sequences), dimension))
    for i, sequence in  enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorise_dataset(train_data)
x_test = vectorise_dataset(test_data)

one_hot_train_labels=to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

model= models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val=x_train[:1000]
partial_x_train=x_train[1000:]

y_val=one_hot_train_labels[:1000]
partial_y_train=one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

loss=history.history['loss']
accuracy=history.history['acc']
val_loss=history.history['val_loss']
val_acc = history.history['val_acc']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('training and validation loss')
plt.legend()
plt.show()

plt.plot(epochs, accuracy, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('training and validation Accuracy')
plt.legend()
plt.show()

predictions = model.predict(x_test)
for i in range(1):
    convert_to_word(test_data[i])
    print(np.argmax(predictions[i]))
