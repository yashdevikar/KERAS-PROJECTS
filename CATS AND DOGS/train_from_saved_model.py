from keras import models
from keras.models import load_model
import os
from helper import visualise_losses

original_dataset_dir = '/home/yash/Desktop/opencv/KERAS/CATS AND DOGS/yash'
base_dir='/home/yash/Desktop/opencv/KERAS/CATS AND DOGS/smaller_datasets'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

model = load_model('cats_and_dogs_small_1.h5')
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=20, class_mode='binary')
validation_generator=test_datagen.flow_from_directory(validation_dir, target_size=(150,150), batch_size=20, class_mode='binary')
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history=model.fit_generator(
    train_generator, steps_per_epoch=100,
    epochs=30, validation_data=validation_generator,
    validation_steps=50
)
model.save('cats_and_dogs_small_1.h5')
with open('history.json', 'w') as fp:
        json.dump(history.history, fp)
visualise_losses()
