from keras import layers, models
import os
import json
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



def initialize_model():

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(256, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(512, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    from keras import optimizers
    model.compile(loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(lr=1e-4),
                    metrics=['acc'])

    #DATA PROCESSING
    #1. Read pictures
    #2. Convert to tensors
    #3. decode to rgb grid
    #4. Rescale to single point from 0-255 to 0-1

    from keras.preprocessing.image import ImageDataGenerator
    train_datagen=ImageDataGenerator(
                                    rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True
                                    )
    test_datagen=ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(150,150), batch_size=20,
                                                        class_mode='binary')
    validation_generator=test_datagen.flow_from_directory(
                                                        validation_dir,            target_size=(150,150),
                                                        batch_size=20,
                                                        class_mode='binary'
                                                        )
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
    return history

initialize_model()
visualise_losses()
