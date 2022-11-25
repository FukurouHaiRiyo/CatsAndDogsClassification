import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

PATH = 'path/to/cats_and_dogs'

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))


#variables for pre-processing and training
batch_size = 32
epochs = 15
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

train_image_generator = ImageDataGenerator(
      rescale = 1./255,
      shear_range = 0.2,
      zoom_range = 0.2,
      horizontal_flip = True
)

train_data_gen = train_image_generator.flow_from_directory(
      directory=train_dir,
      batch_size = batch_size,
      target_size = (IMAGE_HEIGHT, IMAGE_WIDTH), 
      class_mode = 'binary'
)

validation_image_generator = ImageDataGenerator(rescale=1./255)
val_data_gen = validation_image_generator.flow_from_directory(
      directory=validation_dir,
      batch_size = batch_size,
      target_size = (IMAGE_HEIGHT, IMAGE_WIDTH), 
      class_mode = 'binary'
)


test_image_generator = ImageDataGenerator(rescale=1./255)
test_data_gen = test_image_generator.flow_from_directory(
      directory=PATH,
      target_size = (IMAGE_HEIGHT, IMAGE_WIDTH),
      batch_size = batch_size,
      class_mode = 'binary'
)

def plotImages(images_arr, probabilities=False):
      fig, axes = plt.subplots(len(images_arr), 1, figsize=(5, len(images_arr) * 3))

      if probabilities == False:
            for img, ax in zip(images_arr, axes):
                  ax.imshow(img)
                  ax.axis('off')
      else:
            for img, probability, ax in zip(images_arr, probabilities, axes):
                  ax.imshow(img)
                  ax.axis('off')
                  if probability > 0.5:
                        ax.set_title('%.2f' % (probability * 100) + '% dog')
                  else:
                        ax.set_title('%.2f' % (probability * 100) + '% cat')

      plt.show()

sample_training_images, _ = next(train_data_gen)
plotImages(sample_training_images[:5])


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', 
loss='binary_crossentropy',
metrics=['accuracy']
)
model.summary()

history = model.fit(
      x=train_data_gen,
      steps_per_epoch=total_train // batch_size,
      epochs = epochs,
      validation_data=val_data_gen,
      validation_steps = total_val // batch_size
)

model.save('model.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training accuracy')
plt.plot(epochs_range, val_acc, label='Validation accuracy')
plt.legend(loc='lower right')
plt.title('Training and validation accuracy')

plt.plot(epochs_range, loss, label='Training loss')
plt.plot(epochs_range, val_loss, label='Validation loss')
plt.legend(loc='upper right')
plt.title('Training and validation loss')
plt.show()
