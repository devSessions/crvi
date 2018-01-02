import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import PIL

import keras
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras import activations

from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Model Difinitions
# -------

def set_trainable(model, flag=False):
    for layer in model.layers:
        layer.trainable = flag
        print("{0}:\t{1}".format(layer.trainable, layer.name))

def load_images(image_paths):
    # Load the images from disk.
    images = [plt.imread(path) for path in image_paths]

    # Convert to a numpy array and return it.
    return np.asarray(images)

def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]

def predict(image_path):
    # Load and resize the image using PIL.
    img = PIL.Image.open(image_path)
    img_resized = img.resize(input_shape, PIL.Image.LANCZOS)

    # Plot the image.
    plt.imshow(img_resized)
    plt.show()

    # Convert the PIL image to a numpy-array with the proper shape.
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    # Use the VGG16 model to make a prediction.
    # This outputs an array with 1000 numbers corresponding to
    # the classes of the ImageNet-dataset.
    pred = pre_model.predict(img_array)
    
    # Decode the output of the VGG16 model.
    pred_decoded = decode_predictions(pred)[0]

    # Print the predictions.
    for code, name, score in pred_decoded:
        print("{0:>6.2%} : {1}".format(score, name))

def predict_transferred(model,image_path):
    # Load and resize the image using PIL.
    img = PIL.Image.open(image_path)
    img_resized = img.resize(input_shape, PIL.Image.LANCZOS)

    # Plot the image.
    plt.imshow(img_resized)
    plt.show()

    # Convert the PIL image to a numpy-array with the proper shape.
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    # Use the VGG16 model to make a prediction.
    # This outputs an array with 1000 numbers corresponding to
    # the classes of the ImageNet-dataset.
    pred = model.predict(img_array)
    #cls_pred = np.argmax(pred,axis=1)
    print("Rs.10: {0}, Rs.20: {1}".format(pred[0][0]*100, pred[0][1]*100))
    
    # Decode the output of the VGG16 model.
    #pred_decoded = decode_predictions(pred)[0]

    # Print the predictions.
    #for code, name, score in pred_decoded:
        #print("{0:>6.2%} : {1}".format(score, name))


# Network Architecture 
# Loading pretrained model
pre_model = keras.applications.vgg19.VGG19(include_top=True, weights='imagenet')

# Data augmentation and pre-processing

datagen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=90,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=[0.9, 1.5],
      horizontal_flip=True,
      vertical_flip=True,
      fill_mode='nearest')

datagen_test = ImageDataGenerator(rescale=1./255)


# Directory setup
train_dir = 'data/train/'
test_dir = 'data/valid/'

input_shape = pre_model.layers[0].output_shape[1:3]
batch_size = 20

# Making data generator
generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True)

generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  target_size=input_shape,
                                                  batch_size=batch_size,
                                                  shuffle=False)

# Setting epoch size for batch step calculation
epochs = 20

# Steps per epoch setup
steps_per_epoch = generator_train.n / batch_size
steps_test = generator_test.n / batch_size

# joing directory and filenames
image_paths_train = path_join(train_dir, generator_train.filenames)
image_paths_test = path_join(test_dir, generator_test.filenames)

# classes available
cls_train = generator_train.classes
cls_test = generator_test.classes

class_names = list(generator_train.class_indices.keys())

num_classes = generator_train.num_classes

# In case of unequal number of dataset, use class weight balancing
class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(cls_train),
                                    y=cls_train)


# Architecture Definition

# Get last pool/conv layer of pre-trained model
last_conv_block = pre_model.get_layer('block5_pool')
# setup a reference model which as input and conv layer output of pretrained model
ref_model = Model(inputs=pre_model.input, outputs=last_conv_block.output)

# create new model
transfer_model = keras.models.Sequential()
# add reference model to it
transfer_model.add(ref_model)
# flatten pretrained layers output for dense layer
transfer_model.add(Flatten())

# add custom hidden layer
transfer_model.add(Dense(1024, activation='relu'))
# Output layer
transfer_model.add(Dense(num_classes, activation='softmax'))

# setting learning rate
optimizer = Adam(lr=1e-5)
# Loss definition
loss = 'categorical_crossentropy'
# Metrics for measurement
metrics = ['categorical_accuracy']

# for transfer learning set all the layers of pre-trained model to false
set_trainable(ref_model, flag=False)

# Compile the model
transfer_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Begin training for transfer learning
history = transfer_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

# Save Model weights
transfer_model.save('transfer_18.model')
# Save Model
transfer_model.save_weights('transfer_18_weights.hdf5')

# For finetune set all the layers of pre-trained model to true
set_trainable(ref_model, True)

# Optimize learning rate 
optimizer_fine = Adam(lr=1e-7)
# Compile the model
transfer_model.compile(optimizer=optimizer_fine, loss=loss, metrics=metrics)

# Train again for fine tuning
history = transfer_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)
# Save Model
transfer_model.save('finetune_18.model')
# Save weights
transfer_model.save_weights('finetune_18_weights.hdf5')
