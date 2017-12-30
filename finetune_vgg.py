import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import PIL

import keras
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras import activations

from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

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

pre_model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet')

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

train_dir = 'data/train/'
test_dir = 'data/valid/'

save_to_dir='augmented_images/'

input_shape = pre_model.layers[0].output_shape[1:3]

batch_size = 20

generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    save_to_dir=save_to_dir)

generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  target_size=input_shape,
                                                  batch_size=batch_size,
                                                  shuffle=False)

epochs = 20
steps_per_epoch = generator_train.n / batch_size
steps_test = generator_test.n / batch_size

image_paths_train = path_join(train_dir, generator_train.filenames)
image_paths_test = path_join(test_dir, generator_test.filenames)

cls_train = generator_train.classes
cls_test = generator_test.classes

class_names = list(generator_train.class_indices.keys())

num_classes = generator_train.num_classes

class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(cls_train),
                                    y=cls_train)

last_conv_block = pre_model.get_layer('block5_pool')

ref_model = Model(inputs=pre_model.input, outputs=last_conv_block.output)

transfer_model = keras.models.Sequential()

transfer_model.add(ref_model)

transfer_model.add(Flatten())

transfer_model.add(Dense(1024, activation='relu'))

transfer_model.add(Dropout(0.5))

transfer_model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(lr=1e-5)

loss = 'categorical_crossentropy'

metrics = ['categorical_accuracy']

set_trainable(ref_model, flag=False)

transfer_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

history = transfer_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

transfer_model.save('transfer_13.model')

transfer_model.save_weights('transfer_13_weights.hdf5')

set_trainable(ref_model, True)

optimizer_fine = Adam(lr=1e-7)

transfer_model.compile(optimizer=optimizer_fine, loss=loss, metrics=metrics)

history = transfer_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

transfer_model.save('finetune_13.model')

transfer_model.save_weights('finetune_13_weights.hdf5')
