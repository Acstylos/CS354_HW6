import numpy as np
import os.path
#import keras.utils as ku
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation
import copy
import random

from data_objects import LabeledImage, Image, Label

# Constants
images_filename = 'images.npy'
labels_filename = 'labels.npy'
training_size = .6
validation_size = .15
test_size = .25

# Fields
labeled_image_list = []
classifications = []
class_image_dictionary = {}
training_data = []
validation_data = []
test_data = []

# ***** Load Data Files *****
# size of each nparray should be the same once converted into individual
# "units"
images = np.load(images_filename)
labels = np.load(labels_filename)

# ***** Create List of all Images and a list of Classifications *****
# each image has an associated label, so get them in the one-hot format
# we want
labels_as_one_hot = to_categorical(labels, dtype='int32')
# get unique classifications
for one_hot_label in labels_as_one_hot:
    label = Label(one_hot_label)
    if label.classification in classifications:
        continue
    else:
        classifications.append(label.classification)
print('Classifications: {}'.format(classifications))

i = 0
for image_matrix in images:
    image = Image(image_matrix)
    # index for labels correspond to each NxM image-matrix
    label = Label(labels_as_one_hot[i])
    labeled_image = LabeledImage(label, image)
    labeled_image_list.append(labeled_image)
    i += 1
print("ImageList Size: {}\n".format(len(labeled_image_list)))

# ***** Straify the Data, and assign to Sets *****
# Sort images by classification into strata
for classification in classifications:
    images_in_class = filter(lambda image: image.label.classification 
                             == classification, labeled_image_list)
    class_image_dictionary[classification] = list(images_in_class)

# Randomize each strata, select appropriately sized subsets and add 
# subsets to final lists
for classification, images in class_image_dictionary.items():
    total_length = len(images)
    # pseudo-randomize the entire list
    random.shuffle(images)
    # take a subset based on proportionally sized indicies, taking the 
    # floor as size of subset. int automatically truncates/floors 
    # floats, which size*len should be, since size is a float
    training_end_index = int(training_size*total_length)
    validation_end_index = (training_end_index 
                            + int(validation_size*total_length))
    training_subset = images[0:training_end_index]
    training_data += training_subset
    validation_subset = images[training_end_index:validation_end_index]
    validation_data += validation_subset
    # except for the test set, which we just assign as the rest of the 
    # list
    test_subset = images[validation_end_index:]
    test_data += test_subset
    print("Class:{}\n\tSize:\t\t{:3d}\n\tTrainSize:\t{:3d}:: {:.3f}%"
          "\n\tValidSize:\t{:3d}:: {:.3f}%\n\tTestSize:\t{:3d}::"
          " {:.3f}%".format(classification, total_length, 
                            len(training_subset), 
                            len(training_subset)/total_length, 
                            len(validation_subset), 
                            len(validation_subset)/total_length, 
                            len(test_subset), 
                            len(test_subset)/total_length))
print("Total:\t\t{}\nTraining:\t{}\nValidation:\t{}\nTest:\t\t{}"
      .format(len(training_data)+len(validation_data)+len(test_data), 
              len(training_data), len(validation_data), len(test_data)))

# declare model - don't change this
model = Sequential()

# Experiment by modifying first and subsequent layers in the ANN by:
# 1. initializing weights randomly for every layer
# 2. Using ReLu, SeLu, and Tanh activation units
# 3. number of layers and neurons per layer (including the first)
# first layer
model.add(Dense(10, input_shape=(28*28, ), 
          kernel_initializer='he_normal'))
model.add(Activation('relu'))

#
#
#
# Fill in Model Here
#
#

# last layer - don't change this
model.add(Dense(10, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

# Compile Model - don't change this
model.compile(optimizer='sgd', loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train, validation_data = (x_val, y_val), 
                    epochs=10, batch_size=512)

# Report Results
print(history.history)
model.predict()