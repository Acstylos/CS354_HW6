import numpy as np
import tensorflow as tf
import os
import random
import contextlib

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.tree import DecisionTreeClassifier
from data_objects import (LabeledImage, Image, Label, ConfusionMatrix,  
                          TestingResult)

# get rid of annoying tf warnings that I can't change
tf.logging.set_verbosity(tf.logging.ERROR)

# Constants
images_filename = 'images.npy'
labels_filename = 'labels.npy'
output_filename = 'output.txt'
training_size = .6
validation_size = .15
test_size = .25
training_epochs = 256
training_batch_size = 128

# Fields
labeled_image_list = []
classifications = []
class_image_dictionary = {}
training_data = []
validation_data = []
test_data = []

# Helper methods
def log(message):
    print(message)
    with open(output_filename, "a") as output_file:
        output_file.write(message)

# remove previous output file
with contextlib.suppress(FileNotFoundError):
    os.remove(output_filename)

# ***** Load Data Files *****
# size of each nparray should be the same once converted into individual
# "units"
images = np.load(images_filename)
labels = np.load(labels_filename)

# ***** Create list of all Images and a list of Classifications *****
# each image has an associated label, so get them in the one-hot format
# we want for keras classification
labels_as_one_hot = to_categorical(labels, dtype='int32')
# get unique classifications
for one_hot_label in labels_as_one_hot:
    label = Label(one_hot_label)
    if label.classification in classifications:
        continue
    else:
        classifications.append(label.classification)
# be sure to sort classifications to make it easier to understand later
classifications.sort()
log('Classifications: {}\n'.format(classifications))

i = 0
for image_matrix in images:
    image = Image(image_matrix)
    # index for labels correspond to each NxM image-matrix
    label = Label(labels_as_one_hot[i])
    labeled_image = LabeledImage(label, image)
    labeled_image_list.append(labeled_image)
    i += 1
log("ImageList Size: {}\n".format(len(labeled_image_list)))

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
    # floor as size of subset. int() automatically truncates/floors 
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
    log("\nClass:{}\n\tSize:\t\t{:3d}\n\tTrainSize:\t{:3d}:: {:.3f}%"
          "\n\tValidSize:\t{:3d}:: {:.3f}%\n\tTestSize:\t{:3d}::"
          " {:.3f}%".format(classification, total_length, 
                            len(training_subset), 
                            len(training_subset)/total_length, 
                            len(validation_subset), 
                            len(validation_subset)/total_length, 
                            len(test_subset), 
                            len(test_subset)/total_length))
log("\nTotal:\t\t{}\nTraining:\t{}\nValidation:\t{}\nTest:\t\t{}\n\n"
      .format(len(training_data)+len(validation_data)+len(test_data), 
              len(training_data), len(validation_data), len(test_data)))


# ***** Do ANN Training and Prediction *****
# we need to "reshape" our data for Keras input
data_train = np.asarray(list(image.image.flat_image for image 
                             in training_data))
label_train = np.asarray(list(image.label.one_hot for image in training_data))
data_valid = np.asarray(list(image.image.flat_image for image 
                             in validation_data))
label_valid = np.asarray(list(image.label.one_hot for image 
                              in validation_data))
data_test = np.asarray(list(image.image.flat_image for image in test_data))

# declare model - don't change this
model = Sequential()

# Experiment by modifying first and subsequent layers in the ANN by:
# 1. initializing weights randomly for every layer
# 2. Using ReLu, SeLu, and Tanh activation units
# 3. number of layers and neurons per layer (including the first)

# first layer (technically I think this is hidden/middle 1, and first 
# layer is actually a mock copy of the training data)
model.add(Dense(64, input_shape = (Image._image_size,), 
                kernel_initializer = 'glorot_normal', activation = 'relu'))
# mid layers                
model.add(Dense(64, kernel_initializer = 'lecun_uniform', activation = 'tanh'))
model.add(Dense(64, kernel_initializer = 'glorot_normal', activation = 'relu'))
model.add(Dense(64, kernel_initializer = 'lecun_uniform', activation = 'tanh'))
model.add(Dense(64, kernel_initializer = 'glorot_normal', activation = 'relu'))
model.add(Dense(64, kernel_initializer = 'lecun_uniform', activation = 'tanh'))
# last layer - don't change this
model.add(Dense(10, kernel_initializer = 'he_normal', activation = 'softmax'))

# Compile Model - don't change this
model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
# debug usage
# model.summary()

# Train Model
history = model.fit(data_train, label_train, 
                    validation_data = (data_valid, label_valid), 
                    epochs = training_epochs, batch_size = training_batch_size)
# Report Results
# Use history data to graph performance of ANN over each epoch
log(str(history.history))
log("\n\nKeras Training Neural Network Accuracy: {:.6f}"
    .format(history.history.get('acc')[-1]))
predictions = model.predict(data_test)
confusion_matrix = ConfusionMatrix(predictions, test_data, classifications)
log("\n\nTested Neural Network Accuracy: {:.6f}"
    .format(confusion_matrix.get_accuracy()))
log("\nNeural Network Confusion Matrix:\n{}".format(confusion_matrix))

# ***** Baseline Decision Tree *****
baseline_classifier = DecisionTreeClassifier()
baseline_classifier = baseline_classifier.fit(data_train, label_train)
baseline_tree_prediction = baseline_classifier.predict(data_test)
baseline_tree_confusion_matrix = ConfusionMatrix(baseline_tree_prediction, 
                                                 test_data, classifications)
log("\n\nTested Baseline Tree Accuracy: {:.6f}"
    .format(baseline_tree_confusion_matrix.get_accuracy()))
log("\nBaseline Tree Confusion Matrix:\n{}"
    .format(baseline_tree_confusion_matrix))

# ***** Variation Decision Tree *****
variation_classifier = DecisionTreeClassifier(max_depth=12, 
                                              min_samples_leaf=2, 
                                              max_leaf_nodes=128,
                                              criterion='entropy')
variation_classifier = variation_classifier.fit(data_train, label_train)
variation_tree_prediction = variation_classifier.predict(data_test)
variation_tree_confusion_matrix = ConfusionMatrix(variation_tree_prediction, 
                                                 test_data, classifications)
log("\n\nTested Variation Tree Accuracy: {:.6f}"
    .format(variation_tree_confusion_matrix.get_accuracy()))
log("\nVariation Tree Confusion Matrix:\n{}"
    .format(variation_tree_confusion_matrix))

# ***** Hand-Engineered Features Decision Tree *****


