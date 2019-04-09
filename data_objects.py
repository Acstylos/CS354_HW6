import numpy as np

class LabeledImage:
    def __init__(self, label, image):
        self.label = label
        self.image = image
    
    def __repr__(self):
        return "LabeledImage"
    
    def __str__(self):
        return "Label: {}\nImage: {}\n".format(self.label, self.image)

class Image:
    # each image is 28x28 integer-pixel values
    _rows = 28
    _columns = 28
    _image_size = _rows*_columns
    # a flat-image-matrix is 1xN
    _flat_rows = 1
    def __init__(self, image_matrix):
        self.image_matrix = image_matrix
        # reshape the matrix into the flat-format we want
        self.flat_image = image_matrix.reshape(self._image_size)
    
    def __repr__(self):
        return "Image"

    def __str__(self):
        return "Flat-Image: {}".format(self.flat_image)

class Label:
    def __init__(self, one_hot):
        self.one_hot = one_hot
        self.classification = one_hot.tolist().index(1)
    
    def __repr__(self):
        return "Label"

    def __str__(self):
        return "Label: {}".format(self.classification)

class ConfusionMatrix:
    def __init__(self, predictions, test_data, classifications):
        testing_results = []
        x = 0
        for prediction in predictions:
            # we can use argmax to get the index of the highest probability, 
            # which should translate directly into predicted class, since class
            # is sorted by number, which automatically corresponds to the index
            predicted_class = np.argmax(prediction)
            expected_class = test_data[x].label.classification
            test_result = TestingResult(predicted_class, expected_class)
            testing_results.append(test_result)
            x += 1
        self.results = testing_results
        self.classifications = classifications

    def __repr__(self):
        return "ConfusionMatrix"
    
    def __str__(self):
        returnString = []
        returnString.append("---0---1---2---3---4---5---6---7---8---9--\n")
        for classification in self.classifications:
            matrix_row = list(filter(lambda result: result.expected_class 
                                     == classification, self.results))
            returnString.append("{:1d}|".format(classification))
            for classification_2 in self.classifications:
                matrix_box = list(filter(lambda result_2: 
                                         result_2.predicted_class 
                                         == classification_2, matrix_row))
                returnString.append("{:3d}".format(len(matrix_box)))
                returnString.append("|")
            returnString.append("\n------------------------------------------"
                                "\n")
        return ''.join(returnString)
    
    def get_accuracy(self):
        diagonal_count = 0
        accurate_list = list(filter(lambda result: result.is_correct(), 
                                    self.results))
        diagonal_count += len(accurate_list)
        return diagonal_count/len(self.results)

class TestingResult:
    def __init__(self, predicted_class, expected_class):
        self.predicted_class = predicted_class
        self.expected_class = expected_class

    def __repr__(self):
        return "TestingResult"
    
    def __str__(self):
        return "Pre/Exp: {}/{}".format(self.predicted_class, 
                                       self.expected_class)

    def is_correct(self):
        return self.predicted_class == self.expected_class
