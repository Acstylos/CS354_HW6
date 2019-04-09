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
        self.flat_image = image_matrix.reshape(self._flat_rows, 
                                                self._image_size)
    
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
