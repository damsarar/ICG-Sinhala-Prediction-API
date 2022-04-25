from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
import numpy as np

# initializing an InceptionV3 model with imagenet pre-trained weights
inception_v3_model = InceptionV3(weights='imagenet')

# initializing a model to extract image features
# output will be 2048 vector since the last softmax layer is eliminated
# softmax layer - 1000 classes of the imagenet dataset
image_feature_extraction_model = Model(
    inception_v3_model.input, inception_v3_model.layers[-2].output)


# function to pre-process the images
def preprocess(image_path):
    # loading the images with modified dimensions
    # InceptionV3 model only accept 299x299 images as input
    img = load_img(image_path, target_size=(299, 299))

    # convert the image file to an array
    image_array = img_to_array(img)

    # expand the shape of the image array
    image_array = np.expand_dims(image_array, axis=0)

    # preprocess the image array using the inbuilt function of Keras InceptionV3
    image_array = preprocess_input(image_array)

    return image_array


# function to encode the images
def encode(image):
    # calling the image pre-processing function
    preprocessed_image = preprocess(image)

    # predict the image features using the image feature extraction model
    image_feature_vector = image_feature_extraction_model.predict(
        preprocessed_image)

    # reshaping the image feature vector
    image_feature_vector = np.reshape(
        image_feature_vector, image_feature_vector.shape[1])

    return image_feature_vector
