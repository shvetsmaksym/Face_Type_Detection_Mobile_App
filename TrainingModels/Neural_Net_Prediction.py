from tensorflow import keras
import numpy as np
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input


def make_prediction(img_path, model_path='TrainingModels\models_h5\CNN_2.h5'):
    classes = {0: "Heart", 1: "Oblong", 2: "Oval", 3: "Round", 4: "Square"}
    model = keras.models.load_model(model_path)
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)

    pred = model.predict(img)
    print("Your type of face is {}.".format(classes[np.argmax(pred)]))
    return


if __name__ == "__main__":
    make_prediction(img_path='Data/testing_set/Oval/oval (2).jpg')
