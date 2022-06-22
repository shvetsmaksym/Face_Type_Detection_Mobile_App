from tensorflow import keras
import cv2
import numpy as np


def scaler(x_array):
    x_array = (x_array * 0.8) + 0.1
    return x_array


def make_prediction(img_path, model_path='TrainingModels\models_h5\CNN_2.h5'):
    classes = {0: "Heart", 1: "Oblong", 2: "Oval", 3: "Round", 4: "Square"}
    model = keras.models.load_model(model_path)
    img = cv2.imread(img_path)
    processed_img = process_image(cv_image=img)
    if processed_img is not None:
        processed_img = scaler(processed_img)
        processed_img = processed_img.reshape((1, processed_img.shape[0], processed_img.shape[1]))
        # processed_img = preprocess_input(processed_img)

        pred = model.predict(processed_img)
        return classes[np.argmax(pred)]
    else:
        print("Can't process image.")


def add_lacking_px(image, target_size=(168, 224)):
    """Add missing pixels to the edges so the image is of target_size.
    : param image: cv2 image
    : param target_size: target width and hegth of output image"""

    temp_im = (np.array(image) / 255).astype(np.float32)
    w, h = target_size

    def solve_px_partition(ax_size, target):
        if ax_size == target:
            return (0, 0)

        resid = target - ax_size % target

        if resid % 2 == 0:
            return resid // 2, resid // 2
        else:
            return resid // 2, resid // 2 + 1

    # Define margins to add to the edges.
    top, down = solve_px_partition(temp_im.shape[0], h)
    left, right = solve_px_partition(temp_im.shape[1], w)

    px_top = np.ones((top, temp_im.shape[1]))
    px_down = np.ones((down, temp_im.shape[1]))
    temp_im = np.vstack((px_top, temp_im, px_down))

    px_left = np.ones((temp_im.shape[0], left))
    px_right = np.ones((temp_im.shape[0], right))
    temp_im = np.column_stack((px_left, temp_im, px_right))

    return temp_im


def process_image(cv_image):

    try:
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        return None

    height, width = img.shape[0], img.shape[1]
    while height > 224 or width > 168:
        height, width = int(height * 0.5), int(width * 0.5)

    img = cv2.resize(img, (width, height))

    # ADAPTIVE THRESHOLDING
    img = cv2.GaussianBlur(img, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 3)

    # ADD LACKING PIXELS TO THE EDGES
    processed = add_lacking_px(thresh, target_size=(168, 224))

    return processed


if __name__ == "__main__":
    print(make_prediction(img_path='Data/testing_set/Round/round (368).jpg'))
