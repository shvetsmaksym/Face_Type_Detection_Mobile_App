from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.core.window import Window

from tensorflow import keras
import numpy as np
import time
import os
import cv2

# CONSTANTS
GLASSES_IMAGES_PATH = os.path.join('app_data', 'glasses_images')
TMP_PATH = os.path.join('app_data', 'tmp')
MODEL_PATH = 'model.h5'
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

Builder.load_file(os.path.join('app_data', 'layout', 'frontend.kv'))
Window.size = (500, 700)


class MenuScreen(Screen):

    menu_background_source = os.path.join('app_data', 'layout', 'images', 'background.jpeg')
    title_image_source = os.path.join('app_data', 'layout', 'images', 'title.png')


class GalleryUploadScreen(Screen):

    gallery_upload_background_source = os.path.join('app_data', 'layout', 'images', 'background.jpeg')

    def selected(self, filename):
        try:
            self.ids.my_image.source = filename[0]
        except Exception as e:
            pass


class MakePhotoScreen(Screen):

    camera_image_source = os.path.join('app_data', 'layout', 'images', 'camera.png')

    def capture(self):
        camera = self.ids['camera']

        timestr = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(TMP_PATH, f'IMG_{timestr}.png')
        camera.export_to_png(path)
        print("Captured")

        time.sleep(1)
        self.propagate_to_show_screen(path)

    def propagate_to_show_screen(self, path):
        try:
            self.manager.get_screen('show_photo').ids.my_image.source = path
        except Exception as e:
            pass


class ShowPhotoScreen(Screen):
    pass


class GlassesScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_img_path = None
        self.predicted_type = None
        self.paths = []  # paths to glasses images
        self.model = keras.models.load_model(MODEL_PATH)

    def selected(self, filename):
        """
        Save path for selected image.
        """
        try:
            self.ids.my_image.source = filename[0]
            self.base_img_path = filename[0]
        except Exception as e:
            pass

    def predict(self):
        """
        Predict face type and select path to appropriate glasses images.
        """
        classes = {0: "Heart", 1: "Oblong", 2: "Oval", 3: "Round", 4: "Square"}
        processed_img = self.img_processing()
        if processed_img is not None:
            processed_img = processed_img.reshape((1, processed_img.shape[0], processed_img.shape[1]))

            pred = self.model.predict(processed_img)
            self.predicted_type = classes[np.argmax(pred)]
            self.ids.face_type_label.text = self.predicted_type + ' Shape'
            self.select_glasses_img()
            return
        else:
            print("Can't process image.")
            return

    def select_glasses_img(self):
        """
        Select path to appropriate glasses images. Used in predict method.
        """
        path = os.path.join(GLASSES_IMAGES_PATH, self.predicted_type)
        self.paths = [os.path.join(path, os.listdir(path)[i]) for i in range(len(os.listdir(path)))]
        self.ids.button_glasses_0.background_normal = self.paths[0]
        self.ids.button_glasses_1.background_normal = self.paths[1]
        self.ids.button_glasses_2.background_normal = self.paths[2]
        self.ids.button_glasses_3.background_normal = self.paths[3]

    def add_glasses_to_img(self, img_path_id):
        """
        Apply selected glasses to face photo and save new image into temp folder.
        """
        self.ids.my_image.source = self.base_img_path
        img_with_glasses = self.add_glasses(glasses_path=self.paths[img_path_id])

        temp_filename = f'temp{time.strftime("%Y%m%d_%H%M%S")}.png'
        temp_filepath = os.path.join(TMP_PATH, temp_filename)
        cv2.imwrite(temp_filepath, img_with_glasses)
        self.ids.my_image.source = temp_filepath

    def img_processing(self):
        """
        Process image before making predictions by a model. Used in predict method.
        """
        try:
            img = cv2.imread(self.base_img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            return None

        height, width = img.shape[0], img.shape[1]
        while height > 224 or width > 168:
            height, width = int(height * 0.5), int(width * 0.5)

        img = cv2.resize(img, (width, height))

        # ADAPTIVE THRESHOLDING
        img = cv2.GaussianBlur(img, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 3)

        # ADD LACKING PIXELS TO THE EDGES AND SCALE PX RANGE
        processed = self.add_lacking_px(thresh, target_size=(168, 224))
        processed = self.scaler(processed)

        return processed

    def add_glasses(self, glasses_path):
        """
        Apply glasses into face. Used in add_glasses_to_img method.
        """
        face_img = cv2.imread(self.ids.my_image.source)
        glass_img = cv2.imread(glasses_path)
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        centers = []
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 9)

        # iterating over the face detected
        for (x, y, w, h) in faces:

            # create two Regions of Interest.
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = face_img[y:y + h, x:x + w]
            eyes = EYE_CASCADE.detectMultiScale(roi_gray)

            # Store the coordinates of eyes in the image to the 'center' array
            for (ex, ey, ew, eh) in eyes:
                centers.append((x + int(ex + 0.5 * ew), y + int(ey + 0.5 * eh)))

        if len(centers) > 0:
            # change the given value of 2.15 according to the size of the detected face
            glasses_width = 2.16 * abs(centers[1][0] - centers[0][0])
            overlay_img = np.ones(face_img.shape, np.uint8) * 255
            h, w = glass_img.shape[:2]
            scaling_factor = glasses_width / w

            overlay_glasses = cv2.resize(glass_img, None, fx=scaling_factor, fy=scaling_factor,
                                         interpolation=cv2.INTER_AREA)

            x = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0]

            # The x and y variables below depend upon the size of the detected face.
            x -= 0.26 * overlay_glasses.shape[1]
            y += 0.85 * overlay_glasses.shape[0]

            # Slice the height, width of the overlay image.
            h, w = overlay_glasses.shape[:2]
            overlay_img[int(y):int(y + h), int(x):int(x + w)] = overlay_glasses
            # Create a mask and generate it's inverse.
            gray_glasses = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(gray_glasses, 110, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            temp = cv2.bitwise_and(face_img, face_img, mask=mask)

            temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
            final_img = cv2.add(temp, temp2)

            return final_img

    # OTHER METHODS USED IN PREPROCESSING
    @staticmethod
    def scaler(x_array):
        x_array = (x_array * 0.8) + 0.1
        return x_array

    @staticmethod
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


class GlassesApp(App):

    def build(self):
        sm = ScreenManager()
        sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(GalleryUploadScreen(name='gallery_upload'))
        sm.add_widget(MakePhotoScreen(name='make_photo'))
        sm.add_widget(ShowPhotoScreen(name='show_photo'))
        sm.add_widget(GlassesScreen(name="glasses_screen"))

        return sm


if __name__ == '__main__':
    if os.listdir(TMP_PATH):
        for filename in os.listdir(TMP_PATH):
            os.remove(os.path.join(TMP_PATH, filename))
    GlassesApp().run()
