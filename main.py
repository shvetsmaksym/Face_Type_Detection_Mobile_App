from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.core.window import Window
import time
import os
import cv2

from TrainingModels.Neural_Net_Prediction import make_prediction
from GlassManipulations.AddGlassesToImage import add_glasses

GLASSES_IMAGES_PATH = "GlassManipulations/glasses_images"
TMP_PATH = 'GlassManipulations/tmp'

Builder.load_file('Layout_Files/fronted2.kv')
Window.size = (500, 700)


class MenuScreen(Screen):
    pass


class GalleryUploadScreen(Screen):
    def selected(self, filename):
        try:
            self.ids.my_image.source = filename[0]
        except Exception as e:
            pass


class MakePhotoScreen(Screen):
    def capture(self):
        camera = self.ids['camera']

        timestr = time.strftime("%Y%m%d_%H%M%S")
        path = f"saved_images/IMG_{timestr}.png"
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
        self.paths = []  # paths to glasses images

    def selected(self, filename):
        try:
            self.ids.my_image.source = filename[0]
            self.base_img_path = filename[0]
        except Exception as e:
            pass

    def predict(self):
        predicted = make_prediction(img_path=self.ids.my_image.source)
        self.ids.face_type_label.text = predicted + ' Shape'

        # Select appropriate glasses images
        path = os.path.join(GLASSES_IMAGES_PATH, predicted)
        self.paths = [os.path.join(path, os.listdir(path)[i]) for i in range(len(os.listdir(path)))]
        self.ids.button_glasses_0.background_normal = self.paths[0]
        self.ids.button_glasses_1.background_normal = self.paths[1]
        self.ids.button_glasses_2.background_normal = self.paths[2]
        self.ids.button_glasses_3.background_normal = self.paths[3]

    def add_glasses_to_img(self, img_path_id):
        self.ids.my_image.source = self.base_img_path
        img = cv2.imread(self.ids.my_image.source)
        glasses_img = cv2.imread(self.paths[img_path_id])
        img_with_glasses = add_glasses(img, glasses_img)

        temp_filename = f'temp{time.strftime("%Y%m%d_%H%M%S")}.png'
        temp_filepath = os.path.join(TMP_PATH, temp_filename)
        cv2.imwrite(temp_filepath, img_with_glasses)
        self.ids.my_image.source = temp_filepath


class GlassesApp(App):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self):
        sm = ScreenManager()
        sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(GalleryUploadScreen(name='gallery_upload'))
        sm.add_widget(MakePhotoScreen(name='make_photo'))
        sm.add_widget(ShowPhotoScreen(name='show_photo'))
        sm.add_widget(GlassesScreen(name="glasses_screen"))

        return sm


if __name__ == '__main__':
    # Create empty temp dir for images with glasses or clear temp dir if it already exists
    if not os.path.exists(TMP_PATH):
        os.mkdir(TMP_PATH)
    for filename in os.listdir(TMP_PATH):
        os.remove(os.path.join(TMP_PATH, filename))

    # Run app
    GlassesApp().run()
