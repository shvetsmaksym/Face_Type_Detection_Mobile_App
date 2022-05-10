from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.core.window import Window
import time

from TrainingModels.Neural_Net_Prediction import make_prediction

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


class PredictionScreen(Screen):
    def selected(self, filename):
        try:
            self.ids.my_image.source = filename[0]
        except Exception as e:
            pass

    def predict(self):
        predicted = make_prediction(img_path=self.ids.my_image.source)
        self.ids.face_type_label.text = predicted


class GlassesApp(App):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_for_predict = None

    def build(self):
        sm = ScreenManager()
        sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(GalleryUploadScreen(name='gallery_upload'))
        sm.add_widget(MakePhotoScreen(name='make_photo'))
        sm.add_widget(ShowPhotoScreen(name='show_photo'))
        sm.add_widget(PredictionScreen(name="predict_face_type"))


        return sm


if __name__ == '__main__':
    GlassesApp().run()
