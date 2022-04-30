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


class GaleryUploadScreen(Screen):
    def selected(self, filename):
        try:
            self.ids.my_image.source = filename[0]
        except Exception as e:
            pass

    def predict(self):
        pred = make_prediction(img_path=self.ids.my_image.source)
        self.ids.face_type_label.text = pred


class TakePhotoScreen(Screen):
    def capture(self):
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.png".format(timestr))
        print("Captured")


class PredictionScreen(Screen):
    pass


class GlassesApp(App):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.predicted_face_type = None

    def build(self):
        self.predicted_face_type = None
        sm = ScreenManager()
        sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(GaleryUploadScreen(name='gallery_upload'))
        sm.add_widget(PredictionScreen(name="predict_face_type"))
        # sm.add_widget(TakePhotoScreen(name='take_photo'))

        return sm


if __name__ == '__main__':
    GlassesApp().run()
