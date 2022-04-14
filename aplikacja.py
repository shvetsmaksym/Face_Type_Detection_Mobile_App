from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.core.window import Window
import time

Builder.load_file('C:/Users/Евгей/PycharmProjects/pythonProject/menu.kv')

class MenuScreen(Screen):
    Window.size = (500, 700)
    pass
class GaleryUploadScreen(Screen):
    def selected(self, filename):
        try:
            self.ids.my_image.source = filename[0]
        except:
            pass


class TakePhotoScreen(Screen):
    def capture(self):
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.png".format(timestr))
        print("Captured")


class GlassesApp(App):

    def build(self):
        sm = ScreenManager()
        sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(GaleryUploadScreen(name='gallery_upload'))
        sm.add_widget(TakePhotoScreen(name='camera'))

        return sm

if __name__ == '__main__':
    GlassesApp().run()