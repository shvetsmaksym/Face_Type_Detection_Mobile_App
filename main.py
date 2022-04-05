from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
Builder.load_file('Layout_Files/frontend.kv')

class MenuScreen(Screen):
    pass
class GaleryUploadScreen(Screen):
    pass

class TakePhotoScreen(Screen):
    pass


class GlassesApp(App):

    def build(self):
        sm = ScreenManager()
        sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(TakePhotoScreen(name='take_photo'))
        sm.add_widget(GaleryUploadScreen(name='gallery_upload'))

        return sm

if __name__ == '__main__':
    GlassesApp().run()