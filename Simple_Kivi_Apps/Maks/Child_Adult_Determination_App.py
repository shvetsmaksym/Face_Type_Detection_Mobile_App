from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput


class MyGridLayout(GridLayout):

    def __init__(self, **kwargs):
        super(MyGridLayout, self).__init__(**kwargs)

        self.cols = 2

        self.add_widget(Label(text="Wzrost: "))
        self.h = TextInput()
        self.add_widget(self.h)

        self.add_widget(Label(text="Waga: "))
        self.w = TextInput()
        self.add_widget(self.w)

        self.submit = Button(text="Pokaż wynik", font_size=32)
        self.submit.bind(on_press=self.press)
        self.add_widget(self.submit)

    def press(self, instance):
        height = self.h.text
        weight = self.w.text

        if int(height) > 170 or int(weight) > 60:
            result = 'Dorosły'
        else:
            result = 'Dziecko'
        self.add_widget(Label(text=result))


class MyApp(App):
    def build(self):
        return MyGridLayout()


if __name__ == '__main__':
    MyApp().run()
