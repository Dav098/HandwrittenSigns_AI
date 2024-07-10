from kivy.uix.dropdown import DropDown
from kivy.uix.popup import Popup
from kivymd.uix.label import MDLabel
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.core.window import Window
from kivymd.app import MDApp
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.uix.progressbar import ProgressBar
from kivy.metrics import dp
from kivy.core.text import LabelBase
import image
import learn
import os

from kivy.lang import Builder

Builder.load_file('main.kv')
font_path = './arialuni.ttf'
LabelBase.register(name='arialuni', fn_regular=font_path)
class PaintWidget(Widget):
    def on_touch_down(self, touch):
        if self.check_touch_in_buttons_layout(touch):
            return False

        with self.canvas:
            Color(1, 1, 1)  # Set brush color to white
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=self.get_pen_width())

        return super(PaintWidget, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.check_touch_in_buttons_layout(touch) or self.check_touch_out_of_bounds(touch):
            return False

        touch.ud['line'].points += [touch.x, touch.y]
        return super(PaintWidget, self).on_touch_move(touch)

    def check_touch_in_buttons_layout(self, touch):
        buttons_layout = self.parent.buttons_layout
        return buttons_layout.collide_point(touch.x, touch.y)

    def check_touch_out_of_bounds(self, touch):
        return not self.collide_point(touch.x, touch.y)

    def get_pen_width(self):
        #return Window.width * 0.02         #The pen size depends on the size of the window.
        return 15                            #The pen size is constant

    def clear_canvas(self):
        self.canvas.clear()


class MyPaintWidget(BoxLayout):
    orientation = 'vertical'
    paint_widget = PaintWidget()

    def __init__(self, **kwargs):
        super(MyPaintWidget, self).__init__(**kwargs)
        self.dropdown_button = Button()

    def clear_canvas(self, instance):
        self.paint_widget.clear_canvas()

    def on_size(self, *args):
        min_side = min(self.width, self.height)
        self.paint_widget.size = (min_side, min_side)

    def save_canvas(self, instance):
        popup = MyPopup(dropdown_button=self.dropdown_button, paint_widget=self.paint_widget)
        popup.open()

    def predict(self, instance):
        if not os.path.exists("mnist_model.h5"):
            self.show_popup()
            return 0
        else:
            save_path = "predict.jpg"
            self.paint_widget.export_to_png(save_path)
            img = image.Image(save_path, "")
            img.create_new_img()
            m = learn.Mnist('../data/train.csv', '../data')
            m.load_model()
            m.compile_model()
            box_layout = BoxLayout(orientation='vertical', padding=10)
            unicode_code = m.predict_mnist()
            decoded_char = chr(int(unicode_code[2:], 16))
            #print(decoded_char)
            label = Label(text=decoded_char, font_name='arialuni', font_size=20)
            popup = Popup(title="Predicted",
                          content=box_layout,
                          size_hint=(0.6, 0.6))

            # Create a Button to dismiss the Popup
            dismiss_button = Button(text="Dismiss")
            dismiss_button.bind(on_press=popup.dismiss)

            # Add the dismiss button to the Popup content
            box_layout.add_widget(label)
            box_layout.add_widget(dismiss_button)

            # Open the Popup
            popup.open()

    def show_popup(self):
        box_layout = BoxLayout(orientation='vertical', padding=10)
        # Create a Popup with a Label as its content
        label = Label(text="mnist_model.h5 not found")
        popup = Popup(title="No model",
                      content=box_layout,
                      size_hint=(0.6, 0.6))

        # Create a Button to dismiss the Popup
        dismiss_button = Button(text="Dismiss")
        dismiss_button.bind(on_press=popup.dismiss)

        # Add the dismiss button to the Popup content
        box_layout.add_widget(label)
        box_layout.add_widget(dismiss_button)

        # Open the Popup
        popup.open()


class DropdownMenu(BoxLayout):
    def __init__(self, **kwargs):
        super(DropdownMenu, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 10

        # Label to display selection
        self.selected_label = Label(
            text='Select label of symbol',
            size_hint=(1, None),
            height=44,
            pos_hint={'center_x': 0.5, 'y': 0}  # Set position to center in x and 0 in y
        )
        self.add_widget(self.selected_label)

        # Button to trigger dropdown
        self.dropdown_button = Button(text='Select', size_hint=(0.5, None),font_name='arialuni',
            font_size=20, pos_hint={"center_x": 0.5})
        self.dropdown_button.bind(on_release=self.show_dropdown)
        self.add_widget(self.dropdown_button)

        # Dropdown content
        self.dropdown = DropDown()
        for option in ['U+0030', 'U+0031', 'U+0032', 'U+0033', 'U+0034', 'U+0035', 'U+0036', 'U+0037', 'U+0038',
                       'U+0039', 'U+002B','U+002D','U+002A','U+002F','U+00B1','U+003C','U+003D','U+003E','U+0394',
                       'U+03A3','U+03C0','U+0283', 'U+221E', 'U+221A', 'U+2260', 'U+2264', 'U+2265', 'U+2208',
                       'U+2209', 'U+221D', 'U+2205', 'U+2286', 'U+2282', 'U+2287', 'U+2283']:
            decoded_char = chr(int(option[2:], 16))

            btn = Button(text=decoded_char, font_name='arialuni', font_size=20, size_hint_y=None, height=40)
            btn.bind(on_release=lambda btn: self.select_option(btn.text))
            self.dropdown.add_widget(btn)

    def show_dropdown(self, instance):
        self.dropdown.open(instance)

    def select_option(self, option):
        self.dropdown_button.text = option
        self.dropdown.dismiss()


class MyPopup(Popup):
    dropdown = None

    def __init__(self, dropdown_button, paint_widget, **kwargs):
        super(MyPopup, self).__init__(**kwargs)
        self.size_hint = (0.8, 0.8)
        self.title = "Save drawing"
        self.dropdown_menu = DropdownMenu()
        self.dropdown_button = dropdown_button
        self.paint_widget = paint_widget

        # Adding labels, dropdown menu, and buttons to a BoxLayout
        box_layout = BoxLayout(orientation='vertical', padding=10)

        label_confirmation = MDLabel(
            text="Are you sure you want to save?",
            theme_text_color="Custom",
            text_color=(1, 1, 1, 1)
        )

        btn_yes = Button(
            text="Yes",
            size_hint=(1, None),
            height=44,
            on_release=self.save_drawing  # Bind to save_drawing method
        )
        btn_no = Button(
            text="No",
            size_hint=(1, None),
            height=44,
            on_release=self.dismiss
        )

        # Adding widgets to the BoxLayout
        box_layout.add_widget(self.dropdown_menu)
        box_layout.add_widget(label_confirmation)
        box_layout.add_widget(btn_yes)
        box_layout.add_widget(btn_no)

        # Set BoxLayout as the content of the popup
        self.content = box_layout

    def save_drawing(self, instance):
        #filename = self.dropdown_menu.dropdown_button.text + ".jpg"
        unicode_char = 'U+{:04x}'.format(ord(self.dropdown_menu.dropdown_button.text))
        filename = unicode_char + '.jpg'

        save_path = os.path.join("../data/", filename)  # Path to the "data" folder
        self.paint_widget.export_to_png(save_path)
        img = image.Image(save_path, unicode_char)
        img.create_new_img()
        os.remove(save_path)
        self.dismiss()  # Close the popup after saving


class HoldButton(BoxLayout):

    def __init__(self, **kwargs):
        super(HoldButton, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.register_event_type('on_hold')
        self.pressed = False
        self.hold_time = 2  
        self.clock_event = None
        self.progress_bar = ProgressBar(max=1, size_hint=(1, None), height=5)
        self.add_widget(self.progress_bar)


        # Add a button as content
        self.button = Button(text='Run learning', height=95)
        self.button.bind(on_release=self.on_button_release)
        self.add_widget(self.button)

    def on_button_release(self, instance):
        self.on_hold()

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.pressed = True
            self.start_timer()
            return True
        return super(HoldButton, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        if self.pressed:
            self.pressed = False
            self.stop_timer()
            self.progress_bar.value = 0  # Reset progressbar
            return True
        return super(HoldButton, self).on_touch_up(touch)

    def start_timer(self):
        self.clock_event = Clock.schedule_interval(self.update_progress, 1 / 30.)

    def stop_timer(self):
        if self.clock_event:
            self.clock_event.cancel()
        self.progress_bar.value = 0

    def update_progress(self, dt):
        self.progress_bar.value += dt / self.hold_time
        if self.progress_bar.value >= 1:
            self.stop_timer()
            self.dispatch('on_hold')

    def on_hold(self, *args):
        # Run after the button has been held down.
        m = learn.Mnist('../data/train.csv', '../data')
        m.compile_model()
        m.train_model()
        m.save_model()


class MyPaintApp(MDApp):
    def build(self):
        self.title = 'App'
        return MyPaintWidget()


if __name__ == '__main__':
    MyPaintApp().run()
