import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2


class CamApp(App):

    def build(self):
        layout = BoxLayout(orientation = 'horizontal', spacing = 0, padding=0)

        self.webCam = Image(size_hint = (1,1))
        self.acceptBtn = Button(text = "Accept", size_hint =(0.5,.1))
        self.deleteBtn = Button(text="Delete", size_hint=(0.5, .1))


        layout.add_widget(self.webCam)
        layout.add_widget(self.acceptBtn)
        layout.add_widget(self.deleteBtn)

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout

    def update(self, *args):

        ret, frame = self.capture.read()
        #cv2.imshow("CV2 Image", frame)
        # convert it to texture
        buf1 = cv2.flip(frame, -1)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(640,480), colorfmt='bgr')
        # if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer.
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.webCam.texture = texture1

if __name__ == '__main__':
    CamApp().run()

