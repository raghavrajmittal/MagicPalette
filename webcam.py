'''Webcam class to support threading'''

import cv2
from threading import Thread
import time

class Webcam:

    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.current_frame = self.video_capture.read()[1]
        self.running = True

    # create thread for capturing images
    def start(self):
        self.th = Thread(target=self._update_frame, args=())
        self.th.daemon = True
        self.th.start()

    def _update_frame(self):
        while(self.running):
            self.current_frame = self.video_capture.read()[1]
        return

    def get_current_frame(self):
        return self.current_frame

    # clean up
    def end(self):
        self.running = False
        self.video_capture.release()
        cv2.destroyAllWindows()
