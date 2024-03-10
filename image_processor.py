import cv2
from tkinter import IntVar
import numpy as np

class ImageProcessor:
    def __init__(self, image):
        # Images
        self.image = image

        # Attributes
        self.blurred = None 
        self.hsv = None
        self.mask = None
        self.edges = None
        self.contours = None
        self.contour_img = None

        # Parameters for Gaussian Blur
        self.gaussian_blur_ksize = IntVar(value=3)
        self.gaussian_blur_sigmaX = IntVar(value=0)

        # Parameters for inRange (HSV Thresholding)
        self.lower_h = IntVar(value=0)
        self.lower_s = IntVar(value=0)
        self.lower_v = IntVar(value=0)
        self.upper_h = IntVar(value=180)
        self.upper_s = IntVar(value=255)
        self.upper_v = IntVar(value=255)

        # Parameters for Canny Edge Detection
        self.canny_threshold1 = IntVar(value=50)
        self.canny_threshold2 = IntVar(value=150)

        # Parameter for Contour Filtering
        self.min_contour_area = IntVar(value=100)


    
    # Processing Image
    def apply_gaussian_blur(self):
        self.blurred = cv2.GaussianBlur(self.image, (self.gaussian_blur_ksize.get(), self.gaussian_blur_ksize.get()), self.gaussian_blur_sigmaX.get())
    
    def convert_to_hsv(self):
        self.hsv = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2HSV)

    def apply_inrange_threshold(self):
        lower_bound = np.array([self.lower_h.get(), self.lower_s.get(), self.lower_v.get()], dtype=np.uint8)
        upper_bound = np.array([self.upper_h.get(), self.upper_s.get(), self.upper_v.get()], dtype=np.uint8)
        self.mask = cv2.inRange(self.hsv, lower_bound, upper_bound)

    def apply_canny_edge_detection(self):
        self.edges = cv2.Canny(self.mask, self.canny_threshold1.get(), self.canny_threshold2.get())

    def find_and_filter_contours(self):
        contours, _ = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area.get()]

    def draw_contours(self):
        self.contour_img = self.image.copy()
        cv2.drawContours(self.contour_img, self.contours, -1, (0, 255, 0), 2)

    def process_image(self):
        self.apply_gaussian_blur()
        self.convert_to_hsv()
        self.apply_inrange_threshold()
        self.apply_canny_edge_detection()
        self.find_and_filter_contours()
        self.draw_contours()
        return self.contour_img
