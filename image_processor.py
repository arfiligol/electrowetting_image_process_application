import cv2
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
        self.gaussian_blur_ksize = 3  # Assuming slider controls odd values 1, 3, 5, etc.
        self.gaussian_blur_sigmaX = 0  # Sigma value can be adjusted if needed, often set to 0

        # Parameters for inRange (HSV Thresholding)
        self.lower_h = 0
        self.lower_s = 0
        self.lower_v = 0
        self.upper_h = 180  # HSV range is 0-180 for H in OpenCV
        self.upper_s = 255
        self.upper_v = 255

        # Parameters for Canny Edge Detection
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150

        # Parameter for Contour Filtering
        self.min_contour_area = 100


    # Updating Parameter Values
    # 更新高斯模糊核大小
    def update_gaussian_blur_ksize(self, ksize: int):
        self.gaussian_blur_ksize = ksize
        # 可以在這裡調用應用高斯模糊的函數，並更新圖像
        self.process_image()

    # 更新HSV閾值化參數
    def update_hsv_threshold(self, lower_h=None, lower_s=None, lower_v=None, upper_h=None, upper_s=None, upper_v=None):
        if lower_h is not None: self.lower_h = lower_h
        if lower_s is not None: self.lower_s = lower_s
        if lower_v is not None: self.lower_v = lower_v
        if upper_h is not None: self.upper_h = upper_h
        if upper_s is not None: self.upper_s = upper_s
        if upper_v is not None: self.upper_v = upper_v

        # 可以在這裡調用應用HSV閾值化的函數，並更新圖像
        self.process_image()

    # 更新Canny邊緣檢測閾值
    def update_canny_thresholds(self, threshold1: int, threshold2: int):
        if threshold1 is not None: self.canny_threshold1 = threshold1
        if threshold2 is not None: self.canny_threshold2 = threshold2
        # 可以在這裡調用應用Canny邊緣檢測的函數，並更新圖像
        self.process_image()

    # 更新輪廓過濾的最小面積閾值
    def update_min_contour_area(self, min_area: int):
        self.min_contour_area = min_area
        # 可以在這裡調用找尋並過濾輪廓的函數，並更新圖像
        self.process_image()
    
    # Processing Image
    def apply_gaussian_blur(self):
        self.blurred = cv2.GaussianBlur(self.image, (self.gaussian_blur_ksize, self.gaussian_blur_ksize), self.gaussian_blur_sigmaX)
    
    def convert_to_hsv(self):
        self.hsv = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2HSV)

    def apply_inrange_threshold(self):
        lower_bound = np.array([self.lower_h, self.lower_s, self.lower_v], dtype = np.uint8)
        upper_bound = np.array([self.upper_h, self.upper_s, self.upper_v], dtype = np.uint8)
        self.mask = cv2.inRange(self.hsv, lower_bound, upper_bound)

    def apply_canny_edge_detection(self):
        self.edges = cv2.Canny(self.mask, self.canny_threshold1, self.canny_threshold2)

    def find_and_filter_contours(self):
        contours, _ = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]

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
