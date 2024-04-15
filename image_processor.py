import cv2
from tkinter import IntVar, DoubleVar
import numpy as np
import pickle

class ImageProcessor:
    def __init__(self, image):
        # Images
        self.image = image
        image_height, image_width, image_channels = self.image.shape

        # Attributes
        self.blurred = None 
        self.gray_binary_img = None
        self.hsv = None
        self.mask = None
        self.edges = None
        self.contours = None
        self.contour_img = None

        # Parameters for Gaussian Blur
        self.gaussian_blur_ksize = IntVar(value=3)
        self.gaussian_blur_sigmaX = IntVar(value=0)

        # Parameters for Gray to Binary ( Threshold)
        self.gray_threshold1 = IntVar(value=0)
        self.gray_threshold2 = IntVar(value=255)

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
        self.contour_area = DoubleVar()
        self.min_contour_area = IntVar(value=0)
        self.max_contour_area = IntVar(value=(image_height * image_width))


    
    # Processing Image
    def apply_gaussian_blur(self):
        self.blurred = cv2.GaussianBlur(self.gray, (self.gaussian_blur_ksize.get(), self.gaussian_blur_ksize.get()), self.gaussian_blur_sigmaX.get())
    
    def convert_to_gray(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def convert_to_hsv(self):
        self.hsv = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2HSV)

    def apply_inrange_threshold(self):
        lower_bound = np.array([self.lower_h.get(), self.lower_s.get(), self.lower_v.get()], dtype=np.uint8)
        upper_bound = np.array([self.upper_h.get(), self.upper_s.get(), self.upper_v.get()], dtype=np.uint8)
        self.mask = cv2.inRange(self.hsv, lower_bound, upper_bound)

    def apply_gray_threshold(self):
        ret, binary_im = cv2.threshold(self.blurred, self.gray_threshold1.get(), self.gray_threshold2.get(), cv2.THRESH_BINARY)
        self.gray_binary_img = binary_im

    def apply_canny_edge_detection(self):
        self.edges = cv2.Canny(self.gray_binary_img, self.canny_threshold1.get(), self.canny_threshold2.get())

    def find_and_filter_contours(self):
        contours, _ = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_filted = [c for c in contours if cv2.contourArea(c) <= self.max_contour_area.get()]
        min_filted = [c for c in max_filted if cv2.contourArea(c) >= self.min_contour_area.get()]
        self.contours = min_filted

    def draw_contours(self):
        self.contour_img = self.image.copy()
        cv2.drawContours(self.contour_img, self.contours, -1, (0, 255, 0), 2)
    
    def get_information_of_contours(self):
        if (len(self.contours) > 1):
            pass
        else:
            for cnt in self.contours:
                # 面積
                area = cv2.contourArea(cnt)
                print("Area: ", area)
                self.contour_area.set(area)

                # 周長
                perimeter = cv2.arcLength(cnt, True)
                print("Perimeter: ", perimeter)

                # 輪廓的近似形狀
                approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
                print("Approx Shape: ", approx)

                # 輪廓的邊界矩形
                x, y, w, h = cv2.boundingRect(cnt)
                print("Bounding Rect x: ", x, " y: ", y, " w: ", w, " h: ", h)

                # 獲取最小矩形區域
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                print("Rect: ", rect)
                print("Box: ", box)

    def fit_ellipse_and_draw_it(self):
        if (len(self.contours) == 1):
            print("Only one contour")
            for cnt in self.contours:

                # 獲取輪廓點的所有y值
                y_values = [point[0][1] for point in cnt]

                # 計算y值的直方圖
                hist, bin_edges = np.histogram(y_values, bins=range(int(min(y_values)), int(max(y_values)) + 1))

                # 找到最大頻率的y值，這將是最可能的平面邊緣的y值
                c = bin_edges[np.argmax(hist)]
                print("水平線 y 值: " + str(c))

                # 將y_values序列化並存儲到檔案
                with open('y_values_dump.pkl', 'wb') as file:
                    pickle.dump(y_values, file)

                # 繪製水平線
                cv2.line(self.contour_img, (0, int(c)), (self.contour_img.shape[1], int(c)), (255,0,0), 2)


                ellipse = cv2.fitEllipse(cnt)

                # Extract the center, axis lengths, and rotation angle from the ellipse parameters.
                (x0, y0), (MA, ma), angle = ellipse

                # 計算接觸點
                a = MA / 2
                b = ma / 2
                # contact_x1 = 
                

                # 繪製 fit 橢圓 and most point 到 contour image 上
                cv2.ellipse(self.contour_img, ellipse, (0,0,255), 2)

                

    def process_image(self):
        self.convert_to_gray()
        self.apply_gaussian_blur()
        self.apply_gray_threshold()
        # self.convert_to_hsv()
        # self.apply_inrange_threshold()
        self.apply_canny_edge_detection()
        self.find_and_filter_contours()
        self.draw_contours()
        # self.get_information_of_contours()
        self.fit_ellipse_and_draw_it()
        return self.contour_img
