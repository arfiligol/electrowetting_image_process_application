import cv2
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
from numpy import ndarray
import random


def create_random_color_image():
    """創建一個填充隨機顏色的圖像"""
    # 生成一個隨機顏色
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # 創建一個 200x200 的圖像
    img = Image.new("RGB", (200, 200), color)
    return img

class StepFrame(ttkb.Frame):
    """ 代表每個步驟處理的框架 """
    def __init__(self, parent, step_name, *args, **kwargs):
        super().__init__(parent, padding=5, *args, **kwargs)
        self.step_name = step_name
        self.init_ui()
        
    def init_ui(self):
        # 显示步骤名称
        step_label = ttkb.Label(self, text=self.step_name, bootstyle=INFO)
        step_label.pack(pady=(0, 5))
        
        # 创建并显示图像
        img = create_random_color_image()
        self.img_tk = ImageTk.PhotoImage(img)  # 需要保持对ImageTk.PhotoImage对象的引用
        image_label = ttkb.Label(self, image=self.img_tk)
        image_label.pack()


class ImageProcessingApp:
    """ Main App Class """
    def __init__(self, window):
        self.window = window
        self.steps = [
            "原始圖像",
            "高斯模糊",
            "轉換至 HSV",
            "顏色 Threshold",
            "Canny 邊緣檢測",
            "找尋輪廓",
            "篩選",
            "繪製輪廓"
        ]
        self.init_ui()
        
    def init_ui(self):
        # 设置窗口布局
        self.window.title("Electrowetting Image Processing")
        self.window.geometry("1280x920")
        
        for i in range(3):
            self.window.columnconfigure(i, weight=1)
        for i in range(2):
            self.window.rowconfigure(i, weight=1)
        
        # 为每个处理步骤创建框架
        for i, step in enumerate(self.steps):
            frame = StepFrame(self.window, step)
            frame.grid(row=i // 3, column=i % 3, padx=10, pady=10, sticky="nsew")





# Image Processing Functions
def gaussianBlur(img: Image, ksize: tuple) -> Image: # 2 sliders
    return cv2.GaussianBlur(img, ksize, 0) # 0 means 'sigmaX' set to 0, the algorithm will automatically choose the standart derivation


def hsv(img: Image) -> Image:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def inRange(img: Image, lower: ndarray, upper: ndarray) -> Image: # 6 sliders
    return cv2.inRange(img, lower, upper)

def cannyEdgeDetection(img: Image, low_threshold: float, high_threshold: float): # 2 sliders
    return cv2.Canny(img, low_threshold, high_threshold)

def findContours(img: Image, lowest_area: float) -> list: # 1 sliders
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > lowest_area]

def drawContours(img: Image, contours: list) -> Image:
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    return contour_img



# Create a frame for each processing step, include one Label for illustrating and one image
for i, step in enumerate(steps):
    # Create a frame
    frame = ttkb.Frame(window, padding = 5)
    frame.grid(row = i // 3, column = i % 3, padx = 10, pady = 10, sticky = "nsew")

    step_label = ttkb.Label(frame, text = step, bootstyle = INFO)
    step_label.pack(pady = (0, 5))

    # Create and show the image
    img = create_random_color_image()
    img_tk = ImageTk.PhotoImage(img)
    image_label = ttkb.Label(frame, image = img_tk)
    image_label.image = img_tk
    image_label.pack()

# Start Window loop
window.mainloop()