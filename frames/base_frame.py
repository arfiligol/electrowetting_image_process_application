import cv2
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import random

def create_random_color_image():
    """創建一個填充隨機顏色的圖像"""
    # 生成一個隨機顏色
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # 創建一個 200x200 的圖像
    img = Image.new("RGB", (350, 300), color)
    return img

class BaseFrame(ttkb.Frame):
    """所有步驟Frame的基礎類別"""
    def __init__(self, parent, title, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.image_label = None # Used to show the image
        self.init_ui(title)
        
    def init_ui(self, title):
        self.titleLabel = ttkb.Label(self, text=title, bootstyle=INFO)
        self.titleLabel.pack(pady=(0, 5))

        self.update_image(create_random_color_image())

    def update_image(self, img):
        """更新顯示的影像"""
        # 將 PIL 影像轉換為 Tkinter 可用的影像格式
        if (type(img) is not Image.Image):
            # print(type(img))
            img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img)
        if self.image_label is None:
            self.image_label = ttkb.Label(self, image = img_tk)
            self.image_label.pack()
        else:
            self.image_label.configure(image=img_tk)
        
        self.image_label.image = img_tk
