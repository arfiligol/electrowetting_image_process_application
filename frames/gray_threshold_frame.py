import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from tkinter import filedialog

from .base_frame import BaseFrame
from custom_widget import IntScale

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from image_processor import ImageProcessor

class GrayThresholdFrame(BaseFrame):
    """灰階轉二值圖"""
    def __init__(self, parent, processor: "ImageProcessor", *args, **kwargs):
        self.processor = processor
        super().__init__(parent, "Gray Threshold", *args, **kwargs)


        # Threshold 1
        threshold1_container = ttkb.Frame(self)
        threshold1_container.pack(fill="x", padx=10, pady=5)

        ttkb.Label(threshold1_container, text='Threshold 1').pack(side="left")
        ttkb.Label(threshold1_container, textvariable=self.processor.gray_threshold1).pack(side="left", padx=(10, 0))

        self.threshold1Slider = IntScale(self, 
                                             from_=0, 
                                             to=255,
                                             orient='horizontal',
                                             variable=self.processor.gray_threshold1,
                                             command=lambda event: self.update_gray_to_binary_threshold())
        self.threshold1Slider.pack(fill='x', padx=10, pady=5)

        # Threshold 2
        threshold2_container = ttkb.Frame(self)
        threshold2_container.pack(fill="x", padx=10, pady=5)

        ttkb.Label(threshold2_container, text="Threshold 2").pack(side="left")
        ttkb.Label(threshold2_container, textvariable=self.processor.gray_threshold2).pack(side="left", padx=(10, 0))

        self.threshold2Slider = IntScale(self, 
                                              from_=0, 
                                              to=255,
                                              orient='horizontal',
                                              variable=self.processor.gray_threshold2,
                                             command=lambda event: self.update_gray_to_binary_threshold())
        self.threshold2Slider.pack(fill='x', padx=10, pady=5)
    

    def save_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            pass


    def update_gray_to_binary_threshold(self):
        self.processor.process_image()
        self.notify_observers()

    # Override
    def update_image(self, img = None):
        if img is None:
            img = self.processor.gray_binary_img
        super().update_image(img)