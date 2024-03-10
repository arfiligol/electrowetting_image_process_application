import tkinter as tk
import ttkbootstrap as ttkb

from .base_frame import BaseFrame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from image_processor import ImageProcessor

class GaussianBlurFrame(BaseFrame):
    """高斯模糊步驟的Frame"""
    def __init__(self, parent, processor: 'ImageProcessor', *args, **kwargs):
        super().__init__(parent, "Gaussian Blur", *args, **kwargs)
        self.processor = processor
        
        # Since the ksize should be odd number, we can't use ttk widget. It doesn't support args like 'resolution'.
        self.ksizeSlider = tk.Scale(self, 
                                      from_=1, 
                                      to=31, 
                                      resolution=2,
                                      length=31,
                                      orient='horizontal',
                                      variable=self.processor.gaussian_blur_ksize,
                                      command = lambda event: self.update_ksizeSlider_value())
        self.ksizeSlider.pack(fill='x', padx=10, pady=5)
        self.update_image(self.processor.blurred)
    
    def update_ksizeSlider_value(self):
        self.processor.process_image()
        self.update_image(self.processor.blurred)
        