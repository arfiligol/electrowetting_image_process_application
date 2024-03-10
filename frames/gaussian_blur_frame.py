import tkinter as tk
import ttkbootstrap as ttkb

from .base_frame import BaseFrame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from image_processor import ImageProcessor

class GaussianBlurFrame(BaseFrame):
    """高斯模糊步驟的Frame"""
    def __init__(self, parent, processor: 'ImageProcessor', *args, **kwargs):
        self.processor = processor
        super().__init__(parent, "Gaussian Blur", *args, **kwargs)
        

        container = ttkb.Frame(self)
        container.pack(fill='x', padx=10, pady=5)

        ttkb.Label(container, text='Gaussian Blur ksize').pack(side="left")
        ttkb.Label(container, textvariable=self.processor.gaussian_blur_ksize).pack(side="left", padx=(10, 0))


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
    
    def update_ksizeSlider_value(self):
        self.processor.process_image()
        self.notify_observers()
    
    def update_image(self, img = None):
        if img is None:
            img = self.processor.blurred
        super().update_image(img)