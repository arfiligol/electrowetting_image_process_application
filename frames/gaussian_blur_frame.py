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
        
        self.ksizeSlider = ttkb.Scale(self, from_=1, to=31, orient='horizontal')
        self.ksizeSlider.pack(fill='x', padx=10, pady=5)
        print(type(self.processor.blurred))
        self.update_image(self.processor.blurred)
    
    def update_ksizeSlider_value(self):
        