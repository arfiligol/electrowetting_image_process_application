import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

from .base_frame import BaseFrame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from image_processor import ImageProcessor


class HSVThresholdFrame(BaseFrame):
    """HSV閾值化步驟的Frame"""
    def __init__(self, parent, processor: 'ImageProcessor', *args, **kwargs):
        super().__init__(parent, "HSV Threshold", *args, **kwargs)
        self.processor = processor
        
        # 為HSV閾值化創建滑桿和標籤
        ttkb.Label(self, text='Lower H').pack(fill='x', padx=10, pady=(5, 0))
        self.lowerHSlider = ttkb.Scale(self, from_=0, to=180, orient='horizontal', command=lambda event: self.update_lowerHSlider_value(), value = self.processor.lower_h)
        self.lowerHSlider.pack(fill='x', padx=10, pady=5)

        ttkb.Label(self, text='Upper H').pack(fill='x', padx=10, pady=(5, 0))
        self.upperHSlider = ttkb.Scale(self, from_=0, to=180, orient='horizontal', command=lambda event: self.update_upperHSlider_value(), value = self.processor.upper_h)
        self.upperHSlider.pack(fill='x', padx=10, pady=5)

        ttkb.Label(self, text='Lower S').pack(fill='x', padx=10, pady=(5, 0))
        self.lowerSSlider = ttkb.Scale(self, from_=0, to=255, orient='horizontal')
        self.lowerSSlider.pack(fill='x', padx=10, pady=5)

        ttkb.Label(self, text='Upper S').pack(fill='x', padx=10, pady=(5, 0))
        self.upperSSlider = ttkb.Scale(self, from_=0, to=255, orient='horizontal')
        self.upperSSlider.pack(fill='x', padx=10, pady=5)

        ttkb.Label(self, text='Lower V').pack(fill='x', padx=10, pady=(5, 0))
        self.lowerVSlider = ttkb.Scale(self, from_=0, to=255, orient='horizontal')
        self.lowerVSlider.pack(fill='x', padx=10, pady=5)

        ttkb.Label(self, text='Upper V').pack(fill='x', padx=10, pady=(5, 0))
        self.upperVSlider = ttkb.Scale(self, from_=0, to=255, orient='horizontal')
        self.upperVSlider.pack(fill='x', padx=10, pady=5)

    

    # Slider Value Update Commands
    def update_lowerHSlider_value(self):
        if (self.lowerHSlider.get() >= self.upperHSlider.get()):
            self.upperHSlider.set(self.lowerHSlider.get())
        self.processor.update_hsv_threshold(lower_h = self.lowerHSlider.get())
    
    def update_upperHSlider_value(self):
        if (self.upperHSlider.get() < self.lowerHSlider.get()):
            self.lowerHSlider.set(self.upperHSlider.get())
        self.processor.update_hsv_threshold(upper_h = self.upperHSlider.get())
