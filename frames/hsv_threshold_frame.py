from tkinter import *
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

from .base_frame import BaseFrame
from custom_widget import IntScale

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from image_processor import ImageProcessor


class HSVThresholdFrame(BaseFrame):
    """HSV閾值化步驟的Frame"""
    def __init__(self, parent, processor: 'ImageProcessor', *args, **kwargs):
        self.processor = processor
        super().__init__(parent, "HSV Threshold", *args, **kwargs)
        
        # 為HSV閾值化創建滑桿和標籤
        # Lower H
        lower_h_container = ttkb.Frame(self)
        lower_h_container.pack(fill='x', padx=10, pady=5)

        ttkb.Label(lower_h_container, text='Lower H').pack(side="left")
        ttkb.Label(lower_h_container, textvariable=self.processor.lower_h).pack(side="left", padx=(10, 0))
        self.lowerHSlider = IntScale(self,
                                       from_=0, 
                                       to=180, 
                                       length=360,
                                       orient='horizontal',
                                       variable=self.processor.lower_h,
                                       command=lambda event: self.update_lowerHSlider_value())
        self.lowerHSlider.pack(fill='x', padx=10, pady=5)

        # Upper H
        upper_h_container = ttkb.Frame(self)
        upper_h_container.pack(fill='x', padx=10, pady=5)

        ttkb.Label(upper_h_container, text='Upper H').pack(side="left")
        ttkb.Label(upper_h_container, textvariable=self.processor.upper_h).pack(side="left", padx=(10, 0))
        self.upperHSlider = IntScale(self,
                                    from_=0, 
                                    to=180, 
                                    length=360,
                                    orient='horizontal',
                                    variable=self.processor.upper_h,
                                    command=lambda event: self.update_upperHSlider_value())
        self.upperHSlider.pack(fill='x', padx=10, pady=5)

        # Lower S
        lower_s_container = ttkb.Frame(self)
        lower_s_container.pack(fill='x', padx=10, pady=5)

        ttkb.Label(lower_s_container, text='Lower S').pack(side="left")
        ttkb.Label(lower_s_container, textvariable=self.processor.lower_s).pack(side="left", padx=(10, 0))
        self.lowerSSlider = IntScale(self, 
                                    from_=0, 
                                    to=255, 
                                    length=256,
                                    orient='horizontal',
                                    variable=self.processor.lower_s,
                                    command=lambda event: self.update_lowerSSlider_value())
        self.lowerSSlider.pack(fill='x', padx=10, pady=5)

        # Upper S
        upper_s_container = ttkb.Frame(self)
        upper_s_container.pack(fill='x', padx=10, pady=5)

        ttkb.Label(upper_s_container, text='Upper S').pack(side="left")
        ttkb.Label(upper_s_container, textvariable=self.processor.upper_s).pack(side="left", padx=(10, 0))
        self.upperSSlider = IntScale(self, 
                                    from_=0, 
                                    to=255, 
                                    length=256,
                                    orient='horizontal',
                                    variable=self.processor.upper_s,
                                    command=lambda event: self.update_upperSSlider_value())
        self.upperSSlider.pack(fill='x', padx=10, pady=5)

        # Lower V
        lower_v_container = ttkb.Frame(self)
        lower_v_container.pack(fill='x', padx=10, pady=5)

        ttkb.Label(lower_v_container, text='Lower V').pack(side="left")
        ttkb.Label(lower_v_container, textvariable=self.processor.lower_v).pack(side="left", padx=(10, 0))
        self.lowerVSlider = IntScale(self, 
                                    from_=0, 
                                    to=255, 
                                    length=256,
                                    orient='horizontal',
                                    variable=self.processor.lower_v,
                                    command=lambda event: self.update_lowerVSlider_value())
        self.lowerVSlider.pack(fill='x', padx=10, pady=5)

        # Upper V
        upper_v_container = ttkb.Frame(self)
        upper_v_container.pack(fill='x', padx=10, pady=5)

        ttkb.Label(upper_v_container, text='Upper V').pack(side="left")
        ttkb.Label(upper_v_container, textvariable=self.processor.upper_v).pack(side="left", padx=(10, 0))
        self.upperVSlider = IntScale(self, 
                                    from_=0, 
                                    to=255, 
                                    length=256,
                                    orient='horizontal',
                                    variable=self.processor.upper_v,
                                    command=lambda event: self.update_upperVSlider_value())
        self.upperVSlider.pack(fill='x', padx=10, pady=5)

    

    # Slider Value Update Commands
    def update_lowerHSlider_value(self):
        if (self.lowerHSlider.get() >= self.upperHSlider.get()):
            self.upperHSlider.set(self.lowerHSlider.get())
        self.processor.process_image()
        self.notify_observers()

    def update_upperHSlider_value(self):
        if (self.upperHSlider.get() < self.lowerHSlider.get()):
            self.lowerHSlider.set(self.upperHSlider.get())
        self.processor.process_image()
        self.notify_observers()

    def update_lowerSSlider_value(self):
        if (self.lowerSSlider.get() >= self.upperSSlider.get()):
            self.upperSSlider.set(self.lowerSSlider.get())
        self.processor.process_image()
        self.notify_observers()
    
    def update_upperSSlider_value(self):
        if (self.upperSSlider.get() < self.lowerSSlider.get()):
            self.lowerSSlider.set(self.upperSSlider.get())
        self.processor.process_image()
        self.notify_observers()

    def update_lowerVSlider_value(self):
        if (self.lowerVSlider.get() >= self.upperVSlider.get()):
            self.upperVSlider.set(self.lowerVSlider.get())
        self.processor.process_image()
        self.notify_observers()

    def update_upperVSlider_value(self):
        if (self.upperVSlider.get() < self.lowerVSlider.get()):
            self.lowerVSlider.set(self.upperVSlider.get())
        self.processor.process_image()
        self.notify_observers()



    # Override
    def update_image(self, img = None):
        if img is None:
            img = self.processor.mask
        super().update_image(img)