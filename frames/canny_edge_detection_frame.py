import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

from .base_frame import BaseFrame
from custom_widget import IntScale

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from image_processor import ImageProcessor


class CannyEdgeDetectionFrame(BaseFrame):
    """Canny邊緣檢測步驟的Frame"""
    def __init__(self, parent, processor: 'ImageProcessor', *args, **kwargs):
        self.processor = processor
        super().__init__(parent, "Canny Edge Detection", *args, **kwargs)

        # Threshold 1
        threshold1_container = ttkb.Frame(self)
        threshold1_container.pack(fill='x', padx=10, pady=5)

        ttkb.Label(threshold1_container, text='Threshold 1').pack(side="left")
        ttkb.Label(threshold1_container, textvariable=self.processor.canny_threshold1).pack(side="left", padx=(10, 0))

        self.lowThresholdSlider = IntScale(self, 
                                             from_=0, 
                                             to=100,
                                             orient='horizontal',
                                             variable=self.processor.canny_threshold1,
                                             command=lambda event: self.update_slider_value())
        self.lowThresholdSlider.pack(fill='x', padx=10, pady=5)

        # Threshold 2
        threshold2_container = ttkb.Frame(self)
        threshold2_container.pack(fill="x", padx=10, pady=5)

        ttkb.Label(threshold2_container, text="Threshold 2").pack(side="left")
        ttkb.Label(threshold2_container, textvariable=self.processor.canny_threshold2).pack(side="left", padx=(10, 0))

        self.highThresholdSlider = IntScale(self, 
                                              from_=100, 
                                              to=200,
                                              orient='horizontal',
                                              variable=self.processor.canny_threshold2,
                                             command=lambda event: self.update_slider_value())
        self.highThresholdSlider.pack(fill='x', padx=10, pady=5)

        # Contour Minium Area
        min_contour_area_container = ttkb.Frame(self)
        min_contour_area_container.pack(fill="x", padx=10, pady=5)

        ttkb.Label(min_contour_area_container, text="Minimum Contour Area").pack(side="left")
        ttkb.Label(min_contour_area_container, textvariable=self.processor.min_contour_area).pack(side="left", padx=(10, 0))

        self.minContourAreaSlider = IntScale(self,
                                             from_=0,
                                             to_=500,
                                             orient="horizontal",
                                             variable=self.processor.min_contour_area,
                                             command=lambda event: self.update_slider_value())
        self.minContourAreaSlider.pack(fill="x", padx=10, pady=5)

    # Slider Value Update Commands
    def update_slider_value(self):
        self.processor.process_image()
        self.notify_observers()
    

    # Override
    def update_image(self, img = None):
        if img is None:
            img = self.processor.contour_img
        super().update_image(img)
