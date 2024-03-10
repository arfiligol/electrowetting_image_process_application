import cv2

from .base_frame import BaseFrame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..image_processor import ImageProcessor

class OriginImageFrame(BaseFrame):
    """Origin Image's Frame"""
    def __init__(self, parent, processor: 'ImageProcessor', *args, **kwargs):
        self.processor = processor
        super().__init__(parent, "Origin Image", *args, **kwargs)

    def update_image(self, img = None):
        if img is None:
            img = cv2.cvtColor(self.processor.image, cv2.COLOR_BGR2RGB)
        super().update_image(img)