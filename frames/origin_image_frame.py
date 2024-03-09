import cv2

from .base_frame import BaseFrame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..image_processor import ImageProcessor

class OriginImageFrame(BaseFrame):
    """Origin Image's Frame"""
    def __init__(self, parent, processor: 'ImageProcessor', *args, **kwargs):
        super().__init__(parent, "Origin Image", *args, **kwargs)
        self.processor = processor

        self.update_image(cv2.cvtColor(self.processor.image, cv2.COLOR_BGR2RGB))
