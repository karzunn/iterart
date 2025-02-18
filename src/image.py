import numpy as np
import cv2



class Image:

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.data = np.zeros(width * height, dtype=np.uint32)

    def save(self, filename: str):
        log_image = np.log1p(self.data)
        max_val = np.max(log_image)
        normalized = (log_image / max_val * 65535).astype(np.uint16)
        reshaped = normalized.reshape((self.height, self.width))
        cv2.imwrite(filename, reshaped)