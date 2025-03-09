import cv2

class TextSegment:
    def __init__(self,
                 image_path: str,
                 type: str,  # title, text, figure, table
                 ):
        self.image_path = image_path
        self.type = type
        self.content = "" # markdown 格式

    def get_image(self):
        return cv2.imread(self.image_path)
