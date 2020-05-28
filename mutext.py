from CRNN_TEXTREG.model.model_predict import TEXTREG
from BOX_MODEL.text_detection import TEXTDETECTION

class MUTEX(object):
    def __init__(self):
        self.model_box = TEXTDETECTION()
        self.model_reg = TEXTREG()
        self.target_frame = 100

    def get_content_image(self, image):
        list_img, result_box = self.model_box.predict_box(image)
        if len(list_img) == 0:
            return "", []
        text = self.model_reg.predict_batch(list_img)
        return text, result_box