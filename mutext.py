from CRNN_TEXTREG.model.model_predict import TEXTREG
from PIL import Image
import os
import glob
from BOX_MODEL.text_detection import TEXTDETECTION
import ffmpeg
import time, cv2, subprocess
import numpy as np
from io import BytesIO
from scipy import spatial

class MUTEX(object):
    def __init__(self):
        self.model_box = TEXTDETECTION()
        self.model_reg = TEXTREG()
        self.target_frame = 100

    def get_video_size(self, filename):
        probe = ffmpeg.probe(filename)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        duration = float(video_info['duration'])
        fps = video_info['avg_frame_rate']
        if '/' in fps:
            split = fps.split('/')
            fps = float(split[0]) / float(split[1])
        else:
            fps = float(fps)
        output_rate = self.target_frame / duration
        output_rate = min(output_rate, fps)
        return width, height, output_rate
    
    def start_ffmpe_process1(self, in_filename, output_framerate):
        args = (
            ffmpeg.input(in_filename).video
                .filter('fps', fps=output_framerate)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .compile()
        )

        return subprocess.Popen(args, stdout=subprocess.PIPE)

    def read_frame(self, process1, width, height):
        # Note: RGB24 == 3 bytes per pixel.
        frame_size = width * height * 3
        in_bytes = process1.stdout.read(frame_size)
        if len(in_bytes) == 0:
            frame = None
        else:
            assert len(in_bytes) == frame_size
            frame = (
                np
                    .frombuffer(in_bytes, np.uint8)
                    .reshape([height, width, 3])
            )
        
        return frame

    def compute_iou(self, box1, box2):

        x_min_inter = max(box1[0], box2[0])
        y_min_inter = max(box1[1], box2[1])
        x_max_inter = min(box1[2], box2[2])
        y_max_inter = min(box1[3], box2[3])

        inter_area = max(0, x_max_inter - x_min_inter + 1) * max(0, y_max_inter - y_min_inter + 1)

        s1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        s2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        iou = float(inter_area / (s1 + s2 - inter_area))
        return iou

    def get_iou_list_box(self, list1, list2):
        iou_avg = 0
        for index, box1 in enumerate(list1[0:4]):
            box2 = list2[index]
            iou_avg = iou_avg / 4.0
            return iou_avg

    def get_content_image(self, image):
        list_img, result_box = self.model_box.predict_box(image)
        if len(list_img) == 0:
            return ""
        text = self.model_reg.predict_batch(list_img)
        return text

    def get_content_video(self, in_filename):
        width, height, output_rate = self.get_video_size(in_filename)
        output_rate = 1
        process1 = self.start_ffmpe_process1(in_filename, output_rate)
        # print('Process1: ', process1)
        result = []
        list_box = []
        total_time = 0
        count_error = 0
        while True:
            total_time += 1
            in_frame = self.read_frame(process1, width, height)
            print("In frame: ", in_frame)
            if in_frame is None:
                break
            in_frame = Image.fromarray(cv2.cvtColor(in_frame, cv2.COLOR_BGR2RGB))
            # image = Image.open(in_frame)
            # print(image)
            list_img, result_box = self.model_box.predict_box(in_frame)
            if len(list_img) < 4:
                count_error += 1
                continue
            text = self.model_reg.predict_batch(list_img)
            print("Text: ", text)
            result.append(text)
            list_box.append(result_box)

        return result, total_time