import colorsys
import os
from timeit import default_timer as timer
import time
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from BOX_MODEL.model import yolo_eval, yolo_body, tiny_yolo_body
from BOX_MODEL.utils import letterbox_image
from keras.utils import multi_gpu_model
import tensorflow as tf
from xml.etree import ElementTree

class TEXTDETECTION(object):
    # model_path = "/home/quynhnguyen/Documents/project/uet/khoa_luan/mutex_video/weights/box/weights_box.h5"
    model_path = "/home/quynhnguyen/Documents/project/uet/khoa_luan/KLTN/weights/box/yolo_weights.h5"
    anchors_path = '/home/quynhnguyen/Documents/project/uet/khoa_luan/KLTN/BOX_MODEL/yolo_anchors.txt'
    classes_path = '/home/quynhnguyen/Documents/project/uet/khoa_luan/KLTN/BOX_MODEL/yolo.names'
    score = 0.5
    iou = 0.45
    model_image_size = (416, 416)
    gpu_num = 1

    def __init__(self):
        self.class_names = self._get_class()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.06
        self.anchors = self._get_anchors()
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            with self.sess.as_default():
                self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = self.model_path
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def predict_box(self, image):
        img = image.copy()
        self.width, self.height = image.size
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required.'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required.'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                            image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        # print(image_data)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)
        # print(image_data)
        with self.graph.as_default():
            with self.sess.as_default():
                out_boxes, out_scores, out_classes = self.sess.run(
                    [self.boxes, self.scores, self.classes],
                    feed_dict={
                        self.yolo_model.input: image_data,
                        self.input_image_shape: [image.size[1], image.size[0]],
                        K.learning_phase(): 0
                    })

        # print("Out boxes: ", out_boxes)
        # print("Out scores: ", out_scores)
        # print("Out classes: ", out_classes)
        result = []
        for index, label in enumerate(out_classes):
            print(label)
            box = out_boxes[index]
            result.append([int(box[1]), int(box[0]), int(box[3]), int(box[2])])
        # print("Result: ", result)
        result_sort = self.sort_line(result)
        # print(result_sort)
        result_img = []
        result_box = []
        for box in result_sort:
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]
            result_box.append([xmin, ymin, xmax, ymax])
            result_img.append(
                img.copy().crop((max(0, xmin), max(0, ymin), xmax, ymax)).convert('L')
            )

        return result_img, result_box
        
    def sort_line(self, boxes):
        if len(boxes) == 0:
            return []
        boxes = sorted(boxes, key=lambda x: x[1])
        print(boxes)
        lines = [[]]

        print(boxes[0])
        y_center = (boxes[0][1] + boxes[0][3]) / 2.0
        i = 0
        for box in boxes:
            if box[1] < y_center:
                lines[i].append(box)
            else:
                lines[i] = sorted(lines[i], key=lambda x: x[0])
                y_center = (box[1] + box[3]) / 2.0
                lines.append([])
                i += 1
                lines[i].append(box)

        temp = []

        for line in lines:
            temp.append(line[0][1])
        index_sort = np.argsort(np.array(temp)).tolist()
        lines_new = [lines[i] for i in index_sort]

        result = []
        for line in lines_new:
            result.extend(self.remove(line))

        return result
    
    def remove(self, line):
        line = sorted(line, key=lambda x: x[0])
        result = []
        check_index = -1
        for index in range(len(line)):
            if check_index == index:
                pass
            else:
                result.append(line[index])
                check_index = index
            if index == len(line) - 1:
                break
            if self.compute_iou(line[index], line[index + 1]) > 0.4:
                s1 = (line[index][2] - line[index][0] + 1) * (line[index][3] - line[index][1] + 1)
                s2 = (line[index + 1][2] - line[index + 1][0] + 1) * (line[index + 1][3] - line[index + 1][1] + 1)
                if s2 > s1:
                    del(result[-1])
                check_index = index + 1
        result = sorted(result, key=lambda x: x[0])
        return result

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

    def close_session(self):
        self.sess.close()

def get_real_boxes(file_name):
    dom = ElementTree.parse(file_name)
    objects = dom.findall("object")

    real_boxes = []
    for o in objects:
        xmin = o.find("bndbox").find("xmin").text
        xmax = o.find("bndbox").find("xmax").text
        ymin = o.find("bndbox").find("ymin").text
        ymax = o.find("bndbox").find("ymax").text

        real_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])

    print("Real boxes: ", real_boxes)
    return real_boxes

def get_accuracy(model, path_files_test):
    files = [os.path.join(path_files_test, f) for f in os.listdir(path_files_test) if os.path.isfile(os.path.join(path_files_test, f))]
    count = 0
    total_real_boxes = 0
    total_images = 0

    for index, file in enumerate(files):
        if file.endswith('.jpg') or file.endswith('.png'):
            total_images += 1
            if file.endswith('.jpg'):
                txt_file = file.replace(".jpg", ".xml")
            else:
                txt_file = file.replace(".png", ".xml")
            _, boxes_predict = model.predict_box(Image.open(file))
            with open(txt_file, 'r') as txt:
                real_boxes = get_real_boxes(txt)
            total_real_boxes += len(real_boxes)
            # print(boxes_predict)
            for box_predict in boxes_predict:
                for i, box_real in enumerate(real_boxes):
                    iou = model.compute_iou(box_predict, box_real)
                    if iou >= 0.8:
                        # print(iou)
                        del real_boxes[i]
                        count += 1
                        break

    print('Count: ', count)
    print("Total boxes: ", total_real_boxes)
    print("Total images: ", total_images)
    print('Accuracy: {:.2f}%'.format((count / total_real_boxes) * 100))

# path_files = "/home/quynhnguyen/Documents/project/uet/khoa_luan/data/test"
# model = TEXTDETECTION()
#
# get_accuracy(model, path_files)
# model = TEXTDETECTION()
# boxes, scores, classes = model.generate()
# print("Boxes: ", boxes)
# print("Scores: ", scores)
# print("Classes: ", classes)
# model.predict_box(Image.open("/home/quynhnguyen/Documents/project/uet/khoa_luan/data/test/ffd58655-0fd6-43ad-8731-36da79be5c8b.png"))