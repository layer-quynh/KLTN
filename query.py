from flask import Flask, jsonify, request
import time
from PIL import Image
import cv2
from mutext import MUTEX

model = MUTEX()

def jsonify_str(output_list):
    with app.app_context():
        with app.test_request_context():
            result = jsonify(output_list)
    return result

app = Flask(__name__)

def create_query_result(input_url, results, error=None):
    if error is not None:
        results = 'Error: ' + str(error)
    query_result = {
        'results': results
    }
    return query_result

def compute_iou(box1, box2):
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    inter_area = max(0, x_max_inter - x_min_inter + 1) * max(0, y_max_inter - y_min_inter + 1)

    s1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    s2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = float(inter_area / (s1 + s2 - inter_area))
    return iou

def get_iou_list_box(list1, list2):
    iou_avg = 0
    if list1 is not None and list2 is not None and len(list1) != len(list2):
        return 0
    for index, box1 in enumerate(list1):
        box2 = list2[index]
        iou_avg += compute_iou(box1, box2)
    iou_avg = iou_avg / len(list1)
    if iou_avg > 0.7:
        return 1
    return 0

@app.route("/image")
def queryimg():
    try:
        image_url = request.args.get('url', default='', type=str)
        img = Image.open(image_url).convert('RGB')
    except Exception as ex:
        return jsonify_str(create_query_result("", "", ex))
    start = time.time()
    result_text, result_box = model.get_content_image(img)
    result = {"result: ": result_text, "time": time.time() - start}
    return jsonify_str(result)

@app.route("/video")
def query():
    try:
        url = request.args.get('url', default='', type=str)
    except Exception as ex:
        return jsonify_str(ex)
    start = time.time()
    result = []
    list_boxes = []
    vidcap = cv2.VideoCapture(url)
    success, image = vidcap.read()
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
        success, image = vidcap.read()
        if image is not None:
            in_frame = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            text, result_box = model.get_content_image(in_frame)
            result.append(text)
            list_boxes.append(result_box)
        count += 1

    # lines_result = [[]]
    # count = 0
    # # remove duplicate line text
    # leng = len(list_boxes) - 1
    # for index, boxes in enumerate(list_boxes[:leng]):
    #     if list_boxes[index] != list_boxes[leng]:
    #         iou_avg = get_iou_list_box(boxes, list_boxes[index + 1])
    #         print('Boxes: ', boxes)
    #         print('List boxes: ', list_boxes[index + 1])
    #         if iou_avg == 1:
    #             # del list_box_final[-1]
    #             del list_boxes[index + 1]
    #             lines_result[count].append(result[index + 1])
    #             # list_box_final.append(list_boxes[index + 1])
    #         else:
    #             count = count + 1
    #             lines_new = [result[index + 1]]
    #             lines_result.append(lines_new)
    #         # list_box_final.append(list_boxes[index + 1])
    #
    # result_final = []
    # for line in lines_result:
    #     if len(line) > 2:
    #         result_final.append(line[-2])

    result = list(dict.fromkeys(result))

    rs = {"result": result, "time": str(time.time() - start)}
    return jsonify_str(rs)

app.run("localhost", 8080, threaded=False, debug=False)
