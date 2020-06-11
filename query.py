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

    result = list(dict.fromkeys(result))

    rs = {"result": result, "time": str(time.time() - start)}
    return jsonify_str(rs)


app.run("localhost", 8080, threaded=False, debug=False)
