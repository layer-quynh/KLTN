from flask import Flask, jsonify, request
import time, requests
from PIL import Image
from CRNN_TEXTREG.model.model_predict import TEXTREG
import os
import glob
from BOX_MODEL.text_detection import TEXTDETECTION
import cv2
from io import BytesIO
import uuid
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

@app.route("/image", methods=['GET'])
def queryimg():
    try:
        image_url = request.args.get('url', default='', type=str)
        img = Image.open(image_url).convert('RGB')
    except Exception as ex:
        return jsonify_str(create_query_result("", "", ex))
    result_text = model.get_content_image(img)
    result = {"result: ": result_text}
    return jsonify_str(result)

@app.route("/video", methods=['GET'])
def query():
    try:
        url = request.args.get('url', default='', type=str)
    except Exception as ex:
        return jsonify_str(ex)
    # vidcap = cv2.VideoCapture(url)
    # success, image = vidcap.read()
    # count = 0
    # while success:
    #     # cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
    #     success, image = vidcap.read()
    #     print('Read a new frame: ', success)
    #     print("Image: ", image)
    #     count += 1
    # print("Count: ", count)
    start = time.time()

    rs, totaltime = model.get_content_video(url)
    result = {"result": rs, "time": str(time.time() - start)}
    return jsonify_str(result)

app.run("localhost", 1709, threaded=False, debug=False)
