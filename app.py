import logging
from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import base64
import cv2
import numpy as np
from crowd_count import Counting
from matplotlib import cm as c
import matplotlib.pyplot as plt

config = {}
config["IMAGE_UPLOADS"] = "data"
config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("hello.html")


@app.route('/input')
def input():
    return render_template("input.html")


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        data = request.files["file1"]
        filename = secure_filename(data.filename)
        data.save(os.path.join(config["IMAGE_UPLOADS"], filename))
        path = os.path.join(config["IMAGE_UPLOADS"], filename)
        result = detect_persons(path)
        people_count = np.sum(result, dtype=np.float32)
        print("People count is ", people_count)
        if people_count == 0:
            faceDetected = False
            num_faces = 0
            to_send = ''
        else:
            faceDetected = True
            num_faces = int(people_count)
            # In memory
            image = plt.imshow(result, cmap=c.jet)
            plt.savefig('data/output_image.jpg')
            image = cv2.imread('data/output_image.jpg')
            image_content = cv2.imencode('.jpg', image)[1].tostring()
            encoded_image = base64.encodestring(image_content)
            to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
            print("Final count is ", num_faces)
        return render_template('index.html', faceDetected=faceDetected, num_faces=num_faces, image_to_show=to_send,
                               init=True)


def detect_persons(img_path):
    model_file = "files/model_reduce_filter.json"
    weight_file = "files/model_weights_1_rmsprop.h5"
    obj = Counting(model_file, weight_file)
    result = obj.predict_img(img_path)
    return result


def draw_rectangle(img, rect):
    '''Draw a rectangle on the image'''
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)


if __name__ == '__main__':

    app.run(debug=True)
