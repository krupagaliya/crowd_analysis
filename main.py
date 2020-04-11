from flask import Flask, request
import os
from werkzeug.utils import secure_filename
from flask import jsonify
import base64
import cv2
import numpy as np
from controller import Counting
from matplotlib import cm as c
import matplotlib.pyplot as plt
from flask_ngrok import run_with_ngrok
from flask_cors import CORS

config = {}
config["IMAGE_UPLOADS"] = "data"
config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]

app = Flask(__name__)
run_with_ngrok(app)
CORS(app)


@app.route('/')
def hello_world():
    return {
        'done':'API for crowd counting by KSH'
    }


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        try:
            data = request.files["file"]
            filename = secure_filename(data.filename)
            data.save(os.path.join(config["IMAGE_UPLOADS"], filename))
            path = os.path.join(config["IMAGE_UPLOADS"], filename)
            result = detect_persons(path)
            people_count = np.sum(result, dtype=np.float32)
        except Exception as e:
            print(e)
            people_count = 100

        print("People count is ", people_count)
        if people_count == 0:
            faceDetected = False
            num_faces = 0
            to_send = ''
        else:
            faceDetected = True
            num_faces = int(people_count)
            # In memory
            plt.imshow(result, cmap=c.jet)
            plt.savefig('data/output_image.jpg')
            image = cv2.imread('data/output_image.jpg')
            image_content = cv2.imencode('.jpg', image)[1].tostring()
            encoded_image = base64.encodestring(image_content)
            to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
        print("Final count is ", num_faces)
        final_result = {
            'faceDetected': faceDetected,
            'num_faces':  num_faces,
            'image_to_show': to_send

        }
        return jsonify(final_result)


def detect_persons(img_path):
    model_file = "SAnet/SANet.json"
    weight_file = "SAnet/SANet_best.hdf5"

    # Inception model
    # model_file = "files/model_reduce_filter.json"
    # weight_file = "files/model_weights_1_rmsprop.h5"

    obj = Counting(model_file, weight_file)
    result = obj.predict_img(img_path)
    return result


if __name__ == '__main__':
    app.run()
