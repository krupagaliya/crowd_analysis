import logging

from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import cv2

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
        image = cv2.imread(os.path.join(config["IMAGE_UPLOADS"], filename))
        print(image.shape)
        res = image.shape
        # detect_person(image)
        return str(res)


if __name__ == '__main__':
    app.run(debug=True)
