import logging

from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import base64
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
        faces = detect_faces(image)

        if len(faces) == 0:
            faceDetected = False
            num_faces = 0
            to_send = ''
        else:
            faceDetected = True
            num_faces = len(faces)

            # Draw a rectangle
            for item in faces:
                draw_rectangle(image, item['rect'])

            # Save
            # cv2.imwrite(filename, image)

            # In memory
            image_content = cv2.imencode('.jpg', image)[1].tostring()
            encoded_image = base64.encodestring(image_content)
            to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
            print("imshow", type(to_send))
            print(to_send)

        return render_template('index.html', faceDetected=faceDetected, num_faces=num_faces, image_to_show=to_send,
                               init=True)


# ----------------------------------------------------------------------------------
# Detect faces using OpenCV
# ----------------------------------------------------------------------------------
def detect_faces(img):
    '''Detect face in an image'''

    faces_list = []

    # Convert the test image to gray scale (opencv face detector expects gray images)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load OpenCV face detector (LBP is faster)
    face_cascade = cv2.CascadeClassifier('files/lbpcascade_frontalface.xml')

    # Detect multiscale images (some images may be closer to camera than others)
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # If not face detected, return empty list
    if len(faces) == 0:
        return faces_list

    for i in range(0, len(faces)):
        (x, y, w, h) = faces[i]
        face_dict = {}
        face_dict['face'] = gray[y:y + w, x:x + h]
        face_dict['rect'] = faces[i]
        faces_list.append(face_dict)

    # Return the face image area and the face rectangle
    return faces_list


def draw_rectangle(img, rect):
    '''Draw a rectangle on the image'''
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)


if __name__ == '__main__':
    app.run(debug=True)
