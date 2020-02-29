from keras.models import model_from_json
import os
import cv2
import glob
import h5py
import pandas as pd
import scipy.io as io
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils


mse = tf.keras.losses.MeanSquaredError(reduction=losses_utils.ReductionV2.SUM)
mae = tf.keras.losses.MeanAbsoluteError(reduction=losses_utils.ReductionV2.SUM)

json_file = open('files/model_reduce_filter.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("files/model_weights_1_rmsprop.h5")


def create_img(path):
    # Function to load,normalize and return image
    im = cv2.imread(path)
    im = np.array(im)
    im = im / 255.0
    # cv2.imwrite('data/IMG_20171020_083226809.jpg', im)

    im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
    im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
    im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225

    # print(im.shape)
    # im = np.expand_dims(im,axis  = 0)
    return im

path = "data/IMG_20171020_083226809.jpg"
input = create_img(path)
x = np.asarray(input)
input = np.expand_dims(x, axis=0)
print(input.shape)
output = loaded_model.predict(input)
output_s = np.squeeze(output)

from matplotlib import cm as c
import matplotlib.pyplot as plt
plt.imshow(output_s,cmap = c.jet)
print("Predicted Count:-",np.sum(output,dtype=np.float32))
plt.savefig('output_image.png')
plt.show()