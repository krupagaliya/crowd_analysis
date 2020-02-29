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

class Counting:

    def __init__(self, model_file, weight_file):
        self.mse = tf.keras.losses.MeanSquaredError(reduction=losses_utils.ReductionV2.SUM)
        self.mae = tf.keras.losses.MeanAbsoluteError(reduction=losses_utils.ReductionV2.SUM)

        self.json_file = open(model_file, 'r')
        self.loaded_model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = model_from_json(self.loaded_model_json)
        self.loaded_model.load_weights(weight_file)


    def create_img(self, path):
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
    def predict_img(self,img_path):
        input = self.create_img(img_path)
        x = np.asarray(input)
        input = np.expand_dims(x, axis=0)
        print(input.shape)
        output = self.loaded_model.predict(input)
        output_s = np.squeeze(output)
        return output_s

    def show_result(self, input_img):
        from matplotlib import cm as c
        import matplotlib.pyplot as plt
        plt.imshow(input_img, cmap=c.jet)
        count = np.sum(input_img, dtype=np.float32)
        print("Predicted Count:-", count)
        plt.savefig('output_image.png')
        plt.show()
        return count


if __name__ == '__main__':
    model_file = "files/model_reduce_filter.json"
    weight_file = "files/model_weights_1_rmsprop.h5"
    obj = Counting(model_file, weight_file)
    path = "data/IMG_20171020_083226809.jpg"
    result = obj.predict_img(path)
    obj.show_result(result)



