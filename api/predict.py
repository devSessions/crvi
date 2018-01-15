import os
import numpy as np
import keras
from PIL import Image
# import requests
# from io import BytesIO


class predict(object):
    def __init__(self,url):
        self.url=url

    def predict_only(self):
        # Load and resize the image using PIL.
        # response = requests.get(self.url)
        # img = Image.open(BytesIO(response.content))
        img = Image.open(self.url)

        # Set Input Shape
        input_shape = (224, 224)

        # img = PIL.Image.open(image_path)
        img_resized = img.resize(input_shape, Image.LANCZOS)

        # Convert the PIL image to a numpy-array with the proper shape.
        img_array = np.expand_dims(np.array(img_resized), axis=0)

        # Load Model from File
        model = keras.models.load_model('finetune_18.model')


        # Make Predictions
        pred = model.predict_classes(img_array)

        if (pred[0] == 0 ):
            #print("Rs.10")
            return ("Rs.10")
        elif (pred[0] == 1):
            #print("Rs.20")
            return ("Rs.20")
