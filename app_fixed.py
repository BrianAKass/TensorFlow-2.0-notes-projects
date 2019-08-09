# 1 import dependencies

import os
import requests
import numpy as np
import tensorflow as tf
import imageio 
#imsave is depreciated. will use imwite to compensate.
from flask import Flask, request, jsonify

# 2 Load pretrained model



# Load the model structure

with open("fashion_model_flask.json","r") as f:
    model_json = f.read()
    
model = tf.keras.models.model_from_json(model_json)

# Load model weights

model.load_weights("fashion_model_flask.h5")

# 3 Create flask API

# Defining the falsk Application

app = Flask(__name__)

# defining the classify_image function

@app.route("/api/v1/<string:img_name>", methods=["POST"])
def classify_image(img_name):
    
    upload_dir = "uploads/"
    
    image = imageio.imread(upload_dir + img_name)
    
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    prediction = model.predict([image.reshape(1,28*28)])
    
    return jsonify({"object_detcted":classes[np.argmax(prediction[0])]})

# Start the flask API and make predictions
app.env = 'development'   
if __name__ == '__main__':
    app.run(port=5000)