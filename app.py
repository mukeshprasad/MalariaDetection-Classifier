from cv2 import imread
import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)


# Model path
model_path = 'cells.h5'

# load model
model = load_model(model_path)


def convert_to_array(img):
    im = imread(img)
    img_ = Image.fromarray(im, 'RGB')
    image = img_.resize((50, 50))
    return np.array(image)


def get_cell_name(label):
    if label == 0:
        return "Parasitized"
    if label == 1:
        return "Uninfected"


def model_predict(image_path, model):
    # Steps here
    model = load_model(model_path)
    arr = convert_to_array(image_path)
    arr = arr / 255
    label = 1
    a = []
    a.append(arr)
    a = np.array(a)
    score = model.predict(a, verbose=1)
    # print(score)
    label_index = np.argmax(score)
    # print(label_index)
    acc = np.max(score)
    cell = get_cell_name(label_index)
    return str(cell) + ". The predicted cell is " + cell + " with accuracy = " + str(round(acc * 100, 2)) + '%'


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        # print(preds)
        os.remove(file_path)
        return result
    return None


if __name__ == '__main__':
    app.run()
