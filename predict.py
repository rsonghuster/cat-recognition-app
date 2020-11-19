import time, os
import tensorflow as tf
from skimage import io, transform
from scipy.misc import imresize
import numpy as np

start = time.time()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from keras.models import model_from_json
print("import keras time = ", time.time() - start)

model = None
# Getting model
with open('model/model.json', 'r') as f:
    model_content = f.read()
    model = model_from_json(model_content)
    # Getting weights
    model.load_weights("model/weights.h5")

graph = None
graph = tf.get_default_graph()


def predict(file_path):
    start = time.time()
    img_size = 64
    image = io.imread(file_path)

    if image.shape[2] == 4:  #ARGB
        tmp = transform.resize(image, (img_size, img_size, 3))
        img = imresize(tmp, (img_size, img_size, 3))
    else:
        img = imresize(image, (img_size, img_size, 3))

    X = np.zeros((1, 64, 64, 3), dtype='float64')
    X[0] = img

    global model, graph
    with graph.as_default():
        Y = model.predict(X)
        end = time.time()
        print("dog: {:.2}, cat: {:.2}; elapsedTime: {:.3} seconds".format(
            Y[0][1], Y[0][0], end - start))
        return float(Y[0][0])
